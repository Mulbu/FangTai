"""组合规划器：单餐搭配与多人宴请整桌优化（评分项②）。

维度：
- 荤素比：按人数推荐 荤:素 ≈ 6:4（四菜一汤→2荤1素1蛋/豆+1汤）
- 冷热比：默认以热菜为主，允许至多 1 冷
- 烹饪方式多样性：蒸/炒/炖/烤/拌 至少 3 种
- 整桌营养均衡：按人汇总（nutrition.py），输出偏差报告
- 多人约束交集：constraint_engine.merge_constraints 已在检索前置过滤
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.nutrition import (
    Nutrition, recipe_nutrition, balance_report, adaptive_servings,
)
from app.core.recipe_store import Recipe, get_store

SOUP_PAT = __import__("re").compile(r"汤|羹|粥|煲")
STAPLE_PAT = __import__("re").compile(r"饭|面|粥|馒头|包子|饺子|馄饨|卷饼|意面|米线|年糕")


def is_soup(rec: Recipe) -> bool:
    return bool(SOUP_PAT.search(rec.name)) or rec.category["cook_method"] in ("炖", "煮")


def is_staple(rec: Recipe) -> bool:
    return bool(STAPLE_PAT.search(rec.name))


@dataclass
class ComboPlan:
    dishes: list[Recipe]
    roles: dict[int, str]
    diversity: dict
    nutrition_per_person: Nutrition
    balance: dict
    notes: list[str]


def default_dish_count(people: int, requested: int | None = None, meal: str | None = None) -> int:
    if requested:
        return min(requested, 12)
    if meal == "早餐":
        return 2 if people <= 2 else 3
    if people <= 2:
        return 2
    if people <= 4:
        return 3
    if people <= 6:
        return 5
    return people + 1  # 宴请按人数+1


def build_combo(
    candidates: list[Recipe],
    people: int,
    dish_count: int | None = None,
    soup_needed: bool | None = None,
    meal: str | None = None,
    labor: str = "中",
) -> ComboPlan:
    """规则组合优化：从 LLM 选菜/检索候选中构造结构合理的整桌方案。"""
    dish_count = dish_count or default_dish_count(people, meal=meal)
    want_soup = True if soup_needed is None else soup_needed
    if dish_count <= 2 and soup_needed is None:
        want_soup = meal != "早餐"

    picked: list[Recipe] = []
    roles: dict[int, str] = {}
    notes: list[str] = []
    pool = list(candidates)
    used_methods: set[str] = set()

    def take(rec: Recipe, role: str):
        if rec.id in roles:
            return False
        picked.append(rec)
        roles[rec.id] = role
        used_methods.add(rec.category["cook_method"])
        return True

    # 1) 汤
    soups = [r for r in pool if is_soup(r)]
    if want_soup:
        if soups:
            take(soups[0], "汤")
        else:
            want_soup = False
            notes.append("候选中没有合适的汤品，已调整为全菜组合")

    # 2) 荤菜（占剩余的 60%）
    rest = dish_count - len(picked)
    meat_n = max(1, round(rest * 0.6)) if rest > 0 else 0
    meats = [r for r in pool if r.category["meat_type"] == "荤" and not is_soup(r)]
    for r in meats[:meat_n]:
        take(r, "荤菜")

    # 3) 素菜 / 蛋豆
    veg_n = dish_count - len(picked)
    vegs = [r for r in pool if r.category["meat_type"] in ("素", "蛋", "豆制品")
            and not is_soup(r) and r.id not in roles]
    # 烹饪方式多样性优先：优先选未出现过的做法
    vegs.sort(key=lambda r: (r.category["cook_method"] in used_methods, r.category["temp"] == "冷"))
    for r in vegs[:veg_n]:
        role = "素菜" if r.category["meat_type"] == "素" else ("蛋类" if r.category["meat_type"] == "蛋" else "豆制品")
        take(r, role)

    # 4) 兜底补齐（不限荤素，优先多样性）
    for r in pool:
        if len(picked) >= dish_count:
            break
        if r.id not in roles:
            take(r, "汤" if is_soup(r) else "荤菜" if r.category["meat_type"] == "荤" else "素菜")

    # 5) 冷热比检查
    cold_n = sum(1 for r in picked if r.category["temp"] == "冷")
    if cold_n > max(1, dish_count // 4):
        notes.append(f"冷菜 {cold_n} 道，建议冷热比约 1:{max(1, dish_count - cold_n)}")

    # 6) 烹饪方式多样性
    methods = [r.category["cook_method"] for r in picked]
    if len(set(methods)) < min(3, len(methods)):
        notes.append(f"烹饪方式 {len(set(methods))} 种（{','.join(dict.fromkeys(methods))}），可再丰富")

    # 7) 营养（按人，体量大锅菜按更多份额折算）
    total = Nutrition()
    for r in picked:
        total.add(recipe_nutrition(r, servings=adaptive_servings(r, people)))
    balance = balance_report(total, labor)
    meat_cnt = sum(1 for r in picked if r.category["meat_type"] == "荤")
    veg_cnt = sum(1 for r in picked if r.category["meat_type"] in ("素", "蛋", "豆制品"))
    diversity = {
        "荤素比": f"{meat_cnt}:{veg_cnt}",
        "烹饪方式": list(dict.fromkeys(methods)),
        "冷热": f"{cold_n}冷:{len(picked) - cold_n}热",
        "含汤": want_soup and bool(soups and picked),
    }
    return ComboPlan(
        dishes=picked, roles=roles, diversity=diversity,
        nutrition_per_person=total, balance=balance, notes=notes,
    )


def diversity_penalty(candidates: list[Recipe]) -> float:
    """候选集合的多样性得分（用于检索后过滤 参考），越大越多样。"""
    if not candidates:
        return 0.0
    methods = len({r.category["cook_method"] for r in candidates})
    meats = len({r.category["meat_type"] for r in candidates})
    return methods * 0.5 + meats * 0.3
