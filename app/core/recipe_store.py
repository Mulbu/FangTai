"""菜谱库：CSV 加载、标签清洗规范化、食材结构化解析、荤素/烹饪方式/冷热分类。"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Optional

from app import config

# ---------------------------------------------------------------- 标签体系
# 43 个规范标签（按分析报告归并后的高频核心词表）+ 同义词映射
CANONICAL_TAGS = {
    "餐次": ["早餐", "午餐", "晚餐", "下午茶"],
    "菜系": ["川湘菜", "粤菜", "江浙菜", "东北菜", "西餐风味", "地方风味"],
    "口味": ["清淡", "甜", "咸", "辣", "微辣", "酸", "香", "鲜", "咸香", "咸鲜",
            "蒜香", "奶香", "酱香", "豉香", "香脆", "苦"],
    "节日": ["春节", "除夕", "元宵", "清明节", "端午", "七夕", "中秋节", "重阳",
            "感恩节", "圣诞节", "腊八", "冬至", "情人节", "复活节"],
    "人群": ["儿童", "上班族", "老人", "更年期", "哺乳期"],
    "功效": ["健胃消食", "养胃", "减脂", "改善便秘", "补血", "助眠"],
}

TAG_SYNONYMS = {
    "中秋节": "中秋节", "中秋": "中秋节",
    "湘菜": "川湘菜", "湘菜风味": "川湘菜", "川菜": "川湘菜",
    "台菜": "地方风味", "台湾风味": "地方风味", "台式风味": "地方风味",
    "闽菜": "地方风味", "福建菜": "地方风味", "鲁菜": "地方风味",
    "西北风味": "地方风味", "华北风味": "地方风味", "客家风味": "地方风味",
    "韩式风味": "地方风味", "韩式": "地方风味", "日式风味": "地方风味",
    "海鲜风味": "地方风味", "海鲜菜": "地方风味", "海鲜": "地方风味",
    "中式面点": "地方风味", "中式点心": "地方风味", "家常菜": "地方风味",
    "家常风味": "地方风味", "小吃": "地方风味", "甜品风味": "地方风味",
    "甜点": "地方风味", "甜品": "地方风味",
    "鲜美": "鲜", "鲜香": "鲜", "香辣": "辣", "麻辣": "辣", "微微辣": "微辣",
    "清爽": "清淡", "清新": "清淡", "原味": "清淡", "软糯": "清淡",
    "哺乳": "哺乳期",
    "贫血": "补血", "便秘": "改善便秘",
    "椒盐": "香脆", "脆": "香脆",
}

# 噪声标签特征（LLM 推理文本 / JSON 碎片）
_NOISE_TAG_PAT = re.compile(
    r"根据|标签|建议|true|false|\{|\}|【|】|菜谱|我们|分析|\":\"|^\s*$|^\d+$"
)


def parse_label_field(label_str: str) -> dict[str, list[str]]:
    """将 label 原始字段解析为六维规范标签。"""
    result = {dim: [] for dim in CANONICAL_TAGS}
    if not label_str:
        return result
    valid = set()
    for tags in CANONICAL_TAGS.values():
        valid.update(tags)
    for raw in re.split(r"[、，,；;\s]+", str(label_str).strip()):
        raw = raw.strip()
        if not raw or _NOISE_TAG_PAT.search(raw):
            continue
        tag = TAG_SYNONYMS.get(raw, raw)
        if tag in valid:
            for dim, dim_tags in CANONICAL_TAGS.items():
                if tag in dim_tags and tag not in result[dim]:
                    result[dim].append(tag)
    return result


# ---------------------------------------------------------------- 食材解析
_GROUP_PAT = re.compile(r"^(主料|辅料|[A-Z]料)[:：]")
_QTY_PAT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(克|g|G|千克|kg|KG|公斤|斤|毫升|ml|ML|mL|L|"
    r"个|根|片|瓣|只|条|颗|张|勺|包|串|段|块|把|碗|杯|滴|份|盒|罐|袋|小勺|大勺)(?![\dA-Za-z])"
)
_VAGUE_QTY = {"适量", "少许", "若干", "少量"}

# 常见计数单位食材的默认克重（每单位，克）
_DEFAULT_UNIT_GRAMS = {"鸡蛋": 50, "蛋清": 33, "蛋黄": 17, "鹌鹑蛋": 10}


@dataclass
class Ingredient:
    name: str
    grams: Optional[float] = None  # None = 适量等未知量
    unit: str = ""                  # 克/g 时为空串归一，其余保留计数单位
    count: Optional[float] = None   # 计数单位数量
    group: str = "主料"


@dataclass
class Recipe:
    id: int
    name: str
    ingredients: list[Ingredient] = field(default_factory=list)
    ingredient_text: str = ""        # 原始食材清单（清洗括号后）
    steps: list[str] = field(default_factory=list)
    tags: dict[str, list[str]] = field(default_factory=dict)
    category: dict[str, str] = field(default_factory=dict)  # meat_type/cook_method/temp

    @property
    def tag_list(self) -> list[str]:
        return [t for ts in self.tags.values() for t in ts]

    def brief(self, with_steps: bool = False) -> str:
        ing = "、".join(dict.fromkeys(i.name for i in self.ingredients)) or self.ingredient_text
        tags = "、".join(self.tag_list)
        s = f"[{self.id}] {self.name}｜食材：{ing}｜标签：{tags}"
        if with_steps and self.steps:
            s += "｜做法：" + "；".join(self.steps)
        return s

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "ingredients": [
                {"name": i.name, "grams": i.grams, "count": i.count, "unit": i.unit, "group": i.group}
                for i in self.ingredients
            ],
            "steps": self.steps,
            "tags": self.tags,
            "category": self.category,
        }


# 荤素判定关键词（按主料克重最大者判定）
_MEAT_PAT = re.compile(
    r"猪|牛|羊|鸡|鸭|鹅|排骨|火腿|腊肉|培根|咸肉|肉末|肉馅|五花肉|里脊|蹄|牛筋|肥牛|"
    r"鱼|虾|蟹|蛤|花甲|蛏|扇贝|鲍鱼|龙虾|牡蛎|海蛎|鱿鱼|墨鱼|章鱼|带鱼|黄鱼|鲈鱼|鳕鱼|"
    r"三文鱼|金枪鱼|龙利鱼|巴沙鱼|银鱼|泥鳅|黄鳝|田鸡|腊肉|香肠|午餐肉|肉丸"
)
_EGG_PAT = re.compile(r"鸡蛋|蛋清|蛋黄|鹌鹑蛋|蛋液|蛋")
_TOFU_PAT = re.compile(r"豆腐|香干|豆干|千张|腐竹|豆皮|素鸡")

# 烹饪方式（按步骤文本命中优先级）
_COOK_METHODS = [
    ("蒸", re.compile(r"蒸箱|蒸锅|蒸盘|清蒸|蒸制|上汽蒸|放入蒸|隔水蒸|蒸[0-9]*分钟|普通蒸")),
    ("烤", re.compile(r"烤箱|烤盘|烧烤模式|烤制|烘烤|烘焙|烤[0-9]+分钟|空气炸")),
    ("炒", re.compile(r"炒锅|煸炒|翻炒|炒香|爆炒|热锅|下锅炒|炒[0-9]*分钟")),
    ("炸", re.compile(r"油炸|入油锅|炸至| deep|油煎炸")),
    ("煎", re.compile(r"煎至|平底锅煎|少油煎|煎[0-9]*分钟")),
    ("炖", re.compile(r"炖|煲|煮汤|熬汤|煨")),
    ("煮", re.compile(r"煮熟|煮沸|焯水|下面|煮面|煮开|水煮|白灼|涮")),
    ("拌", re.compile(r"凉拌|拌匀|调制|混合搅拌|沙拉|蘸")),
    ("榨汁/打浆", re.compile(r"榨汁|打浆|破壁|搅拌机|料理机")),
]
_COLD_PAT = re.compile(r"凉拌|冷盘|冰镇|冷藏后食|奶昔|冰淇淋|沙拉|凉菜")
_MIN_PAT = re.compile(r"(\d+)\s*分钟")
_HOUR_PAT = re.compile(r"(\d+)\s*小时")


def estimate_minutes(rec: Recipe) -> int:
    """从步骤文本估算烹饪总时长（分钟，粗估：取步骤分钟数之和与最大值的折中）。"""
    text = "".join(rec.steps) + rec.name
    mins = [int(m) for m in _MIN_PAT.findall(text)]
    hours = [int(h) * 60 for h in _HOUR_PAT.findall(text)]
    vals = mins + hours
    if not vals:
        return 30  # 无时间信息按默认
    return min(sum(vals), max(max(vals) * 2, sum(vals) // 2))


def classify_recipe(rec: Recipe) -> dict[str, str]:
    """荤素 / 烹饪方式 / 冷热 分类，供搭配多样性计算。"""
    # 荤素：取克重最大的主料判定
    mains = [i for i in rec.ingredients if i.group == "主料"]
    pool = mains or rec.ingredients
    def weight(i: Ingredient) -> float:
        if i.grams:
            return i.grams
        if i.count:
            return i.count * _DEFAULT_UNIT_GRAMS.get(i.name, 60)
        return 0
    pool = [i for i in pool if weight(i) > 0] or pool
    top = max(pool, key=weight) if pool else None
    if top and _MEAT_PAT.search(top.name):
        meat_type = "荤"
    elif top and _EGG_PAT.search(top.name):
        meat_type = "蛋"
    elif top and _TOFU_PAT.search(top.name):
        meat_type = "豆制品"
    elif _MEAT_PAT.search(rec.name):
        meat_type = "荤"
    else:
        meat_type = "素"
    # 烹饪方式
    text = rec.name + "".join(rec.steps)
    method = "其他"
    for name, pat in _COOK_METHODS:
        if pat.search(text):
            method = name
            break
    temp = "冷" if _COLD_PAT.search(rec.name + rec.ingredient_text[:50]) else "热"
    return {"meat_type": meat_type, "cook_method": method, "temp": temp}


def _clean_parens(s: str) -> str:
    return s.replace("（（", "（").replace("））", "）")


def parse_ingredients(text: str) -> list[Ingredient]:
    """解析食材清单为结构化列表。"""
    ingredients: list[Ingredient] = []
    if not text:
        return ingredients
    text = _clean_parens(str(text))
    group = "主料"
    for seg in re.split(r"[；;]", text):
        seg = re.sub(r"（[^）]*）|\([^)]*\)", "", seg).strip()
        if not seg:
            continue
        m = _GROUP_PAT.match(seg)
        if m:
            group = m.group(1)
            seg = seg[m.end():].strip()
            if not seg:
                continue
        # 去掉可能残留的前缀分隔
        seg = re.sub(r"^[:：、\s]+", "", seg)
        qty_m = _QTY_PAT.search(seg)
        ing = Ingredient(name=seg, group=group)
        if qty_m:
            num, unit = float(qty_m.group(1)), qty_m.group(2).lower()
            name_part = seg[: qty_m.start()].strip("、，, ")
            if name_part:
                ing.name = name_part
                if unit in ("克", "g"):
                    ing.grams = num
                elif unit in ("千克", "kg", "公斤"):
                    ing.grams = num * 1000
                elif unit == "斤":
                    ing.grams = num * 500
                elif unit in ("毫升", "ml", "l"):
                    ing.grams = num * (1000 if unit == "l" else 1)
                else:
                    ing.count, ing.unit = num, unit
        elif any(seg.startswith(v) or seg.endswith(v) for v in _VAGUE_QTY):
            for v in _VAGUE_QTY:
                seg = seg.replace(v, "")
            ing = Ingredient(name=seg.strip() or seg, group=group)
        ing.name = re.sub(r"[（）()\s]+$|^[（）()\s]+", "", ing.name).strip()
        if ing.name:
            ingredients.append(ing)
    return ingredients


def parse_steps(text: str) -> list[str]:
    if not text:
        return []
    text = _clean_parens(str(text))
    parts = re.split(r"第\d+步[：:]?", text)
    steps = [re.sub(r"^\d+[.、]\s*", "", p).strip() for p in parts]
    return [s for s in steps if s]


class RecipeStore:
    def __init__(self):
        self.recipes: list[Recipe] = []
        self.by_id: dict[int, Recipe] = {}
        self.by_name: dict[str, list[Recipe]] = {}

    # ---------- 加载 ----------
    def load(self, csv_path=None) -> "RecipeStore":
        path = csv_path or (config.DATA_DIR / "recipes_sample_2000.csv")
        with open(path, encoding="gb18030", newline="") as f:
            rows = list(csv.DictReader(f))
        for idx, row in enumerate(rows):
            name = (row.get("名称") or "").strip()
            if not name:
                continue
            rec = Recipe(
                id=idx,
                name=name,
                ingredients=parse_ingredients(row.get("食材清单", "")),
                ingredient_text=_clean_parens(row.get("食材清单", "")),
                steps=parse_steps(row.get("烹饪步骤", "")),
                tags=parse_label_field(row.get("label", "")),
            )
            rec.category = classify_recipe(rec)
            self.recipes.append(rec)
            self.by_id[rec.id] = rec
            self.by_name.setdefault(rec.name, []).append(rec)
        return self

    # ---------- 查询 ----------
    def get(self, rid: int) -> Optional[Recipe]:
        return self.by_id.get(rid)

    def find_by_name(self, name: str) -> list[Recipe]:
        """精确名匹配；找不到再做去空白匹配。"""
        name = name.strip()
        if name in self.by_name:
            return self.by_name[name]
        compact = re.sub(r"\s+", "", name)
        return [r for r in self.recipes if re.sub(r"\s+", "", r.name) == compact]

    def exists(self, name: str) -> bool:
        return bool(self.find_by_name(name))

    # ---------- 过滤 ----------
    def filter_ids(self, banned_patterns: re.Pattern | None) -> set[int]:
        """返回食材文本不命中违禁模式的菜谱 id 集合。"""
        if banned_patterns is None:
            return set(self.by_id.keys())
        ok = set()
        for rec in self.recipes:
            text = rec.name + "｜" + "、".join(i.name for i in rec.ingredients)
            if not banned_patterns.search(text):
                ok.add(rec.id)
        return ok

    def stats(self) -> dict:
        from collections import Counter
        return {
            "total": len(self.recipes),
            "tagged": sum(1 for r in self.recipes if r.tag_list),
            "meat_type": dict(Counter(r.category["meat_type"] for r in self.recipes)),
            "cook_method": dict(Counter(r.category["cook_method"] for r in self.recipes)),
        }


_store: Optional[RecipeStore] = None


def get_store() -> RecipeStore:
    global _store
    if _store is None:
        _store = RecipeStore().load()
    return _store
