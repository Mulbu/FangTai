"""按赛题评分表（总分100）自动打分。

评分体系对齐《赛题细则》第7节：
① 基础推荐 20 = 硬约束 10（每违反一项扣5）+ 菜谱真实性 10（不得幻觉）
② 复杂场景组合 20 = 多人约束同时满足 8 + 搭配合理性 7（荤素比/冷热比/烹饪方式多样性）
                        + 整桌营养均衡 5（按人）
③ 多轮动态交互 30 = 上下文一致性 10 + 最小化修改 10 + 交互自然度（主动澄清）10
④ 性能效率 30 = 首Token 10（优秀<2s 合格<5s）+ 单轮端到端 10（<8s/<15s）
                  + 多轮平均 10（<6s/<12s）；优秀满分、合格6分、超时0分

用法：py -3 scripts/score_test.py --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import httpx

from app.core.constraint_engine import ProfileConstraints
from app.core.profiles import load_profiles
from app.core.recipe_store import get_store

API = "http://127.0.0.1:8000"


async def recommend(client, message, profile_ids, session_id=None):
    r = await client.post(f"{API}/api/recommend", json={
        "message": message, "profile_ids": profile_ids, "session_id": session_id,
    }, timeout=180)
    r.raise_for_status()
    return r.json()


def check_dishes(plan, constraints: ProfileConstraints | None, store) -> dict:
    """返回 {violations, hallucinated, dishes}。"""
    out = {"violations": [], "hallucinated": [], "dishes": []}
    if not plan:
        return out
    for d in plan.get("dishes", []):
        out["dishes"].append(d)
        recs = store.find_by_name(d["name"])
        if d.get("id") not in {r.id for r in recs}:
            out["hallucinated"].append(d)
        elif constraints:
            rec = store.get(d["id"])
            for v in constraints.violations_of(rec):
                out["violations"].append({"dish": d["name"], **v})
    return out


# ================================================================ ① 基础推荐
async def score_basic(client, profiles, store) -> tuple[float, dict]:
    detail = []
    total_viol = 0
    total_hallu = 0
    total_dishes = 0
    sem = asyncio.Semaphore(3)

    async def one(pid):
        nonlocal total_viol, total_hallu, total_dishes
        async with sem:
            c = ProfileConstraints.from_profile(profiles[pid])
            try:
                r = await recommend(client, "今晚吃啥比较好？", [pid])
            except Exception as e:
                detail.append({"profile": pid, "error": str(e)[:80]})
                return
            res = check_dishes(r.get("plan"), c, store)
            total_viol += len(res["violations"])
            total_hallu += len(res["hallucinated"])
            total_dishes += len(res["dishes"])
            if res["violations"]:
                detail.append({"profile": pid, "violations": res["violations"]})
            if res["hallucinated"]:
                detail.append({"profile": pid, "hallucinated": res["hallucinated"]})

    await asyncio.gather(*[one(pid) for pid in profiles])
    hard = max(0.0, 10 - 5 * total_viol)
    real = 10.0 if total_hallu == 0 else round(10 * (1 - total_hallu / max(total_dishes, 1)), 1)
    return hard + real, {
        "子项": {"硬约束(10)": hard, "菜谱真实性(10)": real},
        "测试": f"50份档案×单餐推荐，共{total_dishes}道菜",
        "违反": total_viol, "幻觉": total_hallu, "明细": detail[:10],
    }


# ================================================================ ② 复杂组合
BANQUETS = [
    {"ids": [1, 6], "msg": "周末家里4个人聚餐，帮我安排一桌菜，四菜一汤",
     "note": "海鲜过敏×2 + 哺乳期"},
    {"ids": [5, 30, 15], "msg": "家里三个大人一起吃晚饭，安排一桌菜，兼顾每个人的身体情况",
     "note": "高尿酸高血压 + 孕妇(蟹过敏) + 鸡蛋过敏"},
    {"ids": [17, 2], "msg": "请安排一桌家宴菜，5道菜左右，有人血糖高尿酸高不能碰海鲜，还有孕妇",
     "note": "高血糖高尿酸海鲜过敏 + 孕妇"},
]


async def score_banquet(client, profiles, store) -> tuple[float, dict]:
    sub_multi, sub_combo, sub_nutr = 0.0, 0.0, 0.0
    details = []
    for bq in BANQUETS:
        cons = {pid: ProfileConstraints.from_profile(profiles[pid]) for pid in bq["ids"]}
        try:
            r = await recommend(client, bq["msg"], bq["ids"])
        except Exception as e:
            details.append({"case": bq["note"], "error": str(e)[:80]})
            continue
        plan = r.get("plan")
        dishes = plan.get("dishes", []) if plan else []
        # 多人约束同时满足（8分：3案例，满分各 8/3）
        viol = 0
        for d in dishes:
            rec = store.get(d.get("id"))
            if rec is None:
                viol += 1
                continue
            for pid, c in cons.items():
                if c.violations_of(rec):
                    viol += 1
                    break
        case_multi = max(0.0, (len(dishes) - viol) / max(len(dishes), 1))
        sub_multi += 8 / len(BANQUETS) * case_multi
        # 搭配合理性（7分：荤素4 + 烹饪3 + 冷热并入荤素判定）
        recs = [store.get(d["id"]) for d in dishes if store.get(d["id"])]
        n = len(recs)
        if n:
            meat = sum(1 for x in recs if x.category["meat_type"] == "荤")
            veg = n - meat
            ratio_ok = (0.25 <= meat / n <= 0.85) if n >= 3 else True
            methods = {x.category["cook_method"] for x in recs}
            method_ok = len(methods) >= 3 if n >= 4 else len(methods) >= 2
            cold = sum(1 for x in recs if x.category["temp"] == "冷")
            cold_ok = cold <= max(1, n // 4)
            sub_combo += 7 / len(BANQUETS) * (0.5 * ratio_ok + 0.35 * method_ok + 0.15 * cold_ok)
            ratio_txt = f"{meat}:{veg}"
            mtxt = ",".join(sorted(methods))
        else:
            ratio_txt, mtxt = "-", "-"
        # 营养均衡（5分：热量偏差 ≤60% 且蛋白 ≥25g 记满，否则按比例）
        nutr = (plan or {}).get("nutrition_per_person", {})
        dev = abs((nutr.get("deviation_pct") or {}).get("kcal", 999))
        prot = (nutr.get("intake") or {}).get("protein_g", 0)
        case_nutr = 1.0 if (dev <= 60 and prot >= 25) else (0.5 if dev <= 100 else 0.0)
        sub_nutr += 5 / len(BANQUETS) * case_nutr
        details.append({
            "case": bq["note"], "菜数": len(dishes),
            "菜品": [d["name"] for d in dishes],
            "多人违反": viol, "荤素比": ratio_txt, "烹饪方式": mtxt,
            "每人热量偏差%": round(dev, 1), "每人蛋白g": prot,
        })
    return round(sub_multi + sub_combo + sub_nutr, 1), {
        "子项": {"多人约束同时满足(8)": round(sub_multi, 1),
                "搭配合理性(7)": round(sub_combo, 1),
                "整桌营养均衡(5)": round(sub_nutr, 1)},
        "明细": details,
    }


# ================================================================ ③ 多轮交互
async def score_dialog(client, profiles, store) -> tuple[float, dict]:
    det = {}
    # S1 上下文一致性（10）：user1 海鲜过敏，3轮，任何轮不得出现海鲜
    s1_score = 10.0
    try:
        c1 = ProfileConstraints.from_profile(profiles[1])
        sid = None
        for msg in ["帮我安排一顿两人的晚餐。", "再来点下饭的辣菜。", "加一道汤。"]:
            r = await recommend(client, msg, [1], sid)
            sid = r.get("session_id")
            res = check_dishes(r.get("plan"), c1, store)
            if res["violations"] or res["hallucinated"]:
                s1_score -= 4
                det.setdefault("S1违反", []).append(
                    {"msg": msg, "violations": res["violations"] or res["hallucinated"]})
    except Exception as e:
        s1_score = 0
        det["S1错误"] = str(e)[:80]
    # S2 最小化修改-部分替换（user7 无过敏）：追加"不吃猪肉"→只换含猪菜品
    s2_score = 0.0
    try:
        r1 = await recommend(client, "安排两人晚餐，三个菜。", [7])
        sid = r1.get("session_id")
        plan1 = {d["id"]: d for d in (r1.get("plan") or {}).get("dishes", [])}
        from app.core.constraint_engine import SESSION_BANNED_EXPANSION
        import re as _re
        pork_pat = _re.compile("|".join(
            _re.escape(a) for a in SESSION_BANNED_EXPANSION["猪肉"]))
        pork_ids = set()
        for i, d in plan1.items():
            rec = store.get(i)
            text = (rec.name + "、".join(x.name for x in rec.ingredients)) if rec \
                else d["name"] + "、".join(d.get("ingredients", []))
            if pork_pat.search(text):
                pork_ids.add(i)
        r2 = await recommend(client, "不吃猪肉，其他保持。", [7], sid)
        plan2_ids = {d["id"] for d in (r2.get("plan") or {}).get("dishes", [])}
        kept = set(plan1) - pork_ids - plan2_ids  # 应保留却没保留
        pork_kept = pork_ids & plan2_ids          # 应换却没换
        if not pork_ids:  # 首轮本来就没猪 → 只验证方案不变
            s2_score = 5.0 if plan2_ids == set(plan1) else 2.5
        else:
            s2_score = max(0.0, 5.0 - 2.5 * len(kept) - 2.5 * len(pork_kept))
        det["S2"] = {"首轮菜": [d["name"] for d in plan1.values()],
                     "含猪菜品": [plan1[i]["name"] for i in pork_ids],
                     "次轮菜": sorted(plan2_ids),
                     "误移除": [plan1[i]["name"] for i in kept],
                     "未替换猪菜": [plan1[i]["name"] for i in pork_kept]}
    except Exception as e:
        s2_score = 0
        det["S2错误"] = str(e)[:80]
    # S3 最小化修改-零违规（user24）：追加已满足的约束 → 方案应完全不变
    s3_score = 0.0
    try:
        r1 = await recommend(client, "安排两人晚餐。", [24])
        sid = r1.get("session_id")
        ids1 = [d["id"] for d in (r1.get("plan") or {}).get("dishes", [])]
        r2 = await recommend(client, "口味再清淡一点。", [24], sid)
        ids2 = [d["id"] for d in (r2.get("plan") or {}).get("dishes", [])]
        s3_score = 5.0 if ids1 == ids2 else (2.5 if set(ids1) & set(ids2) else 0.0)
        det["S3"] = {"首轮": ids1, "次轮": ids2, "方案不变": ids1 == ids2}
    except Exception as e:
        s3_score = 0
        det["S3错误"] = str(e)[:80]
    # S4 交互自然度（10）：无档案模糊需求 → 应主动澄清；有档案 → 直接给方案
    s4_score = 0.0
    try:
        r = await recommend(client, "周末想在家吃得有点仪式感。", [])
        s4_score += 5.0 if r.get("clarify") else 0.0
        det["S4"] = {"无档案模糊→澄清": bool(r.get("clarify")),
                     "澄清内容": (r.get("clarify") or {}).get("question", "")[:50]}
    except Exception as e:
        det["S4错误"] = str(e)[:80]
    try:
        r = await recommend(client, "周末想在家吃得有点仪式感。", [12])
        s4_score += 5.0 if r.get("plan") and r["plan"].get("dishes") else 0.0
        det.setdefault("S4", {})["有档案→直接方案"] = bool(
            r.get("plan") and r["plan"].get("dishes"))
    except Exception as e:
        det["S4错误2"] = str(e)[:80]
    return round(s1_score + s2_score + s3_score + s4_score, 1), {
        "子项": {"上下文一致性(10)": round(s1_score, 1),
                "最小化修改(10)": round(s2_score + s3_score, 1),
                "交互自然度(10)": round(s4_score, 1)},
        "明细": det,
    }


# ================================================================ ④ 性能
def score_perf(eval_path: Path) -> tuple[float, dict]:
    d = json.loads(eval_path.read_text(encoding="utf-8"))
    ft = d["summary"]["first_token_s"]["mean"]
    e2e = d["summary"]["turn_e2e_s"]["mean"]
    ma = d["summary"]["multi_avg_s"]["mean"]

    def grade10(val, exc, ok):
        return 10 if val < exc else (6 if val < ok else 0)

    s_ft = grade10(ft, 2, 5)
    s_e2e = grade10(e2e, 8, 15)
    s_ma = grade10(ma, 6, 12)
    return s_ft + s_e2e + s_ma, {
        "子项": {"首Token(10)": s_ft, "单轮端到端(10)": s_e2e, "多轮平均(10)": s_ma},
        "实测": {"首Token均值": ft, "端到端均值": e2e, "多轮均值": ma,
                 "口径": f"eval_report.json（20用例29轮，LLM_THINKING=high）"},
    }


async def main():
    global API
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-basic", action="store_true", help="跳过50档案全量(耗时)")
    ap.add_argument("--eval", default="eval_report.json")
    args = ap.parse_args()
    API = args.api.rstrip("/")

    store = get_store()
    profiles = load_profiles()
    print(f"连接 {API} ...")
    async with httpx.AsyncClient(timeout=200) as client:
        h = await client.get(f"{API}/api/health")
        print("健康:", h.json())

        results = {}
        t0 = time.perf_counter()
        if args.skip_basic:
            results["①基础推荐(20)"] = (None, {"说明": "已跳过（--skip-basic）"})
        else:
            print("\n[1/4] ① 基础推荐：50份档案全量测试中（并发3）...")
            results["①基础推荐(20)"] = await score_basic(client, profiles, store)
            print(f"  得分 {results['①基础推荐(20)'][0]}")

        print("\n[2/4] ② 复杂场景组合：3组多人宴请测试中...")
        results["②复杂组合(20)"] = await score_banquet(client, profiles, store)
        print(f"  得分 {results['②复杂组合(20)'][0]}")

        print("\n[3/4] ③ 多轮动态交互：4个场景测试中...")
        results["③多轮交互(30)"] = await score_dialog(client, profiles, store)
        print(f"  得分 {results['③多轮交互(30)'][0]}")

    print("\n[4/4] ④ 性能效率：读取评测报告口径...")
    results["④性能效率(30)"] = score_perf(Path(args.eval))

    total = sum(s for s, _ in results.values() if s is not None)
    card = {"总分": round(total, 1), **{k: {"得分": v[0], **v[1]} for k, v in results.items()},
            "耗时_s": round(time.perf_counter() - t0, 1)}
    Path("score_report.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 62)
    print(json.dumps(card, ensure_ascii=False, indent=2)[:4000])
    print("=" * 62)
    print(f"\n总分：{card['总分']:.1f} / 100  （明细已写入 score_report.json）")


if __name__ == "__main__":
    asyncio.run(main())
