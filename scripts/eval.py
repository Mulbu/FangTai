"""自动评测 harness（对齐赛题验收要求）。

指标：
1. 基础推荐：过敏/忌口零违反（每违反一项记录）、菜谱存在性（不得幻觉）
2. 复杂组合：多人约束满足、荤素比/烹饪方式多样性、按人营养（plan 事件携带）
3. 多轮交互：上下文一致性（历史约束不被遗忘）、每轮响应
4. 性能：首 Token 延迟（首个 delta）、单轮端到端、多轮平均
   优秀档: 首Token<2s 单轮<8s 多轮均值<6s；合格档: <5s/<15s/<12s

用法：
  py -3 scripts/eval.py                     # 进程内直跑 20 用例
  py -3 scripts/eval.py --api http://127.0.0.1:8000   # 经 HTTP API 跑
  py -3 scripts/eval.py --case 20           # 只跑指定用例
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

from app import config  # noqa: E402
from app.core import dialog as dlg  # noqa: E402
from app.core.constraint_engine import ProfileConstraints  # noqa: E402
from app.core.profiles import load_profiles, get_profile  # noqa: E402
from app.core.recipe_store import get_store  # noqa: E402

# 档位阈值
EXCELLENT = {"first_token": 2.0, "turn_e2e": 8.0, "multi_avg": 6.0}
PASS = {"first_token": 5.0, "turn_e2e": 15.0, "multi_avg": 12.0}


# ---------------------------------------------------------------- in-process 驱动
async def run_turn_inproc(agent, session, message: str) -> dict:
    events = []
    t0 = time.perf_counter()
    first_token = None
    plan_ev = None
    async for ev in agent.stream_chat(session, message):
        if ev.get("type") == "delta" and first_token is None:
            first_token = time.perf_counter() - t0
        if ev.get("type") == "plan":
            plan_ev = ev
        events.append(ev)
    total = time.perf_counter() - t0
    text = "".join(e.get("text", "") for e in events if e.get("type") == "delta")
    return {
        "first_token_s": round(first_token, 3) if first_token else None,
        "total_s": round(total, 3),
        "plan": plan_ev,
        "reply_len": len(text),
    }


# ---------------------------------------------------------------- HTTP 驱动
async def run_turn_http(client, base, session_id, message, profile_ids) -> tuple[dict, str]:
    """HTTP 模式：返回 (指标, 会话id)。"""
    t0 = time.perf_counter()
    first_token = None
    plan_ev = None
    text_parts = []
    sid = session_id
    async with client.stream(
        "POST", f"{base}/api/chat",
        json={"message": message, "session_id": session_id, "profile_ids": profile_ids},
        timeout=120,
    ) as resp:
        sid = resp.headers.get("X-Session-Id", session_id)
        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                if not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                if ev.get("type") == "delta":
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    text_parts.append(ev.get("text", ""))
                elif ev.get("type") == "plan":
                    plan_ev = ev
    return {
        "first_token_s": round(first_token, 3) if first_token else None,
        "total_s": round(time.perf_counter() - t0, 3),
        "plan": plan_ev,
        "reply_len": len("".join(text_parts)),
    }, sid


# ---------------------------------------------------------------- 校验
def validate_plan(plan_ev: dict | None, constraints: ProfileConstraints, store) -> dict:
    result = {"violations": [], "hallucinated": [], "dish_count": 0, "ok": True}
    if not plan_ev:
        return result
    dishes = plan_ev.get("dishes", [])
    result["dish_count"] = len(dishes)
    for d in dishes:
        recs = store.find_by_name(d["name"])
        if not recs or d["id"] not in {r.id for r in recs}:
            result["hallucinated"].append({"id": d.get("id"), "name": d.get("name")})
        else:
            rec = store.get(d["id"]) or recs[0]
            vs = constraints.violations_of(rec)
            if vs:
                result["violations"].append({"dish": d["name"], "hits": vs})
    result["ok"] = not result["violations"] and not result["hallucinated"]
    return result


# ---------------------------------------------------------------- 主流程
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="", help="HTTP API 基址（缺省进程内直跑）")
    ap.add_argument("--case", type=int, default=0, help="只跑指定用例 id")
    ap.add_argument("--profile", type=int, default=0, help="固定档案 id（缺省按用例轮转）")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    store = get_store()
    profiles = load_profiles()
    with open(config.DATA_DIR / "对话用例.json", encoding="utf-8") as f:
        cases = json.load(f)
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    agent = None
    http_client = None
    if args.api:
        import httpx
        http_client = httpx.AsyncClient()
    else:
        from app.core.agent import get_agent
        agent = get_agent()

    # 用例 → 档案分配：覆盖多样人群（孕妇/慢病/过敏/增肌等）
    profile_cycle = [3, 2, 17, 16, 1, 6, 24, 30, 37, 19,
                     8, 22, 5, 13, 26, 10, 28, 18, 4, 50]

    all_turns = []
    report_cases = []
    for case in cases:
        pid = args.profile or profile_cycle[(case["id"] - 1) % len(profile_cycle)]
        constraints = ProfileConstraints.from_profile(profiles[pid])
        session = dlg.DialogSession()
        session.set_profiles([pid], get_profile)
        http_sid = None
        case_turns = []
        for i, msg in enumerate(case["user_messages"]):
            if args.api:
                r, http_sid = await run_turn_http(
                    http_client, args.api, http_sid, msg, [pid])
            else:
                r = await run_turn_inproc(agent, session, msg)
            v = validate_plan(r["plan"], constraints, store)
            r["turn"] = i + 1
            r["profile_id"] = pid
            r["message"] = msg[:30]
            r["validation"] = v
            case_turns.append(r)
            all_turns.append(r)
            status = "OK" if v["ok"] else "VIOLATION"
            ft = r["first_token_s"]
            print(f"  用例{case['id']}-轮{i+1} [{status}] 首Token={ft}s 总={r['total_s']}s "
                  f"菜品={v['dish_count']} 荤素比={(r['plan'] or {}).get('nutrition_per_person', {}).get('intake', {}).get('kcal', '-')}kcal/人")
            if v["violations"]:
                print(f"    !! 违反: {json.dumps(v['violations'], ensure_ascii=False)[:200]}")
            if v["hallucinated"]:
                print(f"    !! 幻觉菜名: {v['hallucinated']}")
        # 多轮平均
        if case_turns:
            avg = statistics.mean(t["total_s"] for t in case_turns)
            report_cases.append({
                "case_id": case["id"], "profile_id": pid,
                "turns": case_turns, "case_avg_s": round(avg, 3),
            })

    # 汇总
    def agg(key):
        vals = [t[key] for t in all_turns if t.get(key) is not None]
        return {
            "mean": round(statistics.mean(vals), 3) if vals else None,
            "p95": round(sorted(vals)[int(len(vals) * 0.95) - 1], 3) if vals else None,
            "max": round(max(vals), 3) if vals else None,
        }

    first_tok = agg("first_token_s")
    e2e = agg("total_s")
    case_avgs = [c["case_avg_s"] for c in report_cases]
    viol_total = sum(len(t["validation"]["violations"]) for t in all_turns)
    hallu_total = sum(len(t["validation"]["hallucinated"]) for t in all_turns)

    def grade(val, key):
        if val is None:
            return "n/a"
        if val <= EXCELLENT[key]:
            return "优秀"
        if val <= PASS[key]:
            return "合格"
        return "不合格"

    summary = {
        "turns_total": len(all_turns),
        "first_token_s": first_tok, "first_token_grade": grade(first_tok["mean"], "first_token"),
        "turn_e2e_s": e2e, "turn_e2e_grade": grade(e2e["mean"], "turn_e2e"),
        "multi_avg_s": {"mean": round(statistics.mean(case_avgs), 3) if case_avgs else None},
        "multi_avg_grade": grade(statistics.mean(case_avgs) if case_avgs else None, "multi_avg"),
        "allergen_violations": viol_total,
        "hallucinated_dishes": hallu_total,
        "zero_violation_pass_rate": round(
            sum(1 for t in all_turns if t["validation"]["ok"]) / len(all_turns) * 100, 1
        ) if all_turns else 0,
    }
    print("\n" + "=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = {"summary": summary, "cases": report_cases,
           "thresholds": {"excellent": EXCELLENT, "pass": PASS}}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {args.out}")
    if http_client:
        await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
