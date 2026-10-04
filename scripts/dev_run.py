"""开发调试脚本：进程内跑单个对话流程，打印全部 SSE 事件与计时。"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


async def run(message: str, profile_ids: list[int], session=None, show_text=True):
    from app.core import dialog as dlg
    from app.core.agent import get_agent
    from app.core.profiles import get_profile

    agent = get_agent()
    if session is None:
        session = dlg.DialogSession()
    session.set_profiles(profile_ids, get_profile)

    t0 = time.perf_counter()
    first_tok = None
    plan = None
    text = []
    async for ev in agent.stream_chat(session, message):
        et = ev.get("type")
        if et == "delta":
            if first_tok is None:
                first_tok = time.perf_counter() - t0
            text.append(ev.get("text", ""))
        elif et == "plan":
            plan = ev
            print(f"  [plan +{(time.perf_counter()-t0):.2f}s] " + ", ".join(
                f"{d['name']}({d['role']})" for d in ev["dishes"]))
        elif et == "clarify":
            print(f"  [clarify] {ev['question']}")
        elif et == "intent":
            print(f"  [intent] {ev['slots']}")
    print(f"  首Token={first_tok and round(first_tok,2)}s 总耗时={time.perf_counter()-t0:.2f}s")
    if show_text and text:
        print("  正文:", "".join(text)[:400].replace(chr(10), " ⏎ "))
    return session, plan


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1"

    if which == "1":
        # 用例1：单轮基础推荐（user3 高血压+高血糖+花生过敏+清淡）
        await run("今晚吃啥比较好？", [3])
    elif which == "2":
        # 用例7：现有食材约束
        await run("家里现在就剩番茄、鸡蛋和土豆了，这顿饭还能怎么弄？要能当正餐。", [15])
    elif which == "3":
        # 用例15：多轮约束追加
        s, _ = await run("帮我想顿晚饭。", [16])
        await run("别做辣的，口味清淡一点。", [16], session=s)
    elif which == "4":
        # 用例14：四菜一汤多人
        await run("想做个四菜一汤，营养均衡一点的，小孩不吃辣，老人牙口不好", [24])
    elif which == "5":
        # 用例20：4轮复杂对话
        s, _ = await run("给我想一顿两个人的晚饭。", [50])
        await run("一个人想吃辣，一个人一点辣都不想碰。", [50], session=s)
        await run("最好大部分食材能共用，我不想分开做两套，主菜可以考虑鱼或者鸡翅。", [50], session=s)
        await run("然后整体别超过45分钟，太麻烦的不行。", [50], session=s)
    elif which == "6":
        # 用例12：四菜一汤
        await run("想做个四菜一汤，营养均衡一点的", [13])
    elif which == "7":
        # 宴请（无档案多人）
        await run("周末想请几个人来家里吃饭，大概六个人，稍微正式点，但别整得太难做。", [])


if __name__ == "__main__":
    asyncio.run(main())
