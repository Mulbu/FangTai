"""REST/SSE 路由。"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from app.core import dialog as dlg
from app.core.agent import get_agent
from app.core.profiles import load_profiles, get_profile
from app.core.recipe_store import get_store

router = APIRouter(prefix="/api")

# 会话注册表（生产可换 Redis；评测单进程内存即可）
_sessions: dict[str, dlg.DialogSession] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    profile_ids: list[int] = Field(default_factory=list)
    reset: bool = False


class RecommendRequest(ChatRequest):
    """单次推荐：非流式，直接返回结构化方案+说明。"""


def _get_session(req: ChatRequest) -> dlg.DialogSession:
    if req.reset and req.session_id and req.session_id in _sessions:
        _sessions.pop(req.session_id, None)
    sid = req.session_id or dlg.uuid.uuid4().hex[:12]
    sess = _sessions.get(sid)
    if sess is None:
        sess = dlg.DialogSession(session_id=sid)
        _sessions[sid] = sess
    if req.profile_ids:
        current = set(sess.profile_ids)
        if current != set(req.profile_ids):
            sess.set_profiles(req.profile_ids, get_profile)
    elif not sess.profile_ids:
        sess.set_profiles([], get_profile)
    return sess


@router.post("/chat")
async def chat(req: ChatRequest):
    """多轮对话（SSE 流式）。事件类型：intent / plan / delta / clarify / done。"""
    sess = _get_session(req)
    agent = get_agent()

    async def gen():
        try:
            async for ev in agent.stream_chat(sess, req.message):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'session_id': sess.session_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "X-Session-Id": sess.session_id},
    )


@router.post("/recommend")
async def recommend(req: RecommendRequest):
    """单轮推荐（非流式 JSON）：便于程序化评测/其他应用集成。"""
    sess = _get_session(req)
    agent = get_agent()
    events = []
    t0 = time.perf_counter()
    async for ev in agent.stream_chat(sess, req.message):
        events.append(ev)
    plan = next((e for e in events if e.get("type") == "plan"), None)
    clarify = next((e for e in events if e.get("type") == "clarify"), None)
    text = "".join(e.get("text", "") for e in events if e.get("type") == "delta")
    return {
        "session_id": sess.session_id,
        "plan": plan,
        "clarify": clarify,
        "reply": text,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


@router.get("/recipes")
async def recipes(q: str = "", limit: int = 20, offset: int = 0):
    store = get_store()
    items = store.recipes
    if q:
        ql = q.lower()
        items = [r for r in store.recipes if ql in r.name.lower()
                 or any(ql in i.name for i in r.ingredients)]
    total = len(items)
    return {
        "total": total, "limit": limit, "offset": offset,
        "items": [r.to_dict() for r in items[offset: offset + limit]],
    }


@router.get("/recipes/{rid}")
async def recipe_detail(rid: int):
    rec = get_store().get(rid)
    if not rec:
        raise HTTPException(404, "recipe not found")
    return rec.to_dict()


@router.get("/profiles")
async def profiles(kind: str = "detailed"):
    data = load_profiles("detailed" if kind == "detailed" else "simple")
    return {"total": len(data), "items": list(data.values())}


@router.get("/sessions/{sid}")
async def session_state(sid: str):
    sess = _sessions.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    store = get_store()
    return {
        "session_id": sess.session_id,
        "profile_ids": sess.profile_ids,
        "meal": sess.meal,
        "people": sess.people,
        "time_limit_min": sess.time_limit_min,
        "constraints": sess.constraints.summary(),
        "current_plan": [
            {"id": d.recipe_id, "name": d.name, "role": d.role, "locked": d.locked}
            for d in sess.current_plan
        ],
        "history_count": len(sess.history),
    }


@router.get("/health")
async def health():
    from app.core.retriever import get_retriever
    t0 = time.perf_counter()
    try:
        ret = get_retriever()
        latency = round((time.perf_counter() - t0) * 1000)
        return {
            "status": "ok", "recipes": len(get_store().recipes),
            "rag_provider": ret._provider_label(),
            "vector": "ready" if ret._embed_ready else "bm25-only",
            "rerank": "on" if ret._reranker else "off",
            "index_load_ms": latency,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
