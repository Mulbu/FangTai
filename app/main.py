"""FastAPI 入口：启动时预加载菜谱库/索引/模型（保证首请求低延迟）。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import config
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 预热：菜谱库 + 检索索引（含 GPU 模型加载）
    from app.core.recipe_store import get_store
    from app.core.retriever import get_retriever
    store = get_store()
    print(f"[startup] 菜谱库 {len(store.recipes)} 条")
    ret = get_retriever()
    ret.search("清淡的晚餐", topk=3)  # 首次编码预热 CUDA
    print("[startup] 就绪")
    yield


app = FastAPI(title="方太个性化膳食规划 Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
async def index():
    web = Path(__file__).parent / "web" / "index.html"
    return FileResponse(web)
