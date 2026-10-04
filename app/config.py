"""全局配置：全部来自环境变量（docker-compose 注入），不使用 .env 文件。"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "index"))

# ---------------- LLM（OpenAI 兼容协议，默认智谱 GLM-5.3） ----------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-5.3")
# 思考模式：disabled / enabled / low / high / max（GLM-5.x 常思考模型最低用 low）
LLM_THINKING = os.getenv("LLM_THINKING", "high")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
# 意图识别等轻量调用可用更小模型（留空则同 LLM_MODEL）
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "")

# ---------------- RAG 嵌入/重排（GPU 推理） ----------------
# A100-80G 档推荐: BAAI/bge-m3 + BAAI/bge-reranker-v2-m3
# A100-12G 档推荐: BAAI/bge-large-zh-v1.5 + （可留空禁用重排或用轻量重排）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto/cuda/cpu
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # 留空禁用
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
# bge 系列 retrieval 指令前缀
EMBED_QUERY_INSTRUCTION = os.getenv("EMBED_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关菜谱：")

# HF 镜像（国内环境加速）
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# ---------------- 检索参数 ----------------
RECALL_TOPK = int(os.getenv("RECALL_TOPK", "60"))       # 混合召回条数
RERANK_TOPK = int(os.getenv("RERANK_TOPK", "20"))       # 重排后送入 LLM 的候选数
FINAL_CANDIDATES = int(os.getenv("FINAL_CANDIDATES", "12"))  # 注入 prompt 的最终候选数

# ---------------- 服务 ----------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
