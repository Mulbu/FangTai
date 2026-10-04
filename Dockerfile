# 方太个性化膳食规划 Agent
# 云端 RAG 模式（默认）：无需 GPU/CUDA，轻量镜像
# 本地 GPU 模式：改用 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime 基础镜像，
#               并额外安装 requirements-local.txt（见 docs/部署文档.md）
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 业务代码与数据
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

RUN mkdir -p /app/index

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# 启动即预热（菜谱库/BM25/向量索引；云端 RAG 首次启动会调 API 构建索引后缓存）
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
