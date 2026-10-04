# 方太个性化膳食规划 Agent
# 基础镜像自带 CUDA 12.1 + PyTorch；评测服务器 A100 直接可用
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Python 依赖（torch 由基础镜像提供）
COPY requirements.txt .
RUN grep -vE "^\s*torch" requirements.txt > /tmp/req-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/req-no-torch.txt

# 业务代码与数据
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# 预建向量索引可在启动时自动完成（GPU）；此处创建缓存目录
RUN mkdir -p /app/index /models

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# 启动即预热（加载菜谱/索引/模型），保证首个请求低延迟
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
