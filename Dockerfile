# 图片+书籍 RAG 服务镜像
# 多阶段构建：先用 builder 装 torch 等重依赖，再拷到运行镜像，减小最终体积
FROM python:3.11-slim AS builder

# 系统依赖（Pillow/torch 编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# 装到独立 prefix，方便后续拷贝
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== 运行镜像 =====
FROM python:3.11-slim

# 运行时系统依赖（Pillow 运行需要 libjpeg，torch 运行需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 拷贝已装好的 Python 依赖
COPY --from=builder /install /usr/local

# 拷贝项目代码
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# 数据目录（运行时通过 volume 挂载持久化）
RUN mkdir -p data/raw data/processed data/images data/pdf data/chroma data/eval

EXPOSE 8000

# uvicorn 启动，host 0.0.0.0 让容器外可访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
