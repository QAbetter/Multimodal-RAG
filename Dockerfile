# 图片+书籍 RAG 服务镜像
# 多阶段构建：先用 builder 装 torch 等重依赖，再拷到运行镜像，减小最终体积
FROM python:3.11-slim AS builder

# 换阿里云 Debian 源（国内服务器拉 deb.debian.org 很慢）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# 系统依赖（Pillow/torch 编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# 分两步装依赖：torch 单独从 CPU 索引源装，避免拉下 2GB 的 CUDA 版
# PyPI 默认的 torch 是 CUDA 版本（约 2GB），CPU 版只有约 200MB
# 必须先装 torch CPU 版，再装其余依赖（pip 会识别已装版本并跳过）
# --retries/--timeout：国内访问 PyPI/pytorch.org 不稳定，增加重试避免单次超时失败
# --extra-index-url 清华源：torch 的依赖（networkx/jinja2 等）从国内镜像下载，避免 pythonhosted.org 超时
# torch 本身仍走 pytorch 官方 CPU 源（2.13.0+cpu 的 local version 高于清华源的 CUDA 版，pip 会选 CPU 版）
RUN pip install --no-cache-dir --prefix=/install \
    --retries 5 --timeout 300 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "torch>=2.0.0"

# 装其余依赖到独立 prefix，方便后续拷贝
# torch 已在上一步装好，pip 检测到版本满足 requirements.txt 会跳过，不会重装 CUDA 版
# 清华源加速国内下载，retries/timeout 防止超时
RUN pip install --no-cache-dir --prefix=/install \
    --retries 5 --timeout 300 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# ===== 运行镜像 =====
FROM python:3.11-slim

# 换阿里云 Debian 源（国内服务器拉 deb.debian.org 很慢）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# 运行时系统依赖（Pillow 运行需要 libjpeg，torch 运行需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 拷贝已装好的 Python 依赖
COPY --from=builder /install /usr/local

# 拷贝项目代码（tests/ 已在 .dockerignore 中排除，运行时不需要）
COPY app/ ./app/
COPY scripts/ ./scripts/

# 数据目录（运行时通过 volume 挂载持久化）
RUN mkdir -p data/raw data/processed data/images data/pdf data/chroma data/eval

EXPOSE 8000

# uvicorn 启动，host 0.0.0.0 让容器外可访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
