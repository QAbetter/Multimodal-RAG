"""
FastAPI 应用入口。第二步只挂载 /chat 路由；/books（书籍注册/索引管理）留给后续接口化，
当前阶段可先用 scripts/verify_single_book_index.py 里的 register_book/index_book 走通数据准备。

图片 RAG（第二周新增）：挂载 /image 路由 + 静态文件服务，与书籍 RAG 共存。
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import chat, dify, image
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时重建 BM25 索引（进程内内存结构，重启后需热身恢复）
    from app.core.bm25_store import warm_up
    warm_up()
    # 启动时重建图片 caption BM25 索引（PDF 提取的图片对应的文本索引）
    from app.core.image_bm25_store import warm_up as warm_up_image_bm25
    warm_up_image_bm25()
    # 确保图片存储目录存在
    settings = get_settings()
    Path(settings.image_storage_dir).mkdir(parents=True, exist_ok=True)
    # 确保 PDF 原始文件目录存在（API 上传和脚本扫描都用此目录）
    Path(settings.pdf_raw_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Book RAG", version="0.1.0", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(image.router)
app.include_router(dify.router)

# 静态文件服务：图片可通过 /images/{path} 访问（如 /images/raw/xxx.jpg）
settings = get_settings()
Path(settings.image_storage_dir).mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=settings.image_storage_dir), name="images")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
