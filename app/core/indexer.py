"""
单书索引主流程：注册元数据 -> 加载分块 -> 写入向量库 -> 落盘 BookMetadata。

对应改造方案第一步的验证目标：
    index_book(book_metadata) 跑通后，能在向量库里查到带 book_id 的向量点，
    且 payload 里包含 chapter_title/page 等 metadata。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.loader import split_book_into_documents
from app.core.bm25_store import build_bm25_index, delete_bm25_index
from app.core.query_cache import invalidate_book_cache
from app.core.vectorstore import delete_book_vectors, get_vectorstore
from app.models.schemas import BookMetadata, BookStatus

logger = logging.getLogger(__name__)


def _books_registry_path() -> Path:
    settings = get_settings()
    path = Path(settings.processed_data_dir) / "books.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_registered_books() -> dict[str, BookMetadata]:
    path = _books_registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {book_id: BookMetadata(**data) for book_id, data in raw.items()}


def save_registered_books(books: dict[str, BookMetadata]) -> None:
    path = _books_registry_path()
    payload = {book_id: json.loads(book.model_dump_json()) for book_id, book in books.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def register_book(book: BookMetadata) -> BookMetadata:
    books = load_registered_books()
    books[book.book_id] = book
    save_registered_books(books)
    return book


def get_book(book_id: str) -> BookMetadata | None:
    return load_registered_books().get(book_id)


def index_book(book_id: str) -> BookMetadata:
    """对已注册的书籍执行索引：分块 -> embedding -> 写入原文向量库（单向量索引，第一步用）。"""
    books = load_registered_books()
    book = books.get(book_id)
    if book is None:
        raise ValueError(f"书籍未注册: {book_id}")

    book.status = BookStatus.INDEXING
    books[book_id] = book
    save_registered_books(books)

    try:
        delete_book_vectors(book_id)  # 重新索引前先清理旧向量，避免重复
        delete_bm25_index(book_id)    # 同步清理旧 BM25 索引
        documents = split_book_into_documents(book)
        if not documents:
            raise ValueError("解析结果为空，请检查源文件内容")

        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents)
        build_bm25_index(book_id, documents)  # 同步构建 BM25 索引，与向量库保持一致

        chapter_indices = {doc.metadata["chapter_index"] for doc in documents}
        book.total_chapters = len(chapter_indices)
        book.status = BookStatus.READY
        logger.info("书籍索引完成: %s，共 %d 个 chunk，%d 个章节", book_id, len(documents), book.total_chapters)
    except Exception:
        book.status = BookStatus.FAILED
        logger.exception("书籍索引失败: %s", book_id)
        raise
    finally:
        books[book_id] = book
        save_registered_books(books)
        invalidate_book_cache(book_id)  # 索引内容变化后，旧缓存答案可能已过时

    return book
