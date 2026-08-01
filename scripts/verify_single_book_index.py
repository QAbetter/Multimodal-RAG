"""
第一步验证脚本：跑通"单本书注册 -> 索引 -> 检索"的最小闭环。

用法：
    python scripts/verify_single_book_index.py data/raw/your_book.txt "书名"

前置条件：
    1. 已 pip install -r requirements.txt
    2. 已配置 .env（至少 OPENAI_API_KEY）
    3. 本地或远程已启动 Qdrant（默认 http://localhost:6333，
       本地最快方式：docker run -p 6333:6333 qdrant/qdrant，
       若暂不便用 docker，也可用 Qdrant Cloud 的免费实例替代，把 QDRANT_URL 指向云端即可）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.indexer import get_book, index_book, register_book
from app.core.vectorstore import book_filter, get_vectorstore
from app.models.schemas import BookFormat, BookMetadata


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: python scripts/verify_single_book_index.py <文件路径> <书名>")
        sys.exit(1)

    file_path = sys.argv[1]
    title = sys.argv[2]
    suffix = Path(file_path).suffix.lower().lstrip(".")
    fmt = {"pdf": BookFormat.PDF, "epub": BookFormat.EPUB, "txt": BookFormat.TXT}.get(suffix)
    if fmt is None:
        print(f"不支持的文件格式: {suffix}")
        sys.exit(1)

    book_id = Path(file_path).stem
    book = get_book(book_id)
    if book is None:
        book = BookMetadata(book_id=book_id, title=title, format=fmt, source_path=file_path)
        register_book(book)
        print(f"[1/3] 已注册书籍: {book_id}")
    else:
        print(f"[1/3] 书籍已注册，跳过: {book_id}")

    print("[2/3] 开始索引...")
    book = index_book(book_id)
    print(f"      索引完成，状态={book.status}，章节数={book.total_chapters}")

    print("[3/3] 用 book_id 过滤做一次检索验证...")
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(
        query=title,
        k=3,
        filter=book_filter(book_id),
    )
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"  [{i}] 章节={meta.get('chapter_title')} page={meta.get('page')} 内容片段={doc.page_content[:60]!r}")

    print("\n验证通过：单书索引与元数据过滤检索流程已跑通。")


if __name__ == "__main__":
    main()
