"""
工程索引层优化第 2 项之一：进程内 BM25 索引。

BM25 是关键词稀疏检索算法，与向量检索互补：
- 向量检索：语义相似，对近义词/改写表达召回好，对精确词汇区分度弱
- BM25：关键词精确匹配，对人名/书名/专有名词/数字等精确词召回强

按 book_id 分组存储各书的独立索引，支持单书精读（只搜一本书）和跨书问答（跨所有书搜索）。

生命周期：
- 进程内内存结构，服务重启后需通过 warm_up() 重建（由 app/main.py 在 startup 事件里调用）
- 书籍重新索引后通过 build_bm25_index() 覆盖旧索引
- 书籍删除时通过 delete_bm25_index() 清理

分词策略：
- 中文：jieba 精确模式分词（按词语而非字切分，提升词语粒度的 BM25 召回）
- 英文/其他：按空格切词
"""
from __future__ import annotations

import logging
from threading import Lock

from langchain_core.documents import Document

from app.models.schemas import BookStatus

logger = logging.getLogger(__name__)

# book_id -> (corpus_tokens, documents)
_bm25_indexes: dict[str, tuple[list[list[str]], list[Document]]] = {}
_lock = Lock()


def _tokenize(text: str) -> list[str]:
    """分词：中文用 jieba 精确模式，英文/混合按空格兜底。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return text.split()


def build_bm25_index(book_id: str, documents: list[Document]) -> None:
    """为指定书籍构建/更新 BM25 索引，在 indexer.py 完成向量库写入后调用。"""
    if not documents:
        return
    corpus_tokens = [_tokenize(doc.page_content) for doc in documents]
    with _lock:
        _bm25_indexes[book_id] = (corpus_tokens, documents)
    logger.info("BM25 索引构建完成: %s，共 %d 个 chunk", book_id, len(documents))


def delete_bm25_index(book_id: str) -> None:
    """书籍删除或重索引开始前清理旧 BM25 索引。"""
    with _lock:
        _bm25_indexes.pop(book_id, None)


def search_bm25(book_id: str | None, query: str, top_k: int) -> list[Document]:
    """BM25 检索。

    book_id 不为 None 时只搜指定书籍（单书精读）；
    为 None 时跨所有已索引书籍合并搜索（跨书问答）。
    未命中或索引尚未构建时返回空列表，不抛出异常。
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 未安装，BM25 检索跳过")
        return []

    query_tokens = _tokenize(query)

    with _lock:
        if book_id is not None:
            entry = _bm25_indexes.get(book_id)
            if entry is None:
                return []
            corpus_tokens, documents = entry
            return _search_one(BM25Okapi, corpus_tokens, documents, query_tokens, top_k)

        # 跨书：合并所有书的语料后统一检索，保留跨书相对排名
        all_corpus: list[list[str]] = []
        all_docs: list[Document] = []
        for ct, docs in _bm25_indexes.values():
            all_corpus.extend(ct)
            all_docs.extend(docs)

    if not all_corpus:
        return []
    return _search_one(BM25Okapi, all_corpus, all_docs, query_tokens, top_k)


def _search_one(BM25Okapi, corpus_tokens, documents, query_tokens, top_k):
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices if scores[i] > 0]


def warm_up() -> None:
    """服务启动时调用，遍历所有 READY 书籍重建 BM25 索引，恢复重启前的状态。"""
    from app.core.indexer import load_registered_books
    from app.core.loader import split_book_into_documents

    books = load_registered_books()
    for book in books.values():
        if book.status != BookStatus.READY:
            continue
        if book.book_id in _bm25_indexes:
            continue
        try:
            docs = split_book_into_documents(book)
            build_bm25_index(book.book_id, docs)
        except Exception:
            logger.exception("BM25 warm-up 失败: %s，跳过该书", book.book_id)
