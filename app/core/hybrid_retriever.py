"""
工程索引层优化第 2 项之二：向量检索 + BM25 混合检索，RRF 融合两路结果。

动机：
- 纯向量检索对精确词汇（人名、书名、数字）区分度不足
- BM25 对精确词强但无法做语义近似匹配
- 两者 RRF 融合后互补，召回质量优于任意单路

实现约定：
- 混合检索器实现标准的 invoke(query) 接口，可直接替换现有 vectorstore.as_retriever()
- RRF 融合复用 fusion.py 中的 _doc_key / _RRF_K，保持一致的去重和评分逻辑
- 候选量：向量和 BM25 各取 top_k * 2（扩大召回后融合截断）
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.core.bm25_store import search_bm25
from app.core.fusion import _RRF_K, _doc_key


class HybridRetriever:
    """向量 + BM25 混合检索器，实现 invoke(query) 接口，可作为 LangChain retriever 使用。

    直接接收两个子检索器（向量端已在外部按候选量构造），invoke 时并行调两路，RRF 融合后返回。
    """

    def __init__(self, vector_retriever, book_id: str | None, top_k: int, candidate_k: int) -> None:
        self._vector_retriever = vector_retriever
        self._book_id = book_id
        self._top_k = top_k
        self._candidate_k = candidate_k

    def invoke(self, query: str) -> list[Document]:
        vector_docs: list[Document] = []
        try:
            vector_docs = self._vector_retriever.invoke(query) or []
        except Exception:
            pass  # 向量检索失败时退化到只用 BM25

        bm25_docs = search_bm25(self._book_id, query, self._candidate_k)
        return _rrf_merge(vector_docs, bm25_docs, self._top_k)


def _rrf_merge(vector_docs: list[Document], bm25_docs: list[Document], top_k: int) -> list[Document]:
    """RRF 融合向量结果和 BM25 结果，去重后截断到 top_k。"""
    rrf_scores: dict[tuple, float] = {}
    doc_lookup: dict[tuple, Document] = {}

    for ranked_list in (vector_docs, bm25_docs):
        for rank, doc in enumerate(ranked_list):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            doc_lookup.setdefault(key, doc)

    fused_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    return [doc_lookup[key] for key in fused_keys]


def get_hybrid_book_retriever(book_id: str, top_k: int) -> HybridRetriever:
    """单书精读场景：向量限 book_id 范围 + BM25 限同一书。"""
    from app.core.vectorstore import book_filter, get_vectorstore
    candidate_k = top_k * 2
    vector_retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": candidate_k, "filter": book_filter(book_id)}
    )
    return HybridRetriever(vector_retriever, book_id=book_id, top_k=top_k, candidate_k=candidate_k)


def get_hybrid_multi_book_retriever(top_k: int) -> HybridRetriever:
    """跨书问答场景：向量全库检索 + BM25 跨所有书检索。"""
    from app.core.retriever import _get_fusion_base_retriever
    # 向量端复用 _get_fusion_base_retriever（已按 fusion_retrieval_k 取候选），BM25 端用 top_k * 2
    vector_retriever = _get_fusion_base_retriever()
    candidate_k = top_k * 2
    return HybridRetriever(vector_retriever, book_id=None, top_k=top_k, candidate_k=candidate_k)
