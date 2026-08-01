"""
第五步：Rerank，对 RAG-Fusion 融合后的候选文档做真实语义相关性重排序。

思路：
RRF 融合排序只反映"多路检索排名的一致性"，不代表文档与原始问题的真实语义相关度
（例如某个变体 query 引入的噪声文档，也可能因为在某一路排名靠前而获得较高 RRF 分）。
用本地 cross-encoder（Flashrank）对 (query, document) 直接打相关性分数，
在融合候选集基础上二次精排并截断到最终 top_k，作为送入 LLM 上下文前的最后一道过滤。

仅用于跨书问答（multi_book）场景，与 fusion.py 配套使用：先 fuse_retrieval() 扩大召回，
再用 rerank_documents() 收窄到高精度的最终结果集。
"""
from __future__ import annotations

import logging
from functools import lru_cache

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_ranker() -> Ranker:
    settings = get_settings()
    return Ranker(model_name=settings.rerank_model)


def rerank_documents(query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
    """用 Flashrank 对候选文档按与 query 的真实相关性重新排序，截断到 top_n 条。

    documents 为空时直接返回空列表，避免空请求触发 Flashrank 内部报错。

    异常兜底：模型加载或 rerank 调用失败时（如模型文件缺失、内存不足）不应让
    整个检索链路失败，退化为不做重排、直接按融合排序截断前 top_n 条返回。
    """
    if not documents:
        return []

    settings = get_settings()
    top_n = top_n or settings.retrieval_top_k

    try:
        ranker = _get_ranker()
        passages = [
            {"id": idx, "text": doc.page_content} for idx, doc in enumerate(documents)
        ]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        ranked = sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]
        return [documents[r["id"]] for r in ranked]
    except Exception:
        logger.exception("Flashrank 重排序失败，退化为不重排，直接截断前 top_n 条")
        return documents[:top_n]
