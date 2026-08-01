"""
第五步：RAG-Fusion，提升跨书问答场景的检索召回质量。

思路：
1. 用 LLM 把用户原始问题改写成 N 个语义相近但表述不同的变体 query
   （例如换用词、换角度提问），弥补单一 query 可能遗漏的相关文档。
2. 对原始 query + 每个变体 query 分别做一次向量检索，得到多路结果列表。
3. 用 RRF（Reciprocal Rank Fusion）算法融合多路排名：
   每个文档在每一路结果中的排名贡献 1/(k+rank) 分数，按 doc_id 累加后重新排序。
   RRF 是 RAG-Fusion 论文提出的标准融合算法，不依赖各路检索分数的绝对数值
   （不同路的相似度分数量纲可能不一致，用排名融合比直接加权分数更稳健）。

仅用于跨书问答（multi_book）场景：跨书检索的查询意图更宽泛、候选书籍更多，
多路召回能显著降低"因为用词不同而漏检某本书相关内容"的概率；单书精读场景
范围已经用 book_id 收窄，多路 query 收益有限，暂不接入。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_QUERY_VARIANTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是检索查询改写助手。请针对用户的问题，生成 {count} 个语义相近但表述不同的"
            "改写版本，用于扩大向量检索的召回范围。每行一个改写结果，不要编号、不要解释，"
            "不要输出原始问题本身。",
        ),
        ("human", "{query}"),
    ]
)

# RRF 融合常数：越大则不同排名位置之间的分数差异越平滑，60 是 RAG-Fusion 论文的常用默认值
_RRF_K = 60


def _generate_query_variants(query: str, llm) -> list[str]:
    """调用 LLM 生成 multi_query_count 个改写变体，解析失败或空结果时容错为空列表。

    异常兜底：LLM 调用本身失败（网络问题、限流、网关故障等）不应让整个
    RAG-Fusion 流程失败，退化为只用原始 query 检索（返回空变体列表）。
    """
    settings = get_settings()
    try:
        chain = _QUERY_VARIANTS_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"query": query, "count": settings.multi_query_count})
    except Exception:
        logger.exception("生成查询改写变体失败，退化为仅使用原始 query 检索")
        return []
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    return variants[: settings.multi_query_count]


def _doc_key(doc: Document) -> tuple:
    """用 (book_id, page_content) 作为文档去重/融合的唯一键。

    向量检索返回的 Document 没有稳定 id 字段，同一段原文可能在不同 query
    路次中被重复命中，用内容本身做 key 足以在单次请求的融合窗口内去重。
    """
    return (doc.metadata.get("book_id"), doc.page_content)


def _retrieve_one_query(retriever, query: str) -> list[Document]:
    """单路检索，失败时记录日志并返回空列表，不让整个融合流程失败。"""
    try:
        return retriever.invoke(query)
    except Exception:
        logger.exception("RAG-Fusion 单路检索失败，跳过该 query 变体: %r", query)
        return []


def fuse_retrieval(query: str, retriever, llm) -> list[Document]:
    """RAG-Fusion 主流程：多路 query 并发检索 + RRF 融合排序，返回融合后的文档列表（降序）。

    各路 query 之间互相独立、无数据依赖，用线程池并发检索以降低整体检索延迟
    （原始 query + multi_query_count 个改写变体，默认共 4 路，串行时延迟接近串行 4 次的总和）。

    异常兜底：单路检索失败（如某个改写 query 触发向量库瞬时异常）不应让整个
    融合流程失败，该路结果记为空列表跳过；只要原始 query 那一路能正常返回，
    最终结果集依然可用。
    """
    queries = [query] + _generate_query_variants(query, llm)

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results = executor.map(lambda q: _retrieve_one_query(retriever, q), queries)

    rrf_scores: dict[tuple, float] = {}
    doc_lookup: dict[tuple, Document] = {}

    for ranked_docs in results:
        for rank, doc in enumerate(ranked_docs):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            doc_lookup.setdefault(key, doc)

    fused_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [doc_lookup[key] for key in fused_keys]
