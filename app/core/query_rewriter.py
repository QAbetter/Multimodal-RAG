"""
预检索优化：单书精读 query 改写 + 查询意图分类。

query 改写（单书场景）：
跨书问答已有 RAG-Fusion（多路 query 改写 + RRF 融合）。单书精读过去直接用原始 query
做检索，遇到"换用近义词提问"的场景（如用户问"核心思想"，但书中原文是"中心论点"）
会漏检相关段落。

改写策略：专门针对书籍场景——"换用书中可能出现的具体术语/说法"，而不是通用语义改写；
生成 2 个变体（比跨书场景的 multi_query_count 少，因为单书范围已收窄，减少 LLM 调用开销）。
LLM 调用失败时返回空列表，调用方降级为只用原始 query 检索。

意图分类：
- fact（事实检索型）：答案就在某个 chunk 里，如"XX 是谁""XX 发生在第几章"
  → top_k 用较小值（intent_fact_top_k=4），精度优先，减少无关 chunk 干扰 LLM 生成
- reasoning（归纳推理型）：需要跨多章节综合，如"这本书的主旨是什么""分析 XX 的性格"
  → top_k 用较大值（intent_reasoning_top_k=10），召回优先，保证有足够的上下文覆盖
LLM 调用失败时返回 "fact"（保守兜底，使用较小 top_k，避免召回过多无关内容）。
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

IntentType = Literal["fact", "reasoning"]

_SINGLE_BOOK_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是书籍检索改写助手。用户正在阅读《{book_title}》，请针对用户的问题，"
            "生成 2 个检索改写版本，改写时优先换用书中可能出现的专有术语、人名、章节名等具体表达，"
            "而不是泛化的近义词。每行一个，不要编号、不要解释，不要输出原始问题本身。",
        ),
        ("human", "{query}"),
    ]
)

_INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "判断用户的问题属于哪种检索意图，只输出单个单词：\n"
            "- fact：事实型，答案在书中某个具体段落，如人名、时间、事件、引用原文\n"
            "- reasoning：归纳型，需要综合多处内容，如主旨分析、人物评价、风格比较\n"
            "只输出 fact 或 reasoning，不要输出其他任何内容。",
        ),
        ("human", "{query}"),
    ]
)


def rewrite_query_for_single_book(query: str, book_title: str, llm) -> list[str]:
    """为单书精读生成 2 个检索改写变体。

    LLM 调用失败时返回空列表，调用方降级为只用原始 query 检索，不影响正常流程。
    """
    try:
        chain = _SINGLE_BOOK_REWRITE_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"query": query, "book_title": book_title})
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        return variants[:2]
    except Exception:
        logger.exception("单书 query 改写失败，退化为仅使用原始 query")
        return []


def classify_query_intent(query: str, llm) -> IntentType:
    """判断 query 是事实检索型还是归纳推理型，决定检索 top_k。

    LLM 调用失败时返回 "fact"（保守兜底：使用较小 top_k，避免过多无关 chunk 干扰生成）。
    """
    try:
        chain = _INTENT_CLASSIFY_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()
        return "reasoning" if result == "reasoning" else "fact"
    except Exception:
        logger.exception("查询意图分类失败，退化为 fact 类型")
        return "fact"
