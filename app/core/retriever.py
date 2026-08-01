"""
检索层：单书精读 + 跨书问答的核心逻辑。

对应改造方案第二步 / 第三步：
- get_book_retriever(book_id)：基于 book_filter() 构造只在该书范围内检索的 retriever（单书精读）
- get_multi_book_retriever()：不加过滤，全库检索（跨书问答）
- ask_single_book / ask_multi_book：分别对应两种模式的问答入口，
  共用同一套 LangGraph MemorySaver 多轮记忆机制，session_id 作为 thread_id 区分会话

第四步（父子索引）：
- get_book_parent_child_retriever(book_id)：LangChain 官方 ParentDocumentRetriever，
  vectorstore 存子块（细粒度，检索精度高），docstore 存父块（大粒度，回表取完整上下文），
  二者通过官方内置的单一 doc_id 一对一关联，天然规避了手写 MultiVectorRetriever 时
  id_key 只支持单个字符串、不支持数组聚合的协议限制。

第五步（RAG-Fusion + Rerank，仅跨书问答场景接入）：
- _get_multi_book_graph_cached() 不再是单路 retriever.invoke()，而是先用
  fusion.fuse_retrieval() 做多路 query 检索 + RRF 融合扩大召回，再用
  rerank.rerank_documents() 做 Flashrank 精排截断到 retrieval_top_k。
  单书精读 / 父子索引场景检索范围已用 book_id 收窄，收益有限，保持原有单路检索不变。

v1.9 改造：
1. 修复多轮对话路径 bug：引入 SingleBookState 扩展 MessagesState，新增 injected_docs 字段；
   call_model 优先使用注入的 docs，有注入则跳过内部二次检索。ask_single_book 多轮分支
   将融合好的 docs 通过 injected_docs 注入 graph，不再出现检索两次的问题。
2. 第4项跨粒度去重：_dedup_by_parent() 在 RRF 融合后按"chapter_index + chunk_index//3"
   作为父块 key 去重，同父块只保留 RRF 分最高的一个子块，避免相邻子块重复占用 context。
3. 第5a 项单书精排：ask_single_book 在去重后调用 rerank_documents() 精排，与跨书场景对齐。
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from app.core.config import get_settings
from app.core.fusion import _doc_key, _RRF_K, fuse_retrieval
from app.core.rerank import rerank_documents
from app.core.vectorstore import book_filter, get_vectorstore
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph 状态：扩展 MessagesState，支持外部注入 docs 跳过内部检索
# ---------------------------------------------------------------------------

_NO_ANSWER_PATTERNS = [
    "无法从原文", "原文中没有", "找不到答案", "原文中找不到",
    "无法在原文", "没有相关内容", "cannot find", "no relevant",
    "not found in", "no information",
]


class SingleBookState(MessagesState):
    """单书精读图的状态。

    injected_docs：外部已检索好的文档列表（意图分类 + query改写 + RRF融合后的结果）。
    call_model 优先使用此字段，不为空时跳过内部检索，避免多轮对话下检索两次的问题。
    """
    injected_docs: Optional[list[Any]]

_SINGLE_BOOK_PROMPT = (
    "你是一名书籍精读助手，只能依据下面提供的书籍原文片段回答用户问题。\n"
    "如果片段中没有足够信息回答，请直接说明无法从原文中找到答案，不要编造内容。\n"
    "回答时可以引用章节标题帮助用户定位原文。\n\n"
    "书籍原文片段：\n{context}"
)

_MULTI_BOOK_PROMPT = (
    "你是一名跨书问答助手，下面提供的片段可能来自多本不同的书。\n"
    "只能依据这些片段回答用户问题，如果片段中没有足够信息回答，请直接说明无法找到答案，不要编造内容。\n"
    "回答时请明确指出信息分别来自哪本书（用书名标注），帮助用户区分不同书籍的观点或内容。\n\n"
    "书籍原文片段：\n{context}"
)


def _build_prompt(system_text: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("placeholder", "{messages}"),
        ]
    )


_single_book_prompt = _build_prompt(_SINGLE_BOOK_PROMPT)
_multi_book_prompt = _build_prompt(_MULTI_BOOK_PROMPT)


@lru_cache
def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key or None,       # 复用同一个智谱 Key
        base_url=settings.openai_base_url,              # 智谱的 base_url
    )


def _resolve_top_k(top_k: int | None) -> int:
    return top_k or get_settings().retrieval_top_k


def get_book_retriever(book_id: str, top_k: int | None = None):
    """返回只在指定 book_id 范围内检索的混合 retriever（向量 + BM25，单书精读用）。"""
    from app.core.hybrid_retriever import get_hybrid_book_retriever
    return get_hybrid_book_retriever(book_id, top_k=_resolve_top_k(top_k))


def _get_fusion_base_retriever():
    """RAG-Fusion 每路检索用的底层 retriever：不加 book_id 过滤，检索条数为 fusion_retrieval_k。

    fusion_retrieval_k 通常大于最终的 retrieval_top_k，因为融合 + rerank 之后还要二次收窄，
    初始召回阶段适当放宽候选量能降低漏检概率。
    """
    settings = get_settings()
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": settings.fusion_retrieval_k})


def retrieve_multi_book_with_fusion(query: str, llm) -> list:
    """跨书问答专用检索策略：RAG-Fusion 多路混合召回 + RRF 融合 -> Flashrank 精排截断。

    底层检索器已升级为 HybridRetriever（向量 + BM25），RAG-Fusion 在此基础上再做多路 query 扩展，
    三层召回增强：BM25 精确词 × 向量语义 × 多路 query 改写。

    fuse_retrieval() 与 rerank_documents() 内部已分别对"单路检索失败""LLM 改写失败"
    "Flashrank 重排失败"做了降级处理，保证两者本身不会向外抛出异常。
    """
    from app.core.hybrid_retriever import get_hybrid_multi_book_retriever
    settings = get_settings()
    hybrid_retriever = get_hybrid_multi_book_retriever(top_k=settings.fusion_retrieval_k)
    fused_docs = fuse_retrieval(query, hybrid_retriever, llm)
    return rerank_documents(query, fused_docs, top_n=settings.retrieval_top_k)


def _format_context(docs) -> str:
    blocks = []
    for doc in docs:
        meta = doc.metadata
        header = f"[章节: {meta.get('chapter_title') or '未知'}]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _docs_to_sources(docs) -> list[SourceChunk]:
    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append(
            SourceChunk(
                book_id=meta["book_id"],
                book_title=meta["book_title"],
                chapter_title=meta.get("chapter_title"),
                page=meta.get("page"),
                content=doc.page_content,
            )
        )
    return sources


_checkpointer = MemorySaver()


def _build_graph(retrieve_fn, prompt: ChatPromptTemplate):
    """构造多轮对话图（跨书问答用）：检索 -> 拼 prompt -> LLM。

    retrieve_fn(query, llm) -> list[Document]：检索策略以回调形式传入。
    跨书问答场景使用 RAG-Fusion + Rerank；不支持外部注入 docs（跨书无需注入）。
    """
    llm = get_llm()

    def call_model(state: MessagesState) -> dict:
        last_human = state["messages"][-1].content
        docs = retrieve_fn(last_human, llm)
        context = _format_context(docs)
        rendered = prompt.invoke({"context": context, "messages": state["messages"]})
        response = llm.invoke(rendered)
        return {"messages": [response], "sources": _docs_to_sources(docs)}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(checkpointer=_checkpointer)


def _build_single_book_graph(retrieve_fn, prompt: ChatPromptTemplate):
    """构造单书精读多轮对话图，使用 SingleBookState 支持外部注入 docs。

    当 state.injected_docs 不为空时，call_model 直接使用注入的文档，跳过内部检索；
    为空时（历史轮次对话）降级为 retrieve_fn 重新检索，保持多轮上下文一致性。
    """
    llm = get_llm()

    def call_model(state: SingleBookState) -> dict:
        last_human = state["messages"][-1].content
        # 优先用外部注入的 docs（当前轮次意图分类+改写+RRF融合后的结果）
        docs = state.get("injected_docs") or retrieve_fn(last_human, llm)
        context = _format_context(docs)
        rendered = prompt.invoke({"context": context, "messages": state["messages"]})
        response = llm.invoke(rendered)
        # injected_docs 置 None，不持久化到下一轮（下一轮用 retrieve_fn 重新检索）
        return {"messages": [response], "sources": _docs_to_sources(docs), "injected_docs": None}

    graph = StateGraph(SingleBookState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(checkpointer=_checkpointer)


@lru_cache
def _get_single_book_graph_cached(book_id: str):
    retriever = get_book_retriever(book_id)
    return _build_single_book_graph(lambda query, llm: retriever.invoke(query), _single_book_prompt)


@lru_cache
def _get_multi_book_graph_cached():
    return _build_graph(retrieve_multi_book_with_fusion, _multi_book_prompt)

def _invoke_graph(graph, query: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("human", query)]}, config=config)
    answer: BaseMessage = result["messages"][-1]
    sources = result.get("sources", [])
    return {"answer": answer.content, "sources": sources}


def _dedup_by_parent(rrf_scores: dict, doc_lookup: dict, top_k: int) -> list[Document]:
    """按父块 key 去重后截断到 top_k，同父块只保留 RRF 分最高的一个子块。

    父块 key = (chapter_index, chunk_index // 3)：每 3 个相邻 chunk 视作同一父块粒度。
    因为 rrf_scores 按降序遍历，第一次命中 parent_key 的子块分数一定最高，直接保留。
    """
    parent_best_doc_key: dict[tuple, Any] = {}

    for doc_key in sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True):
        doc = doc_lookup[doc_key]
        meta = doc.metadata
        parent_key = (meta.get("chapter_index", 0), meta.get("chunk_index", 0) // 3)
        if parent_key not in parent_best_doc_key:
            parent_best_doc_key[parent_key] = doc_key

    # 保持原始 RRF 降序，按 parent_best_doc_key 的插入顺序即可（Python 3.7+ dict 有序）
    return [doc_lookup[doc_key] for doc_key in list(parent_best_doc_key.values())[:top_k]]


def _check_answer_quality(answer: str) -> str:
    """轻量检测答案是否为"无法回答"类，返回 quality flag。

    ok：正常回答；no_answer：检测到"原文中找不到答案"类表达。
    只做字符串匹配，无 LLM 调用，几乎无额外延迟。
    """
    lower = answer.lower()
    if any(p in lower for p in _NO_ANSWER_PATTERNS):
        return "no_answer"
    return "ok"


def _safe_invoke(retriever, query: str):
    """单路检索，失败时返回空列表，不中断多路融合流程。"""
    try:
        return retriever.invoke(query) or []
    except Exception:
        logger.exception("单书多路检索失败，跳过该 query 变体: %r", query)
        return []


def ask_single_book(query: str, book_id: str, session_id: str | None = None) -> dict:
    """单书精读问答入口，返回 answer + sources + answer_quality。

    检索链路（v1.9）：
    1. 意图分类 + query 改写并发执行（各一次 LLM 调用，互不依赖）
    2. 多路并发混合检索 + RRF 融合
    3. 父块级去重（同父块只保留 RRF 分最高的子块）
    4. Flashrank 精排
    5. LLM 生成答案 + 答案质量检测

    多轮对话（session_id 不为空）：检索结果通过 injected_docs 注入 LangGraph 图，
    graph 内部直接使用注入的 docs，不再触发二次检索（修复 v1.8 的 bug）。
    """
    from app.core.hybrid_retriever import get_hybrid_book_retriever
    from app.core.indexer import load_registered_books
    from app.core.query_rewriter import classify_query_intent, rewrite_query_for_single_book

    llm = get_llm()
    settings = get_settings()

    # 1. 意图分类 + query 改写并发执行（两次 LLM 调用互不依赖，并发节省约一半延迟）
    books = load_registered_books()
    book = books.get(book_id)
    book_title = book.title if book else ""

    with ThreadPoolExecutor(max_workers=2) as pre_executor:
        intent_fut = pre_executor.submit(classify_query_intent, query, llm)
        variants_fut = pre_executor.submit(rewrite_query_for_single_book, query, book_title, llm)
        intent = intent_fut.result()
        variants = variants_fut.result()

    top_k = settings.intent_reasoning_top_k if intent == "reasoning" else settings.intent_fact_top_k
    queries = [query] + variants  # 改写失败时 variants=[]，退化为只用原始 query

    # 2. 多路混合检索 + RRF 融合
    candidate_k = top_k * 3  # 加宽候选量，给去重 + rerank 留充足空间
    retriever = get_hybrid_book_retriever(book_id, top_k=candidate_k)

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results = list(executor.map(lambda q: _safe_invoke(retriever, q), queries))

    rrf_scores: dict[Any, float] = {}
    doc_lookup: dict[Any, Document] = {}
    for ranked_docs in results:
        for rank, doc in enumerate(ranked_docs):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            doc_lookup.setdefault(key, doc)

    # 3. 父块级去重，截断到 top_k * 2 作为 rerank 候选
    deduped = _dedup_by_parent(rrf_scores, doc_lookup, top_k=top_k * 2)

    # 4. Flashrank 精排，截断到最终 top_k
    docs = rerank_documents(query, deduped, top_n=top_k)

    # 5. LLM 生成 + 答案质量检测
    if session_id:
        # 多轮对话：通过 injected_docs 注入检索结果，graph 不会二次检索
        graph = _get_single_book_graph_cached(book_id)
        config = {"configurable": {"thread_id": session_id}}
        result = graph.invoke(
            {"messages": [("human", query)], "injected_docs": docs},
            config=config,
        )
        answer: BaseMessage = result["messages"][-1]
        return {
            "answer": answer.content,
            "sources": _docs_to_sources(docs),
            "answer_quality": _check_answer_quality(answer.content),
        }

    context = _format_context(docs)
    rendered = _single_book_prompt.invoke({"context": context, "messages": [("human", query)]})
    response = llm.invoke(rendered)
    return {
        "answer": response.content,
        "sources": _docs_to_sources(docs),
        "answer_quality": _check_answer_quality(response.content),
    }


def ask_multi_book(query: str, session_id: str | None = None) -> dict:
    """跨书问答入口，全库检索，返回 answer + sources + answer_quality。

    多轮对话（session_id 不为空）：通过 LangGraph MemorySaver 保持历史上下文。
    单轮（session_id 为空）：生成独立的 thread_id，每次调用相互隔离，不会累积无关历史。
    """
    graph = _get_multi_book_graph_cached()
    thread_id = session_id or uuid.uuid4().hex
    result = _invoke_graph(graph, query, thread_id)
    result["answer_quality"] = _check_answer_quality(result["answer"])
    return result
