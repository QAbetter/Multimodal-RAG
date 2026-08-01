"""
/chat 路由：接入语义路由，支持单书精读 + 跨书问答两种模式。

工程性优化：单轮问答（无 session_id）命中查询缓存时直接返回，跳过检索与 LLM 调用；
未命中则正常走检索链路，并把结果写入缓存供后续相同问题复用。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.indexer import get_book
from app.core.query_cache import get_cached_answer, set_cached_answer
from app.core.retriever import ask_multi_book, ask_single_book
from app.core.router import route_query
from app.models.schemas import BookStatus, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    route_result = route_query(request.query, request.book_id)

    cached = get_cached_answer(route_result.route, route_result.book_id, request.query, request.session_id)
    if cached is not None:
        return ChatResponse(
            answer=cached["answer"],
            sources=cached["sources"],
            route=route_result.route,
            answer_quality=cached.get("answer_quality", "ok"),
        )

    if route_result.route == "multi_book":
        result = ask_multi_book(request.query, request.session_id)
        set_cached_answer(route_result.route, route_result.book_id, request.query, request.session_id, result)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            route=route_result.route,
            answer_quality=result.get("answer_quality", "ok"),
        )

    book_id = route_result.book_id
    book = get_book(book_id)
    if book is None:
        raise HTTPException(status_code=404, detail=f"书籍不存在: {book_id}")
    if book.status != BookStatus.READY:
        raise HTTPException(status_code=409, detail=f"书籍尚未索引完成，当前状态: {book.status}")

    result = ask_single_book(request.query, book_id, request.session_id)
    set_cached_answer(route_result.route, book_id, request.query, request.session_id, result)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        route=route_result.route,
        answer_quality=result.get("answer_quality", "ok"),
    )
