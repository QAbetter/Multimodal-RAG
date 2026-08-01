"""
语义路由：判断一次问答该走"单书精读"还是"跨书问答"。

策略（对应改造方案第三步，按当前书籍量级 <10 本，先用最简单的方式，不引入
额外的 semantic-router 依赖，避免过度设计）：
1. 如果请求已显式传入 book_id，直接判定 single_book，不需要路由。
2. 否则，先尝试从用户问题文本里模糊匹配已注册书籍的书名/book_id——
   命中则判定为 single_book，并回填匹配到的 book_id。
   注意：若问题里同时提到 ≥2 本书（如跨书比较类问题），判定为 multi_book，
   避免"先命中即返回"把跨书问题误路由成单书精读。
3. 模糊匹配不命中时，判定为 multi_book（跨书问答，全库检索）。

书籍数量增长到 10-30 本后，可以在这一层换成关键词 Layer0 + semantic-router
的分层方案（历史经验中记录过该规模的选型建议），当前阶段无需引入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouteType = Literal["single_book", "multi_book"]


@dataclass
class RouteResult:
    route: RouteType
    book_id: str | None = None


def route_query(query: str, book_id: str | None) -> RouteResult:
    if book_id:
        return RouteResult(route="single_book", book_id=book_id)

    matched = _match_book_by_title(query)
    if matched:
        return RouteResult(route="single_book", book_id=matched)

    return RouteResult(route="multi_book", book_id=None)


def _match_book_by_title(query: str) -> str | None:
    """在已注册书籍中查找标题或 book_id 是否出现在用户问题里。

    若命中 ≥2 本书（跨书比较类问题，如"《A》和《B》有什么区别"），返回 None
    让调用方路由到 multi_book，避免先命中即返回导致的误路由。
    """
    # 延迟导入以打破 indexer -> query_cache -> router -> indexer 的循环导入
    from app.core.indexer import load_registered_books

    books = load_registered_books()
    matched = [
        book.book_id
        for book in books.values()
        if (book.title and book.title in query) or book.book_id in query
    ]
    return matched[0] if len(matched) == 1 else None
