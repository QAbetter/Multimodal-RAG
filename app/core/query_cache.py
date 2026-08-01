"""
工程性优化第6项之一：查询结果缓存。

问题背景：相同/相似问题被重复问到时（如多个用户问同一本书的常见问题、
或前端重试请求），每次都要重新走一遍"检索 + LLM 生成"的完整链路，
既浪费向量检索算力，也重复消耗 LLM token、增加响应延迟。

方案：进程内 LRU + TTL 缓存，key 由 (route, book_id, 归一化后的 query) 构成。
只做精确匹配缓存（不做语义相似度缓存），原因：
- 精确匹配足以覆盖"同一问题被重复问"这个最常见、最高频的场景
- 语义相似度缓存需要额外一次 embedding 调用判断是否命中，
  在缓存本身要解决的"降低调用成本"目标上反而多引入一次调用，性价比不高，
  相似问题走一遍完整检索链路也能保证结果始终基于最新索引内容

不缓存多轮对话场景（session_id 不为空时跳过缓存）：LangGraph 的 MemorySaver
会话记忆机制依赖完整的对话历史来生成回答，同一句话在不同会话轮次上下文里
可能有不同含义，缓存会破坏这个语义，因此仅对无 session_id 的单轮问答做缓存。
"""
from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import NamedTuple

from app.core.config import get_settings
from app.core.router import RouteType


class _CacheKey(NamedTuple):
    route: RouteType
    book_id: str | None
    query: str


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _make_key(route: RouteType, book_id: str | None, query: str) -> _CacheKey:
    return _CacheKey(route, book_id, _normalize_query(query))


def _is_cacheable(session_id: str | None) -> bool:
    """多轮对话（有 session_id）不缓存；缓存功能关闭时也不缓存。"""
    return get_settings().query_cache_enabled and not session_id


class _TTLLRUCache:
    """线程安全的 LRU + TTL 缓存，容量和过期时间均可配置。"""

    def __init__(self, max_size: int, ttl_seconds: int) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[_CacheKey, tuple[float, dict]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: _CacheKey) -> dict | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: _CacheKey, value: dict) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + self._ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear_book(self, book_id: str) -> None:
        """某本书重新索引后，清掉所有涉及该 book_id 的缓存项（包括跨书问答缓存，因为跨书结果可能引用了这本书）。"""
        with self._lock:
            stale_keys = [key for key in self._store if key.book_id == book_id or key.route == "multi_book"]
            for key in stale_keys:
                del self._store[key]

    def clear_all(self) -> None:
        with self._lock:
            self._store.clear()


_cache_init_lock = Lock()
_cache: _TTLLRUCache | None = None


def _get_cache() -> _TTLLRUCache:
    global _cache
    if _cache is None:
        with _cache_init_lock:
            if _cache is None:
                settings = get_settings()
                _cache = _TTLLRUCache(settings.query_cache_max_size, settings.query_cache_ttl_seconds)
    return _cache


def get_cached_answer(route: RouteType, book_id: str | None, query: str, session_id: str | None) -> dict | None:
    if not _is_cacheable(session_id):
        return None
    return _get_cache().get(_make_key(route, book_id, query))


def set_cached_answer(route: RouteType, book_id: str | None, query: str, session_id: str | None, result: dict) -> None:
    if not _is_cacheable(session_id):
        return
    _get_cache().set(_make_key(route, book_id, query), result)


def invalidate_book_cache(book_id: str) -> None:
    """重新索引某本书后调用，避免继续返回基于旧索引内容生成的缓存答案。"""
    if _cache is not None:
        _cache.clear_book(book_id)
