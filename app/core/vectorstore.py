"""
Chroma 向量库封装（本地持久化模式，无需单独起服务进程）。

改造点（对应方案中"单 Collection + metadata 过滤"策略）：
- 所有书籍共用一个 collection，靠 metadata 中的 book_id 字段做租户隔离与检索过滤
- 提供 book_id 过滤器构造函数，供 retriever.py 在"单书精读"模式下使用

注：Chroma 与 Qdrant 都是 LangChain 官方支持的 VectorStore，接口层完全对齐
（add_documents / similarity_search / filter），未来若要切回 Qdrant，
只需替换本文件内容，其余模块（indexer.py / retriever.py）不用改。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )


def _build_chroma(collection_name: str) -> Chroma:
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


@lru_cache
def get_vectorstore() -> Chroma:
    return _build_chroma(get_settings().chroma_collection)


def book_filter(book_id: str) -> dict:
    """构造仅检索指定 book_id 的 Chroma where 过滤条件，用于单书精读模式。"""
    return {"book_id": book_id}


def delete_book_vectors(book_id: str) -> None:
    """删除某本书在原文 collection 中的全部向量（重新索引前清理旧数据）。

    注：langchain_chroma 的 Chroma.delete() 仅透传 ids，不支持 where 过滤，
    因此直接调用底层 chromadb collection 的 delete，用 where 按 book_id 删除。
    """
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where=book_filter(book_id))
