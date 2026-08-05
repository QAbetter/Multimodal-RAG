"""
图片向量库封装（Chroma 本地持久化，独立 collection 与书籍 RAG 隔离）。

对应书籍 RAG 的 vectorstore.py，区别在于：
- 书籍 RAG 用 langchain_chroma.Chroma + OpenAIEmbeddings（文本 embedding 由 LangChain 自动调 API）
- 图片 RAG 的向量由 Chinese-CLIP 本地预计算，不走 OpenAIEmbeddings，
  因此直接操作 Chroma 底层 collection 写入预计算向量。

设计要点：
- 独立 collection（images）：与书籍 collection（books）数据隔离，互不影响。
  复用同一个 chroma_persist_dir，物理上在同一目录但逻辑隔离。
- 单例模式（@lru_cache）：与 get_vectorstore() 一致，避免重复创建 collection 句柄。
- 预计算向量写入：CLIP 向量是本地算好的，用 collection.add(embeddings=...) 直接写入，
  绕过 LangChain 的 embedding_function（它期望文本输入）。
- metadata 过滤：按 category 过滤（单 Collection + metadata 过滤策略，与书籍的 book_id 过滤同思路）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb

from app.core.config import get_settings


@lru_cache
def get_image_collection():
    """单例获取图片向量库的 Chroma collection（底层 chromadb 客户端）。

    返回 chromadb 的 Collection 对象（非 langchain Chroma），
    因为我们要写入预计算的 CLIP 向量，需要用底层 add(embeddings=...) 接口。
    """
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=settings.chroma_image_collection)


def add_image_vector(
    image_id: str,
    embedding: list[float],
    metadata: dict,
) -> None:
    """写入单张图片的向量 + metadata。

    重新索引前应先调用 delete_image_vectors() 清理旧向量，
    否则 Chroma 会因 id 重复报错。
    """
    collection = get_image_collection()
    collection.add(
        ids=[image_id],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def delete_image_vectors(image_id: str) -> None:
    """删除指定 image_id 的向量（重新索引前清理旧数据）。"""
    collection = get_image_collection()
    collection.delete(ids=[image_id])


def batch_delete_image_vectors(image_ids: list[str]) -> None:
    """批量删除向量（重新索引前清理旧数据）。"""
    if not image_ids:
        return
    # 去重（保留顺序）：重复 ID 会导致 Chroma delete 抛 DuplicateIDError
    seen = set()
    unique_ids = [iid for iid in image_ids if not (iid in seen or seen.add(iid))]
    collection = get_image_collection()
    collection.delete(ids=unique_ids)


def batch_add_image_vectors(items: list[dict]) -> None:
    """批量写入图片向量。

    items: [{"id": str, "embedding": list[float], "metadata": dict}, ...]
    比循环调用 add_image_vector 快，减少 Chroma 内部事务次数。
    """
    if not items:
        return
    collection = get_image_collection()
    collection.add(
        ids=[it["id"] for it in items],
        embeddings=[it["embedding"] for it in items],
        metadatas=[it["metadata"] for it in items],
    )


def reset_image_collection() -> None:
    """清空整个 images collection（删除磁盘文件残留的旧向量）。

    场景：用户从 data/images/raw/ 删除了图片文件，但 Chroma 里的向量仍残留，
    导致检索结果出现已删除的图片。此函数重建一个空 collection，彻底清理。

    注意：会清空所有图片向量，需重新索引。get_image_collection 的 lru_cache
    会缓存旧 collection 句柄，这里用 cache_clear 强制重建。
    """
    settings = get_settings()
    client = chromadb.PersistentClient(path=str(Path(settings.chroma_persist_dir)))
    try:
        client.delete_collection(name=settings.chroma_image_collection)
    except Exception:
        pass  # collection 不存在时忽略
    get_image_collection.cache_clear()


def search_by_vector(
    embedding: list[float],
    top_k: int,
    where: dict | None = None,
) -> list[dict]:
    """向量相似度检索，返回 top_k 结果。

    返回格式：[{"id", "score", "metadata"}]，score 为余弦相似度（0~1，越大越相似）。

    Chroma 默认用 L2 距离（平方欧几里得）。对 L2 归一化的 CLIP 向量：
        L2_distance = 2 × (1 - 余弦相似度)
        余弦相似度 = 1 - L2_distance / 2
    """
    collection = get_image_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
    )
    ids = results["ids"][0] if results["ids"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return [
        {
            "id": id_,
            "score": 1.0 - dist / 2,  # L2距离转余弦相似度（CLIP向量已L2归一化）
            "metadata": meta,
        }
        for id_, dist, meta in zip(ids, distances, metadatas)
    ]


def category_filter(category: str) -> dict:
    """构造类别过滤条件，与 vectorstore.py 的 book_filter() 同思路。"""
    return {"category": category}


def count_images() -> int:
    """当前 collection 中的图片向量数量（用于健康检查 / 评测）。"""
    return get_image_collection().count()


def get_by_ids(image_ids: list[str]) -> dict[str, dict]:
    """根据 image_id 列表批量获取 metadata，返回 {image_id: metadata}。

    混合检索的标签召回路只返回 image_id（无 metadata），
    需要此函数补充 metadata，用于格式化最终结果。

    返回的 metadata 与 search_by_vector 中的 metadata 字段一致（含 tags 逗号分隔字符串）。
    """
    if not image_ids:
        return {}
    # 去重（保留顺序）：混合检索的标签路和 caption 路可能召回同一张图，
    # 合并后存在重复 ID，Chroma 的 get() 不接受重复 ID 会抛 DuplicateIDError
    seen = set()
    unique_ids = [iid for iid in image_ids if not (iid in seen or seen.add(iid))]
    collection = get_image_collection()
    results = collection.get(ids=unique_ids)
    return {
        id_: meta for id_, meta in zip(results.get("ids", []), results.get("metadatas", []))
    }
