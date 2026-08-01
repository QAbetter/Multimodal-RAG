"""
图片 caption 的进程内 BM25 索引：基于文本的关键词稀疏检索。

对应书籍 RAG 的 bm25_store.py，区别在于：
- 书籍 bm25_store：按 book_id 分组，索引书籍文本 chunk
- 图片 image_bm25_store：单 corpus，索引图片的 caption（PDF提取的图文对应文本）

为什么需要 BM25：
- CLIP 文本向量：语义相似检索，对"图里有剑"这种描述能匹配到"古剑"图片
- BM25：关键词精确匹配，对人名/地名/专有名词（如"李白""匡山"）召回更准
- 两者互补：CLIP 负责"语义像"，BM25 负责"关键词命中"，用 RRF 融合效果最佳

生命周期（与 bm25_store.py 一致）：
- 进程内内存结构，服务重启后需通过 warm_up() 重建（由 main.py startup 调用）
- 图片索引完成后调用 build_image_bm25() 增量加入
- 图片删除时调用 remove_image_bm25() 清理
- 全量重建时调用 reset_image_bm25()

分词策略：与 bm25_store.py 一致，中文 jieba + 英文空格兜底。
"""
from __future__ import annotations

import logging
from threading import Lock

logger = logging.getLogger(__name__)

# 进程内 BM25 索引：image_id -> caption 文本
# 单 corpus 设计（图片无 book_id 概念，所有图片统一检索）
# 用 dict 存储 image_id -> caption，检索时动态构建 BM25Okapi
# （图片数量通常远少于书籍 chunk，动态构建开销可接受，避免增量更新 BM25 的复杂性）
_image_captions: dict[str, str] = {}
_lock = Lock()


def _tokenize(text: str) -> list[str]:
    """分词：中文用 jieba 精确模式，英文/混合按空格兜底。

    与 bm25_store.py 的 _tokenize 保持一致，确保两套 BM25 行为统一。
    """
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return text.split()


def build_image_bm25(image_id: str, caption: str) -> None:
    """添加/更新单张图片的 caption 到 BM25 索引。

    在 image_indexer.index_image 完成向量库写入后调用。
    caption 为空时跳过（无文本的图片不参与 BM25 检索，但仍可走 CLIP 向量检索）。
    """
    if not caption or not caption.strip():
        return
    with _lock:
        _image_captions[image_id] = caption.strip()
    logger.debug("图片 caption BM25 索引更新: %s", image_id)


def remove_image_bm25(image_id: str) -> None:
    """删除单张图片的 caption（图片删除或重索引前清理）。"""
    with _lock:
        _image_captions.pop(image_id, None)


def reset_image_bm25() -> None:
    """清空所有图片的 caption 索引（全量重建场景）。"""
    with _lock:
        _image_captions.clear()


def search_by_caption(query: str, top_k: int) -> list[str]:
    """BM25 检索：query → 相关 image_id 列表（按 BM25 分数降序）。

    对应书籍 RAG 的 search_bm25，区别在于返回 image_id 而非 Document
    （图片检索只需 id，metadata 由 image_vectorstore.get_by_ids 补充）。

    无 caption 或 rank_bm25 未安装时返回空列表，不抛异常（降级为纯向量检索）。
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 未安装，图片 caption BM25 检索跳过")
        return []

    with _lock:
        if not _image_captions:
            return []
        # 快照当前 corpus（锁内拷贝，避免检索过程中被其他线程修改）
        items = list(_image_captions.items())  # [(image_id, caption), ...]

    # 锁外执行 BM25 计算（rank_bm25 内部会建索引，耗时与 corpus 大小成正比）
    image_ids = [iid for iid, _ in items]
    corpus_tokens = [_tokenize(cap) for _, cap in items]
    query_tokens = _tokenize(query)

    if not any(corpus_tokens):  # 所有 caption 都为空 token
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    # 按分数降序取 top_k，过滤掉 0 分（无任何关键词命中）
    ranked = sorted(
        [(iid, score) for iid, score in zip(image_ids, scores) if score > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    return [iid for iid, _ in ranked[:top_k]]


def get_caption(image_id: str) -> str:
    """查询单张图片的 caption（用于结果展示）。"""
    with _lock:
        return _image_captions.get(image_id, "")


def get_caption_stats() -> dict:
    """统计信息：有 caption 的图片数（用于 /image/stats 健康检查）。"""
    with _lock:
        return {"caption_indexed": len(_image_captions)}


def warm_up() -> None:
    """服务启动时调用，从 images.json 注册表重建 caption 索引。

    与 bm25_store.warm_up 同思路：遍历所有 READY 图片，把 caption 加入内存索引。
    服务重启后进程内索引丢失，需此函数恢复。
    """
    from app.core.image_indexer import load_registered_images

    images = load_registered_images()
    count = 0
    for image_id, image in images.items():
        if image.status.value != "ready":
            continue
        if image.caption:
            build_image_bm25(image_id, image.caption)
            count += 1
    if count:
        logger.info("图片 caption BM25 索引热身完成: %d 张图片", count)
