"""
标签倒排索引：标签 → image_id 列表，用于标签精确检索与混合检索。

对应书籍 RAG 的 BM25 倒排索引（rank_bm25），区别在于：
- 书籍 RAG：文档全文分词 → BM25 倒排索引 → 文本关键词检索
- 图片 RAG：GLM-4V 提取的离散标签 → 倒排索引 → 标签精确检索

为什么需要标签倒排索引：
1. CLIP 向量检索是语义模糊匹配，对"青铜剑"这种精确品类词召回可能不准
   （CLIP 训练数据以英文为主，中文图文对齐能力有限）
2. GLM-4V 提取的标签是精确的关键词，用倒排索引做精确召回，与向量召回互补
3. 两路召回用 RRF 融合，向量负责语义相关、标签负责精确命中，提升整体检索质量

存储格式（tag_index.json）：
    {
      "青铜剑": ["7096fdcf1df96617", "415bdb16ec08ad9a"],
      "青铜": ["7096fdcf1df96617", "415bdb16ec08ad9a", "47f29a758ec191c2"],
      ...
    }

设计要点：
- 持久化到 data/processed/tag_index.json，与 images.json 同级
- 索引图片时自动写入，删除图片时自动清理
- 模块级缓存（_tag_index）：进程内只加载一次，写操作同步更新缓存
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.locks import tag_index_lock

logger = logging.getLogger(__name__)

# 模块级缓存：进程内只加载一次，写操作同步更新
_tag_index: dict[str, list[str]] | None = None


def _tag_index_path() -> Path:
    """标签倒排索引文件路径：data/processed/tag_index.json。"""
    settings = get_settings()
    path = Path(settings.processed_data_dir) / "tag_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_tag_index() -> dict[str, list[str]]:
    """加载标签倒排索引（带模块级缓存）。"""
    global _tag_index
    if _tag_index is not None:
        return _tag_index

    path = _tag_index_path()
    if not path.exists():
        _tag_index = {}
        return _tag_index

    try:
        _tag_index = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("标签倒排索引加载失败，重置为空")
        _tag_index = {}
    return _tag_index


def save_tag_index(index: dict[str, list[str]]) -> None:
    """保存标签倒排索引到 JSON，并更新模块级缓存。

    注意：此函数本身不加锁，调用方需用 tag_index_lock 包裹"读-改-写"整体。
    """
    global _tag_index
    path = _tag_index_path()
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    _tag_index = index


def add_image_tags(image_id: str, tags: list[str]) -> None:
    """索引图片时调用：把 image_id 加到每个 tag 的倒排列表中。

    幂等：同一 image_id 重复添加同一 tag 不会重复（先移除再加）。
    并发安全：用 tag_index_lock 保护"读-改-写"整体，避免并发索引丢标签。
    """
    if not tags:
        return
    with tag_index_lock:
        index = load_tag_index()
        for tag in tags:
            tag = tag.strip()
            if not tag:
                continue
            bucket = index.setdefault(tag, [])
            # 幂等：先移除再加，避免重复
            if image_id in bucket:
                bucket.remove(image_id)
            bucket.append(image_id)
        save_tag_index(index)
    logger.debug("标签索引写入: %s → %s", image_id, tags)


def batch_add_image_tags(items: list[tuple[str, list[str]]]) -> None:
    """批量写入多个图片的标签（一次 load + 一次 save）。

    比逐个调用 add_image_tags 快 N 倍：N 次磁盘 IO → 2 次磁盘 IO。
    幂等：同一 image_id 重复添加同一 tag 不会重复（先移除再加）。
    并发安全：用 tag_index_lock 保护"读-改-写"整体。

    items: [(image_id, tags), ...]
    """
    if not items:
        return
    with tag_index_lock:
        index = load_tag_index()
        for image_id, tags in items:
            if not tags:
                continue
            for tag in tags:
                tag = tag.strip()
                if not tag:
                    continue
                bucket = index.setdefault(tag, [])
                # 幂等：先移除再加，避免重复
                if image_id in bucket:
                    bucket.remove(image_id)
                bucket.append(image_id)
        save_tag_index(index)
    logger.debug("批量标签索引写入: %d 张图片", len(items))


def batch_remove_image_tags(image_ids: list[str]) -> None:
    """批量从所有 tag 的倒排列表中移除多个 image_id（一次 load + 一次 save）。

    比逐个调用 remove_image_tags 快 N 倍。
    并发安全：用 tag_index_lock 保护"读-改-写"整体。
    """
    if not image_ids:
        return
    id_set = set(image_ids)
    with tag_index_lock:
        index = load_tag_index()
        removed = False
        for tag in list(index.keys()):
            new_bucket = [iid for iid in index[tag] if iid not in id_set]
            if len(new_bucket) != len(index[tag]):
                removed = True
                if new_bucket:
                    index[tag] = new_bucket
                else:
                    del index[tag]
        if removed:
            save_tag_index(index)
    logger.debug("批量标签索引清理: %d 张图片", len(image_ids))


def remove_image_tags(image_id: str) -> None:
    """删除图片时调用：从所有 tag 的倒排列表中移除该 image_id。

    并发安全：用 tag_index_lock 保护"读-改-写"整体。
    """
    with tag_index_lock:
        index = load_tag_index()
        removed = False
        for tag in list(index.keys()):
            if image_id in index[tag]:
                index[tag].remove(image_id)
                removed = True
                # 标签下已无图片，移除空标签
                if not index[tag]:
                    del index[tag]
        if removed:
            save_tag_index(index)
    logger.debug("标签索引清理: %s", image_id)


def search_by_tags(tags: list[str], top_k: int | None = None) -> list[str]:
    """标签精确检索：返回包含任意指定标签的 image_id 列表（按命中标签数降序）。

    用于混合检索的标签召回路：与向量召回并行，结果用 RRF 融合。

    返回的 image_id 按"命中标签数"降序排列（命中越多越相关）。
    命中数相同的按 image_id 字典序（稳定排序）。

    top_k 为 None 时返回全部命中结果。
    """
    if not tags:
        return []
    index = load_tag_index()

    # 统计每个 image_id 命中的标签数
    hit_count: dict[str, int] = {}
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        for image_id in index.get(tag, []):
            hit_count[image_id] = hit_count.get(image_id, 0) + 1

    # 按命中数降序排列
    sorted_ids = sorted(hit_count.keys(), key=lambda iid: (-hit_count[iid], iid))
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]
    return sorted_ids


def get_tags_by_image(image_id: str) -> list[str]:
    """查询某个图片的所有标签（反向查询，用于调试/展示）。"""
    index = load_tag_index()
    return [tag for tag, ids in index.items() if image_id in ids]


def reset_tag_index() -> None:
    """清空整个标签倒排索引（全量重建时调用）。

    并发安全：用 tag_index_lock 保护，避免与并发 add/remove 竞态。
    """
    with tag_index_lock:
        global _tag_index
        _tag_index = {}
        path = _tag_index_path()
        if path.exists():
            path.unlink()
    logger.info("标签倒排索引已清空")


def get_tag_stats() -> dict:
    """标签索引统计信息（用于 /image/stats 接口）。"""
    index = load_tag_index()
    total_tags = len(index)
    total_relations = sum(len(ids) for ids in index.values())
    avg_images_per_tag = total_relations / total_tags if total_tags else 0
    return {
        "total_tags": total_tags,
        "total_relations": total_relations,
        "avg_images_per_tag": round(avg_images_per_tag, 2),
    }
