"""
图片检索主流程：支持文本搜图、以图搜图、标签过滤、混合检索（RRF 融合）。

对应书籍 RAG 的 retriever.py，区别在于：
- 书籍 RAG：query(文本) → 文本 embedding → 向量检索 → BM25 混合 → RRF 融合 → rerank → LLM 生成答案
- 图片 RAG：query(文本/图片) → CLIP embedding → 向量检索 → [标签召回 + caption BM25 + RRF 融合] → 返回图片结果

图片 RAG 是纯检索任务（输入图片/query → 输出相关图片+标签），不涉及 LLM 答案生成，
因此比书籍 RAG 简单，不需要 LangGraph 多轮对话、不需要 rerank 精排。

检索路由：
- text_to_image：文本 query → CLIP 文本向量 → 向量检索
- image_to_image：图片 → CLIP 图像向量 → 向量检索
- text_to_image_hybrid / image_to_image_hybrid：传入 tags 或有 caption 索引时启用混合检索
  （向量召回 + 标签召回 + caption BM25 召回 + RRF 融合，对应书籍 RAG 的 向量+BM25+RRF）

混合检索设计（对应书籍 RAG 的 fusion.py）：
- 向量召回路：CLIP 语义匹配，负责"看起来像"的语义相关召回
- 标签召回路：tag_store 倒排索引精确匹配，负责"标签命中"的精确召回
- caption BM25 召回路：image_bm25_store 关键词匹配，负责"图文对应文本命中"的精确召回
  （PDF 提取的图片有 caption，基于文本查图时这一路是主要召回来源）
- RRF 融合：用排名融合多路结果，不依赖分数绝对值（与 fusion.py 的 _RRF_K=60 一致）

设计要点：
- CLIP 图文同空间：文本向量和图片向量在同一空间，用同一个 search_by_vector 检索。
- 相似度分数：纯向量检索的 score 为余弦相似度（0~1）；混合检索的 score 为 RRF 分数。
- metadata 过滤：按 category 过滤（单 Collection + metadata 过滤，与书籍 book_id 过滤同思路）。
- caption 自动启用：文本搜图时若 caption BM25 索引非空，自动启用混合检索（无需传 tags）。
"""
from __future__ import annotations

import base64
import io
import logging
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.core.image_bm25_store import get_caption_stats, search_by_caption
from app.core.image_embedder import embed_image, embed_text
from app.core.image_vectorstore import category_filter, get_by_ids, search_by_vector
from app.core.tag_store import search_by_tags
from app.models.image_schemas import ImageResult, ImageSearchResponse

logger = logging.getLogger(__name__)

# RRF 融合常数：与 fusion.py 一致，60 是 RAG-Fusion 论文常用默认值
_RRF_K = 60


def _resolve_top_k(top_k: int | None) -> int:
    """与 retriever.py 的 _resolve_top_k 同思路：未传则用配置默认值。"""
    return top_k or get_settings().image_retrieval_top_k


def _build_where(category: str | None) -> dict | None:
    """构造 Chroma where 过滤条件（仅 category，tags 走混合检索而非 where 过滤）。

    tags 不在这里处理：tags 作为独立召回路（tag_store.search_by_tags）与向量召回并行，
    用 RRF 融合，比硬过滤效果更好（硬过滤会丢掉语义相关但标签未命中的图片）。
    """
    if category:
        return category_filter(category)
    return None


def _filter_by_tags(tags: list[str], top_k: int) -> dict | None:
    """构造标签硬过滤的 where 条件：{"image_id": {"$in": candidate_ids}}。

    用于"纯标签过滤检索"场景：只在包含指定标签的图片中做向量检索。
    与 _hybrid_retrieve 的区别：这里是硬过滤（缩小范围），后者是软融合（两路并行 + RRF）。

    当前 search_by_text/search_by_image 默认走混合检索，此函数保留供 API 层
    显式指定 tag_filter 模式时使用。
    """
    if not tags:
        return None
    candidate_ids = search_by_tags(tags, top_k=top_k)
    if not candidate_ids:
        return None
    return {"image_id": {"$in": candidate_ids}}


def _hybrid_retrieve(
    query_vector: list[float],
    tags: list[str] | None,
    top_k: int,
    where: dict | None,
    query_text: str | None = None,
) -> list[dict]:
    """混合检索：向量召回 + 标签召回 + caption BM25 召回 + RRF 融合。

    对应书籍 RAG 的 fuse_retrieval（向量+BM25+RRF），图片 RAG 用三路召回：
    - 向量路（必有）：CLIP 语义匹配
    - 标签路（可选）：传入 tags 时启用
    - caption BM25 路（可选）：传入 query_text 且 caption 索引非空时启用

    流程：
    1. 向量召回：search_by_vector 获取 top_k*2 候选（带余弦相似度 score 和 metadata）
    2. 标签召回：tag_store.search_by_tags 获取 top_k*2 image_id（按命中标签数降序）
    3. caption BM25 召回：image_bm25_store.search_by_caption 获取 top_k*2 image_id
    4. category 过滤：标签/caption 召回的 image_id 通过 get_by_ids 补 metadata 后按 category 过滤
    5. RRF 融合：每路按排名贡献 1/(k+rank+1)，按 image_id 累加后降序排序
    6. 补充 metadata：非向量召回的 image_id 用 get_by_ids 取 metadata
    7. 取 top_k

    返回格式与 search_by_vector 一致：[{"id", "score", "metadata"}]，
    但 score 为 RRF 融合分数（非余弦相似度，量纲不同，不与 image_score_threshold 比较）。

    无 tags 且无 caption 召回时退化为纯向量检索（直接返回向量召回 top_k）。
    """
    # 召回窗口放大到 2 倍，确保 RRF 融合后有足够候选
    recall_k = top_k * 2

    # 1. 向量召回（where 已包含 category 过滤）
    vector_results = search_by_vector(query_vector, top_k=recall_k, where=where)

    # 2. 标签召回（无 tags 时跳过）
    tag_ids: list[str] = []
    if tags:
        tag_ids = search_by_tags(tags, top_k=recall_k)

    # 3. caption BM25 召回（无 query_text 或 caption 索引为空时跳过）
    caption_ids: list[str] = []
    if query_text and get_caption_stats().get("caption_indexed", 0) > 0:
        caption_ids = search_by_caption(query_text, top_k=recall_k)

    # 无标签召回也无 caption 召回，退化为纯向量检索
    if not tag_ids and not caption_ids:
        return vector_results[:top_k]

    # 4. 补充 metadata + category 过滤（标签路和 caption 路只返回 image_id）
    cat = where.get("category") if where else None
    extra_ids = [iid for iid in tag_ids + caption_ids if iid not in {r["id"] for r in vector_results}]
    extra_meta = get_by_ids(extra_ids) if extra_ids else {}

    if cat:
        # category 过滤标签路
        tag_ids = [iid for iid in tag_ids if extra_meta.get(iid, {}).get("category") == cat]
        # category 过滤 caption 路（caption 路的 image_id 也可能在向量召回中，需统一过滤）
        caption_ids = [
            iid for iid in caption_ids
            if iid in {r["id"] for r in vector_results if r["metadata"].get("category") == cat}
            or extra_meta.get(iid, {}).get("category") == cat
        ]
        if not tag_ids and not caption_ids:
            return vector_results[:top_k]

    # 5. RRF 融合：三路按排名累加 1/(k+rank+1)
    rrf_scores: dict[str, float] = {}
    meta_lookup: dict[str, dict] = {}

    # 向量路（search_by_vector 已按相似度降序，rank 从 0 开始）
    for rank, item in enumerate(vector_results):
        iid = item["id"]
        rrf_scores[iid] = rrf_scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        meta_lookup[iid] = item["metadata"]

    # 标签路（search_by_tags 已按命中标签数降序，rank 从 0 开始）
    for rank, iid in enumerate(tag_ids):
        rrf_scores[iid] = rrf_scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if iid not in meta_lookup:
            meta_lookup[iid] = extra_meta.get(iid, {})

    # caption BM25 路（search_by_caption 已按 BM25 分数降序，rank 从 0 开始）
    for rank, iid in enumerate(caption_ids):
        rrf_scores[iid] = rrf_scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if iid not in meta_lookup:
            meta_lookup[iid] = extra_meta.get(iid, {})

    # 6. 按 RRF 分数降序，取 top_k
    sorted_ids = sorted(rrf_scores, key=lambda iid: rrf_scores[iid], reverse=True)[:top_k]
    return [
        {"id": iid, "score": rrf_scores[iid], "metadata": meta_lookup.get(iid, {})}
        for iid in sorted_ids
    ]


def _format_results(raw_results: list[dict], base_url: str = "") -> list[ImageResult]:
    """将 Chroma 检索结果格式化为 ImageResult 列表。

    base_url 用于拼接图片访问 URL（API 层会传入，脚本直调时为空）。
    tags 在存储时是逗号分隔字符串（Chroma 不支持 list），这里 split 回 list。
    caption 直接从 metadata 读取（存储时就是 str，无需转换）。
    """
    results = []
    for item in raw_results:
        meta = item["metadata"] or {}
        image_id = item["id"]
        file_path = meta.get("file_path", "")
        thumbnail = meta.get("thumbnail_path")
        # tags 存储为逗号分隔字符串，解析回 list
        tags_str = meta.get("tags", "")
        tags = tags_str.split(",") if tags_str else []
        results.append(
            ImageResult(
                image_id=image_id,
                product_id=meta.get("product_id", ""),
                image_url=f"{base_url}/images/{file_path}" if base_url else file_path,
                thumbnail_url=f"{base_url}/images/{thumbnail}" if thumbnail and base_url else thumbnail,
                tags=tags,
                category=meta.get("category"),
                score=round(item["score"], 4),
                caption=meta.get("caption"),
                pdf_source=meta.get("pdf_source"),
            )
        )
    return results


def _check_quality(results: list[ImageResult], is_hybrid: bool = False) -> str:
    """检测检索结果质量，与 retriever.py 的 _check_answer_quality 同思路。

    ok=有结果且最高分达标；low_confidence=有结果但最高分低于阈值；no_result=无结果。

    混合检索（is_hybrid=True）的 score 是 RRF 分数（量纲与余弦相似度不同），
    不与 image_score_threshold 比较，只要有结果即视为 ok。
    """
    if not results:
        return "no_result"
    if is_hybrid:
        return "ok"
    if results[0].score < get_settings().image_score_threshold:
        return "low_confidence"
    return "ok"


def search_by_text(
    query: str,
    category: str | None = None,
    tags: list[str] | None = None,
    top_k: int | None = None,
    base_url: str = "",
) -> ImageSearchResponse:
    """文本搜图：文本 query → CLIP 文本向量 → 向量检索 / 混合检索。

    CLIP 把文本和图片映射到同一向量空间，文本向量可直接匹配图片向量。

    混合检索启用条件（满足任一即启用）：
    - 传入 tags：向量召回 + 标签召回 + RRF 融合
    - caption BM25 索引非空（有 PDF 提取的图片）：向量召回 + caption BM25 召回 + RRF 融合
      这一路是"基于文本查图"的核心：用户输入描述性文本，通过 caption 关键词命中相关图片。
    """
    k = _resolve_top_k(top_k)
    query_vector = embed_text(query)
    where = _build_where(category)

    # 判断是否启用混合检索：有 tags 或有 caption 索引
    has_caption_index = get_caption_stats().get("caption_indexed", 0) > 0
    is_hybrid = bool(tags) or has_caption_index
    if is_hybrid:
        raw = _hybrid_retrieve(
            query_vector, tags, top_k=k, where=where, query_text=query
        )
        route = "text_to_image_hybrid"
    else:
        raw = search_by_vector(query_vector, top_k=k, where=where)
        route = "text_to_image"

    results = _format_results(raw, base_url)
    return ImageSearchResponse(
        results=results,
        route=route,
        total=len(results),
        answer_quality=_check_quality(results, is_hybrid=is_hybrid),
    )


def search_by_image(
    image_base64: str,
    category: str | None = None,
    tags: list[str] | None = None,
    top_k: int | None = None,
    base_url: str = "",
) -> ImageSearchResponse:
    """以图搜图：图片 base64 → CLIP 图像向量 → 向量检索 / 混合检索。

    图片与图片在同一模态，CLIP 图像编码器直接处理。

    传入 tags 时启用混合检索（向量召回 + 标签召回 + RRF 融合）。
    """
    k = _resolve_top_k(top_k)
    # base64 解码为 PIL Image
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data))
    if img.mode != "RGB":
        img = img.convert("RGB")

    query_vector = embed_image(img)
    where = _build_where(category)

    # 传入 tags 走混合检索，否则走纯向量检索
    # 以图搜图无文本 query，caption BM25 路不启用（query_text=None）
    is_hybrid = bool(tags)
    if is_hybrid:
        raw = _hybrid_retrieve(
            query_vector, tags, top_k=k, where=where, query_text=None
        )
        route = "image_to_image_hybrid"
    else:
        raw = search_by_vector(query_vector, top_k=k, where=where)
        route = "image_to_image"

    results = _format_results(raw, base_url)
    return ImageSearchResponse(
        results=results,
        route=route,
        total=len(results),
        answer_quality=_check_quality(results, is_hybrid=is_hybrid),
    )


def search(
    query: str | None = None,
    image_base64: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    top_k: int | None = None,
    base_url: str = "",
) -> ImageSearchResponse:
    """统一检索入口：根据输入类型路由到文本搜图或以图搜图。

    与 retriever.py 的 ask_single_book / ask_multi_book 路由分发同思路。

    传入 tags 时自动启用混合检索（向量召回 + 标签召回 + RRF 融合）。
    """
    if image_base64:
        return search_by_image(image_base64, category, tags, top_k, base_url)
    if query:
        return search_by_text(query, category, tags, top_k, base_url)
    # 两者都为空：返回空结果（API 层会拦截 400，这里兜底）
    return ImageSearchResponse(
        results=[], route="empty", total=0, answer_quality="no_result"
    )
