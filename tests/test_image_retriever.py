"""
image_retriever 图片检索主流程单元测试。

覆盖核心功能：
- _hybrid_retrieve：混合检索（向量召回 + 标签召回 + RRF 融合）
  - 无 tags 退化为纯向量检索
  - 标签无命中退化为纯向量检索
  - 两路融合：双路命中 > 单路命中
  - category 过滤
  - top_k 截断
- _format_results：结果格式化（tags 字符串解析、URL 拼接）
- _check_quality：质量检测（ok / low_confidence / no_result / hybrid 特殊处理）
- _build_where：category 过滤条件构造
- search 路由分发（通过 mock embed 函数避免加载 CLIP 模型）

测试策略：
- mock embed_text / embed_image / search_by_vector / get_by_ids，避免依赖 CLIP 模型和 Chroma
- tag_store 用真实实现（依赖 conftest 的临时目录隔离）
- 重点验证 RRF 融合逻辑，这是混合检索的核心
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core import image_retriever
from app.core.image_retriever import (
    _build_where,
    _check_quality,
    _format_results,
    _hybrid_retrieve,
    search,
    search_by_text,
    _RRF_K,
)
from app.models.image_schemas import ImageResult


# ---------------------------------------------------------------------------
# 辅助函数：构造 mock 数据
# ---------------------------------------------------------------------------

def _make_vector_result(image_id: str, score: float, category: str = None, tags: list[str] = None) -> dict:
    """构造 search_by_vector 返回格式的 mock 数据。"""
    metadata = {
        "image_id": image_id,
        "product_id": f"prod_{image_id}",
        "file_path": f"raw/{image_id}.jpg",
    }
    if category:
        metadata["category"] = category
    if tags:
        metadata["tags"] = ",".join(tags)
    return {"id": image_id, "score": score, "metadata": metadata}


# ---------------------------------------------------------------------------
# _build_where 测试
# ---------------------------------------------------------------------------

class TestBuildWhere:
    """_build_where：category 过滤条件构造。"""

    def test_with_category(self):
        where = _build_where("clothing")
        assert where == {"category": "clothing"}

    def test_without_category(self):
        where = _build_where(None)
        assert where is None

    def test_with_empty_category(self):
        where = _build_where("")
        assert where is None  # 空字符串视为无过滤


# ---------------------------------------------------------------------------
# _format_results 测试
# ---------------------------------------------------------------------------

class TestFormatResults:
    """_format_results：结果格式化。"""

    def test_basic_format(self):
        """基本格式化：image_id、score、metadata 字段正确映射。"""
        raw = [_make_vector_result("img1", 0.85, tags=["青铜", "兵器"])]
        results = _format_results(raw)
        assert len(results) == 1
        r = results[0]
        assert r.image_id == "img1"
        assert r.product_id == "prod_img1"
        assert r.score == 0.85
        assert r.tags == ["青铜", "兵器"]
        assert r.image_url == "raw/img1.jpg"  # base_url 为空时用 file_path

    def test_tags_string_parsing(self):
        """tags 存储为逗号分隔字符串，解析回 list。"""
        raw = [_make_vector_result("img1", 0.9, tags=["a", "b", "c"])]
        results = _format_results(raw)
        assert results[0].tags == ["a", "b", "c"]

    def test_empty_tags(self):
        """无 tags 字段时返回空列表。"""
        raw = [_make_vector_result("img1", 0.9)]
        results = _format_results(raw)
        assert results[0].tags == []

    def test_base_url_concatenation(self):
        """base_url 拼接图片访问 URL。"""
        raw = [_make_vector_result("img1", 0.9)]
        results = _format_results(raw, base_url="http://localhost:8000")
        assert results[0].image_url == "http://localhost:8000/images/raw/img1.jpg"

    def test_thumbnail_url(self):
        """缩略图 URL 拼接。"""
        raw = [{
            "id": "img1",
            "score": 0.9,
            "metadata": {
                "image_id": "img1",
                "product_id": "p1",
                "file_path": "raw/img1.jpg",
                "thumbnail_path": "thumbnails/img1.jpg",
            },
        }]
        results = _format_results(raw, base_url="http://localhost:8000")
        assert results[0].thumbnail_url == "http://localhost:8000/images/thumbnails/img1.jpg"

    def test_score_rounding(self):
        """score 保留 4 位小数。"""
        raw = [_make_vector_result("img1", 0.123456789)]
        results = _format_results(raw)
        assert results[0].score == 0.1235  # round(0.123456789, 4)


# ---------------------------------------------------------------------------
# _check_quality 测试
# ---------------------------------------------------------------------------

class TestCheckQuality:
    """_check_quality：质量检测。"""

    def test_no_result(self):
        """无结果：no_result。"""
        assert _check_quality([]) == "no_result"

    def test_ok_high_score(self):
        """高分结果：ok（score >= threshold）。"""
        results = [ImageResult(image_id="img1", product_id="p", image_url="", score=0.5)]
        assert _check_quality(results) == "ok"

    def test_low_confidence(self):
        """低分结果：low_confidence（score < threshold，默认 0.2）。"""
        results = [ImageResult(image_id="img1", product_id="p", image_url="", score=0.1)]
        assert _check_quality(results) == "low_confidence"

    def test_hybrid_always_ok(self):
        """混合检索只要有结果就是 ok（RRF 分数量纲不同，不与阈值比较）。"""
        results = [ImageResult(image_id="img1", product_id="p", image_url="", score=0.001)]
        assert _check_quality(results, is_hybrid=True) == "ok"

    def test_hybrid_no_result_still_no_result(self):
        """混合检索无结果仍是 no_result。"""
        assert _check_quality([], is_hybrid=True) == "no_result"


# ---------------------------------------------------------------------------
# _hybrid_retrieve 测试（核心：RRF 融合逻辑）
# ---------------------------------------------------------------------------

class TestHybridRetrieve:
    """_hybrid_retrieve：混合检索 RRF 融合。"""

    def test_no_tags_degrades_to_vector(self):
        """无 tags 时退化为纯向量检索（直接返回向量召回 top_k）。"""
        vector_results = [
            _make_vector_result("img1", 0.9),
            _make_vector_result("img2", 0.8),
        ]
        with patch("app.core.image_retriever.search_by_vector", return_value=vector_results):
            result = _hybrid_retrieve([0.1] * 10, tags=None, top_k=2, where=None)

        assert len(result) == 2
        assert result[0]["id"] == "img1"
        assert result[1]["id"] == "img2"

    def test_tag_no_hit_degrades_to_vector(self):
        """标签无命中时退化为纯向量检索。"""
        vector_results = [_make_vector_result("img1", 0.9)]
        with patch("app.core.image_retriever.search_by_vector", return_value=vector_results), \
             patch("app.core.image_retriever.search_by_tags", return_value=[]):
            result = _hybrid_retrieve([0.1] * 10, tags=["不存在的标签"], top_k=5, where=None)

        assert len(result) == 1
        assert result[0]["id"] == "img1"

    def test_both_routes_hit_ranks_higher(self):
        """双路命中的图片 RRF 分数更高，排名靠前。

        场景：向量召回 img1 排第1，标签召回 img1 也排第1
              向量召回 img2 排第2，标签召回无 img2
        预期：img1 双路命中，RRF 分数 = 2/(60+0+1) > img2 的 1/(60+1+1)
        """
        vector_results = [
            _make_vector_result("img1", 0.9),
            _make_vector_result("img2", 0.8),
        ]
        tag_ids = ["img1"]  # 标签只召回 img1
        with patch("app.core.image_retriever.search_by_vector", return_value=vector_results), \
             patch("app.core.image_retriever.search_by_tags", return_value=tag_ids), \
             patch("app.core.image_retriever.get_by_ids", return_value={"img1": {"category": None, "file_path": "raw/img1.jpg"}}):
            result = _hybrid_retrieve([0.1] * 10, tags=["青铜"], top_k=2, where=None)

        assert result[0]["id"] == "img1"  # 双路命中，排第1
        # 验证 RRF 分数：img1 = 1/61 + 1/61 ≈ 0.0328
        expected_img1_score = 1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 1)
        assert abs(result[0]["score"] - expected_img1_score) < 0.0001

    def test_tag_only_image_included(self):
        """只在标签路命中的图片也会出现在结果中（用 get_by_ids 补 metadata）。"""
        vector_results = [_make_vector_result("img1", 0.9)]
        tag_ids = ["img1", "img2"]  # img2 只在标签路命中
        tag_meta = {
            "img1": {"category": None, "file_path": "raw/img1.jpg", "product_id": "p1"},
            "img2": {"category": None, "file_path": "raw/img2.jpg", "product_id": "p2"},
        }
        with patch("app.core.image_retriever.search_by_vector", return_value=vector_results), \
             patch("app.core.image_retriever.search_by_tags", return_value=tag_ids), \
             patch("app.core.image_retriever.get_by_ids", return_value=tag_meta):
            result = _hybrid_retrieve([0.1] * 10, tags=["青铜"], top_k=5, where=None)

        ids = [r["id"] for r in result]
        assert "img2" in ids  # img2 虽然不在向量召回中，但标签召回命中
        # img2 的 metadata 来自 get_by_ids
        img2_result = next(r for r in result if r["id"] == "img2")
        assert img2_result["metadata"]["file_path"] == "raw/img2.jpg"

    def test_category_filter_applied_to_tag_results(self):
        """category 过滤应用到标签召回结果（tag_store 不存 category，需 get_by_ids 补充后过滤）。"""
        vector_results = [_make_vector_result("img1", 0.9, category="clothing")]
        tag_ids = ["img1", "img2"]
        # img2 的 category 是 electronics，应被过滤掉
        tag_meta = {
            "img1": {"category": "clothing", "file_path": "raw/img1.jpg"},
            "img2": {"category": "electronics", "file_path": "raw/img2.jpg"},
        }
        with patch("app.core.image_retriever.search_by_vector", return_value=vector_results), \
             patch("app.core.image_retriever.search_by_tags", return_value=tag_ids), \
             patch("app.core.image_retriever.get_by_ids", return_value=tag_meta):
            result = _hybrid_retrieve(
                [0.1] * 10, tags=["衣服"], top_k=5, where={"category": "clothing"}
            )

        ids = [r["id"] for r in result]
        assert "img2" not in ids  # img2 因 category 不匹配被过滤

    def test_top_k_truncation(self):
        """top_k 截断：融合后只返回 top_k 个结果。"""
        vector_results = [_make_vector_result(f"img{i}", 0.9 - i * 0.1) for i in range(5)]
        tag_ids = [f"img{i}" for i in range(5)]
        tag_meta = {f"img{i}": {"file_path": f"raw/img{i}.jpg"} for i in range(5)}
        with patch("app.core.image_retriever.search_by_vector", return_value=vector_results), \
             patch("app.core.image_retriever.search_by_tags", return_value=tag_ids), \
             patch("app.core.image_retriever.get_by_ids", return_value=tag_meta):
            result = _hybrid_retrieve([0.1] * 10, tags=["test"], top_k=3, where=None)

        assert len(result) == 3  # 只返回 top_k=3 个


# ---------------------------------------------------------------------------
# search 路由分发测试
# ---------------------------------------------------------------------------

class TestSearchRouting:
    """search 函数的路由分发。"""

    def test_empty_input_returns_empty(self):
        """query 和 image_base64 都为空：返回空结果，route=empty。"""
        resp = search()
        assert resp.route == "empty"
        assert resp.total == 0
        assert resp.results == []

    def test_text_search_uses_text_vector(self):
        """文本搜图：调用 embed_text 生成向量。"""
        mock_vector = [0.1] * 10
        mock_results = [_make_vector_result("img1", 0.9)]
        with patch("app.core.image_retriever.embed_text", return_value=mock_vector), \
             patch("app.core.image_retriever.search_by_vector", return_value=mock_results):
            resp = search(query="青铜器")

        assert resp.route == "text_to_image"
        assert resp.total == 1
        assert resp.results[0].image_id == "img1"

    def test_text_search_with_tags_uses_hybrid(self):
        """文本搜图 + tags：启用混合检索，route=text_to_image_hybrid。"""
        mock_vector = [0.1] * 10
        mock_results = [_make_vector_result("img1", 0.9)]
        with patch("app.core.image_retriever.embed_text", return_value=mock_vector), \
             patch("app.core.image_retriever.search_by_vector", return_value=mock_results), \
             patch("app.core.image_retriever.search_by_tags", return_value=[]):  # 标签无命中，退化
            resp = search(query="青铜器", tags=["青铜"])

        assert resp.route == "text_to_image_hybrid"

    def test_image_search_uses_image_vector(self):
        """以图搜图：调用 embed_image 生成向量。"""
        import base64
        mock_vector = [0.1] * 10
        mock_results = [_make_vector_result("img1", 0.95)]
        # 构造一个 1x1 红色 PNG 的 base64
        from PIL import Image
        import io
        img = Image.new("RGB", (10, 10), "red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        with patch("app.core.image_retriever.embed_image", return_value=mock_vector), \
             patch("app.core.image_retriever.search_by_vector", return_value=mock_results):
            resp = search(image_base64=img_b64)

        assert resp.route == "image_to_image"
        assert resp.results[0].image_id == "img1"

    def test_image_takes_priority_over_text(self):
        """image_base64 优先级高于 query（同时传入时走以图搜图）。"""
        mock_vector = [0.1] * 10
        mock_results = [_make_vector_result("img1", 0.9)]
        import base64
        from PIL import Image
        import io
        img = Image.new("RGB", (10, 10), "red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        with patch("app.core.image_retriever.embed_image", return_value=mock_vector), \
             patch("app.core.image_retriever.search_by_vector", return_value=mock_results):
            resp = search(query="青铜器", image_base64=img_b64)

        assert resp.route == "image_to_image"  # 走以图搜图，不是文本搜图
