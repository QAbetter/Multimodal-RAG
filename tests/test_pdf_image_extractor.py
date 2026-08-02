"""
PDF 插图提取的单元测试：验证 MinerU middle.json 解析和 PDF 切分逻辑（不依赖网络和MinerU API）。

测试覆盖：
1. middle.json 解析：图片+图注配对（MinerU 已在模型层面完成配对，无需坐标匹配）
2. content_list.json 解析：简化版格式的图文配对
3. 图片文件名提取：从 CDN URL 和相对路径提取文件名
4. PDF 切分判断：超阈值切分、未超阈值不切分
5. JSON 文件查找：middle.json / content_list.json 的查找优先级
6. 边界情况：无 image block、无 figure_text、非法 JSON

运行：
    .venv\\Scripts\\python.exe -m pytest tests/test_pdf_image_extractor.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.pdf_image_extractor import (
    ExtractedImage,
    _build_image_caption_map_from_markdown,
    _extract_image_name_from_url,
    _find_middle_json,
    _match_by_coordinates,
    _parse_content_list,
    _parse_middle_json,
    _parse_middle_json_full,
    _should_split_pdf,
    extract_images_from_pdf,
)


# ===========================================================================
# 辅助函数：构造 MinerU middle.json 结构
# ===========================================================================

def _make_image_block(
    image_path: str,
    caption: str = "",
    page_idx: int = 0,
) -> dict:
    """构造 MinerU middle.json 中的一个 image para_block。

    结构对应真实 MinerU 输出：
        {
            "type": "image",
            "blocks": [
                {"type": "image_body", "lines": [{"spans": [{"image_path": "..."}]}]},
                {"type": "image_footnote", "lines": [{"spans": [{"content": "..."}]}]}
            ]
        }
    """
    sub_blocks = [
        {
            "type": "image_body",
            "lines": [{"spans": [{"image_path": image_path}]}],
        }
    ]
    if caption:
        sub_blocks.append({
            "type": "image_footnote",
            "lines": [{"spans": [{"content": caption}]}],
        })
    return {
        "type": "image",
        "bbox": [0, 0, 100, 100],
        "blocks": sub_blocks,
    }


def _make_middle_json(pages: list[dict]) -> dict:
    """构造完整的 middle.json 结构。

    pages 是 list[dict]，每个 dict 形如 {"page_idx": 0, "para_blocks": [...]}
    """
    return {
        "pdf_info": [
            {"page_idx": p["page_idx"], "para_blocks": p["para_blocks"]}
            for p in pages
        ],
        "_backend": "vlm",
        "_version_name": "2.5",
    }


def _write_json(data, path: Path) -> None:
    """把数据写入 JSON 文件。"""
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ===========================================================================
# 1. middle.json 解析测试（核心：MinerU 已配对图文，直接读取）
# ===========================================================================

class TestParseMiddleJson:
    """测试 _parse_middle_json：从 MinerU middle.json 提取图片和图注。"""

    def test_single_image_with_caption(self, tmp_path):
        """单张图片+图注：MinerU 已配对，直接读取。"""
        middle_data = _make_middle_json([
            {"page_idx": 0, "para_blocks": [
                _make_image_block("images/fig1.jpg", "图1 須摩提女請佛故事之三"),
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json(json_path)
        assert "fig1.jpg" in result
        caption, page = result["fig1.jpg"]
        assert caption == "图1 須摩提女請佛故事之三"
        assert page == 0

    def test_image_without_caption(self, tmp_path):
        """图片无图注：caption 为空字符串。"""
        middle_data = _make_middle_json([
            {"page_idx": 0, "para_blocks": [
                _make_image_block("images/no_cap.jpg", caption=""),
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json(json_path)
        assert "no_cap.jpg" in result
        caption, _ = result["no_cap.jpg"]
        assert caption == ""

    def test_cdn_url_image_path(self, tmp_path):
        """图片路径是远程 CDN URL：应正确提取文件名。"""
        cdn_url = "https://cdn-mineru.openxlab.org.cn/result/2026-07-31/xxx/c5d13ec2b30d4143c65d86f4102bd616087554da2ac15f90cb709011617c7a23.jpg"
        middle_data = _make_middle_json([
            {"page_idx": 5, "para_blocks": [
                _make_image_block(cdn_url, "内景 第二六八窟"),
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json(json_path)
        expected_name = "c5d13ec2b30d4143c65d86f4102bd616087554da2ac15f90cb709011617c7a23.jpg"
        assert expected_name in result
        caption, page = result[expected_name]
        assert caption == "内景 第二六八窟"
        assert page == 5

    def test_multiple_images_multiple_pages(self, tmp_path):
        """多页多图片：每个图片的图注和页码都正确。"""
        middle_data = _make_middle_json([
            {"page_idx": 0, "para_blocks": [
                _make_image_block("images/p0_img.jpg", "第0页图注"),
            ]},
            {"page_idx": 1, "para_blocks": [
                _make_image_block("images/p1_img1.jpg", "第1页图注1"),
                _make_image_block("images/p1_img2.jpg", "第1页图注2"),
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json(json_path)
        assert len(result) == 3
        assert result["p0_img.jpg"] == ("第0页图注", 0)
        assert result["p1_img1.jpg"] == ("第1页图注1", 1)
        assert result["p1_img2.jpg"] == ("第1页图注2", 1)

    def test_non_image_blocks_ignored(self, tmp_path):
        """非 image 类型的 para_block 应被忽略。"""
        middle_data = _make_middle_json([
            {"page_idx": 0, "para_blocks": [
                {"type": "text", "bbox": [0, 0, 100, 50], "blocks": []},
                _make_image_block("images/fig1.jpg", "图注"),
                {"type": "table", "bbox": [0, 0, 100, 50], "blocks": []},
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json(json_path)
        assert len(result) == 1
        assert "fig1.jpg" in result

    def test_invalid_json(self, tmp_path):
        """非法 JSON 应返回空 dict，不抛异常。"""
        json_path = tmp_path / "bad_middle.json"
        json_path.write_text("这不是合法JSON", encoding="utf-8")
        assert _parse_middle_json(json_path) == {}

    def test_nonexistent_file(self, tmp_path):
        """文件不存在应返回空。"""
        assert _parse_middle_json(tmp_path / "nonexistent.json") == {}

    def test_real_structure_from_user(self, tmp_path):
        """模拟用户提供的真实 middle.json 结构（image_body + image_footnote）。"""
        middle_data = {
            "pdf_info": [
                {
                    "page_idx": 32,
                    "para_blocks": [
                        {
                            "type": "image",
                            "bbox": [40, 80, 474, 603],
                            "blocks": [
                                {
                                    "type": "image_body",
                                    "bbox": [40, 80, 474, 603],
                                    "lines": [{
                                        "spans": [{
                                            "image_path": "https://cdn-mineru.openxlab.org.cn/result/2026-07-31/xxx/ca60e37801412d69a9f951331d55251f756ed365a0d4f5001b0e46c01916da51.jpg"
                                        }]
                                    }],
                                },
                                {
                                    "type": "image_footnote",
                                    "bbox": [63, 610, 159, 622],
                                    "lines": [{
                                        "spans": [{
                                            "content": "一 内景 第二六八窟"
                                        }]
                                    }],
                                },
                            ],
                        }
                    ],
                }
            ],
            "_backend": "vlm",
        }
        json_path = tmp_path / "real_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json(json_path)
        expected_name = "ca60e37801412d69a9f951331d55251f756ed365a0d4f5001b0e46c01916da51.jpg"
        assert expected_name in result
        caption, page = result[expected_name]
        assert caption == "一 内景 第二六八窟"
        assert page == 32


# ===========================================================================
# 2. content_list.json 解析测试（简化版格式）
# ===========================================================================

class TestParseContentList:
    """测试 _parse_content_list：从 MinerU content_list.json 提取图片和图注。"""

    def test_basic_content_list(self, tmp_path):
        """基本 content_list 结构。"""
        data = [
            {"type": "text", "text": "标题", "page_idx": 0},
            {
                "type": "image",
                "img_path": "images/fig1.jpg",
                "image_caption": ["图1 说明文字"],
                "image_footnote": [],
                "page_idx": 1,
            },
        ]
        json_path = tmp_path / "test_content_list.json"
        _write_json(data, json_path)

        result = _parse_content_list(json_path)
        assert "fig1.jpg" in result
        caption, page = result["fig1.jpg"]
        assert caption == "图1 说明文字"
        assert page == 1

    def test_multiple_captions_joined(self, tmp_path):
        """多个图注元素应拼接为一个字符串。"""
        data = [{
            "type": "image",
            "img_path": "images/fig1.jpg",
            "image_caption": ["图1", "局部细节"],
            "page_idx": 0,
        }]
        json_path = tmp_path / "test_content_list.json"
        _write_json(data, json_path)

        result = _parse_content_list(json_path)
        caption, _ = result["fig1.jpg"]
        assert "图1" in caption
        assert "局部细节" in caption

    def test_empty_caption_list(self, tmp_path):
        """image_caption 为空列表：caption 为空字符串。"""
        data = [{
            "type": "image",
            "img_path": "images/fig1.jpg",
            "image_caption": [],
            "page_idx": 0,
        }]
        json_path = tmp_path / "test_content_list.json"
        _write_json(data, json_path)

        result = _parse_content_list(json_path)
        assert result["fig1.jpg"][0] == ""


# ===========================================================================
# 3. 图片文件名提取测试
# ===========================================================================

class TestExtractImageNameFromUrl:
    """测试 _extract_image_name_from_url：从 URL 或路径提取文件名。"""

    def test_cdn_url(self):
        url = "https://cdn-mineru.openxlab.org.cn/result/2026-07-31/xxx/abc123.jpg"
        assert _extract_image_name_from_url(url) == "abc123.jpg"

    def test_relative_path(self):
        assert _extract_image_name_from_url("images/fig1.png") == "fig1.png"

    def test_plain_filename(self):
        assert _extract_image_name_from_url("test.jpg") == "test.jpg"

    def test_url_with_query_params(self):
        url = "https://example.com/path/img.png?token=abc&expires=123"
        assert _extract_image_name_from_url(url) == "img.png"


# ===========================================================================
# 4. JSON 文件查找测试
# ===========================================================================

class TestFindMiddleJson:
    """测试 _find_middle_json：查找 MinerU ZIP 内的 JSON 文件。"""

    def test_find_middle_json(self, tmp_path):
        """优先找到 *_middle.json。"""
        (tmp_path / "doc_middle.json").write_text("{}", encoding="utf-8")
        (tmp_path / "doc_content_list.json").write_text("[]", encoding="utf-8")
        (tmp_path / "full.md").write_text("# title", encoding="utf-8")

        result = _find_middle_json(tmp_path)
        assert result is not None
        assert result.name == "doc_middle.json"

    def test_fallback_to_content_list(self, tmp_path):
        """无 middle.json 时降级找 content_list.json。"""
        (tmp_path / "doc_content_list.json").write_text("[]", encoding="utf-8")
        (tmp_path / "full.md").write_text("# title", encoding="utf-8")

        result = _find_middle_json(tmp_path)
        assert result is not None
        assert result.name == "doc_content_list.json"

    def test_no_json_files(self, tmp_path):
        """无任何 JSON 文件应返回 None。"""
        (tmp_path / "full.md").write_text("# title", encoding="utf-8")
        assert _find_middle_json(tmp_path) is None

    def test_ignores_model_json(self, tmp_path):
        """model.json 不应被选中（既非 middle 也非 content_list）。"""
        (tmp_path / "doc_model.json").write_text("{}", encoding="utf-8")
        assert _find_middle_json(tmp_path) is None


# ===========================================================================
# 5. PDF 切分判断测试（mock文件大小和页数，不依赖真实PDF）
# ===========================================================================

class TestShouldSplitPdf:
    """测试 _should_split_pdf：判断是否需要切分。"""

    def test_small_file_no_split(self, tmp_path):
        """小文件不应切分。"""
        pdf = tmp_path / "small.pdf"
        pdf.write_bytes(b"x" * 100)  # 100字节
        with patch("app.core.pdf_image_extractor.get_settings") as mock_settings:
            mock_settings.return_value.pdf_split_size_mb = 150
            mock_settings.return_value.pdf_split_page_threshold = 150
            with patch("pypdf.PdfReader") as mock_reader:
                mock_reader.return_value.pages = [0] * 10  # 10页
                assert _should_split_pdf(str(pdf)) is False

    def test_large_size_triggers_split(self, tmp_path):
        """超体积阈值应切分。"""
        pdf = tmp_path / "large.pdf"
        pdf.write_bytes(b"x" * (160 * 1024 * 1024))  # 160MB > 150MB
        with patch("app.core.pdf_image_extractor.get_settings") as mock_settings:
            mock_settings.return_value.pdf_split_size_mb = 150
            mock_settings.return_value.pdf_split_page_threshold = 150
            assert _should_split_pdf(str(pdf)) is True

    def test_many_pages_triggers_split(self, tmp_path):
        """超页数阈值应切分。"""
        pdf = tmp_path / "manypages.pdf"
        pdf.write_bytes(b"x" * 100)
        with patch("app.core.pdf_image_extractor.get_settings") as mock_settings:
            mock_settings.return_value.pdf_split_size_mb = 150
            mock_settings.return_value.pdf_split_page_threshold = 150
            with patch("pypdf.PdfReader") as mock_reader:
                mock_reader.return_value.pages = [0] * 200  # 200页 > 150
                assert _should_split_pdf(str(pdf)) is True


# ===========================================================================
# 6. Markdown 降级匹配测试
# ===========================================================================

class TestMarkdownFallback:
    """测试 _build_image_caption_map_from_markdown：middle.json 缺失时的降级方案。"""

    def test_basic_markdown_match(self, tmp_path):
        """简单Markdown：图片引用+前后文本。"""
        md = tmp_path / "test.md"
        md.write_text(
            "这是前文\n"
            "![图1说明](images/fig1.png)\n"
            "这是后文",
            encoding="utf-8",
        )
        result = _build_image_caption_map_from_markdown(md)
        assert "fig1.png" in result
        assert "图1说明" in result["fig1.png"]
        assert "这是前文" in result["fig1.png"]
        assert "这是后文" in result["fig1.png"]

    def test_empty_markdown(self, tmp_path):
        """空Markdown应返回空dict。"""
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        assert _build_image_caption_map_from_markdown(md) == {}

    def test_nonexistent_file(self):
        """文件不存在应返回空。"""
        assert _build_image_caption_map_from_markdown(Path("nonexistent.md")) == {}


# ===========================================================================
# 7. 坐标匹配降级测试（核心：无 caption 时基于空间邻近度匹配）
# ===========================================================================

def _img_dict(name: str, bbox: list, page_idx: int = 0) -> dict:
    """构造 _match_by_coordinates 输入格式的图片 dict。"""
    return {"name": name, "caption": "", "bbox": bbox, "page_idx": page_idx}


def _text_dict(content: str, bbox: list, page_idx: int = 0) -> dict:
    """构造 _match_by_coordinates 输入格式的文本块 dict。"""
    return {"content": content, "bbox": bbox, "page_idx": page_idx}


class TestMatchByCoordinates:
    """测试 _match_by_coordinates：无 caption 图片的坐标匹配降级。"""

    def test_side_by_side_layout(self):
        """并排版式：3图横排 + 3注横排，每个注精准匹配到正上方的图。

        对应用户原始诉求：第1行[图1,图2,图3]，第2行[注1,注2,注3]。
        每个注的 x 范围只与正上方的图重叠，不会误配到相邻图。
        """
        images = [
            _img_dict("fig1.jpg", bbox=[100, 100, 300, 400]),  # x:[100,300], y:[100,400]
            _img_dict("fig2.jpg", bbox=[400, 100, 600, 400]),  # x:[400,600]
            _img_dict("fig3.jpg", bbox=[700, 100, 900, 400]),  # x:[700,900]
        ]
        texts = [
            _text_dict("注1 須摩提女請佛故事之三", bbox=[150, 420, 290, 450]),
            _text_dict("注2 須摩提女請佛故事之二", bbox=[450, 420, 590, 450]),
            _text_dict("注3 第二五七窟西壁", bbox=[750, 420, 890, 450]),
        ]

        result = _match_by_coordinates(images, texts)
        assert result["fig1.jpg"] == "注1 須摩提女請佛故事之三"
        assert result["fig2.jpg"] == "注2 須摩提女請佛故事之二"
        assert result["fig3.jpg"] == "注3 第二五七窟西壁"

    def test_caption_above_image(self):
        """图注在图片上方：应通过上方匹配兜底逻辑命中。"""
        images = [_img_dict("fig.jpg", bbox=[100, 400, 300, 700])]
        texts = [_text_dict("上图说明", bbox=[150, 350, 290, 380])]

        result = _match_by_coordinates(images, texts)
        assert result["fig.jpg"] == "上图说明"

    def test_multiple_paragraphs_one_image(self):
        """图下方多个段落：应按 y 升序拼接为同一图的 caption。"""
        images = [_img_dict("fig.jpg", bbox=[100, 100, 300, 400])]
        # 故意乱序，验证按 y 排序后拼接
        texts = [
            _text_dict("第二段", bbox=[100, 500, 300, 520]),
            _text_dict("第一段", bbox=[100, 420, 300, 440]),
        ]

        result = _match_by_coordinates(images, texts)
        assert result["fig.jpg"] == "第一段 第二段"

    def test_different_pages_not_matched(self):
        """不同页的图和文本不应跨页匹配。"""
        images = [_img_dict("fig.jpg", bbox=[100, 100, 300, 400], page_idx=0)]
        texts = [_text_dict("第1页注", bbox=[150, 420, 290, 450], page_idx=1)]

        result = _match_by_coordinates(images, texts)
        assert result == {}

    def test_no_x_overlap_not_matched(self):
        """x 坐标无重叠的图和文本不应匹配（避免误配到相邻列）。"""
        images = [_img_dict("fig.jpg", bbox=[100, 100, 300, 400])]
        # 文本在图下方但 x 完全不重叠
        texts = [_text_dict("不相干的段落", bbox=[500, 420, 700, 450])]

        result = _match_by_coordinates(images, texts)
        assert result == {}

    def test_nearest_image_selected(self):
        """图下方有多个候选图时，文本应归属综合距离最近的图。

        文本 x 范围与两个图都重叠，但与图2距离更近，应匹配图2。
        """
        images = [
            _img_dict("fig1.jpg", bbox=[100, 100, 400, 400]),  # 中心 x=250
            _img_dict("fig2.jpg", bbox=[300, 100, 600, 400]),  # 中心 x=450
        ]
        # 文本 x:[350,450]，中心 x=400，与两图都重叠，但距图2更近
        texts = [_text_dict("归属图2的注", bbox=[350, 420, 450, 450])]

        result = _match_by_coordinates(images, texts)
        assert result["fig2.jpg"] == "归属图2的注"
        assert "fig1.jpg" not in result

    def test_unmatched_text_skipped(self):
        """无归属图片的文本（如页脚）应被跳过，不出现在结果中。"""
        images = [_img_dict("fig.jpg", bbox=[100, 100, 300, 400])]
        texts = [
            _text_dict("图注", bbox=[150, 420, 290, 450]),
            _text_dict("64", bbox=[1471, 2047, 1489, 2062]),  # 远离图片的页脚
        ]

        result = _match_by_coordinates(images, texts)
        assert result == {"fig.jpg": "图注"}

    def test_empty_inputs(self):
        """空输入应返回空 dict。"""
        assert _match_by_coordinates([], []) == {}
        assert _match_by_coordinates([_img_dict("a.jpg", [0, 0, 10, 10])], []) == {}
        assert _match_by_coordinates([], [_text_dict("x", [0, 0, 10, 10])]) == {}


class TestParseMiddleJsonFull:
    """测试 _parse_middle_json_full：解析完整坐标信息供坐标匹配使用。"""

    def test_image_with_caption_and_text_blocks(self, tmp_path):
        """含 caption 的图片 + 独立文本块：应分别收集。"""
        middle_data = _make_middle_json([
            {"page_idx": 0, "para_blocks": [
                _make_image_block("images/fig1.jpg", "图1 caption"),
                {
                    "type": "text",
                    "bbox": [100, 500, 300, 520],
                    "lines": [{"spans": [{"content": "独立段落"}]}],
                },
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json_full(json_path)
        assert len(result["images"]) == 1
        assert result["images"][0]["name"] == "fig1.jpg"
        assert result["images"][0]["caption"] == "图1 caption"
        assert result["images"][0]["page_idx"] == 0
        assert len(result["text_blocks"]) == 1
        assert result["text_blocks"][0]["content"] == "独立段落"

    def test_image_without_caption_collected_for_matching(self, tmp_path):
        """无 caption 的图片应被收集（caption 为空），供坐标匹配降级使用。"""
        middle_data = _make_middle_json([
            {"page_idx": 0, "para_blocks": [
                _make_image_block("images/no_cap.jpg", caption=""),
            ]},
        ])
        json_path = tmp_path / "test_middle.json"
        _write_json(middle_data, json_path)

        result = _parse_middle_json_full(json_path)
        assert len(result["images"]) == 1
        assert result["images"][0]["caption"] == ""

    def test_invalid_json_returns_empty(self, tmp_path):
        """非法 JSON 应返回空结构，不抛异常。"""
        json_path = tmp_path / "bad.json"
        json_path.write_text("not json", encoding="utf-8")
        result = _parse_middle_json_full(json_path)
        assert result == {"images": [], "text_blocks": []}

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """文件不存在应返回空结构。"""
        result = _parse_middle_json_full(tmp_path / "nonexistent.json")
        assert result == {"images": [], "text_blocks": []}

    def test_full_pipeline_image_to_coordinate_match(self, tmp_path):
        """端到端：middle.json 解析 → 无 caption 图片 → 坐标匹配补充 caption。"""
        # 构造并排版式：2张图横排 + 2个注横排，均无 image_footnote
        middle_data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "type": "image",
                            "bbox": [100, 100, 300, 400],
                            "blocks": [
                                {"type": "image_body", "lines": [{"spans": [{"image_path": "images/fig1.jpg"}]}]},
                            ],
                        },
                        {
                            "type": "image",
                            "bbox": [400, 100, 600, 400],
                            "blocks": [
                                {"type": "image_body", "lines": [{"spans": [{"image_path": "images/fig2.jpg"}]}]},
                            ],
                        },
                        {
                            "type": "text",
                            "bbox": [150, 420, 290, 450],
                            "lines": [{"spans": [{"content": "注1 说明"}]}],
                        },
                        {
                            "type": "text",
                            "bbox": [450, 420, 590, 450],
                            "lines": [{"spans": [{"content": "注2 说明"}]}],
                        },
                    ],
                }
            ],
        }
        json_path = tmp_path / "real_middle.json"
        _write_json(middle_data, json_path)

        # 模拟 _extract_single_pdf 中的降级流程
        full = _parse_middle_json_full(json_path)
        uncaptioned = [img for img in full["images"] if not img["caption"]]
        matched = _match_by_coordinates(uncaptioned, full["text_blocks"])

        assert matched["fig1.jpg"] == "注1 说明"
        assert matched["fig2.jpg"] == "注2 说明"


# ===========================================================================
# 7. 并发解析测试（mock _extract_single_pdf，验证合并顺序和错误隔离）
# ===========================================================================

class TestConcurrentExtract:
    """测试 extract_images_from_pdf 的并发解析逻辑。

    通过 mock _extract_single_pdf 和 _should_split_pdf / _split_pdf / _cleanup_split_tmp，
    验证并发场景下的结果合并顺序和错误隔离，不调真实 MinerU API。
    """

    def test_split_results_merged_in_order(self, tmp_path):
        """切分后多个切片的结果按页码顺序合并（不按完成顺序）。"""
        pdf_path = tmp_path / "big.pdf"
        pdf_path.write_bytes(b"x" * 100)

        # 模拟 3 个切片分别返回不同图片
        def mock_extract(path, stem, storage_root):
            # 按 stem 后缀区分不同切片的返回
            if stem.endswith("part000"):
                return [ExtractedImage(file_path="a.jpg", caption="A", image_name="a.jpg", page_number=0)]
            elif stem.endswith("part001"):
                return [ExtractedImage(file_path="b.jpg", caption="B", image_name="b.jpg", page_number=1)]
            elif stem.endswith("part002"):
                return [ExtractedImage(file_path="c.jpg", caption="C", image_name="c.jpg", page_number=2)]
            return []

        with patch("app.core.pdf_image_extractor._should_split_pdf", return_value=True), \
             patch("app.core.pdf_image_extractor._split_pdf", return_value=[tmp_path / "p0.pdf", tmp_path / "p1.pdf", tmp_path / "p2.pdf"]), \
             patch("app.core.pdf_image_extractor._extract_single_pdf", side_effect=mock_extract), \
             patch("app.core.pdf_image_extractor._cleanup_split_tmp"), \
             patch("app.core.pdf_image_extractor.get_settings") as mock_settings:
            mock_settings.return_value.pdf_concurrent_workers = 2
            mock_settings.return_value.image_storage_dir = str(tmp_path)

            results = extract_images_from_pdf(str(pdf_path))

        # 即使并发执行完成顺序不确定，最终结果也应按 part000→part001→part002 顺序合并
        assert len(results) == 3
        assert results[0].image_name == "a.jpg"
        assert results[1].image_name == "b.jpg"
        assert results[2].image_name == "c.jpg"

    def test_single_part_failure_isolated(self, tmp_path):
        """单切片失败不影响其他切片（错误隔离）。"""
        pdf_path = tmp_path / "big.pdf"
        pdf_path.write_bytes(b"x" * 100)

        call_count = {"n": 0}

        def mock_extract(path, stem, storage_root):
            call_count["n"] += 1
            if stem.endswith("part001"):
                raise RuntimeError("模拟 MinerU API 失败")
            return [ExtractedImage(file_path=f"{stem}.jpg", caption=stem, image_name=f"{stem}.jpg", page_number=0)]

        with patch("app.core.pdf_image_extractor._should_split_pdf", return_value=True), \
             patch("app.core.pdf_image_extractor._split_pdf", return_value=[tmp_path / "p0.pdf", tmp_path / "p1.pdf", tmp_path / "p2.pdf"]), \
             patch("app.core.pdf_image_extractor._extract_single_pdf", side_effect=mock_extract), \
             patch("app.core.pdf_image_extractor._cleanup_split_tmp"), \
             patch("app.core.pdf_image_extractor.get_settings") as mock_settings:
            mock_settings.return_value.pdf_concurrent_workers = 2
            mock_settings.return_value.image_storage_dir = str(tmp_path)

            results = extract_images_from_pdf(str(pdf_path))

        # 3 个切片都被调用
        assert call_count["n"] == 3
        # part001 失败被跳过，part000 和 part002 的结果正常返回
        assert len(results) == 2

    def test_no_split_uses_single_extract(self, tmp_path):
        """无需切分时走单文件直解路径，不走并发。"""
        pdf_path = tmp_path / "small.pdf"
        pdf_path.write_bytes(b"x" * 100)

        single_called = {"n": 0}

        def mock_extract(path, stem, storage_root):
            single_called["n"] += 1
            return [ExtractedImage(file_path="x.jpg", caption="X", image_name="x.jpg", page_number=0)]

        with patch("app.core.pdf_image_extractor._should_split_pdf", return_value=False), \
             patch("app.core.pdf_image_extractor._extract_single_pdf", side_effect=mock_extract), \
             patch("app.core.pdf_image_extractor.get_settings") as mock_settings:
            mock_settings.return_value.image_storage_dir = str(tmp_path)

            results = extract_images_from_pdf(str(pdf_path))

        assert single_called["n"] == 1
        assert len(results) == 1
        assert results[0].image_name == "x.jpg"
