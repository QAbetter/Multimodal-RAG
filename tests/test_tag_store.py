"""
tag_store 标签倒排索引单元测试。

覆盖核心功能：
- add_image_tags / remove_image_tags：写入与清理（含幂等性）
- search_by_tags：精确检索（含命中数排序、top_k 截断）
- get_tags_by_image：反向查询
- get_tag_stats：统计信息
- reset_tag_index：清空
- 持久化：写后重新加载，数据一致
- 边界场景：空标签、空白标签、重复添加、删除不存在的图片
"""
from __future__ import annotations

import pytest

from app.core import tag_store


class TestAddImageTags:
    """add_image_tags 写入测试。"""

    def test_add_single_tag(self):
        """单标签写入：image_id 出现在对应 tag 的倒排列表中。"""
        tag_store.add_image_tags("img1", ["青铜"])
        index = tag_store.load_tag_index()
        assert index == {"青铜": ["img1"]}

    def test_add_multiple_tags(self):
        """多标签写入：image_id 出现在所有 tag 的倒排列表中。"""
        tag_store.add_image_tags("img1", ["青铜", "兵器", "古风"])
        index = tag_store.load_tag_index()
        assert index["青铜"] == ["img1"]
        assert index["兵器"] == ["img1"]
        assert index["古风"] == ["img1"]

    def test_add_same_tag_multiple_images(self):
        """多张图片共享同一标签：倒排列表按添加顺序追加。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.add_image_tags("img2", ["青铜"])
        index = tag_store.load_tag_index()
        assert index["青铜"] == ["img1", "img2"]

    def test_idempotent_add(self):
        """幂等性：同一 image_id 重复添加同一 tag 不会重复出现。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.add_image_tags("img1", ["青铜"])  # 重复添加
        index = tag_store.load_tag_index()
        assert index["青铜"] == ["img1"]  # 只出现一次

    def test_idempotent_add_moves_to_end(self):
        """幂等性的副作用：重复添加会把 image_id 移到列表末尾（先 remove 再 append）。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.add_image_tags("img2", ["青铜"])
        tag_store.add_image_tags("img1", ["青铜"])  # img1 重复添加
        index = tag_store.load_tag_index()
        assert index["青铜"] == ["img2", "img1"]  # img1 被移到末尾

    def test_add_empty_tags(self):
        """空标签列表：不做任何写入。"""
        tag_store.add_image_tags("img1", [])
        index = tag_store.load_tag_index()
        assert index == {}

    def test_add_whitespace_tags_filtered(self):
        """空白标签被过滤（strip 后为空则跳过）。"""
        tag_store.add_image_tags("img1", ["青铜", "   ", ""])
        index = tag_store.load_tag_index()
        assert index == {"青铜": ["img1"]}

    def test_add_tag_with_whitespace_stripped(self):
        """带空白的标签会被 strip。"""
        tag_store.add_image_tags("img1", ["  青铜  "])
        index = tag_store.load_tag_index()
        assert index == {"青铜": ["img1"]}


class TestRemoveImageTags:
    """remove_image_tags 清理测试。"""

    def test_remove_single_image(self):
        """删除图片：从所有 tag 的倒排列表中移除该 image_id。"""
        tag_store.add_image_tags("img1", ["青铜", "兵器"])
        tag_store.add_image_tags("img2", ["青铜"])  # img2 也在青铜下
        tag_store.remove_image_tags("img1")
        index = tag_store.load_tag_index()
        assert index == {"青铜": ["img2"]}  # img1 从青铜移除，兵器 tag 整个删除（空了）

    def test_remove_removes_empty_tag(self):
        """删除图片后若某 tag 倒排列表为空，则该 tag 被移除。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.remove_image_tags("img1")
        index = tag_store.load_tag_index()
        assert index == {}  # 青铜 tag 因空被移除

    def test_remove_nonexistent_image(self):
        """删除不存在的图片：不报错，不影响现有数据。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.remove_image_tags("nonexistent")  # 不存在
        index = tag_store.load_tag_index()
        assert index == {"青铜": ["img1"]}  # 数据不变

    def test_remove_then_readd(self):
        """删除后重新添加：数据恢复。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.remove_image_tags("img1")
        tag_store.add_image_tags("img1", ["青铜"])
        index = tag_store.load_tag_index()
        assert index == {"青铜": ["img1"]}


class TestSearchByTags:
    """search_by_tags 精确检索测试。"""

    def test_search_single_tag(self):
        """单标签检索：返回所有包含该标签的 image_id。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.add_image_tags("img2", ["青铜"])
        tag_store.add_image_tags("img3", ["陶瓷"])  # 不相关
        result = tag_store.search_by_tags(["青铜"])
        assert set(result) == {"img1", "img2"}

    def test_search_multiple_tags_sorted_by_hit_count(self):
        """多标签检索：按命中标签数降序排列。"""
        # img1 命中 2 个标签，img2 命中 1 个标签
        tag_store.add_image_tags("img1", ["青铜", "兵器"])
        tag_store.add_image_tags("img2", ["青铜"])
        result = tag_store.search_by_tags(["青铜", "兵器"])
        assert result[0] == "img1"  # 命中 2 个标签，排第 1
        assert result[1] == "img2"  # 命中 1 个标签，排第 2

    def test_search_with_top_k(self):
        """top_k 截断：只返回前 top_k 个结果。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.add_image_tags("img2", ["青铜"])
        tag_store.add_image_tags("img3", ["青铜"])
        result = tag_store.search_by_tags(["青铜"], top_k=2)
        assert len(result) == 2

    def test_search_empty_tags(self):
        """空标签列表：返回空列表。"""
        result = tag_store.search_by_tags([])
        assert result == []

    def test_search_nonexistent_tag(self):
        """查询不存在的标签：返回空列表。"""
        tag_store.add_image_tags("img1", ["青铜"])
        result = tag_store.search_by_tags(["不存在的标签"])
        assert result == []

    def test_search_stable_sort(self):
        """命中数相同时按 image_id 字典序稳定排序。"""
        tag_store.add_image_tags("zzz", ["青铜"])
        tag_store.add_image_tags("aaa", ["青铜"])
        tag_store.add_image_tags("mmm", ["青铜"])
        result = tag_store.search_by_tags(["青铜"])
        assert result == ["aaa", "mmm", "zzz"]


class TestGetTagsByImage:
    """get_tags_by_image 反向查询测试。"""

    def test_get_tags_for_image(self):
        """查询某图片的所有标签。"""
        tag_store.add_image_tags("img1", ["青铜", "兵器", "古风"])
        tags = tag_store.get_tags_by_image("img1")
        assert set(tags) == {"青铜", "兵器", "古风"}

    def test_get_tags_for_nonexistent_image(self):
        """查询不存在的图片：返回空列表。"""
        tags = tag_store.get_tags_by_image("nonexistent")
        assert tags == []


class TestGetTagStats:
    """get_tag_stats 统计信息测试。"""

    def test_stats_empty(self):
        """空索引统计。"""
        stats = tag_store.get_tag_stats()
        assert stats == {"total_tags": 0, "total_relations": 0, "avg_images_per_tag": 0}

    def test_stats_with_data(self):
        """有数据时的统计。"""
        tag_store.add_image_tags("img1", ["青铜", "兵器"])
        tag_store.add_image_tags("img2", ["青铜"])
        stats = tag_store.get_tag_stats()
        # 2 个标签（青铜、兵器），3 个关系（青铜→img1, 青铜→img2, 兵器→img1）
        assert stats["total_tags"] == 2
        assert stats["total_relations"] == 3
        assert stats["avg_images_per_tag"] == 1.5  # 3/2


class TestResetTagIndex:
    """reset_tag_index 清空测试。"""

    def test_reset_clears_index(self):
        """清空后索引为空。"""
        tag_store.add_image_tags("img1", ["青铜", "兵器"])
        tag_store.reset_tag_index()
        index = tag_store.load_tag_index()
        assert index == {}

    def test_reset_clears_cache(self):
        """清空后模块级缓存也清空。"""
        tag_store.add_image_tags("img1", ["青铜"])
        tag_store.reset_tag_index()
        assert tag_store._tag_index == {}


class TestPersistence:
    """持久化测试：写后重新加载缓存，数据一致。"""

    def test_save_and_reload(self):
        """写入后清空缓存重新加载，数据应一致。"""
        tag_store.add_image_tags("img1", ["青铜", "兵器"])
        tag_store.add_image_tags("img2", ["青铜"])

        # 清空缓存，强制从磁盘重新加载
        tag_store._tag_index = None
        index = tag_store.load_tag_index()

        assert index["青铜"] == ["img1", "img2"]
        assert index["兵器"] == ["img1"]

    def test_corrupted_file_resets_to_empty(self):
        """文件损坏时重置为空索引（容错）。"""
        from app.core.tag_store import _tag_index_path

        # 写入损坏的 JSON
        path = _tag_index_path()
        path.write_text("不是有效的 JSON", encoding="utf-8")
        tag_store._tag_index = None  # 清空缓存

        index = tag_store.load_tag_index()
        assert index == {}  # 容错处理，返回空索引
