"""
image_indexer 图片索引器单元测试。

重点测试 delete_image（删除流程的三库一致性：注册表 + 向量库 + 标签倒排索引），
因为这是最容易出 bug 的地方——三个存储要同时清理，漏一个就会产生脏数据。

测试策略：
- tag_store 用真实实现（依赖 conftest 的临时目录隔离）
- image_vectorstore 和 image_embedder 用 mock（避免依赖 CLIP 模型和 Chroma 初始化）
- 直接操作注册表 JSON 文件构造测试数据，绕过 register_image（它依赖文件系统 hash）
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core import image_indexer
from app.core.image_indexer import delete_image, load_registered_images
from app.core.tag_store import add_image_tags, get_tags_by_image, search_by_tags
from app.models.image_schemas import ImageMetadata, ImageStatus


def _make_image_metadata(image_id: str, tags: list[str] = None) -> ImageMetadata:
    """构造测试用 ImageMetadata（绕过 register_image 的文件系统 hash 计算）。"""
    return ImageMetadata(
        image_id=image_id,
        product_id=f"prod_{image_id}",
        category="test_category",
        file_path=f"raw/{image_id}.jpg",
        tags=tags or [],
        width=100,
        height=100,
        status=ImageStatus.READY,
    )


def _seed_registry(images: dict[str, ImageMetadata]) -> None:
    """直接写入注册表 JSON 文件（绕过 register_image）。"""
    image_indexer.save_registered_images(images)


class TestDeleteImage:
    """delete_image：删除流程的三库一致性。"""

    def test_delete_existing_image(self):
        """删除存在的图片：返回 True，注册表、向量库、标签索引都清理。"""
        _seed_registry({
            "img1": _make_image_metadata("img1", tags=["青铜", "兵器"]),
            "img2": _make_image_metadata("img2", tags=["青铜"]),
        })
        # 手动写入标签索引（绕过 index_image，它依赖 CLIP）
        add_image_tags("img1", ["青铜", "兵器"])
        add_image_tags("img2", ["青铜"])

        with patch("app.core.image_indexer.delete_image_vectors") as mock_delete_vec:
            result = delete_image("img1")

        assert result is True
        mock_delete_vec.assert_called_once_with("img1")  # 向量库被清理

        # 注册表中 img1 已删除，img2 保留
        registry = load_registered_images()
        assert "img1" not in registry
        assert "img2" in registry

        # 标签索引中 img1 已清理
        assert get_tags_by_image("img1") == []
        assert search_by_tags(["青铜"]) == ["img2"]  # img2 保留

    def test_delete_nonexistent_image(self):
        """删除不存在的图片：返回 False，不报错。"""
        _seed_registry({"img1": _make_image_metadata("img1")})

        with patch("app.core.image_indexer.delete_image_vectors") as mock_delete_vec:
            result = delete_image("nonexistent")

        assert result is False
        mock_delete_vec.assert_not_called()  # 没有调用向量删除

        # 注册表不变
        registry = load_registered_images()
        assert "img1" in registry

    def test_delete_idempotent(self):
        """删除是幂等的：第二次删除已删除的图片返回 False。"""
        _seed_registry({"img1": _make_image_metadata("img1", tags=["青铜"])})
        add_image_tags("img1", ["青铜"])

        with patch("app.core.image_indexer.delete_image_vectors"):
            assert delete_image("img1") is True
            assert delete_image("img1") is False  # 第二次删除返回 False

    def test_delete_clears_all_tags(self):
        """删除图片后，该图片的所有标签关联都被清理。"""
        _seed_registry({"img1": _make_image_metadata("img1", tags=["青铜", "兵器", "古风"])})
        add_image_tags("img1", ["青铜", "兵器", "古风"])

        with patch("app.core.image_indexer.delete_image_vectors"):
            delete_image("img1")

        # img1 不应出现在任何标签的倒排列表中
        assert get_tags_by_image("img1") == []
        assert search_by_tags(["青铜"]) == []
        assert search_by_tags(["兵器"]) == []
        assert search_by_tags(["古风"]) == []

    def test_delete_preserves_other_images_tags(self):
        """删除一张图片不影响其他图片的标签索引。"""
        _seed_registry({
            "img1": _make_image_metadata("img1", tags=["青铜"]),
            "img2": _make_image_metadata("img2", tags=["青铜", "陶瓷"]),
        })
        add_image_tags("img1", ["青铜"])
        add_image_tags("img2", ["青铜", "陶瓷"])

        with patch("app.core.image_indexer.delete_image_vectors"):
            delete_image("img1")

        # img2 的标签不受影响
        assert set(get_tags_by_image("img2")) == {"青铜", "陶瓷"}
        assert search_by_tags(["青铜"]) == ["img2"]
        assert search_by_tags(["陶瓷"]) == ["img2"]

    def test_delete_no_tags_image(self):
        """删除无标签的图片：正常清理注册表和向量库，标签索引无需操作。"""
        _seed_registry({"img1": _make_image_metadata("img1", tags=[])})

        with patch("app.core.image_indexer.delete_image_vectors") as mock_delete_vec:
            result = delete_image("img1")

        assert result is True
        mock_delete_vec.assert_called_once_with("img1")
        assert "img1" not in load_registered_images()

    def test_delete_removes_empty_tag_bucket(self):
        """删除图片后，若某 tag 的倒排列表为空，该 tag 被移除（不留空 key）。"""
        _seed_registry({"img1": _make_image_metadata("img1", tags=["唯一标签"])})
        add_image_tags("img1", ["唯一标签"])

        with patch("app.core.image_indexer.delete_image_vectors"):
            delete_image("img1")

        from app.core.tag_store import load_tag_index
        index = load_tag_index()
        assert "唯一标签" not in index  # 空 tag bucket 被移除


class TestLoadSaveRegistry:
    """注册表加载/保存的基础测试。"""

    def test_load_empty_registry(self):
        """注册表不存在时返回空 dict。"""
        registry = load_registered_images()
        assert registry == {}

    def test_save_and_load_roundtrip(self):
        """保存后重新加载，数据一致。"""
        images = {
            "img1": _make_image_metadata("img1", tags=["青铜"]),
            "img2": _make_image_metadata("img2", tags=["陶瓷"]),
        }
        image_indexer.save_registered_images(images)

        loaded = load_registered_images()
        assert set(loaded.keys()) == {"img1", "img2"}
        assert loaded["img1"].tags == ["青铜"]
        assert loaded["img2"].tags == ["陶瓷"]
        assert loaded["img1"].status == ImageStatus.READY

    def test_save_overwrites_existing(self):
        """保存覆盖：第二次保存会完全替换第一次的数据。"""
        _seed_registry({"img1": _make_image_metadata("img1")})
        _seed_registry({"img2": _make_image_metadata("img2")})  # 覆盖

        registry = load_registered_images()
        assert "img1" not in registry
        assert "img2" in registry
