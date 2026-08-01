"""
pytest 公共夹具：用临时目录隔离测试数据，避免污染真实的 data/ 目录。

核心思路：
- monkeypatch Settings 的路径字段（processed_data_dir / image_storage_dir / chroma_persist_dir / eval_dataset_dir）
- 清空 tag_store / image_vectorstore 的模块级缓存（lru_cache / 全局变量），确保每个测试用例拿到干净的临时存储
- 每个测试用例独立的临时目录，互不影响
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core import config, tag_store, image_vectorstore


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """每个测试用例用独立的临时目录，避免污染真实数据。"""
    # 构造临时目录结构
    processed_dir = tmp_path / "processed"
    image_dir = tmp_path / "images"
    chroma_dir = tmp_path / "chroma"
    eval_dir = tmp_path / "eval"
    for d in (processed_dir, image_dir, chroma_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    # monkeypatch Settings 实例的路径字段
    settings = config.get_settings()
    monkeypatch.setattr(settings, "processed_data_dir", str(processed_dir))
    monkeypatch.setattr(settings, "image_storage_dir", str(image_dir))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(chroma_dir))
    monkeypatch.setattr(settings, "eval_dataset_dir", str(eval_dir))

    # 清空 tag_store 模块级缓存（global _tag_index）
    tag_store._tag_index = None

    # 清空 image_vectorstore 的 lru_cache（get_image_collection 单例缓存旧 collection 句柄）
    image_vectorstore.get_image_collection.cache_clear()

    yield

    # 测试结束清理（保险，tmp_path 会自动删除）
    tag_store._tag_index = None
    image_vectorstore.get_image_collection.cache_clear()
