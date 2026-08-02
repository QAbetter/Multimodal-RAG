"""
图片索引主流程：注册元数据 -> 预处理 -> CLIP 向量化 -> 标签提取 -> 写入向量库。

对应书籍 RAG 的 indexer.py，保持完全一致的注册表 + 状态机模式：
- JSON 文件持久化注册表（images.json），与书籍的 books.json 同级
- 状态机迁移：PENDING → EXTRACTING_TAGS → INDEXING → READY / FAILED
- 重新索引前先清理旧向量（delete_image_vectors），与 delete_book_vectors 同思路

索引链路：
    1. 注册 ImageMetadata（写入 images.json，status=PENDING）
    2. 加载图片 + 预处理（EXIF 修正 + 缩放到 224×224）
    3. CLIP 图像 embedding（status=INDEXING）
    4. GLM-4V 标签提取（status=EXTRACTING_TAGS，失败降级为空列表）
    5. 写入 Chroma images collection（向量 + metadata payload）
    6. status=READY，更新 images.json
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import get_settings
from app.core.image_embedder import embed_image, embed_images
from app.core.image_loader import (
    compute_image_id,
    generate_thumbnail,
    get_image_size,
    load_and_preprocess,
)
from app.core.image_vectorstore import (
    add_image_vector,
    batch_add_image_vectors,
    batch_delete_image_vectors,
    delete_image_vectors,
)
from app.core.image_bm25_store import (
    build_image_bm25,
    remove_image_bm25,
    reset_image_bm25,
)
from app.core.cultural_relic_aliases import structured_fields_to_tags
from app.core.locks import registry_lock
from app.core.tag_extractor import extract_relic_metadata, RelicMetadata, _empty_metadata
from app.core.tag_store import (
    add_image_tags,
    remove_image_tags,
    reset_tag_index,
)
from app.models.image_schemas import ImageMetadata, ImageStatus

logger = logging.getLogger(__name__)


def _apply_relic_metadata(image: ImageMetadata, relic: RelicMetadata) -> None:
    """把 GLM-4V 提取的文博元数据应用到 ImageMetadata 对象。

    原字段（category/tags/caption）保留：
    - tags：来自 relic["tags"]（与原 extract_tags 行为一致）
    - caption：若 image.caption 已有值（如 PDF 提取的图注），保留原值不覆盖；
      若为空，用 caption_standard（标准著录描述）填充，用于文本检索
    - category：若 image.category 已有值（手动指定），保留；否则用 category_top 填充
    """
    image.tags = relic["tags"]
    # caption 保留逻辑：PDF 提取的图注优先，否则用 GLM-4V 的标准著录描述
    if not image.caption and relic["caption_standard"]:
        image.caption = relic["caption_standard"]
    # category 保留逻辑：手动指定优先，否则用一级分类
    if not image.category and relic["category_top"]:
        image.category = relic["category_top"]
    # 文博结构化字段
    image.caption_standard = relic["caption_standard"] or None
    image.caption_public = relic["caption_public"] or None
    image.category_top = relic["category_top"] or None
    image.category_sub = relic["category_sub"] or None
    image.dynasty = relic["dynasty"] or None
    image.material = relic["material"] or None
    image.color_feature = relic["color_feature"] or None
    image.craft = relic["craft"] or None
    image.pattern_theme = relic["pattern_theme"]
    image.function_usage = relic["function_usage"] or None
    image.relic_condition = relic["relic_condition"] or None

    # 把结构化字段转为 "命名空间:值" 的标签追加到 tags，并入标签倒排索引
    # 这样查询端用 parse_structured_tags 解析 query 后能精确命中这些字段
    structured_tags = structured_fields_to_tags(
        dynasty=image.dynasty,
        material=image.material,
        category_sub=image.category_sub,
        craft=image.craft,
        function_usage=image.function_usage,
        relic_condition=image.relic_condition,
        color_feature=image.color_feature,
    )
    if structured_tags:
        image.tags = list(image.tags) + structured_tags


def _images_registry_path() -> Path:
    """图片注册表路径：data/processed/images.json，与书籍的 books.json 同级。"""
    settings = get_settings()
    path = Path(settings.processed_data_dir) / "images.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_registered_images() -> dict[str, ImageMetadata]:
    """加载图片注册表（image_id -> ImageMetadata）。"""
    path = _images_registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {image_id: ImageMetadata(**data) for image_id, data in raw.items()}


def save_registered_images(images: dict[str, ImageMetadata]) -> None:
    """保存图片注册表到 JSON。"""
    path = _images_registry_path()
    payload = {image_id: json.loads(img.model_dump_json()) for image_id, img in images.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def register_image(
    file_path: str,
    product_id: str,
    category: str | None = None,
    tags: list[str] | None = None,
    caption: str | None = None,
    pdf_source: str | None = None,
    page_number: int | None = None,
) -> ImageMetadata:
    """注册图片元数据（索引前先注册，与书籍的 register_book 同流程）。

    file_path 为相对 image_storage_dir 的路径。
    如果 tags 手动传入，则跳过 GLM-4V 自动提取。
    caption/pdf_source/page_number 用于 PDF 提取的图片：caption 是图片对应的文本，
    用于基于文本查图（BM25 检索）；pdf_source/page_number 用于溯源。

    并发安全：注册表的读-改-写用 registry_lock 保护，避免并发注册丢失。
    """
    settings = get_settings()
    abs_path = Path(settings.image_storage_dir) / file_path
    image_id = compute_image_id(str(abs_path))
    width, height = get_image_size(str(abs_path))

    image = ImageMetadata(
        image_id=image_id,
        product_id=product_id,
        category=category,
        file_path=file_path,
        tags=tags or [],
        width=width,
        height=height,
        status=ImageStatus.PENDING,
        caption=caption,
        pdf_source=pdf_source,
        page_number=page_number,
    )
    with registry_lock:
        images = load_registered_images()
        images[image_id] = image
        save_registered_images(images)
    return image


def get_image(image_id: str) -> ImageMetadata | None:
    return load_registered_images().get(image_id)


def index_image(image_id: str) -> ImageMetadata:
    """对已注册的图片执行索引，状态机迁移：PENDING → EXTRACTING_TAGS → INDEXING → READY/FAILED。

    与 index_book() 同结构：try 块内完成核心步骤，except 标记 FAILED，finally 落盘。

    并发安全：注册表的读-改-写用 registry_lock 保护；耗时操作（标签提取、CLIP 向量化）
    在锁外执行，避免长时间持锁降低并发度。
    """
    settings = get_settings()

    # 加锁读取注册表，取出 image 对象后释放锁（耗时操作在锁外执行）
    with registry_lock:
        images = load_registered_images()
        image = images.get(image_id)
        if image is None:
            raise ValueError(f"图片未注册: {image_id}")
        # 标记为 EXTRACTING_TAGS 并落盘
        image.status = ImageStatus.EXTRACTING_TAGS
        images[image_id] = image
        save_registered_images(images)

    abs_path = Path(settings.image_storage_dir) / image.file_path

    try:
        # 1. 文博元数据提取（若注册时未手动指定 tags）—— 锁外执行（GLM-4V API 耗时）
        #    一次调用同时产出：结构化字段 + tags + caption
        if not image.tags:
            relic_meta = extract_relic_metadata(str(abs_path))
            _apply_relic_metadata(image, relic_meta)

        # 2. 预处理 + CLIP 向量化 —— 锁外执行（模型前向耗时）
        with registry_lock:
            image.status = ImageStatus.INDEXING
            images = load_registered_images()
            images[image_id] = image
            save_registered_images(images)

        img = load_and_preprocess(str(abs_path))
        vector = embed_image(img)

        # 3. 生成缩略图 —— 锁外执行（IO 耗时）
        thumb_dir = Path(settings.image_storage_dir) / "thumbnails"
        image.thumbnail_path = generate_thumbnail(str(abs_path), str(thumb_dir), image_id)

        # 4. 写入向量库（先清理旧向量，避免 id 重复）
        delete_image_vectors(image_id)
        add_image_vector(image_id, vector, image.to_payload())

        # 5. 写入标签倒排索引（tag_store 内部自带 tag_index_lock）
        remove_image_tags(image_id)
        add_image_tags(image_id, image.tags)

        # 6. 写入 caption BM25 索引（用于基于文本查图，对应书籍 RAG 的 build_bm25_index）
        # 重索引前先清理旧 caption，避免重复（build_image_bm25 是 upsert，但显式清理更清晰）
        remove_image_bm25(image_id)
        if image.caption:
            build_image_bm25(image_id, image.caption)

        image.status = ImageStatus.READY
        logger.info(
            "图片索引完成: %s，标签=%s，类别=%s，caption=%s",
            image_id, image.tags, image.category,
            "有" if image.caption else "无",
        )
    except Exception:
        image.status = ImageStatus.FAILED
        logger.exception("图片索引失败: %s", image_id)
        raise
    finally:
        with registry_lock:
            images = load_registered_images()
            images[image_id] = image
            save_registered_images(images)

    return image


def batch_index_images(
    image_ids: list[str],
    batch_size: int = 32,
    tag_workers: int = 4,
) -> list[ImageMetadata]:
    """批量索引图片（CLIP 批量向量化 + 标签并发提取）。

    与 index_image 的核心区别：
    - CLIP 向量化用 embed_images 批量前向（一次处理 batch_size 张），而非循环 embed_image
    - 标签提取用 ThreadPoolExecutor 并发（tag_workers 个线程），而非串行
    - Chroma 写入用 batch_add_image_vectors，减少事务次数

    流程分两阶段：
    1. 并发提取标签（GLM-4V，tag_workers 并发）
    2. 按 batch_size 分批：批量加载 → 批量 CLIP 向量化 → 批量写 Chroma → 逐张缩略图

    单张图片失败不影响同批其他图片，失败图片标记 FAILED。

    tag_workers 建议：智谱 GLM-4V 有并发限制，4 并发较稳定，8 并发需测试。

    并发安全：注册表读-改-写用 registry_lock 保护；耗时操作（标签提取、CLIP 向量化）
    在锁外执行。每批次落盘前重新加载最新注册表并合并本批修改，避免覆盖其他线程的改动。
    """
    settings = get_settings()
    thumb_dir = Path(settings.image_storage_dir) / "thumbnails"

    # 加锁读取注册表（仅读，确认所有 image_id 都已注册）
    with registry_lock:
        images = load_registered_images()
        for image_id in image_ids:
            if image_id not in images:
                raise ValueError(f"图片未注册: {image_id}")
        # 筛选需要提取标签的图片（已有标签的跳过）
        need_tags = [iid for iid in image_ids if not images[iid].tags]
        # 标记状态
        for iid in need_tags:
            images[iid].status = ImageStatus.EXTRACTING_TAGS
        if need_tags:
            save_registered_images(images)

    # ===== 阶段 1：并发提取文博元数据（锁外执行，GLM-4V API 耗时）=====
    if need_tags:
        def _extract_one(image_id: str) -> tuple[str, RelicMetadata]:
            # 重新读取单张图片元数据（避免引用可能被其他线程修改的对象）
            with registry_lock:
                image = load_registered_images().get(image_id)
            if image is None:
                return image_id, _empty_metadata()
            abs_path = Path(settings.image_storage_dir) / image.file_path
            try:
                return image_id, extract_relic_metadata(str(abs_path))
            except Exception:
                logger.exception("文博元数据提取失败: %s", image_id)
                return image_id, _empty_metadata()  # 降级为空，不阻塞索引

        logger.info("并发提取 %d 张图片文博元数据（workers=%d）", len(need_tags), tag_workers)
        tag_results: dict[str, RelicMetadata] = {}
        with ThreadPoolExecutor(max_workers=tag_workers) as executor:
            for image_id, relic_meta in executor.map(_extract_one, need_tags):
                tag_results[image_id] = relic_meta

        # 加锁合并元数据结果到注册表
        with registry_lock:
            images = load_registered_images()
            for image_id, relic_meta in tag_results.items():
                if image_id in images:
                    _apply_relic_metadata(images[image_id], relic_meta)
            save_registered_images(images)

    # ===== 阶段 2：分批 CLIP 向量化 + 写入 =====
    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start:start + batch_size]
        batch_imgs = []
        valid_ids = []

        # 2a. 批量加载 + 预处理（锁外执行，IO 耗时）
        with registry_lock:
            images = load_registered_images()
            for image_id in batch_ids:
                if image_id not in images:
                    continue
                image = images[image_id]
                abs_path = Path(settings.image_storage_dir) / image.file_path
                try:
                    img = load_and_preprocess(str(abs_path))
                    batch_imgs.append(img)
                    valid_ids.append(image_id)
                    image.status = ImageStatus.INDEXING
                except Exception:
                    image.status = ImageStatus.FAILED
                    logger.exception("图片加载失败: %s", image_id)

            if not valid_ids:
                save_registered_images(images)
                continue

        # 2b. 批量 CLIP 向量化（一次前向，核心优化点，锁外执行）
        vectors = embed_images(batch_imgs)

        # 2c. 批量删除旧向量（锁外执行）
        batch_delete_image_vectors(valid_ids)

        # 2d. 逐张生成缩略图 + 组装写入数据（锁外执行，IO 耗时）
        items = []
        with registry_lock:
            images = load_registered_images()
            for image_id, vector in zip(valid_ids, vectors):
                if image_id not in images:
                    continue
                image = images[image_id]
                abs_path = Path(settings.image_storage_dir) / image.file_path
                try:
                    image.thumbnail_path = generate_thumbnail(str(abs_path), str(thumb_dir), image_id)
                    items.append({"id": image_id, "embedding": vector, "metadata": image.to_payload()})
                    image.status = ImageStatus.READY
                    logger.info("图片索引完成: %s，标签=%s", image_id, image.tags[:3])
                except Exception:
                    image.status = ImageStatus.FAILED
                    logger.exception("索引失败: %s", image_id)

            # 2e. 批量写入 Chroma（锁内调用以保证注册表状态与向量库一致，
            # Chroma 写入通常 <100ms 可接受）
            if items:
                batch_add_image_vectors(items)

            # 2f. 落盘注册表
            save_registered_images(images)

        # 标签索引写入放到锁外（tag_store 自带 tag_index_lock）
        # caption BM25 索引同样放锁外（image_bm25_store 自带 _lock）
        for item in items:
            image_id = item["id"]
            # 重新读取最新的 tags 和 caption（避免引用已被其他线程修改的对象）
            with registry_lock:
                image = load_registered_images().get(image_id)
                tags = list(image.tags) if image and image.tags else []
                caption = image.caption if image else None
            if tags:
                remove_image_tags(image_id)
                add_image_tags(image_id, tags)
            # caption BM25 索引（PDF 提取的图片才有 caption）
            remove_image_bm25(image_id)
            if caption:
                build_image_bm25(image_id, caption)

    # 最终读取一次注册表返回结果
    with registry_lock:
        images = load_registered_images()
        return [images[image_id] for image_id in image_ids if image_id in images]


def delete_image(image_id: str) -> bool:
    """删除图片：清理注册表 + 向量库 + 标签倒排索引。

    与书籍 RAG 的 delete_book 同思路，但图片 RAG 多一个标签倒排索引需要清理。

    返回 True 表示删除成功，False 表示图片不存在。
    删除是幂等的：图片不存在时返回 False，不抛异常。

    并发安全：注册表的读-改-写用 registry_lock 保护；标签索引的清理由
    tag_store 内部的 tag_index_lock 保护；向量库删除是单次调用，无需额外锁。
    """
    with registry_lock:
        images = load_registered_images()
        if image_id not in images:
            return False
        # 先从注册表移除（拿到锁后立即删，缩短窗口）
        del images[image_id]
        save_registered_images(images)

    # 1. 清理向量库（锁外执行，Chroma 操作可能较慢）
    delete_image_vectors(image_id)

    # 2. 清理标签倒排索引（tag_store 内部自带 tag_index_lock）
    remove_image_tags(image_id)

    # 3. 清理 caption BM25 索引（image_bm25_store 自带 _lock）
    remove_image_bm25(image_id)

    logger.info("图片删除完成: %s", image_id)
    return True
