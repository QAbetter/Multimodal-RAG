"""
批量索引脚本：扫描 data/images/raw/ 目录，批量注册并索引所有图片。

支持两种模式：
1. 增量模式（默认）：与 images.json 注册表比对，只索引新增或上次失败的图片，
   跳过已索引且状态为 READY 的图片。适合日常新增图片后快速更新索引。
2. 全量模式（--full）：先清空向量库和注册表，再全量重新索引。适合切换模型、
   修复脏数据、或想彻底重建索引的场景。

product_id 规则：用图片文件名（去扩展名）作为 product_id，如 "青铜剑.jpg" → "青铜剑"。
image_id 规则：用文件内容 MD5 前 16 位，相同图片（不同文件名）不会重复索引。

用法：
    # 增量索引（默认，只索引新增/失败的图片）
    $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
    .venv\\Scripts\\python.exe scripts\\batch_index_images.py

    # 全量重建（先清空再全量索引）
    .venv\\Scripts\\python.exe scripts\\batch_index_images.py --full

    # 指定扫描目录（默认 data/images/raw/）
    .venv\\Scripts\\python.exe scripts\\batch_index_images.py --dir data/images/raw

    # 指定图片类别（所有图片统一打上该类别，不指定则为 None）
    .venv\\Scripts\\python.exe scripts\\batch_index_images.py --category product
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.image_indexer import (
    batch_index_images,
    batch_register_images,
    load_registered_images,
    save_registered_images,
)
from app.core.image_vectorstore import reset_image_collection

# 支持的图片格式
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_dataset_metadata(path: str | None) -> dict:
    """加载 dataset_metadata.json（import_dataset.py 生成）。

    返回 {product_id: {dynasty, material, category_top, ...}} 映射。
    文件不存在或未指定路径时返回空字典（回退到 GLM-4V 模式）。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[!] 元数据文件不存在: {path}（将使用 GLM-4V 提取标签）")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def infer_product_id(file_path: Path, raw_dir: Path) -> str:
    """推断 product_id。

    新结构 raw/{博物馆}/{藏品编号}/{文件名} → 用藏品编号（第二级子目录）
    旧结构 raw/{类别}/{文件名} 或 raw/{文件名} → 用文件名 stem（保持兼容）

    例：
      raw/包头博物馆/000001/xxx.jpg → "000001"
      raw/安阳博物馆/安阳_001/xxx.jpg → "安阳_001"
      raw/瓷器/李白.jpg → "李白"（旧结构兼容）
    """
    rel = file_path.relative_to(raw_dir)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[1]  # 藏品编号
    return file_path.stem  # 旧结构兼容


def scan_image_files(raw_dir: Path) -> list[Path]:
    """扫描目录及所有子目录下的图片文件，按相对路径排序。

    使用 rglob 递归扫描，支持 data/images/raw/瓷器/xxx.jpg 这种子目录结构。
    跳过 pdf/ 子目录（由 batch_index_pdf.py 独立管理，避免覆盖 caption/pdf_source）
    和 PDF 处理的临时目录。
    """
    if not raw_dir.exists():
        return []
    skip_dirs = {"pdf", ".pdf_split_tmp", ".pdf_extract_tmp"}
    results = []
    for p in raw_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
            continue
        rel_parts = p.relative_to(raw_dir).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        results.append(p)
    return sorted(results, key=lambda p: str(p.relative_to(raw_dir)))


def infer_category(file_path: Path, raw_dir: Path, explicit_category: str | None) -> str | None:
    """推断图片类别：显式指定优先，否则用第一级子目录名。

    例：raw/瓷器/李白.jpg → "瓷器"；raw/李白.jpg → None。
    """
    if explicit_category:
        return explicit_category
    rel = file_path.relative_to(raw_dir)
    if len(rel.parts) > 1:  # 在子目录下
        return rel.parts[0]
    return None


def run_incremental(raw_dir: Path, category: str | None, batch_size: int, tag_workers: int,
                    metadata: dict | None = None) -> None:
    """增量索引：只索引新增或失败的图片，跳过已 READY 的图片。"""
    metadata = metadata or {}
    registry = load_registered_images()
    files = scan_image_files(raw_dir)

    if not files:
        print(f"[!] 目录下没有图片: {raw_dir}")
        return

    print(f"[增量模式] 扫描到 {len(files)} 张图片，注册表中已有 {len(registry)} 条记录")
    if metadata:
        print(f"          dataset_metadata.json 提供 {len(metadata)} 条文博元数据（跳过 GLM-4V）")
    print("-" * 60)

    # 第一遍：批量注册（并行算 MD5+尺寸，只读写注册表 1 次）
    # skip_existing_ready=True：已 READY 的图片不会被覆盖
    print(f"批量注册 {len(files)} 张图片（并行计算 MD5+尺寸）...")
    register_items = []
    for file_path in files:
        rel_path = f"raw/{file_path.relative_to(raw_dir)}"
        pid = infer_product_id(file_path, raw_dir)
        cat = infer_category(file_path, raw_dir, category)

        item = {
            "file_path": rel_path,
            "product_id": pid,
            "category": cat,
        }

        # 注入 dataset_metadata.json 的文博字段（跳过 GLM-4V）
        if pid in metadata:
            meta = metadata[pid]
            item["tags"] = meta.get("tags") or []
            item["dynasty"] = meta.get("dynasty")
            item["material"] = meta.get("material")
            item["category_top"] = meta.get("category_top")
            item["category_sub"] = meta.get("category_sub")
            item["relic_condition"] = meta.get("relic_condition")
            item["caption"] = meta.get("caption")
            # category 优先用 metadata 的 category_top（更准确）
            if meta.get("category_top"):
                item["category"] = meta["category_top"]

        register_items.append(item)

    registered, skipped_imgs = batch_register_images(register_items, skip_existing_ready=True)
    print(f"  注册 {len(registered)} | 跳过已索引 {len(skipped_imgs)} | 共 {len(files)}")

    # 去重：相同内容（相同 image_id）在 registered 里只保留第一个
    to_index = []  # (image_id, file_name)
    seen_ids: set[str] = set()
    dup_skipped = 0
    for img in registered:
        if img.image_id in seen_ids:
            dup_skipped += 1
        else:
            to_index.append((img.image_id, Path(img.file_path).name))
            seen_ids.add(img.image_id)

    if not to_index:
        print("-" * 60)
        print(f"[完成] 跳过 {len(skipped_imgs)} | 重复跳过 {dup_skipped} | 共 {len(files)}（无需索引新图片）")
        return

    # 第二遍：批量索引（CLIP 批量向量化 + 标签并发提取）
    print("-" * 60)
    print(f"批量索引 {len(to_index)} 张图片（batch_size={batch_size}, tag_workers={tag_workers}）...")
    image_ids = [iid for iid, _ in to_index]
    name_map = {iid: name for iid, name in to_index}

    results = batch_index_images(image_ids, batch_size=batch_size, tag_workers=tag_workers)

    indexed = sum(1 for r in results if r.status.value == "ready")
    failed = sum(1 for r in results if r.status.value != "ready")

    for r in results:
        status_icon = "✓" if r.status.value == "ready" else "✗"
        tags_str = f"标签={r.tags[:3]}" if r.tags else "无标签"
        print(f"  {status_icon} {name_map.get(r.image_id, r.image_id[:8])} 状态={r.status.value} {tags_str}")

    print("-" * 60)
    print(f"[完成] 成功 {indexed} | 跳过 {len(skipped_imgs)} | 重复跳过 {dup_skipped} | 失败 {failed} | 共 {len(files)}")


def run_full(raw_dir: Path, category: str | None, batch_size: int, tag_workers: int,
             metadata: dict | None = None) -> None:
    """全量重建：清空向量库和注册表，再全量索引。"""
    metadata = metadata or {}
    files = scan_image_files(raw_dir)

    if not files:
        print(f"[!] 目录下没有图片: {raw_dir}")
        return

    print(f"[全量模式] 将清空向量库并重新索引 {len(files)} 张图片")
    if metadata:
        print(f"          dataset_metadata.json 提供 {len(metadata)} 条文博元数据（跳过 GLM-4V）")
    print("-" * 60)

    # 1. 清空向量库
    print("[1/2] 清空 images collection...")
    reset_image_collection()
    print("      ✓ 已清空")

    # 2. 清空注册表
    print("[2/2] 清空注册表 images.json...")
    save_registered_images({})
    print("      ✓ 已清空")

    # 3. 批量注册所有图片（并行算 MD5+尺寸，只读写注册表 1 次）
    print("-" * 60)
    print(f"批量注册 {len(files)} 张图片（并行计算 MD5+尺寸）...")
    register_items = []
    for file_path in files:
        rel_path = f"raw/{file_path.relative_to(raw_dir)}"
        pid = infer_product_id(file_path, raw_dir)
        cat = infer_category(file_path, raw_dir, category)

        item = {
            "file_path": rel_path,
            "product_id": pid,
            "category": cat,
        }

        # 注入 dataset_metadata.json 的文博字段（跳过 GLM-4V）
        if pid in metadata:
            meta = metadata[pid]
            item["tags"] = meta.get("tags") or []
            item["dynasty"] = meta.get("dynasty")
            item["material"] = meta.get("material")
            item["category_top"] = meta.get("category_top")
            item["category_sub"] = meta.get("category_sub")
            item["relic_condition"] = meta.get("relic_condition")
            item["caption"] = meta.get("caption")
            # category 优先用 metadata 的 category_top（更准确）
            if meta.get("category_top"):
                item["category"] = meta["category_top"]

        register_items.append(item)

    registered, _ = batch_register_images(register_items, skip_existing_ready=False)

    # 去重：相同内容（相同 image_id）只索引第一个
    image_ids = []
    name_map = {}
    seen_ids: set[str] = set()
    dup_skipped = 0
    for img in registered:
        if img.image_id in seen_ids:
            dup_skipped += 1
            continue
        seen_ids.add(img.image_id)
        image_ids.append(img.image_id)
        name_map[img.image_id] = Path(img.file_path).name

    # 4. 批量索引（CLIP 批量向量化 + 标签并发提取）
    print(f"批量索引 {len(image_ids)} 张图片（batch_size={batch_size}, tag_workers={tag_workers}）...")
    results = batch_index_images(image_ids, batch_size=batch_size, tag_workers=tag_workers)

    indexed = sum(1 for r in results if r.status.value == "ready")
    failed = sum(1 for r in results if r.status.value != "ready")

    for r in results:
        status_icon = "✓" if r.status.value == "ready" else "✗"
        tags_str = f"标签={r.tags[:3]}" if r.tags else "无标签"
        print(f"  {status_icon} {name_map.get(r.image_id, r.image_id[:8])} 状态={r.status.value} {tags_str}")

    print("-" * 60)
    print(f"[完成] 成功 {indexed} | 重复跳过 {dup_skipped} | 失败 {failed} | 共 {len(files)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量索引图片")
    parser.add_argument("--full", action="store_true", help="全量重建（清空向量库和注册表后重新索引）")
    parser.add_argument("--dir", type=str, default=None, help="图片扫描目录（默认 data/images/raw/）")
    parser.add_argument("--category", type=str, default=None, help="统一设置图片类别（不设置则按第一级子目录名自动推断，如 raw/瓷器/x.jpg → 瓷器）")
    parser.add_argument("--metadata", type=str, default="data/processed/dataset_metadata.json",
                        help="dataset_metadata.json 路径（import_dataset.py 生成，提供文博元数据跳过 GLM-4V）")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP 批量向量化的批大小（默认 32）")
    parser.add_argument("--tag-workers", type=int, default=8, help="GLM-4V 标签提取并发数（默认 8，限流可降到 4）")
    args = parser.parse_args()

    print("=" * 60)
    print("图片批量索引")
    print("=" * 60)

    settings = get_settings()
    raw_dir = Path(args.dir) if args.dir else Path(settings.image_storage_dir) / "raw"
    raw_dir = raw_dir.resolve()
    metadata = load_dataset_metadata(args.metadata)
    print(f"扫描目录: {raw_dir}")
    print(f"元数据: {args.metadata}（{len(metadata)} 条）" if metadata else f"元数据: {args.metadata}（未加载，将用 GLM-4V）")
    print(f"类别设置: {args.category or '(无)'}")
    print(f"批大小: {args.batch_size}")
    print(f"标签并发数: {args.tag_workers}")

    start_time = time.time()

    if args.full:
        run_full(raw_dir, args.category, args.batch_size, args.tag_workers, metadata)
    else:
        run_incremental(raw_dir, args.category, args.batch_size, args.tag_workers, metadata)

    elapsed = time.time() - start_time
    print(f"耗时: {elapsed:.1f} 秒")
    print("=" * 60)

    # Chroma 的 Posthog 遥测线程在 Linux 下是非 daemon 线程，会阻止解释器退出。
    # 数据已全部落盘，直接 _exit(0) 绕过清理，避免脚本结束后挂起。
    # Windows 下该线程是 daemon，无此问题，所以本地不受影响。
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
