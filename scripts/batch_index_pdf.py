"""
批量 PDF 索引脚本：扫描指定目录下的所有 PDF，提取插图+对应文本并索引。

流程（每个 PDF）：
1. 调智谱同步文件解析 API，提取图片 + caption（Markdown 周边文本）
2. 注册到 ImageMetadata（含 caption/pdf_source）
3. 批量索引：CLIP 向量化 + GLM-4V 标签 + caption BM25 索引
4. 清理解析临时目录

用法：
    # 索引单个 PDF
    $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
    .venv\\Scripts\\python.exe scripts\\batch_index_pdf.py --pdf "data/raw/xxx.pdf"

    # 批量索引目录下所有 PDF（默认 data/raw/pdf/）
    .venv\\Scripts\\python.exe scripts\\batch_index_pdf.py --dir data/raw/pdf

    # 指定类别 + 自定义并发
    .venv\\Scripts\\python.exe scripts\\batch_index_pdf.py --dir data/raw/pdf --category 文物 --tag-workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.image_indexer import batch_index_images, register_image
from app.core.pdf_image_extractor import (
    cleanup_extract_temp,
    extract_images_from_pdf,
)


def index_one_pdf(
    pdf_path: Path,
    category: str | None,
    batch_size: int,
    tag_workers: int,
) -> dict:
    """索引单个 PDF，返回统计信息。"""
    print(f"\n[PDF] {pdf_path.name}")
    print("-" * 60)

    # 1. 提取图片 + caption
    print("  [1/3] 调智谱解析 API 提取图片+caption...")
    try:
        extracted = extract_images_from_pdf(str(pdf_path))
    except Exception as e:
        print(f"  [!] 解析失败: {e}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 1, "error": str(e)}

    if not extracted:
        print("  [!] 未提取到图片")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 1}

    with_caption = sum(1 for img in extracted if img.caption)
    print(f"      ✓ 提取 {len(extracted)} 张图片，{with_caption} 张有 caption")

    # 2. 注册（带 caption/pdf_source）+ 去重
    print("  [2/3] 注册图片元数据...")
    image_ids: list[str] = []
    name_map: dict[str, str] = {}
    for img in extracted:
        product_id = Path(img.image_name).stem
        image = register_image(
            file_path=img.file_path,
            product_id=product_id,
            category=category,
            caption=img.caption or None,
            pdf_source=pdf_path.name,
        )
        if image.image_id not in name_map:
            name_map[image.image_id] = img.image_name
            image_ids.append(image.image_id)

    if not image_ids:
        print("  [!] 无新图片需索引（全部已存在）")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 1}

    # 3. 批量索引
    print(f"  [3/3] 批量索引 {len(image_ids)} 张图片（CLIP+标签+caption BM25）...")
    results = batch_index_images(
        image_ids,
        batch_size=batch_size,
        tag_workers=tag_workers,
    )

    success = sum(1 for r in results if r.status.value == "ready")
    failed = len(results) - success

    for r in results:
        status_icon = "✓" if r.status.value == "ready" else "✗"
        cap_icon = "📝" if r.caption else "  "
        tags_str = f"标签={r.tags[:3]}" if r.tags else "无标签"
        name = name_map.get(r.image_id, r.image_id[:8])
        print(f"      {status_icon} {cap_icon} {name} {tags_str}")

    # 4. 清理临时目录
    cleanup_extract_temp(pdf_path.stem)

    print(f"  [完成] 成功 {success} | 失败 {failed} | 共 {len(results)}")
    return {"total": len(results), "success": success, "failed": failed, "skipped": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="批量索引 PDF 中的插图")
    parser.add_argument("--pdf", type=str, default=None, help="单个 PDF 文件路径")
    parser.add_argument("--dir", type=str, default=None, help="PDF 扫描目录（默认配置的 pdf_raw_dir: data/raw/pdf/）")
    parser.add_argument("--category", type=str, default=None, help="统一设置图片类别")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP 批量向量化的批大小")
    parser.add_argument("--tag-workers", type=int, default=8, help="GLM-4V 标签提取并发数（默认 8，限流可降到 4）")
    args = parser.parse_args()

    print("=" * 60)
    print("PDF 批量索引")
    print("=" * 60)

    settings = get_settings()

    # 收集待处理的 PDF 列表
    if args.pdf:
        pdf_files = [Path(args.pdf).resolve()]
        if not pdf_files[0].exists():
            print(f"[!] PDF 文件不存在: {pdf_files[0]}")
            return
    else:
        scan_dir = Path(args.dir).resolve() if args.dir else Path(settings.pdf_raw_dir)
        if not scan_dir.exists():
            print(f"[!] 扫描目录不存在: {scan_dir}")
            return
        pdf_files = sorted(scan_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[!] 目录下没有 PDF 文件: {scan_dir}")
            return

    print(f"待处理 PDF: {len(pdf_files)} 个")
    print(f"类别设置: {args.category or '(无)'}")
    print(f"批大小: {args.batch_size}")
    print(f"标签并发数: {args.tag_workers}")

    start_time = time.time()
    total_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] 处理中...")
        stats = index_one_pdf(pdf_path, args.category, args.batch_size, args.tag_workers)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[全部完成] PDF: {len(pdf_files)} | 图片: 成功 {total_stats['success']} | "
          f"失败 {total_stats['failed']} | 跳过 {total_stats['skipped']}")
    print(f"耗时: {elapsed:.1f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
