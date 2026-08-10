"""包头博物馆导入器。

xlsx 格式：前3行是表头（填报说明+一级表头+二级表头），数据从第4行开始。
图片结构：按藏品编号分目录，每个编号下有多角度图片（A/B/C/F）。

新增包头同类博物馆（同一系统导出的 xlsx）时，可复制此文件修改 MUSEUM_NAME 和 xlsx 文件名。
"""
import shutil
from pathlib import Path

import pandas as pd

from . import IMAGE_EXTS, build_relic_metadata

MUSEUM_NAME = "包头博物馆"
XLSX_FILENAME = "15020311800001内蒙古包头博物馆.xlsx"
SKIP_ROWS = 3  # 前3行是表头


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """导入包头博物馆数据。

    Returns: {product_id: {name, dynasty, material, ...}}
    """
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path, header=None, skiprows=SKIP_ROWS)

    metadata = {}
    for _, row in df.iterrows():
        product_id = str(row[2]).strip() if pd.notna(row[2]) else None
        if not product_id or product_id == "nan":
            continue

        name = str(row[3]).strip() if pd.notna(row[3]) else ""
        if not name:
            continue

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            dynasty=str(row[6]).strip() if pd.notna(row[6]) else None,
            category_top=str(row[10]).strip() if pd.notna(row[10]) else None,
            material=str(row[13]).strip() if pd.notna(row[13]) else None,
            relic_condition=str(row[25]).strip() if pd.notna(row[25]) else None,
            extra_caption_parts=[
                str(row[18]).strip() if pd.notna(row[18]) else None,  # 具体尺寸
            ],
        )

    # 复制图片：data/images/raw/{博物馆}/{藏品编号}/
    count = 0
    products_with_images: set[str] = set()
    for product_id in metadata:
        src_img_dir = src_dir / product_id
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_raw / MUSEUM_NAME / product_id
        if not dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
        has_img = False
        for img in src_img_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                if not dry_run:
                    shutil.copy2(img, dst_img_dir / img.name)
                count += 1
                has_img = True
        if has_img:
            products_with_images.add(product_id)

    # 过滤孤儿记录（xlsx 有记录但磁盘无图片）
    orphan_count = len(metadata) - len(products_with_images)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），复制图片 {count} 张")
    else:
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条，复制图片 {count} 张")
    return metadata
