"""安阳博物馆导入器。

xlsx 格式：标准表头（标题/标题1/文本/文本1/图片链接_保存位置），无跳过行。
图片结构：resource/ 目录下按时间戳命名的 jpeg 文件。

注意：xlsx 里"图片链接_保存位置"是 Windows 路径（如 D:\\八爪鱼下载\\xxx.jpeg），
在 Linux 服务器上需先统一为正斜杠再取 basename。
"""
import shutil
from pathlib import Path

import pandas as pd

from . import build_relic_metadata, copy_image_if_changed

MUSEUM_NAME = "安阳博物馆"
XLSX_FILENAME = "青铜器_安博藏品_安阳博物馆.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """导入安阳博物馆数据。

    Returns: {product_id: {name, caption, ...}}
    """
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    resource_dir = src_dir / "resource"
    metadata = {}
    total = 0
    copied = 0
    products_with_images: set[str] = set()
    orphan_products: set[str] = set()

    for idx, row in df.iterrows():
        name = str(row["标题1"]).strip() if pd.notna(row["标题1"]) else ""
        if not name:
            continue

        # product_id 用"安阳_序号"避免与包头编号冲突
        product_id = f"安阳_{idx + 1:03d}"

        # 从"图片链接_保存位置"提取文件名
        # 注意：Windows 路径在 Linux 上需先 replace("\\", "/")
        save_path = str(row["图片链接_保存位置"]) if pd.notna(row["图片链接_保存位置"]) else ""
        img_filename = Path(save_path.replace("\\", "/")).name if save_path else ""

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            category_top=str(row["标题"]).strip() if pd.notna(row["标题"]) else None,
            extra_caption_parts=[
                str(row["文本"]).strip() if pd.notna(row["文本"]) else None,    # 尺寸
                str(row["文本1"]).strip() if pd.notna(row["文本1"]) else None,  # 详细描述
            ],
        )
        metadata[product_id]["source_file"] = img_filename

        # 复制对应图片
        has_img = False
        if img_filename:
            src_img = resource_dir / img_filename
            if src_img.exists():
                dst_img = dst_raw / MUSEUM_NAME / product_id / src_img.name
                total += 1
                if not dry_run:
                    if copy_image_if_changed(src_img, dst_img):
                        copied += 1
                else:
                    copied += 1
                has_img = True

        if has_img:
            products_with_images.add(product_id)
        else:
            orphan_products.add(product_id)

    # 过滤孤儿记录
    orphan_count = len(orphan_products)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        suffix = f"（复制 {copied} 张变化）" if copied != total else ""
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），图片 {total} 张{suffix}")
    else:
        suffix = f"（复制 {copied} 张变化）" if copied != total else ""
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata
