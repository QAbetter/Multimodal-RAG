"""苏州戏曲博物馆导入器。

xlsx 结构：文物名称 / 图片名称
图片目录：图片/ 子目录，文件名对应 xlsx 的"图片名称"列（如"百灵唱片集（一）.jpg"）

两种数据模式：
  1. 文物名称是分组名（"馆藏文物剪影举隅——XXX"），图片名称是具体文物名
     → 用图片名称作为 name，文物名称拼入 caption
  2. 文物名称是具体文物名（如"清代苏州三善堂《西游记曲谱》"），图片名称是长描述
     → 用文物名称作为 name，图片名称作为 caption

用图片名称查找图片文件（图片名 = 文件名 stem）。
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata, copy_image_if_changed, IMAGE_EXTS

MUSEUM_NAME = "苏州戏曲博物馆"
XLSX_FILENAME = "苏州戏曲博物馆.xlsx"

# 判断"图片名称"是否为长描述（含句号或超过 40 字）
_DESC_PUNCT = {"。", "，", "“", "”", "！", "？"}


def _is_description(text: str) -> bool:
    """判断图片名称是否实际上是描述文本（而非文物名）。"""
    if not text:
        return False
    if len(text) > 40:
        return True
    return any(p in text for p in _DESC_PUNCT)


def _find_image(src_dir: Path, stem: str) -> Path | None:
    """在 src_dir/图片/ 下按 stem 查找图片文件（尝试多种扩展名）。"""
    img_dir = src_dir / "图片"
    for ext in IMAGE_EXTS | {e.upper() for e in IMAGE_EXTS}:
        matched = list(img_dir.glob(f"{stem}{ext}"))
        if matched:
            return matched[0]
    # rglob 兜底
    for ext in IMAGE_EXTS:
        matched = list(src_dir.rglob(f"{stem}{ext}"))
        if matched:
            return matched[0]
    return None


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """苏州戏曲博物馆：根据图片名称判断是文物名还是描述，并用图片名称查找图片。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    metadata = {}
    total = 0
    copied = 0
    products_with_images: set[str] = set()
    orphan_products: set[str] = set()

    for idx, row in df.iterrows():
        group_name = str(row["文物名称"]).strip() if pd.notna(row.get("文物名称")) else ""
        img_name = str(row["图片名称"]).strip() if pd.notna(row.get("图片名称")) else ""
        if not group_name and not img_name:
            continue

        product_id = f"苏州戏曲_{idx + 1:03d}"

        # 判断模式：图片名称是描述还是具体文物名
        if img_name and _is_description(img_name):
            # 模式 2：文物名称是 name，图片名称是描述
            name = group_name
            caption_parts = [img_name] if img_name else []
            img_stem = None  # 无对应图片文件名（描述不是文件名）
        else:
            # 模式 1：图片名称是具体 name，文物名称是分组（拼入 caption）
            name = img_name if img_name else group_name
            caption_parts = [group_name] if group_name and group_name != name else []
            img_stem = img_name  # 图片名称 = 文件名 stem

        if not name:
            continue

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            extra_caption_parts=caption_parts if caption_parts else None,
        )

        # 用 img_stem 查找图片
        has_img = False
        if img_stem:
            src_img = _find_image(src_dir, img_stem)
            if src_img:
                metadata[product_id]["source_file"] = src_img.name
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
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），图片 {total} 张"
              f"{'（复制 ' + str(copied) + ' 张变化）' if copied != total else ''}")
    else:
        suffix = f"（复制 {copied} 张变化）" if copied != total else ""
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata
