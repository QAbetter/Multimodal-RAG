"""华侨博物院导入器。

xlsx 结构：标题 / 图片(内嵌缩略图) / 大小 / 描述
图片来源优先级：
  1. 图片/ 目录的原图（文件名是哈希值，用图片相似度匹配到 xlsx 行）
  2. xlsx 内嵌缩略图（194x275，匹配失败时兜底）

描述字段含完整文物描述，作为 caption 主体。
"""
from pathlib import Path
import pandas as pd

from . import (
    build_relic_metadata,
    extract_xlsx_embedded_images,
    match_embedded_to_dir_images,
    copy_image_if_changed,
)

MUSEUM_NAME = "华侨博物院"
XLSX_FILENAME = "华侨博物院.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """华侨博物院：xlsx 有标题+描述，图片优先用 图片/目录原图，内嵌图兜底。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    # 提取 xlsx 内嵌图片：{行号(0-indexed): 图片bytes}
    row_to_emb = extract_xlsx_embedded_images(xlsx_path)

    # 把内嵌图片匹配到 图片/ 目录的原图
    img_dir = src_dir / "图片"
    row_to_orig = match_embedded_to_dir_images(row_to_emb, img_dir)

    metadata = {}
    total = 0
    copied = 0
    used_original = 0
    used_embedded = 0

    for idx, row in df.iterrows():
        name = str(row["标题"]).strip() if pd.notna(row.get("标题")) else ""
        if not name:
            continue
        product_id = f"华侨_{idx + 1:03d}"
        desc = str(row["描述"]).strip() if pd.notna(row.get("描述")) and row.get("描述") else None
        size = str(row["大小"]).strip() if pd.notna(row.get("大小")) and row.get("大小") else None
        caption_parts = [p for p in [size, desc] if p]
        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            extra_caption_parts=caption_parts if caption_parts else None,
        )

        # xlsx 行号（idx 是 df 的 0-indexed，xlsx 第1行是表头，对应 row=idx+1）
        xlsx_row = idx + 1

        # 优先用 图片/目录原图
        orig_path = row_to_orig.get(xlsx_row)
        if orig_path:
            dst_img = dst_raw / MUSEUM_NAME / product_id / orig_path.name
            metadata[product_id]["source_file"] = orig_path.name
            total += 1
            used_original += 1
            if not dry_run:
                if copy_image_if_changed(orig_path, dst_img):
                    copied += 1
            else:
                copied += 1
            continue

        # 兜底：用内嵌图片
        emb_bytes = row_to_emb.get(xlsx_row)
        if emb_bytes:
            dst_img = dst_raw / MUSEUM_NAME / product_id / "image.jpeg"
            metadata[product_id]["source_file"] = dst_img.name
            total += 1
            used_embedded += 1
            if not dry_run:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if not dst_img.exists() or dst_img.read_bytes() != emb_bytes:
                    dst_img.write_bytes(emb_bytes)
                    copied += 1
            else:
                copied += 1

    print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条，图片 {total} 张"
          f"（原图 {used_original} + 内嵌 {used_embedded}）"
          f"{'，复制 ' + str(copied) + ' 张变化' if copied != total else ''}")
    return metadata
