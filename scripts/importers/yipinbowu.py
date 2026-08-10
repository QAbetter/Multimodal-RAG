"""懿品博悟导入器。

xlsx 结构：名称 / 图片(内嵌) / 年代 / 博物馆 / 简介 / 包含纹样 / 纹样介绍
图片来源优先级：
  1. 图片/ 目录的原图（文件名是哈希值，用图片相似度匹配到 xlsx 行）
  2. xlsx 内嵌图片（300x210，匹配失败时兜底）

特点：
- 已有"年代"字段（如"清康熙"），可直接用作 dynasty
- "包含纹样"含纹样关键词，"纹样介绍"含详细描述，拼入 caption
- "博物馆"列记录来源博物馆（如"故宫博物院"），也拼入 caption
"""
from pathlib import Path
import pandas as pd

from . import (
    build_relic_metadata,
    extract_xlsx_embedded_images,
    match_embedded_to_dir_images,
    copy_image_if_changed,
)

MUSEUM_NAME = "懿品博悟"
XLSX_FILENAME = "懿品博悟.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """懿品博悟：xlsx 有名称+年代+纹样信息，图片优先用 图片/目录原图。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    # 提取 xlsx 内嵌图片
    row_to_emb = extract_xlsx_embedded_images(xlsx_path)

    # 匹配到 图片/ 目录的原图
    img_dir = src_dir / "图片"
    row_to_orig = match_embedded_to_dir_images(row_to_emb, img_dir)

    metadata = {}
    total = 0
    copied = 0
    used_original = 0
    used_embedded = 0

    for idx, row in df.iterrows():
        name = str(row["名称"]).strip() if pd.notna(row.get("名称")) else ""
        if not name:
            continue
        product_id = f"懿品_{idx + 1:03d}"

        dynasty = str(row["年代"]).strip() if pd.notna(row.get("年代")) and row.get("年代") else None
        src_museum = str(row["博物馆"]).strip() if pd.notna(row.get("博物馆")) and row.get("博物馆") else None
        patterns = str(row["包含纹样"]).strip() if pd.notna(row.get("包含纹样")) and row.get("包含纹样") else None
        pattern_intro = str(row["纹样介绍"]).strip() if pd.notna(row.get("纹样介绍")) and row.get("纹样介绍") else None

        caption_parts = [p for p in [src_museum, patterns, pattern_intro] if p and p != "无"]

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            dynasty=dynasty,
            extra_caption_parts=caption_parts if caption_parts else None,
        )

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
