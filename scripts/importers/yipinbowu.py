"""懿品博悟导入器。

xlsx 结构：名称 / 图片(内嵌) / 年代 / 博物馆 / 简介 / 包含纹样 / 纹样介绍
图片：xlsx 单元格内嵌图片（pandas 读为 NaN），通过 extract_xlsx_embedded_images 提取
特点：
- 已有"年代"字段（如"清康熙"），可直接用作 dynasty，不需 GLM-4 补全
- "包含纹样"含纹样关键词，"纹样介绍"含详细描述，拼入 caption
- "博物馆"列记录来源博物馆（如"故宫博物院"），也拼入 caption
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata, extract_xlsx_embedded_images

MUSEUM_NAME = "懿品博悟"
XLSX_FILENAME = "懿品博悟.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """懿品博悟：xlsx 有名称+年代+纹样信息，图片从 xlsx 内嵌提取。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    # 提取 xlsx 内嵌图片
    row_to_img = extract_xlsx_embedded_images(xlsx_path)

    metadata = {}
    total = 0
    copied = 0

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

        # 用 xlsx 行号提取内嵌图片
        img_bytes = row_to_img.get(idx + 1)
        if img_bytes:
            ext = ".jpeg"
            dst_img = dst_raw / MUSEUM_NAME / product_id / f"image{ext}"
            metadata[product_id]["source_file"] = dst_img.name
            total += 1
            if not dry_run:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if not dst_img.exists() or dst_img.read_bytes() != img_bytes:
                    dst_img.write_bytes(img_bytes)
                    copied += 1
            else:
                copied += 1

    suffix = f"（提取 {copied} 张变化）" if copied != total else ""
    print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata
