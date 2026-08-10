"""华侨博物院导入器。

xlsx 结构：标题 / 图片(内嵌) / 大小 / 描述
图片：xlsx 单元格内嵌缩略图（pandas 读为 NaN），通过 extract_xlsx_embedded_images 提取
描述字段含完整文物描述，作为 caption 主体。
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata, extract_xlsx_embedded_images

MUSEUM_NAME = "华侨博物院"
XLSX_FILENAME = "华侨博物院.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """华侨博物院：xlsx 有标题+描述，图片从 xlsx 内嵌提取。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    # 提取 xlsx 内嵌图片：{行号(0-indexed): 图片bytes}
    row_to_img = extract_xlsx_embedded_images(xlsx_path)

    metadata = {}
    total = 0
    copied = 0

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

        # 用 xlsx 行号提取内嵌图片（df.iterrows 的 idx 是 0-indexed，对应 xlsx 行号 idx+1）
        img_bytes = row_to_img.get(idx + 1)  # +1 因为 xlsx 第1行是表头，df idx=0 对应 xlsx row=1
        if img_bytes:
            ext = ".jpeg"  # xlsx 内嵌图片通常是 jpeg
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
