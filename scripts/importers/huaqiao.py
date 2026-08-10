"""华侨博物院导入器。

xlsx 结构：标题 / 图片(NaN) / 大小 / 描述
图片：本数据集无图片（图片列全 NaN）
描述字段含完整文物描述，作为 caption 主体。

无图片的条目也保留（caption 有检索价值），与莆田博物馆同样处理。
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata

MUSEUM_NAME = "华侨博物院"
XLSX_FILENAME = "华侨博物院.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """华侨博物院：xlsx 有标题+描述但图片列全 NaN，仅构造 metadata。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    metadata = {}
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

    print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（无图片，仅 caption 数据）")
    return metadata
