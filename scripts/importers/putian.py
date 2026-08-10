"""莆田博物馆导入器。

xlsx 结构：文物名称 / 介绍
图片：本数据集莆田目录下仅有 xlsx，无图片文件（介绍含完整描述，作 caption 主体）

注意：本数据集无图片，所有条目都会被 xlsx_name_import_museum 的孤儿过滤逻辑过滤掉。
需要让无图片的条目也保留（caption 信息仍有价值，未来可补图），故关闭过滤。
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata

MUSEUM_NAME = "莆田博物馆"
XLSX_FILENAME = "莆田博物馆.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """莆田博物馆：xlsx 有名称+介绍但无图片，直接构造 metadata（不过滤）。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    metadata = {}
    for idx, row in df.iterrows():
        name = str(row["文物名称"]).strip() if pd.notna(row.get("文物名称")) else ""
        if not name:
            continue
        product_id = f"莆田_{idx + 1:03d}"
        intro = str(row["介绍"]).strip() if pd.notna(row.get("介绍")) and row.get("介绍") else None
        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            extra_caption_parts=[intro] if intro else None,
        )

    print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（无图片，仅 caption 数据）")
    return metadata
