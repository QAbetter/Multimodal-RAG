"""懿品博悟导入器。

xlsx 结构：名称 / 图片(NaN) / 年代 / 博物馆 / 简介 / 包含纹样 / 纹样介绍
特点：
- 已有"年代"字段（如"清康熙"、"清光绪"），可直接用作 dynasty，不需 GLM-4 补全
- 图片列全 NaN，无图片
- "简介"多为"无"，"包含纹样"含具体纹样关键词，作为 caption 主体
- "纹样介绍"含详细纹样描述，拼入 caption

无图片的条目也保留（年代+纹样信息有检索价值）。
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata

MUSEUM_NAME = "懿品博悟"
XLSX_FILENAME = "懿品博悟.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """懿品博悟：xlsx 有名称+年代+纹样信息但无图片。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    metadata = {}
    for idx, row in df.iterrows():
        name = str(row["名称"]).strip() if pd.notna(row.get("名称")) else ""
        if not name:
            continue
        product_id = f"懿品_{idx + 1:03d}"

        dynasty = str(row["年代"]).strip() if pd.notna(row.get("年代")) and row.get("年代") else None
        # "博物馆"列记录来源博物馆（如"故宫博物院"），作为 caption 一部分
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

    print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（无图片，含年代+纹样信息）")
    return metadata
