"""苏州戏曲博物馆导入器。

xlsx 结构：文物名称 / 图片名称
两种数据模式：
  1. 文物名称是分组名（"馆藏文物剪影举隅——XXX"），图片名称是具体文物名（短文本）
     → 用图片名称作为 name，文物名称拼入 caption
  2. 文物名称是具体文物名（如"清代苏州三善堂《西游记曲谱》"），图片名称是长描述
     → 用文物名称作为 name，图片名称作为 caption

无图片，仅 caption 数据。
"""
from pathlib import Path
import pandas as pd

from . import build_relic_metadata

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


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """苏州戏曲博物馆：根据图片名称长度判断是文物名还是描述。"""
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path)

    metadata = {}
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
        else:
            # 模式 1：图片名称是具体 name，文物名称是分组（拼入 caption）
            name = img_name if img_name else group_name
            caption_parts = [group_name] if group_name and group_name != name else []

        if not name:
            continue

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            extra_caption_parts=caption_parts if caption_parts else None,
        )

    print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（无图片，仅 caption 数据）")
    return metadata
