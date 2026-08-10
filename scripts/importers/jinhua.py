"""金华市博物馆导入器。

xlsx 结构：标题（文物名称）/ 图片 / 图片_保存位置（Windows 路径）
图片目录：11-11 193832/图片/ 子目录（rglob 自动查找，无需指定）
"""
from pathlib import Path

from . import simple_import_museum

MUSEUM_NAME = "金华市博物馆"
XLSX_FILENAME = "金华市博物馆.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return simple_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        xlsx_filename=XLSX_FILENAME,
        id_prefix="金华",
        dry_run=dry_run,
    )
