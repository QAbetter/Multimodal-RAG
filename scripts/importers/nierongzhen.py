"""聂荣臻元帅网馆藏文物导入器。

xlsx 结构：标题（文物名称）/ 图片 / 图片_保存位置（Windows 路径）/ 字段
图片目录：11-11 165327/图片/ 子目录（rglob 自动查找，无需指定）
"""
from pathlib import Path

from . import simple_import_museum

MUSEUM_NAME = "聂荣臻元帅网 -- 馆藏文物"
XLSX_FILENAME = "聂荣臻元帅网 -- 馆藏文物.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return simple_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        xlsx_filename=XLSX_FILENAME,
        id_prefix="聂荣臻",
        dry_run=dry_run,
    )
