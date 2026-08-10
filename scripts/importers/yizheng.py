"""仪征博物馆导入器。

xlsx 结构：文物名称 / 介绍
图片组织：按类别分子目录（书画/瓷器/金属器/玉石器/陶器/漆竹木骨角牙器）
        另有"仪征市博物馆馆藏精品文物"子目录存放部分精品

xlsx 里的"介绍"字段含尺寸、出土信息、形制描述，作为 caption 主体。
"""
from pathlib import Path

from . import xlsx_name_import_museum

MUSEUM_NAME = "仪征博物馆"
XLSX_FILENAME = "仪征博物馆.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return xlsx_name_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        xlsx_filename=XLSX_FILENAME,
        id_prefix="仪征",
        name_col="文物名称",
        intro_col="介绍",
        dry_run=dry_run,
    )
