"""莆田博物馆导入器。

xlsx 结构：文物名称 / 介绍
图片目录：图片/ 子目录，文件名即文物名（如"清康熙青花釉里红梅雀纹炉.jpg"）

介绍字段含完整文物描述（形制、釉色、胎质等），作为 caption 主体。
"""
from pathlib import Path

from . import xlsx_name_import_museum

MUSEUM_NAME = "莆田博物馆"
XLSX_FILENAME = "莆田博物馆.xlsx"


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return xlsx_name_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        xlsx_filename=XLSX_FILENAME,
        id_prefix="莆田",
        name_col="文物名称",
        intro_col="介绍",
        name_as_image_filename=True,
        dry_run=dry_run,
    )
