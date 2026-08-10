"""青白江博物馆导入器。

无 xlsx，图片按类别分目录存放：
- 玉器/石器/青铜器/瓷器/陶器/其他

文件名即文物名（如"东汉五铢铜钱.jpg"、"清乾隆通宝铜钱.jpg"），
子目录名作为 category_top（映射到标准一级分类）。

青白江博物馆铜钱类文物较多（汉五铢、唐开元、宋各年号、清各年号），
文件名含朝代+钱币名，GLM-4 能从名称提取 dynasty。
"""
from pathlib import Path

from . import category_subdir_import_museum

MUSEUM_NAME = "青白江博物馆"

# 子目录名 → 一级分类映射
_CATEGORY_MAP = {
    "玉器": "玉器",
    "石器": "石刻",
    "青铜器": "青铜器",
    "瓷器": "陶瓷器",
    "陶器": "陶瓷器",
    "其他": "杂项",
}


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return category_subdir_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        id_prefix="青白江",
        category_to_top=_CATEGORY_MAP,
        dry_run=dry_run,
    )
