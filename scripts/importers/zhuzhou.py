"""株洲博物馆导入器。

无 xlsx，图片按类别分目录存放：
- 瓷器/陶器/铜器/玉器/金银器/铁器/石器/木/复合器

文件名即文物名（如"明"五子登科"铜镜.jpg"、"战国单钮铜矛.jpg"），
子目录名作为 category_top（映射到标准一级分类）。
"""
from pathlib import Path

from . import category_subdir_import_museum

MUSEUM_NAME = "株洲博物馆"

# 子目录名 → 一级分类映射（对齐 ImageMetadata 的 category_top 枚举）
_CATEGORY_MAP = {
    "瓷器": "陶瓷器",
    "陶器": "陶瓷器",
    "铜器": "青铜器",
    "玉器": "玉器",
    "金银器": "金银器",
    "铁器": "青铜器",  # 铁器归入金属器，用青铜器作为近似
    "石器": "石刻",
    "木": "漆器",  # 木器归入漆器类
    "复合器": "杂项",
}


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return category_subdir_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        id_prefix="株洲",
        category_to_top=_CATEGORY_MAP,
        dry_run=dry_run,
    )
