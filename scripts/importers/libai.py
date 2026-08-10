"""李白纪念馆导入器。

无 xlsx，图片按类别分目录存放：
- 书画/：大量近现代书画作品（信函、条幅、镜片等）
- 其他/：瓷器、木器、玉石等杂项
- 古籍文献/：刻本、文集、方志等
- 碑刻/：石碑、雕像

文件名即文物名（如"1962年刘君礼李白诗意画《采莲曲》条幅.jpg"），
子目录名作为 category_top。

注意：书画类文件名常含年份前缀（如"1962年..."），这部分信息对检索有意义，
保留在 name 中（不剥离年份），GLM-4 补全时能从年份推断朝代为"现代"。
"""
from pathlib import Path

from . import category_subdir_import_museum

MUSEUM_NAME = "李白纪念馆"

# 子目录名 → 一级分类映射
# 李白纪念馆的子目录：书画/其他/古籍文献/碑刻
_CATEGORY_MAP = {
    "书画": "书画",
    "其他": "杂项",
    "古籍文献": "古籍文献",
    "碑刻": "石刻",
}


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    return category_subdir_import_museum(
        src_dir=src_dir,
        dst_raw=dst_raw,
        museum_name=MUSEUM_NAME,
        id_prefix="李白",
        category_to_top=_CATEGORY_MAP,
        dry_run=dry_run,
    )
