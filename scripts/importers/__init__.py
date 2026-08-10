"""博物馆导入器公共工具。

所有导入器（importers/*.py）共享的工具函数：
- extract_ware_type：从文物名称反查器型（二级分类）
- build_name_tags：名称整体作为标签
- IMAGE_EXTS：支持的图片扩展名
- MUSEUM_NAME 约定：每个导入器模块需定义 MUSEUM_NAME 常量，对应 DataSet/ 下的目录名

新增博物馆只需在 importers/ 下新建 .py 文件，实现 MUSEUM_NAME + import_museum() 即可，
主脚本 import_dataset.py 会自动发现并调用。
"""
import sys
from pathlib import Path

# 让导入器模块能 import app.core.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.cultural_relic_aliases import _WARE_TYPE_ALIASES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_ware_type(name: str) -> str | None:
    """从文物名称反查器型（二级分类）。

    利用 cultural_relic_aliases 的 _WARE_TYPE_ALIASES 别名表反向匹配：
      "商火纹青铜鼎" → "鼎"（匹配别名"鼎"或"青铜鼎"）
      "新石器时代灰陶盆" → "盆"（匹配别名"盆"）
      "编钟" → None（别名表无"编钟"）
    """
    if not name:
        return None
    for canonical, aliases in _WARE_TYPE_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in name:
                return canonical
    return None


def build_name_tags(name: str) -> list[str]:
    """把文物名称整体作为标签返回。

    标签路 search_by_tags 能精确命中"商火纹青铜鼎"这类查询。
    """
    if not name:
        return []
    return [name.strip()]


def build_relic_metadata(
    name: str,
    museum: str,
    dynasty: str | None = None,
    material: str | None = None,
    category_top: str | None = None,
    relic_condition: str | None = None,
    craft: str | None = None,
    color_feature: str | None = None,
    function_usage: str | None = None,
    extra_caption_parts: list[str] | None = None,
) -> dict:
    """构造单条文物元数据（统一字段格式，所有导入器共用）。

    名称字段三处使用（检索核心字段）：
    - tags：名称整体作为标签，标签路精确命中
    - category_sub：从名称提取器型，生成"二级分类:鼎"结构化标签
    - caption：名称 + extra_caption_parts 拼串，BM25 路精确匹配

    Args:
        name: 文物名称（必填）
        museum: 博物馆名（必填，写入 metadata 的 museum 字段）
        dynasty/material/...: 结构化字段，有就传，没有默认 None
        extra_caption_parts: caption 的额外组成部分（如尺寸、描述），会拼到 name 后面
    """
    # caption：名称 + 额外部分
    caption_parts = [name] + (extra_caption_parts or [])
    caption = "。".join(p for p in caption_parts if p)

    return {
        "name": name,
        "tags": build_name_tags(name),
        "category_sub": extract_ware_type(name),
        "dynasty": dynasty or None,
        "material": material or None,
        "category_top": category_top or None,
        "relic_condition": relic_condition or None,
        "craft": craft or None,
        "color_feature": color_feature or None,
        "function_usage": function_usage or None,
        "caption": caption or None,
        "museum": museum,
    }


def simple_import_museum(
    src_dir: Path,
    dst_raw: Path,
    museum_name: str,
    xlsx_filename: str,
    id_prefix: str,
    dry_run: bool = False,
) -> dict:
    """通用简单导入器：适用于"只有标题+图片"的博物馆。

    这类博物馆的 xlsx 共同特征：
    - "标题"列 = 文物名称
    - "图片_保存位置"列 = Windows 路径（含文件名）
    - 图片在 src_dir 下的某个子目录（图片/ 或 {时间戳}/图片/）

    本函数自动递归查找图片，无需关心图片在哪个子目录。
    product_id 用 {id_prefix}_{序号:03d} 格式，避免跨博物馆冲突。

    适用于：福州市林则徐纪念馆、聂荣臻元帅网、苏州丝绸博物馆、金华市博物馆 等。

    Args:
        museum_name: 博物馆名（对应 DataSet/ 下的目录名）
        xlsx_filename: xlsx 文件名
        id_prefix: product_id 前缀（如"林则徐"、"聂荣臻"），避免与其他博物馆冲突
    """
    import shutil
    import pandas as pd

    xlsx_path = src_dir / xlsx_filename
    df = pd.read_excel(xlsx_path)

    # "标题"列必须存在，"图片_保存位置"列可能叫不同名字
    title_col = "标题" if "标题" in df.columns else df.columns[0]
    path_col = None
    for candidate in ["图片_保存位置", "图片保存位置", "图片路径"]:
        if candidate in df.columns:
            path_col = candidate
            break
    if path_col is None:
        # 取最后一列作为路径（多数情况是图片路径）
        path_col = df.columns[-1]

    metadata = {}
    count = 0
    products_with_images: set[str] = set()
    orphan_products: set[str] = set()

    for idx, row in df.iterrows():
        name = str(row[title_col]).strip() if pd.notna(row[title_col]) else ""
        if not name:
            continue

        product_id = f"{id_prefix}_{idx + 1:03d}"

        # 从"图片_保存位置"提取文件名（处理 Windows 路径在 Linux 上的问题）
        save_path = str(row[path_col]) if pd.notna(row[path_col]) else ""
        img_filename = Path(save_path.replace("\\", "/")).name if save_path else ""

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=museum_name,
        )
        if img_filename:
            metadata[product_id]["source_file"] = img_filename

        # 在 src_dir 下递归查找图片文件
        has_img = False
        if img_filename:
            # rglob 查找匹配文件名（不区分大小写）
            matched = list(src_dir.rglob(img_filename))
            if matched:
                src_img = matched[0]
                dst_img_dir = dst_raw / museum_name / product_id
                if not dry_run:
                    dst_img_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img_dir / src_img.name)
                count += 1
                has_img = True

        if has_img:
            products_with_images.add(product_id)
        else:
            orphan_products.add(product_id)

    # 过滤孤儿记录
    orphan_count = len(orphan_products)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        print(f"[{museum_name}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），复制图片 {count} 张")
    else:
        print(f"[{museum_name}] 元数据 {len(metadata)} 条，复制图片 {count} 张")
    return metadata

