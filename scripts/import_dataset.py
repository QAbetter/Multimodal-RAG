"""
数据集导入脚本：把 DataSet/ 真实数据整理成索引系统期望的结构。

输出：
1. 图片复制到 data/images/raw/{博物馆}/{藏品编号}/
2. 元数据 JSON：data/processed/dataset_metadata.json
   {product_id: {dynasty, material, category_top, category_sub, tags, caption, ...}}

名称字段三处使用（检索核心字段）：
- tags：名称整体作为标签，标签路精确命中（搜"商火纹青铜鼎"直接匹配）
- category_sub：从名称提取器型（如"商火纹青铜鼎"→"鼎"），生成"二级分类:鼎"结构化标签
- caption：名称+尺寸+描述拼串，BM25 路精确匹配

用法：
    python scripts/import_dataset.py                    # 导入全部博物馆
    python scripts/import_dataset.py --museum 包头博物馆  # 只导入指定博物馆
    python scripts/import_dataset.py --dry-run         # 只预览不实际复制
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.cultural_relic_aliases import _WARE_TYPE_ALIASES

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_ware_type(name: str) -> str | None:
    """从文物名称反查器型（二级分类）。

    利用 cultural_relic_aliases 的 _WARE_TYPE_ALIASES 别名表反向匹配：
      "商火纹青铜鼎" → "鼎"（匹配别名"鼎"或"青铜鼎"）
      "新石器时代灰陶盆" → "盆"（匹配别名"盆"）
      "编钟" → None（别名表无"编钟"）

    匹配优先级：长别名优先（避免"青铜鼎"被"鼎"短别名先命中导致返回标准值相同，
    长短别名指向同一标准值时结果一致，但长别名更准确）。
    """
    if not name:
        return None
    for canonical, aliases in _WARE_TYPE_ALIASES.items():
        # 别名按长度降序排，优先匹配长串（如"青铜鼎"优先于"鼎"）
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in name:
                return canonical
    return None


def build_name_tags(name: str) -> list[str]:
    """把文物名称整体作为标签返回。

    标签路 search_by_tags 能精确命中"商火纹青铜鼎"这类查询。
    返回 list 是为了与现有 tags 数据结构兼容（GLM-4V 也是返回 list）。

    若名称为空返回空列表，下游注册时跳过 tags 注入。
    """
    if not name:
        return []
    return [name.strip()]


# ========== 包头博物馆导入 ==========

def import_baotou(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """包头博物馆：xlsx（跳过前2行表头）+ 分编号图片目录。

    名称 [3] 三处使用：
    - tags：名称整体作为标签
    - category_sub：extract_ware_type 提取器型
    - caption：名称 + 尺寸 拼串

    xlsx 列索引（跳过2行表头后，0-based）：
      [2] 藏品总登记号 → product_id
      [3] 名称 → tags + category_sub + caption
      [6] 年代 → dynasty
      [10] 文物类别 → category_top
      [13] 质地 → material
      [18] 具体尺寸 → caption 组成
      [25] 完残程度 → relic_condition
    """
    xlsx_path = src_dir / "15020311800001内蒙古包头博物馆.xlsx"
    # 前3行是表头：第1行填报说明、第2行一级表头（藏品编号/名称/年代/质地等）、
    # 第3行二级表头（编号类型/编号/质地类别1/质地类别2/通长/通宽/通高等），数据从第4行开始
    df = pd.read_excel(xlsx_path, header=None, skiprows=3)

    metadata = {}
    for _, row in df.iterrows():
        product_id = str(row[2]).strip() if pd.notna(row[2]) else None
        if not product_id or product_id == "nan":
            continue

        name = str(row[3]).strip() if pd.notna(row[3]) else ""
        dynasty = str(row[6]).strip() if pd.notna(row[6]) else ""
        category_top = str(row[10]).strip() if pd.notna(row[10]) else ""
        material = str(row[13]).strip() if pd.notna(row[13]) else ""
        size = str(row[18]).strip() if pd.notna(row[18]) else ""
        condition = str(row[25]).strip() if pd.notna(row[25]) else ""

        # caption：名称 + 尺寸（用于 BM25 文本检索）
        caption_parts = [p for p in [name, size] if p]
        caption = "。".join(caption_parts)

        # 名称作为标签 + 从名称提取器型
        name_tags = build_name_tags(name)
        ware_type = extract_ware_type(name)

        metadata[product_id] = {
            "name": name,
            "tags": name_tags,                # 名称作为标签
            "category_sub": ware_type,        # 器型（二级分类）
            "dynasty": dynasty or None,
            "category_top": category_top or None,
            "material": material or None,
            "relic_condition": condition or None,
            "caption": caption or None,
            "museum": "包头博物馆",
        }

    # 复制图片：data/images/raw/包头博物馆/{藏品编号}/
    # 同时记录哪些 product_id 实际有图片，过滤掉无图片的孤儿元数据
    count = 0
    products_with_images: set[str] = set()
    for product_id in metadata:
        src_img_dir = src_dir / product_id
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_raw / "包头博物馆" / product_id
        if not dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
        has_img = False
        for img in src_img_dir.iterdir():
            if img.suffix.lower() in _IMAGE_EXTS:
                if not dry_run:
                    shutil.copy2(img, dst_img_dir / img.name)
                count += 1
                has_img = True
        if has_img:
            products_with_images.add(product_id)

    # 过滤孤儿记录：只保留实际有图片的 product_id
    # 避免 dataset_metadata.json 膨胀（如 xlsx 有 10万条但磁盘只有 5 张图片）
    orphan_count = len(metadata) - len(products_with_images)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        print(f"[包头博物馆] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），复制图片 {count} 张")
    else:
        print(f"[包头博物馆] 元数据 {len(metadata)} 条，复制图片 {count} 张")
    return metadata


# ========== 安阳博物馆导入 ==========

def import_anyang(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """安阳博物馆：青铜器 xlsx + resource/ 图片。

    名称(标题1) 三处使用：
    - tags：名称整体作为标签
    - category_sub：extract_ware_type 提取器型（如"商火纹青铜鼎"→"鼎"）
    - caption：名称 + 尺寸 + 描述 拼串

    xlsx 列：
      标题 → category_top（如"青铜器"）
      标题1 → 名称 → tags + category_sub + caption
      文本 → 尺寸信息 → caption 组成
      文本1 → 详细描述 → caption 组成
      图片链接_保存位置 → 提取文件名匹配 resource/ 下的图片
    """
    xlsx_path = src_dir / "青铜器_安博藏品_安阳博物馆.xlsx"
    df = pd.read_excel(xlsx_path)

    resource_dir = src_dir / "resource"
    metadata = {}
    count = 0
    products_with_images_anyang: set[str] = set()
    orphan_products_anyang: set[str] = set()

    for idx, row in df.iterrows():
        name = str(row["标题1"]).strip() if pd.notna(row["标题1"]) else ""
        if not name:
            continue

        # product_id 用"安阳_序号"避免与包头编号冲突
        product_id = f"安阳_{idx + 1:03d}"

        category_top = str(row["标题"]).strip() if pd.notna(row["标题"]) else ""
        size_text = str(row["文本"]).strip() if pd.notna(row["文本"]) else ""
        desc = str(row["文本1"]).strip() if pd.notna(row["文本1"]) else ""
        caption_parts = [p for p in [name, size_text, desc] if p]
        caption = "。".join(caption_parts)

        # 从"图片链接_保存位置"提取文件名
        # 注意：xlsx 里是 Windows 路径（如 D:\八爪鱼下载\xxx.jpeg），
        # 在 Linux 服务器上跑时 Path 不认反斜杠，需先统一为正斜杠再取 basename
        save_path = str(row["图片链接_保存位置"]) if pd.notna(row["图片链接_保存位置"]) else ""
        img_filename = Path(save_path.replace("\\", "/")).name if save_path else ""

        # 名称作为标签 + 从名称提取器型
        name_tags = build_name_tags(name)
        ware_type = extract_ware_type(name)

        metadata[product_id] = {
            "name": name,
            "tags": name_tags,                # 名称作为标签
            "category_sub": ware_type,        # 器型（二级分类）
            "category_top": category_top or None,
            "caption": caption or None,
            "museum": "安阳博物馆",
            "source_file": img_filename,
        }

        # 复制对应图片，同时记录哪些 product_id 实际有图片
        has_img = False
        if img_filename:
            src_img = resource_dir / img_filename
            if src_img.exists():
                dst_img_dir = dst_raw / "安阳博物馆" / product_id
                if not dry_run:
                    dst_img_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img_dir / src_img.name)
                count += 1
                has_img = True

        if has_img:
            products_with_images_anyang.add(product_id)
        else:
            # 标记无图片，稍后过滤
            orphan_products_anyang.add(product_id)

    # 过滤孤儿记录：只保留实际有图片的 product_id
    orphan_count = len(orphan_products_anyang)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images_anyang}
        print(f"[安阳博物馆] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），复制图片 {count} 张")
    else:
        print(f"[安阳博物馆] 元数据 {len(metadata)} 条，复制图片 {count} 张")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="导入 DataSet 真实数据")
    parser.add_argument("--museum", type=str, default=None, help="只导入指定博物馆")
    parser.add_argument("--dry-run", action="store_true", help="只预览不实际复制")
    args = parser.parse_args()

    dataset_dir = Path("DataSet")
    dst_raw = Path("data/images/raw")
    if not args.dry_run:
        dst_raw.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    museums = {
        "包头博物馆": import_baotou,
        "安阳博物馆": import_anyang,
    }

    for name, func in museums.items():
        if args.museum and args.museum != name:
            continue
        src = dataset_dir / name
        if not src.exists():
            print(f"[跳过] {name} 目录不存在")
            continue
        meta = func(src, dst_raw, dry_run=args.dry_run)
        all_metadata.update(meta)

    # 保存元数据 JSON
    out_path = Path("data/processed/dataset_metadata.json")
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 共 {len(all_metadata)} 条元数据 → {out_path}")
    if args.dry_run:
        # 预览前3条
        for pid, meta in list(all_metadata.items())[:3]:
            print(f"  {pid}: name={meta.get('name')} | dynasty={meta.get('dynasty')} | "
                  f"material={meta.get('material')} | category_sub={meta.get('category_sub')} | "
                  f"tags={meta.get('tags')}")


if __name__ == "__main__":
    main()
