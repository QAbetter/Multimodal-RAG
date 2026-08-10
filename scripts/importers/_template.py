"""新博物馆导入模板。

新增博物馆时，复制此文件为 xxx.py（xxx 用博物馆拼音或简称），然后：
1. 修改 MUSEUM_NAME 为 DataSet/ 下对应的目录名
2. 修改 XLSX_FILENAME 为实际 xlsx 文件名
3. 根据实际 xlsx 结构修改 import_museum 函数的字段映射
4. 运行 python scripts/import_dataset.py（自动发现新导入器）

=== 两种常见情况 ===

情况 A：和包头博物馆同一系统导出的 xlsx（字段顺序一致）
  → 直接复制 baotou.py，只改 MUSEUM_NAME 和 XLSX_FILENAME 即可

情况 B：xlsx 结构不同（字段名/列顺序/表头行数不同）
  → 复制此模板，根据实际 xlsx 结构修改字段映射
  → 用 build_relic_metadata() 构造统一格式的元数据

=== 字段映射步骤 ===

1. 用 Excel/pandas 打开 xlsx，确认：
   - 表头占几行（设 SKIP_ROWS）
   - 用列名还是列索引（header=None + skiprows 时用列索引）
   - 哪列是名称、年代、材质等

2. 在 import_museum 中按实际列填入 build_relic_metadata() 参数。

3. 图片复制逻辑根据实际目录结构调整：
   - 按编号分目录（同包头）→ 遍历子目录
   - 统一目录（同安阳）→ 按文件名匹配
   - 其他结构 → 自定义

=== 验证 ===

  python scripts/import_dataset.py --museum xxx博物馆 --dry-run
  # 确认元数据条数和图片复制数正确后再正式导入
"""
import shutil
from pathlib import Path

import pandas as pd

from . import IMAGE_EXTS, build_relic_metadata

# === 必改：博物馆名（对应 DataSet/ 下的目录名）===
MUSEUM_NAME = "xxx博物馆"

# === 必改：xlsx 文件名 ===
XLSX_FILENAME = "xxx.xlsx"

# === 必改：表头行数（数据从第 SKIP_ROWS+1 行开始）===
SKIP_ROWS = 0  # 0=第一行就是数据，1=跳过1行表头，3=包头那种3行表头


def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
    """导入博物馆数据。

    必须实现此函数，返回 {product_id: {name, dynasty, material, ...}}。
    主脚本 import_dataset.py 会自动发现并调用此函数。

    Returns: {product_id: metadata_dict}
    """
    xlsx_path = src_dir / XLSX_FILENAME
    df = pd.read_excel(xlsx_path, header=None, skiprows=SKIP_ROWS)

    metadata = {}
    products_with_images: set[str] = set()

    for idx, row in df.iterrows():
        # === 必改：根据实际 xlsx 列填入字段 ===
        # 示例：假设 [0] 是编号、[1] 是名称、[2] 是年代
        product_id = str(row[0]).strip() if pd.notna(row[0]) else None
        if not product_id or product_id == "nan":
            continue

        name = str(row[1]).strip() if pd.notna(row[1]) else ""
        if not name:
            continue

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=MUSEUM_NAME,
            # === 按实际列索引填入，没有的字段不传即可 ===
            dynasty=str(row[2]).strip() if pd.notna(row[2]) else None,
            material=str(row[3]).strip() if pd.notna(row[3]) else None,
            category_top=str(row[4]).strip() if pd.notna(row[4]) else None,
            relic_condition=str(row[5]).strip() if pd.notna(row[5]) else None,
            extra_caption_parts=[
                str(row[6]).strip() if pd.notna(row[6]) else None,  # 尺寸
                str(row[7]).strip() if pd.notna(row[7]) else None,  # 描述
            ],
        )

    # === 必改：图片复制逻辑（根据实际目录结构调整）===
    # 示例 A：按编号分目录（同包头）
    count = 0
    for product_id in metadata:
        src_img_dir = src_dir / product_id
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_raw / MUSEUM_NAME / product_id
        if not dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
        has_img = False
        for img in src_img_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                if not dry_run:
                    shutil.copy2(img, dst_img_dir / img.name)
                count += 1
                has_img = True
        if has_img:
            products_with_images.add(product_id)

    # 示例 B：统一目录（同安阳）→ 把上面替换为：
    # for product_id, meta in metadata.items():
    #     img_filename = meta.get("source_file")
    #     if img_filename:
    #         src_img = src_dir / img_filename
    #         ...

    # 过滤孤儿记录
    orphan_count = len(metadata) - len(products_with_images)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），复制图片 {count} 张")
    else:
        print(f"[{MUSEUM_NAME}] 元数据 {len(metadata)} 条，复制图片 {count} 张")
    return metadata
