"""博物馆导入器公共工具。

所有导入器（importers/*.py）共享的工具函数：
- extract_ware_type：从文物名称反查器型（二级分类）
- build_name_tags：名称整体作为标签
- IMAGE_EXTS：支持的图片扩展名
- MUSEUM_NAME 约定：每个导入器模块需定义 MUSEUM_NAME 常量，对应 DataSet/ 下的目录名

新增博物馆只需在 importers/ 下新建 .py 文件，实现 MUSEUM_NAME + import_museum() 即可，
主脚本 import_dataset.py 会自动发现并调用。

增量导入支持：
- load_import_state / save_import_state：记录每个博物馆 xlsx 的修改时间，
  下次运行时跳过未变化的博物馆（--force 可强制重跑）
- copy_image_if_changed：只在源图片内容变化时才复制，避免不必要的目标 mtime 更新
"""
import hashlib
import json
import re
import sys
from pathlib import Path

# 让导入器模块能 import app.core.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.cultural_relic_aliases import _WARE_TYPE_ALIASES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 导入状态文件路径（记录每个博物馆 xlsx 的 mtime，用于增量导入判断）
STATE_FILE = Path("data/processed/import_state.json")


def clean_filename(filename: str) -> str:
    """清洗文件名：去首尾空格 + 合并连续点号为单个点。

    修复 RAGFlow 静态文件服务无法访问含首尾空格、连续点号（如 "....jpeg"）的 URL 问题。
    例如：
      " 元龙泉窑粉青釉划花....jpeg" → "元龙泉窑粉青釉划花.jpeg"
      "  明代青花瓷瓶.jpeg  " → "明代青花瓷瓶.jpeg"
      "test...jpg" → "test.jpg"

    Args:
        filename: 原始文件名（可能含空格、连续点号）

    Returns:
        清洗后的文件名；输入为空或仅点号时返回 "unknown_file"
    """
    if not filename:
        return filename
    # 1. 去首尾空格（URL 中 %20 导致静态服务 404）
    cleaned = filename.strip()
    # 2. 合并连续点号（如 "....jpeg" → ".jpeg"，"test...jpg" → "test.jpg"）
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    # 3. 处理只剩点号或空字符串的极端情况
    if not cleaned or cleaned == '.':
        return 'unknown_file'
    return cleaned


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
    name_col: str | None = None,
    dynasty_col: str | None = None,
    intro_col: str | None = None,
    category_col: str | None = None,
    size_col: str | None = None,
    path_col: str | None = None,
    strip_prefix: bool = True,
) -> dict:
    """通用简单导入器：xlsx 有名称列 + 图片路径列，可提取结构化字段。

    支持多种列名：name_col 指定名称列（默认"标题"，常见还有"定名"、"名称"），
    path_col 指定图片保存路径列（默认自动查找"图片_保存位置"/"图片链接_保存位置"等）。

    可选提取字段：dynasty_col（年代）、intro_col（简介）、category_col（类别）、
    size_col（尺寸）。某些博物馆的字段值带前缀（如"年代：清"），strip_prefix=True 时自动去除。

    图片查找：从 path_col 的路径提取文件名，在 src_dir 下 rglob 查找。

    Args:
        name_col: 名称列名，None 则自动选"标题"或第一列
        dynasty_col: 年代列名（如"年代"/"朝代"/"时代"），None 则不提取
        intro_col: 介绍列名（如"简介"/"描述"/"textellipsis"/"bd_con_c"），None 则不提取
        category_col: 类别列名（如"类别"/"分类"/"文物类型"），None 则不提取
        size_col: 尺寸列名，None 则不提取（拼入 caption）
        path_col: 图片保存路径列名，None 则自动查找
        strip_prefix: True=去除字段值的前缀（如"年代：清"→"清"）
    """
    import pandas as pd
    import re

    xlsx_path = src_dir / xlsx_filename
    df = pd.read_excel(xlsx_path)

    # 预建文件名到路径的索引，避免逐行 rglob（O(N) 而非 O(N×M)）
    name_to_path: dict[str, Path] = {}
    for f in src_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            name_to_path[f.name] = f

    # 确定 name_col
    if name_col and name_col in df.columns:
        title_col = name_col
    elif "标题" in df.columns:
        title_col = "标题"
    else:
        title_col = df.columns[0]

    # 确定 path_col（图片保存位置）
    if path_col and path_col in df.columns:
        pass
    else:
        path_col = None
        for candidate in ["图片_保存位置", "图片保存位置", "图片路径",
                          "图片链接_保存位置", "图片链接1_保存位置",
                          "图片链接1_保存位置", "二维_保存位置",
                          "字段1_保存位置", "字段1_保存位置",
                          "资源文件路径"]:
            if candidate in df.columns:
                path_col = candidate
                break
        if path_col is None:
            path_col = df.columns[-1]

    def _strip(val: str) -> str | None:
        """去除字段值前缀（如'年代：清'→'清'）并清理。"""
        if not val:
            return None
        val = val.strip()
        if strip_prefix:
            # 去除"xxx："或"xxx:"前缀
            val = re.sub(r'^[^：:]{1,6}[：:]', '', val).strip()
        return val if val else None

    def _get(row, col: str | None) -> str | None:
        if col and col in df.columns and pd.notna(row.get(col)):
            return _strip(str(row[col]))
        return None

    metadata = {}
    total = 0
    copied = 0
    products_with_images: set[str] = set()
    orphan_products: set[str] = set()

    for idx, row in df.iterrows():
        name = str(row[title_col]).strip() if pd.notna(row[title_col]) else ""
        if not name:
            continue

        product_id = f"{id_prefix}_{idx + 1:03d}"

        # 提取结构化字段
        dynasty = _get(row, dynasty_col)
        intro = _get(row, intro_col)
        category = _get(row, category_col)
        size = _get(row, size_col)

        # caption 组成：name + size + intro
        caption_parts = [p for p in [size, intro] if p]

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=museum_name,
            dynasty=dynasty,
            category_top=category,
            extra_caption_parts=caption_parts if caption_parts else None,
        )

        # 从路径列提取文件名
        save_path = str(row[path_col]) if pd.notna(row[path_col]) else ""
        img_filename = Path(save_path.replace("\\", "/")).name if save_path else ""
        if img_filename:
            metadata[product_id]["source_file"] = img_filename

        # 用预建索引查找图片（O(1)，避免逐行 rglob）
        has_img = False
        if img_filename:
            matched = [name_to_path[img_filename]] if img_filename in name_to_path else []
            if matched:
                src_img = matched[0]
                # 清洗目标文件名，避免首尾空格/连续点号导致静态服务 404
                dst_name = clean_filename(src_img.name)
                dst_img = dst_raw / museum_name / product_id / dst_name
                total += 1
                if not dry_run:
                    if copy_image_if_changed(src_img, dst_img):
                        copied += 1
                else:
                    copied += 1
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
              f"磁盘缺图 {orphan_count} 条已过滤），图片 {total} 张"
              f"{'（复制 ' + str(copied) + ' 张变化）' if copied != total else ''}")
    else:
        suffix = f"（复制 {copied} 张变化）" if copied != total else ""
        print(f"[{museum_name}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata


def info_3d_import_museum(
    src_dir: Path,
    dst_raw: Path,
    museum_name: str,
    id_prefix: str,
    xlsx_filename: str = "info.xlsx",
    dry_run: bool = False,
) -> dict:
    """info.xlsx 标准 3D 格式导入器。

    info.xlsx 列名固定：序号 | 分类 | 定名 | 尺寸 | 年代 | 简介 | 存储类型 | 备注 | 资源文件路径(以#分隔)
    资源文件路径格式："images/001.jpg#images/obj.obj#..."，以#分隔多个文件，只取图片部分。

    图片在 src_dir/resource/ 目录下（或 src_dir/ 下其他目录），文件名为序号（如 001.jpg）。

    适用于：中国南海博物馆、天水市博物馆、广东中国客家博物馆、广东海上丝绸之路博物馆 等。
    """
    import pandas as pd

    xlsx_path = src_dir / xlsx_filename
    if not xlsx_path.exists():
        print(f"[{museum_name}] {xlsx_filename} 不存在，跳过")
        return {}

    df = pd.read_excel(xlsx_path)

    # 列名兼容：资源文件路径列可能叫"资源文件路径"或"资源文件路径(以#分隔)"
    path_col_name = None
    for col in df.columns:
        if "资源文件路径" in str(col):
            path_col_name = col
            break

    # 预建文件名到路径的索引，避免逐行 rglob（O(N) 而非 O(N×M)）
    name_to_path: dict[str, Path] = {}
    for f in src_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            name_to_path[f.name] = f

    metadata = {}
    total = 0
    copied = 0
    products_with_images: set[str] = set()
    orphan_products: set[str] = set()

    for idx, row in df.iterrows():
        name = str(row["定名"]).strip() if pd.notna(row.get("定名")) else ""
        if not name:
            continue

        product_id = f"{id_prefix}_{idx + 1:03d}"

        dynasty = str(row["年代"]).strip() if pd.notna(row.get("年代")) and row.get("年代") else None
        category = str(row["分类"]).strip() if pd.notna(row.get("分类")) and row.get("分类") else None
        size = str(row["尺寸"]).strip() if pd.notna(row.get("尺寸")) and row.get("尺寸") else None
        intro = str(row["简介"]).strip() if pd.notna(row.get("简介")) and row.get("简介") else None

        # 解析资源文件路径：以#分隔，只取图片
        raw_path = ""
        if path_col_name and pd.notna(row.get(path_col_name)):
            raw_path = str(row[path_col_name])
        img_files = []
        for part in raw_path.split("#"):
            part = part.strip()
            if part and any(part.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
                img_files.append(part)

        caption_parts = [p for p in [size, intro] if p and p != "无"]
        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=museum_name,
            dynasty=dynasty,
            category_top=category,
            extra_caption_parts=caption_parts if caption_parts else None,
        )

        # 查找图片：用预建索引查（O(1)），而非逐行 rglob
        has_img = False
        if img_files:
            for img_rel in img_files:
                img_name = Path(img_rel.replace("\\", "/")).name
                src_img = name_to_path.get(img_name)
                if src_img:
                    # 清洗目标文件名，避免首尾空格/连续点号导致静态服务 404
                    dst_name = clean_filename(src_img.name)
                    dst_img = dst_raw / museum_name / product_id / dst_name
                    total += 1
                    if not dry_run:
                        if copy_image_if_changed(src_img, dst_img):
                            copied += 1
                    else:
                        copied += 1
                    has_img = True
            if has_img:
                metadata[product_id]["source_file"] = img_files[0].split("/")[-1]

        if has_img:
            products_with_images.add(product_id)
        else:
            orphan_products.add(product_id)

    orphan_count = len(orphan_products)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        print(f"[{museum_name}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），图片 {total} 张"
              f"{'（复制 ' + str(copied) + ' 张变化）' if copied != total else ''}")
    else:
        suffix = f"（复制 {copied} 张变化）" if copied != total else ""
        print(f"[{museum_name}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata


def extract_xlsx_embedded_images(xlsx_path: Path) -> dict[int, bytes]:
    """从 xlsx 提取内嵌图片，返回 {行号(0-indexed): 图片bytes}。

    Excel 单元格内嵌的图片存在 xl/media/imageN.jpeg，
    通过 xl/drawings/drawing1.xml 的 anchor 记录图片锚定的行号，
    通过 xl/drawings/_rels/drawing1.xml.rels 的 rId 关联到 media 文件。

    适用于 xlsx "图片"列是内嵌图片（pandas 读为 NaN）的情况：
    - 华侨博物院：图片列是单元格内嵌缩略图
    - 懿品博悟：图片列是单元格内嵌图片
    """
    import re
    import zipfile

    if not xlsx_path.exists():
        return {}

    with zipfile.ZipFile(xlsx_path) as z:
        names = z.namelist()

        # 找 drawing 文件
        drawing_files = [n for n in names if n.startswith("xl/drawings/drawing") and n.endswith(".xml")]
        if not drawing_files:
            return {}

        # 解析所有 rels 文件：rId → media 路径
        rid_to_media: dict[str, str] = {}
        for df in drawing_files:
            rels_path = df.replace("xl/drawings/", "xl/drawings/_rels/") + ".rels"
            if rels_path in names:
                rels = z.read(rels_path).decode("utf-8")
                for m in re.finditer(r'Id="(rId\d+)"[^>]*Target="\.\./(media/[^"]+)"', rels):
                    rid_to_media[m.group(1)] = m.group(2)

        if not rid_to_media:
            return {}

        # 解析 drawing：行号 → rId
        row_to_image: dict[int, bytes] = {}
        for df in drawing_files:
            drawing = z.read(df).decode("utf-8")
            # 匹配 <xdr:from>...<xdr:row>N</xdr:row>...</xdr:from> ... r:embed="rIdXXX"
            anchors = re.findall(
                r'<xdr:from>.*?<xdr:row>(\d+)</xdr:row>.*?</xdr:from>.*?r:embed="(rId\d+)"',
                drawing, re.DOTALL
            )
            for row_str, rid in anchors:
                row = int(row_str)
                media_path = rid_to_media.get(rid)
                if media_path and f"xl/{media_path}" in names:
                    row_to_image[row] = z.read(f"xl/{media_path}")

        return row_to_image


def _image_fingerprint(img_bytes: bytes) -> tuple:
    """计算图片的灰度指纹（缩放到 8x8 灰度，返回 64 个像素值 tuple）。

    用于快速相似度比较：指纹相同 = 图片内容基本相同。
    比 MD5 快且能匹配"缩略图 vs 原图"这种尺寸不同但内容相同的情况。
    """
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((8, 8))
    return tuple(img.getdata())


def match_embedded_to_dir_images(
    row_to_emb: dict[int, bytes],
    img_dir: Path,
) -> dict[int, Path]:
    """把 xlsx 内嵌图片匹配到 图片/ 目录里的原图，返回 {行号: 原图Path}。

    策略：
    1. 预计算 图片/ 目录所有文件的指纹（一次遍历）
    2. 对每张内嵌图片，计算指纹后和目录指纹比对
    3. 完全相同 → 直接对应；否则用 16x16 指纹找最相似的（相似度>0.9）

    对懿品博悟 2000+ 张图，用指纹字典预筛 + 精细比较兜底，避免 O(N²)。

    Args:
        row_to_emb: extract_xlsx_embedded_images 的返回值 {行号: 内嵌图片bytes}
        img_dir: 图片/ 目录路径

    Returns:
        {行号: 原图Path} 匹配成功的条目；匹配失败的行号不在结果里
    """
    import io
    from PIL import Image

    if not img_dir.exists():
        return {}

    # 预计算目录图片指纹
    dir_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    dir_fp_to_path: dict[tuple, Path] = {}
    dir_fps: list[tuple[tuple, Path]] = []
    for f in dir_files:
        try:
            fp = _image_fingerprint(f.read_bytes())
            dir_fp_to_path[fp] = f
            dir_fps.append((fp, f))
        except Exception:
            continue

    result: dict[int, Path] = {}
    for row, emb_bytes in row_to_emb.items():
        try:
            emb_fp = _image_fingerprint(emb_bytes)
        except Exception:
            continue

        # 精确匹配（指纹完全相同）
        if emb_fp in dir_fp_to_path:
            result[row] = dir_fp_to_path[emb_fp]
            continue

        # 模糊匹配：和所有目录图片比较，找最相似的
        best_path = None
        best_sim = 0.0
        for dir_fp, path in dir_fps:
            same = sum(1 for a, b in zip(emb_fp, dir_fp) if abs(a - b) < 30)
            sim = same / 64.0
            if sim > best_sim:
                best_sim = sim
                best_path = path
        if best_sim >= 0.85 and best_path:
            result[row] = best_path

    return result


def xlsx_name_import_museum(
    src_dir: Path,
    dst_raw: Path,
    museum_name: str,
    xlsx_filename: str,
    id_prefix: str,
    name_col: str = "文物名称",
    intro_col: str | None = "介绍",
    dynasty_col: str | None = None,
    material_col: str | None = None,
    category_top_col: str | None = None,
    extra_caption_cols: list[str] | None = None,
    img_filename_col: str | None = None,
    name_as_image_filename: bool = True,
    dry_run: bool = False,
) -> dict:
    """通用 xlsx 名称+介绍导入器：读 xlsx 名称列，rglob 在子目录查找同名图片。

    适用于"xlsx 只有文物名称（和介绍/描述）"的博物馆：
    - 仪征博物馆（文物名称/介绍）
    - 莆田博物馆（文物名称/介绍，无图片）
    - 华侨博物院（标题/大小/描述，图片列全 NaN）
    - 懿品博悟（名称/年代/博物馆/简介/...，图片列全 NaN）

    Args:
        name_col: 文物名称所在列名
        intro_col: 介绍/描述列名（作为 caption 主体），None 表示无
        dynasty_col: 年代列名（如懿品博悟有），None 表示无
        material_col: 材质列名，None 表示无
        category_top_col: 一级分类列名，None 表示无
        extra_caption_cols: 额外拼到 caption 的列名列表（如大小、纹样介绍）
        img_filename_col: 图片文件名列（如苏州戏曲的"图片名称"），
            None 表示用文物名称作为图片文件名查找
        name_as_image_filename: True=用 name_col 内容查找图片文件
            （需配合 img_filename_col=None 或同时使用）
    """
    import pandas as pd

    xlsx_path = src_dir / xlsx_filename
    df = pd.read_excel(xlsx_path)

    metadata = {}
    total = 0
    copied = 0
    products_with_images: set[str] = set()
    orphan_products: set[str] = set()

    for idx, row in df.iterrows():
        name = str(row[name_col]).strip() if pd.notna(row.get(name_col)) else ""
        if not name:
            continue

        product_id = f"{id_prefix}_{idx + 1:03d}"

        # 构造 caption：name + 介绍 + 额外列
        caption_parts = [name]
        if intro_col and intro_col in df.columns and pd.notna(row.get(intro_col)):
            caption_parts.append(str(row[intro_col]).strip())
        if extra_caption_cols:
            for col in extra_caption_cols:
                if col in df.columns and pd.notna(row.get(col)):
                    val = str(row[col]).strip()
                    if val:
                        caption_parts.append(val)

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=museum_name,
            dynasty=str(row[dynasty_col]).strip() if dynasty_col and dynasty_col in df.columns and pd.notna(row.get(dynasty_col)) else None,
            material=str(row[material_col]).strip() if material_col and material_col in df.columns and pd.notna(row.get(material_col)) else None,
            category_top=str(row[category_top_col]).strip() if category_top_col and category_top_col in df.columns and pd.notna(row.get(category_top_col)) else None,
            extra_caption_parts=caption_parts[1:],  # name 已在 build_relic_metadata 内加入
        )

        # 确定要查找的图片文件名
        # 优先用 img_filename_col，其次用 name_as_image_filename
        img_filename = ""
        if img_filename_col and img_filename_col in df.columns and pd.notna(row.get(img_filename_col)):
            img_filename = str(row[img_filename_col]).strip()
        elif name_as_image_filename:
            img_filename = f"{name}.jpg"  # 尝试常见扩展名

        if img_filename:
            metadata[product_id]["source_file"] = img_filename

        # rglob 查找图片（尝试多种扩展名）
        has_img = False
        if img_filename:
            # 先尝试精确匹配文件名
            stem = Path(img_filename).stem if "." in img_filename else img_filename
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                try:
                    matched = list(src_dir.rglob(f"{stem}{ext}"))
                except OSError:
                    matched = []  # 文件名过长等异常，跳过
                if matched:
                    src_img = matched[0]
                    # 清洗目标文件名，避免首尾空格/连续点号导致静态服务 404
                    dst_name = clean_filename(src_img.name)
                    dst_img = dst_raw / museum_name / product_id / dst_name
                    total += 1
                    if not dry_run:
                        if copy_image_if_changed(src_img, dst_img):
                            copied += 1
                    else:
                        copied += 1
                    has_img = True
                    break

        if has_img:
            products_with_images.add(product_id)
        else:
            orphan_products.add(product_id)

    # 过滤孤儿记录
    orphan_count = len(orphan_products)
    if orphan_count > 0:
        metadata = {pid: meta for pid, meta in metadata.items() if pid in products_with_images}
        print(f"[{museum_name}] 元数据 {len(metadata)} 条（xlsx 有 {len(metadata) + orphan_count} 条，"
              f"磁盘缺图 {orphan_count} 条已过滤），图片 {total} 张"
              f"{'（复制 ' + str(copied) + ' 张变化）' if copied != total else ''}")
    else:
        suffix = f"（复制 {copied} 张变化）" if copied != total else ""
        print(f"[{museum_name}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata


def category_subdir_import_museum(
    src_dir: Path,
    dst_raw: Path,
    museum_name: str,
    id_prefix: str,
    category_to_top: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict:
    """通用类别子目录导入器：无 xlsx，遍历子目录，文件名作为文物名。

    适用于"无 xlsx，图片按类别分目录存放，文件名即文物名"的博物馆：
    - 李白纪念馆（书画/其他/古籍文献/碑刻）
    - 株洲博物馆（瓷器/陶器/铜器/玉器/金银器/铁器/石器/木/复合器）
    - 青白江博物馆（玉器/石器/青铜器/瓷器/陶器/其他）

    Args:
        category_to_top: 子目录名 → 一级分类映射（如 {"书画": "书画", "瓷器": "陶瓷器"}）。
            None 表示用子目录名直接作为 category_top。
    """
    metadata = {}
    total = 0
    copied = 0

    # 收集一级子目录名，用于确定 category_top
    top_subdirs = {p.name: p for p in src_dir.iterdir() if p.is_dir() and not p.name.startswith(".")}

    # 递归遍历所有图片，category_top 取一级子目录名
    for img in sorted(src_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        # 文件名（去扩展名）作为文物名
        name = img.stem
        if not name:
            continue

        # 找出相对于 src_dir 的第一级子目录名
        try:
            rel = img.relative_to(src_dir)
            first_part = rel.parts[0] if len(rel.parts) > 1 else None
        except ValueError:
            first_part = None
        category_top = None
        if first_part and category_to_top:
            category_top = category_to_top.get(first_part, first_part)
        elif first_part:
            category_top = first_part

        product_id = f"{id_prefix}_{total + 1:04d}"

        metadata[product_id] = build_relic_metadata(
            name=name,
            museum=museum_name,
            category_top=category_top,
        )
        metadata[product_id]["source_file"] = img.name

        # 清洗目标文件名，避免首尾空格/连续点号导致静态服务 404
        dst_name = clean_filename(img.name)
        dst_img = dst_raw / museum_name / product_id / dst_name
        total += 1
        if not dry_run:
            if copy_image_if_changed(img, dst_img):
                copied += 1
        else:
            copied += 1

    suffix = f"（复制 {copied} 张变化）" if copied != total else ""
    print(f"[{museum_name}] 元数据 {len(metadata)} 条，图片 {total} 张{suffix}")
    return metadata


# ========== 增量导入支持 ==========

def load_import_state() -> dict:
    """加载导入状态（记录每个博物馆 xlsx 的 mtime + size）。

    状态文件：data/processed/import_state.json
    格式：{museum_name: {"xlsx_mtime": float, "xlsx_size": int}}

    用于增量导入：如果 xlsx 的 mtime+size 未变，则跳过该博物馆重新导入。
    """
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_import_state(state: dict) -> None:
    """保存导入状态。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def is_xlsx_changed(museum_name: str, xlsx_path: Path, state: dict) -> bool:
    """判断博物馆的 xlsx 是否较上次导入有变化。

    用 mtime + size 双重判断（mtime 可能因 touch 改变但内容没变，加 size 更可靠）。

    Returns:
        True = 有变化或首次导入，需要重新导入
        False = 无变化，可跳过
    """
    if not xlsx_path.exists():
        return False
    stat = xlsx_path.stat()
    key = f"{museum_name}"
    prev = state.get(key)
    if not prev:
        return True  # 首次导入
    return prev.get("xlsx_mtime") != stat.st_mtime or prev.get("xlsx_size") != stat.st_size


def record_import_state(museum_name: str, xlsx_path: Path, state: dict) -> None:
    """记录本次导入的 xlsx 状态。"""
    if xlsx_path.exists():
        stat = xlsx_path.stat()
        state[museum_name] = {
            "xlsx_mtime": stat.st_mtime,
            "xlsx_size": stat.st_size,
        }


def copy_image_if_changed(src: Path, dst: Path) -> bool:
    """只在源图片与目标不同时才复制（避免更新目标 mtime 触发不必要的重新索引）。

    用文件大小快速判断 + MD5 精确判断（大小相同才算 MD5）。

    Returns:
        True = 已复制（内容有变化或目标不存在）
        False = 跳过（内容相同）
    """
    import shutil

    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True

    # 大小不同 → 内容肯定不同
    if src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
        return True

    # 大小相同 → 用 MD5 精确判断
    def md5(p: Path) -> str:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    if md5(src) != md5(dst):
        shutil.copy2(src, dst)
        return True
    return False


# ========== 配置驱动导入 ==========

def _inject_category_top(metadata: dict, category_top: str | None) -> None:
    """多 xlsx 模式：把 file 级的 category_top 强制注入所有 metadata。"""
    if category_top:
        for m in metadata.values():
            m["category_top"] = category_top


def _make_config_import_museum(cfg: dict):
    """根据一条配置 dict 生成 import_museum 闭包。

    支持 simple / xlsx_name / category_subdir 三种 mode，
    单 xlsx（cfg["xlsx"]）和多 xlsx（cfg["files"] 列表）。
    多 xlsx 时每个 file 用 {id_prefix}_{文件stem} 作为子前缀，product_id 天然唯一。
    """
    import types
    from pathlib import Path

    mode = cfg["mode"]
    name = cfg["name"]
    id_prefix = cfg["id_prefix"]
    files = cfg.get("files")  # 多 xlsx 列表，与 cfg["xlsx"] 二选一

    # 收集 museum 级默认字段（可被 file 级覆盖）
    def _field_kwargs(keys: list[str]) -> dict:
        return {k: cfg[k] for k in keys if k in cfg}

    if mode == "simple":
        _keys = ["name_col", "dynasty_col", "intro_col", "category_col", "size_col",
                 "path_col", "strip_prefix"]

        def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
            defaults = _field_kwargs(_keys)
            if files:
                all_meta = {}
                for f in files:
                    sub_prefix = f"{id_prefix}_{Path(f['xlsx']).stem}"
                    f_kwargs = {**defaults}
                    f_kwargs.update({k: f[k] for k in _keys if k in f})
                    meta = simple_import_museum(
                        src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                        xlsx_filename=f["xlsx"], id_prefix=sub_prefix, dry_run=dry_run,
                        **f_kwargs,
                    )
                    _inject_category_top(meta, f.get("category_top"))
                    all_meta.update(meta)
                print(f"[{name}] 多 xlsx 合并：{len(files)} 个文件，共 {len(all_meta)} 条")
                return all_meta
            return simple_import_museum(
                src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                xlsx_filename=cfg["xlsx"], id_prefix=id_prefix, dry_run=dry_run,
                **defaults,
            )
        return import_museum

    elif mode == "info_3d":
        def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
            if files:
                all_meta = {}
                for f in files:
                    sub_prefix = f"{id_prefix}_{Path(f['xlsx']).stem}"
                    meta = info_3d_import_museum(
                        src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                        xlsx_filename=f["xlsx"], id_prefix=sub_prefix, dry_run=dry_run,
                    )
                    _inject_category_top(meta, f.get("category_top"))
                    all_meta.update(meta)
                print(f"[{name}] 多 xlsx 合并：{len(files)} 个文件，共 {len(all_meta)} 条")
                return all_meta
            return info_3d_import_museum(
                src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                xlsx_filename=cfg.get("xlsx", "info.xlsx"), id_prefix=id_prefix,
                dry_run=dry_run,
            )
        return import_museum

    elif mode == "xlsx_name":
        _keys = ["name_col", "intro_col", "dynasty_col", "material_col",
                 "category_top_col", "extra_caption_cols", "img_filename_col",
                 "name_as_image_filename"]

        def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
            defaults = _field_kwargs(_keys)
            if files:
                all_meta = {}
                for f in files:
                    sub_prefix = f"{id_prefix}_{Path(f['xlsx']).stem}"
                    # file 级字段覆盖 museum 级
                    f_kwargs = {**defaults}
                    f_kwargs.update({k: f[k] for k in _keys if k in f})
                    meta = xlsx_name_import_museum(
                        src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                        xlsx_filename=f["xlsx"], id_prefix=sub_prefix,
                        dry_run=dry_run, **f_kwargs,
                    )
                    _inject_category_top(meta, f.get("category_top"))
                    all_meta.update(meta)
                print(f"[{name}] 多 xlsx 合并：{len(files)} 个文件，共 {len(all_meta)} 条")
                return all_meta
            return xlsx_name_import_museum(
                src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                xlsx_filename=cfg["xlsx"], id_prefix=id_prefix,
                dry_run=dry_run, **defaults,
            )
        return import_museum

    elif mode == "category_subdir":
        def import_museum(src_dir: Path, dst_raw: Path, dry_run: bool = False) -> dict:
            return category_subdir_import_museum(
                src_dir=src_dir, dst_raw=dst_raw, museum_name=name,
                id_prefix=id_prefix,
                category_to_top=cfg.get("category_to_top"),
                dry_run=dry_run,
            )
        return import_museum

    else:
        raise ValueError(f"未知的 mode: {mode}（博物馆: {name}）")


def load_config_importers() -> dict[str, object]:
    """从 museums_config.py 加载配置驱动的导入器。

    为每条配置生成一个虚拟模块（types.SimpleNamespace），
    带 MUSEUM_NAME / XLSX_FILENAME / XLSX_FILES / import_museum 属性，
    与 discover_importers 的 .py 扫描结果格式一致。

    Returns: {MUSEUM_NAME: virtual_module}
    """
    import types
    from pathlib import Path

    try:
        # museums_config.py 与本文件同目录，直接用文件路径加载避免包导入问题
        cfg_path = Path(__file__).parent / "museums_config.py"
        if not cfg_path.exists():
            return {}
        import importlib.util
        spec = importlib.util.spec_from_file_location("scripts.importers.museums_config", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[配置] 加载 museums_config.py 失败: {e}")
        return {}

    importers = {}
    for cfg in mod.MUSEUMS:
        name = cfg["name"]
        files = cfg.get("files")
        if files:
            xlsx_files = [f["xlsx"] for f in files]
            xlsx_filename = xlsx_files[0]  # --list 显示用
        else:
            xlsx_files = None
            xlsx_filename = cfg.get("xlsx")

        virtual = types.SimpleNamespace()
        virtual.MUSEUM_NAME = name
        virtual.SRC_SUBDIR = cfg.get("src_subdir", name)  # 实际数据目录名（默认=name）
        virtual.XLSX_FILENAME = xlsx_filename
        virtual.XLSX_FILES = xlsx_files  # 多 xlsx 列表，用于增量判断
        virtual.import_museum = _make_config_import_museum(cfg)
        importers[name] = virtual
    return importers


