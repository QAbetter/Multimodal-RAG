"""
PDF 插图提取：调用 MinerU 精准解析 API，提取 PDF 中的图片及其对应文本。

对应图片 RAG 的 image_loader.py（加载单张图片），但这里处理的是 PDF：
- 调用 MinerU 批量上传接口（/api/v4/file-urls/batch）上传 PDF 并自动提交解析任务
- 轮询批量结果接口（/api/v4/extract-results/batch/{batch_id}）获取结果 ZIP 下载链接
- ZIP 内含：images/ 目录（提取的图片）+ Markdown + middle.json（中间处理结果）
- 优先用 middle.json 的 para_blocks 建立图文映射（MinerU 已将图片与图注配对在同一 block 内）
- middle.json 缺失时降级用 Markdown 的前后行匹配（简单版式）
- 大 PDF（超体积/页数阈值）按页切分后分批解析，合并结果

设计要点：
- MinerU middle.json 的 image block 结构：
  para_blocks[] 中 type=="image" 的 block，其 blocks[] 子块含：
  - image_body（图片本体，image_path 为远程 CDN URL）
  - image_footnote（图注文本，已与图片配对，无需坐标匹配）
- 图片获取：优先从 ZIP 内 images/ 目录取本地文件；文件名不匹配时从 CDN URL 下载
- PDF 切分：MinerU 限制单文件≤200MB/200页，超过阈值按页拆分后逐个解析
- 失败降级：单张图片解析失败不影响其他图片，caption 为空时仍保留图片

API 调用流程（异步批量上传）：
1. POST /api/v4/file-urls/batch 申请上传链接，返回 batch_id + file_urls[]
2. PUT 上传文件到 file_urls[0]（系统自动提交解析任务）
3. GET /api/v4/extract-results/batch/{batch_id} 轮询，state==done 时取 full_zip_url
4. 下载 ZIP，解压到临时目录
5. 优先解析 middle.json 建立图文映射；无 JSON 则降级用 Markdown
6. 遍历 images/ 目录的图片，复制到正式存储目录
"""
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Markdown 图片引用正则：![alt](path)，alt 可空，path 可能含相对目录
_MD_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# 支持的图片扩展名（用于从 ZIP 中筛选图片文件）
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


@dataclass
class ExtractedImage:
    """单张从 PDF 提取的图片及其对应文本。

    file_path: 相对 image_storage_dir 的路径（如 raw/pdf/xxx/img_001.png），用于注册索引
    caption: 图片对应的文本（middle.json 中 image_footnote 的 content）
    image_name: 图片文件名（如 img_001.png），用于日志和去重
    page_number: 图片所在 PDF 页码（middle.json 可知，否则为 None）
    """

    file_path: str
    caption: str
    image_name: str
    page_number: Optional[int] = None


# ===========================================================================
# MinerU API 调用
# ===========================================================================

def _get_mineru_headers() -> dict[str, str]:
    """构造 MinerU API 请求头。"""
    settings = get_settings()
    token = settings.mineru_token
    if not token:
        raise ValueError("未配置 MinerU Token（mineru_token），请在 .env 中设置 MINERU_TOKEN")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _call_mineru_parser(pdf_path: str) -> str:
    """调用 MinerU 精准解析 API，返回结果 ZIP 下载链接。

    流程（批量上传接口，一次只传1个文件）：
    1. POST /file-urls/batch 申请上传链接，返回 batch_id + file_urls[]
    2. PUT 上传文件到 file_urls[0]
    3. GET /extract-results/batch/{batch_id} 轮询，state==done 时取 full_zip_url

    失败时抛异常，由上层调用方决定降级策略（如跳过该 PDF）。
    """
    import time

    settings = get_settings()
    headers = _get_mineru_headers()
    api_base = settings.mineru_api_base

    # 1. 申请上传链接
    file_name = Path(pdf_path).name
    apply_url = f"{api_base}/file-urls/batch"
    apply_data = {
        "files": [{"name": file_name}],
        "model_version": settings.mineru_model_version,
        "is_ocr": settings.mineru_is_ocr,
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }
    logger.info("MinerU 申请上传链接: %s", file_name)
    resp = requests.post(apply_url, headers=headers, json=apply_data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"MinerU 申请上传链接失败: {result.get('msg', result)}")

    batch_id = result["data"]["batch_id"]
    upload_url = result["data"]["file_urls"][0]
    logger.info("MinerU 上传链接已获取, batch_id=%s", batch_id)

    # 2. PUT 上传文件（注意：上传时不设置 Content-Type 头）
    logger.info("MinerU 上传文件: %s", file_name)
    with open(pdf_path, "rb") as f:
        upload_resp = requests.put(upload_url, data=f, timeout=300)
    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"MinerU 文件上传失败, HTTP {upload_resp.status_code}: {upload_resp.text[:200]}"
        )
    logger.info("MinerU 文件上传成功, 等待解析...")

    # 3. 轮询批量结果
    query_url = f"{api_base}/extract-results/batch/{batch_id}"
    poll_interval = settings.mineru_poll_interval
    poll_timeout = settings.mineru_poll_timeout
    start_time = time.time()

    while time.time() - start_time < poll_timeout:
        poll_resp = requests.get(query_url, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        if poll_data.get("code") != 0:
            raise RuntimeError(f"MinerU 查询失败: {poll_data.get('msg', poll_data)}")

        extract_result = poll_data.get("data", {}).get("extract_result", [])
        if extract_result:
            item = extract_result[0]
            state = item.get("state", "")
            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    raise RuntimeError(f"MinerU 解析完成但未返回 full_zip_url: {item}")
                elapsed = time.time() - start_time
                logger.info("MinerU 解析完成（耗时 %.1fs）, ZIP: %s", elapsed, zip_url[:80])
                return zip_url
            elif state == "failed":
                err_msg = item.get("err_msg", "未知错误")
                raise RuntimeError(f"MinerU 解析失败: {err_msg}")
            else:
                # pending / running / converting
                logger.debug("MinerU 解析中, state=%s", state)
        time.sleep(poll_interval)

    raise TimeoutError(f"MinerU 解析超时（{poll_timeout}s），batch_id={batch_id}")


def _download_and_extract_zip(zip_url: str, dest_dir: Path) -> Path:
    """下载结果 ZIP 并解压到 dest_dir，返回解压目录。

    MinerU ZIP 结构通常为：
        result.zip
        ├── images/
        │   ├── xxx.jpg
        ├── full.md
        ├── xxx_middle.json      (中间处理结果，含图文配对)
        ├── xxx_content_list.json (内容列表，简化版)
        └── xxx_model.json       (模型推理结果)
    """
    settings = get_settings()
    resp = requests.get(zip_url, timeout=settings.mineru_download_timeout)
    resp.raise_for_status()

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)
    return dest_dir


# ===========================================================================
# 文件查找工具
# ===========================================================================

def _find_markdown(extract_dir: Path) -> Optional[Path]:
    """在解压目录中查找 Markdown 文件（.md）。"""
    return next(extract_dir.rglob("*.md"), None)


def _find_middle_json(extract_dir: Path) -> Optional[Path]:
    """在解压目录中查找 MinerU 的 middle.json 文件。

    MinerU ZIP 内的命名格式为 {original_filename}_middle.json。
    匹配策略（按优先级）：
    1. 明确的 middle.json：*_middle.json（MinerU 实际命名）
    2. 兜底：任意非 content_list/model 的 .json
    """
    # 1. 优先找 MinerU 的 middle.json
    middle_files = sorted(extract_dir.rglob("*_middle.json"), key=lambda p: p.name)
    if middle_files:
        return middle_files[0]

    # 2. 兜底：找 content_list.json（简化版，也含图文配对）
    content_list_files = sorted(extract_dir.rglob("*_content_list.json"), key=lambda p: p.name)
    if content_list_files:
        return content_list_files[0]

    return None


def _find_images(extract_dir: Path) -> list[Path]:
    """在解压目录中查找所有图片文件（递归，按文件名排序保证顺序稳定）。"""
    return sorted(
        [p for p in extract_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS and p.is_file()],
        key=lambda p: p.name,
    )


# ===========================================================================
# middle.json 解析（MinerU 已将图片与图注配对，无需坐标匹配）
# ===========================================================================

def _extract_image_name_from_url(url_or_path: str) -> str:
    """从图片路径或 URL 中提取文件名。

    MinerU middle.json 中 image_path 可能是：
    - 远程 CDN URL: https://cdn-mineru.openxlab.org.cn/result/.../xxx.jpg
    - 相对路径: images/xxx.jpg
    两种情况都取最后一段作为文件名。
    """
    parsed = urlparse(url_or_path)
    # 如果是 URL，path 部分含文件名；如果是相对路径，直接取 basename
    path = parsed.path if parsed.scheme else url_or_path
    return Path(path).name


def _parse_middle_json(middle_path: Path) -> dict[str, tuple[str, int]]:
    """解析 MinerU middle.json，返回 {图片文件名: (caption, page_idx)}。

    MinerU 已将图片与图注配对在同一个 para_block 内（type=="image"），
    无需自己做坐标匹配。每个 image block 的子 blocks 含：
    - image_body: 图片本体，spans[].image_path 为图片 URL/路径
    - image_footnote: 图注文本，spans[].content 为图注内容

    示例结构：
        para_blocks[].{
            "type": "image",
            "blocks": [
                {"type": "image_body", "lines": [{"spans": [{"image_path": "..."}]}]},
                {"type": "image_footnote", "lines": [{"spans": [{"content": "图注"}]}]}
            ]
        }
    """
    full = _parse_middle_json_full(middle_path)
    return {img["name"]: (img["caption"], img["page_idx"]) for img in full["images"]}


# middle.json 中视为"文本"的 para_block 类型（用于坐标匹配降级）
_TEXT_BLOCK_TYPES = {"text", "title"}


def _parse_middle_json_full(middle_path: Path) -> dict:
    """解析 MinerU middle.json，返回完整的图片和文本块坐标信息。

    返回结构：
        {
            "images": [
                {"name": str, "caption": str, "bbox": [x0,y0,x1,y1], "page_idx": int}
            ],
            "text_blocks": [
                {"content": str, "bbox": [x0,y0,x1,y1], "page_idx": int}
            ]
        }

    图片的 caption 来自 image_footnote（可能为空字符串）；
    text_blocks 收集所有 text/title 类型的 para_block（含坐标和内容），
    供 _match_by_coordinates 对无 caption 的图片做降级匹配。
    """
    if not middle_path or not middle_path.exists():
        return {"images": [], "text_blocks": []}

    try:
        data = json.loads(middle_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        logger.warning("middle.json 解析失败: %s", middle_path)
        return {"images": [], "text_blocks": []}

    images: list[dict] = []
    text_blocks: list[dict] = []

    for page in data.get("pdf_info", []):
        page_idx = page.get("page_idx", 0)
        for block in page.get("para_blocks", []):
            block_type = block.get("type", "")
            bbox = block.get("bbox", [0, 0, 0, 0])

            if block_type == "image":
                img_name = ""
                caption_parts: list[str] = []
                for sub_block in block.get("blocks", []):
                    sub_type = sub_block.get("type", "")
                    for line in sub_block.get("lines", []):
                        for span in line.get("spans", []):
                            if sub_type == "image_body" and span.get("image_path"):
                                img_name = _extract_image_name_from_url(span["image_path"])
                            elif sub_type == "image_footnote" and span.get("content"):
                                caption_parts.append(span["content"].strip())
                if img_name:
                    images.append({
                        "name": img_name,
                        "caption": " ".join(caption_parts).strip(),
                        "bbox": bbox,
                        "page_idx": page_idx,
                    })
            elif block_type in _TEXT_BLOCK_TYPES:
                # 收集文本块的 content（从 lines[].spans[].content 拼接）
                content_parts: list[str] = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("content"):
                            content_parts.append(span["content"].strip())
                content = " ".join(content_parts).strip()
                if content:
                    text_blocks.append({
                        "content": content,
                        "bbox": bbox,
                        "page_idx": page_idx,
                    })

    return {"images": images, "text_blocks": text_blocks}


def _match_by_coordinates(
    uncaptioned_images: list[dict],
    text_blocks: list[dict],
) -> dict[str, str]:
    """基于坐标的空间邻近度匹配：为无 caption 的图片找最近的文本块。

    匹配策略（每个段落归属于"正下方且x坐标有重叠"的图片）：
    - 正下方匹配（主）：段落顶部y >= 图片底部y（段落在图片下方，符合图注惯例）
    - x坐标有重叠：图片x范围与段落x范围有交集（确保段落属于该图而非相邻图）
    - 多候选时选综合距离最近的（垂直距离×1.0 + x中心距离×0.5）
    - 一个段落只归属一个图，避免重复

    能精准处理"图1图2图3一排 / 注1注2注3一排"的并排版式：
    每个注的x范围只与正上方对应的图重叠，不会误配到相邻图。

    无匹配图片的段落（如页脚、独立段落）被跳过。
    段落按 y 坐标升序处理（从上到下），同一图片的多个段落按顺序拼接为 caption。
    """
    if not uncaptioned_images or not text_blocks:
        return {}

    caption_map: dict[str, list[str]] = {}

    # 文本块按 y 坐标升序排序（从上到下，符合阅读顺序）
    sorted_texts = sorted(text_blocks, key=lambda t: t["bbox"][1])

    for text in sorted_texts:
        t_bbox = text["bbox"]
        t_left, t_top, t_right, t_bottom = t_bbox
        t_center_x = (t_left + t_right) / 2
        t_page = text["page_idx"]

        # 找同页的正下方候选图片（文本在图片下方且 x 有重叠）
        candidates: list[tuple[dict, float]] = []
        for img in uncaptioned_images:
            if img["page_idx"] != t_page:
                continue
            i_bbox = img["bbox"]
            i_left, i_top, i_right, i_bottom = i_bbox
            # 文本顶部必须在图片底部下方（允许 10px 容差应对坐标抖动）
            if t_top < i_bottom - 10:
                continue
            # x 坐标有重叠：max(左边界) < min(右边界)
            overlap = min(i_right, t_right) - max(i_left, t_left)
            if overlap <= 0:
                continue
            # 综合距离：垂直距离（主） + x中心距离（辅）
            vert_dist = t_top - i_bottom
            x_dist = abs(t_center_x - (i_left + i_right) / 2)
            score = vert_dist * 1.0 + x_dist * 0.5
            candidates.append((img, score))

        if not candidates:
            # 下方无图，尝试上方匹配（某些文档图注在图上方）
            for img in uncaptioned_images:
                if img["page_idx"] != t_page:
                    continue
                i_bbox = img["bbox"]
                i_left, i_top, i_right, i_bottom = i_bbox
                if t_bottom > i_top + 10:
                    continue
                overlap = min(i_right, t_right) - max(i_left, t_left)
                if overlap <= 0:
                    continue
                vert_dist = i_top - t_bottom
                x_dist = abs(t_center_x - (i_left + i_right) / 2)
                score = vert_dist * 1.0 + x_dist * 0.5
                candidates.append((img, score))

        if not candidates:
            continue  # 文本无归属图片，跳过

        # 选综合距离最近的图
        candidates.sort(key=lambda x: x[1])
        best_img = candidates[0][0]
        caption_map.setdefault(best_img["name"], []).append(text["content"])

    # 合并每个图的文本片段
    return {name: " ".join(texts).strip() for name, texts in caption_map.items()}


def _parse_content_list(content_list_path: Path) -> dict[str, tuple[str, int]]:
    """解析 MinerU content_list.json（简化版），返回 {图片文件名: (caption, page_idx)}。

    content_list.json 是 middle.json 的简化版，扁平结构，每个元素含：
    - type: "image"
    - img_path: "images/xxx.jpg"（相对路径）
    - image_caption: ["图注1", "图注2"]（列表）
    - page_idx: 页码
    """
    if not content_list_path or not content_list_path.exists():
        return {}

    try:
        data = json.loads(content_list_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        logger.warning("content_list.json 解析失败: %s", content_list_path)
        return {}

    result: dict[str, tuple[str, int]] = {}
    for block in data:
        if block.get("type") != "image":
            continue
        img_path = block.get("img_path", "")
        if not img_path:
            continue
        img_name = Path(img_path).name
        captions = block.get("image_caption", [])
        caption = " ".join(captions).strip() if captions else ""
        page_idx = block.get("page_idx", 0)
        result[img_name] = (caption, page_idx)

    return result


# ===========================================================================
# Markdown 匹配（降级方案：middle.json 缺失时使用）
# ===========================================================================

def _build_image_caption_map_from_markdown(markdown_path: Path) -> dict[str, str]:
    """解析 Markdown，建立「图片文件名 → caption」映射（降级方案）。

    策略：对每个 ![](images/xxx.png) 引用，取其 alt 文本 + 前后各 2 行作为 caption。
    - alt 文本：MinerU 解析通常会把图注放入 alt
    - 前后段落：补充上下文

    此方案无法处理并排图片（Markdown 是线性的，丢失了空间位置信息），
    仅在 middle.json 不可用时降级使用。

    若同一图片在 Markdown 中被多次引用，取第一次出现的上下文。
    """
    if not markdown_path or not markdown_path.exists():
        return {}

    text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    caption_map: dict[str, str] = {}
    for i, line in enumerate(lines):
        for match in _MD_IMAGE_PATTERN.finditer(line):
            alt_text = match.group(1).strip()
            img_path = match.group(2).strip()
            img_name = Path(img_path).name

            if img_name in caption_map:
                continue

            # 收集上下文：前 2 行 + 当前行 alt + 后 2 行
            context_parts: list[str] = []
            for j in range(max(0, i - 2), min(len(lines), i + 3)):
                if j == i:
                    if alt_text:
                        context_parts.append(alt_text)
                    continue
                neighbor = lines[j].strip()
                if not neighbor or neighbor.startswith("!"):
                    continue
                neighbor = re.sub(r"^#{1,6}\s*", "", neighbor)
                if neighbor:
                    context_parts.append(neighbor)

            caption_map[img_name] = " ".join(context_parts).strip()
    return caption_map


# ===========================================================================
# PDF 切分（大文件按页拆分，规避 MinerU 200MB/200页限制）
# ===========================================================================

def _should_split_pdf(pdf_path: str) -> bool:
    """判断 PDF 是否需要切分（超过体积或页数阈值）。

    MinerU 限制单文件≤200MB/200页。超过任一阈值则按页切分，分批解析后合并结果。
    阈值在 config 中可调，默认体积 150MB、页数 150 页（留余量确保稳定）。
    """
    settings = get_settings()
    size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
    if size_mb > settings.pdf_split_size_mb:
        return True

    # 页数判断（需 pypdf，项目已有依赖）
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        if len(reader.pages) > settings.pdf_split_page_threshold:
            return True
    except Exception:
        logger.debug("pypdf 读取页数失败，跳过页数阈值判断: %s", pdf_path)
    return False


def _split_pdf(pdf_path: str, pdf_stem: str) -> list[Path]:
    """按页切分 PDF，返回子 PDF 路径列表。

    每个子文件包含 pdf_split_chunk_pages 页（默认 100 页），
    切分到 image_storage_dir/.pdf_split_tmp/{pdf_stem}/ 下，用完由调用方清理。
    """
    from pypdf import PdfReader, PdfWriter

    settings = get_settings()
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    chunk_size = settings.pdf_split_chunk_pages

    tmp_dir = Path(settings.image_storage_dir) / ".pdf_split_tmp" / pdf_stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    for start in range(0, total, chunk_size):
        writer = PdfWriter()
        for i in range(start, min(start + chunk_size, total)):
            writer.add_page(reader.pages[i])
        part_path = tmp_dir / f"part_{start // chunk_size:03d}.pdf"
        with part_path.open("wb") as f:
            writer.write(f)
        parts.append(part_path)
    logger.info("PDF 切分完成: %s（%d 页 → %d 个子文件，每 %d 页）",
                pdf_stem, total, len(parts), chunk_size)
    return parts


def _cleanup_split_tmp(pdf_stem: str) -> None:
    """清理 PDF 切分的临时文件。"""
    import shutil
    settings = get_settings()
    tmp_root = Path(settings.image_storage_dir) / ".pdf_split_tmp" / pdf_stem
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)


# ===========================================================================
# 图片复制到存储目录
# ===========================================================================

def _copy_image_to_storage(
    src_image: Path,
    pdf_stem: str,
    storage_root: Path,
) -> str:
    """把提取的图片复制到 image_storage_dir/raw/pdf/{pdf_stem}/ 下。

    返回相对 image_storage_dir 的路径（如 raw/pdf/xxx/img_001.png），
    与 image_indexer.register_image 期望的 file_path 格式一致。
    """
    dest_dir = storage_root / "raw" / "pdf" / pdf_stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_image.name
    dest_path.write_bytes(src_image.read_bytes())
    return f"raw/pdf/{pdf_stem}/{src_image.name}"


# ===========================================================================
# 主流程：单 PDF 解析（含 middle.json 优先匹配 + Markdown 降级）
# ===========================================================================

def _extract_single_pdf(
    pdf_path: str,
    stem: str,
    storage_root: Path,
) -> list[ExtractedImage]:
    """解析单个 PDF 文件（不切分），返回提取的图片列表。

    主流程：
    1. 调 MinerU 解析 API，拿 ZIP 下载链接
    2. 下载并解压 ZIP
    3. 优先用 middle.json 的 para_blocks 匹配图文（MinerU 已配对）；无 JSON 则降级用 Markdown
    4. 把图片复制到 image_storage_dir/raw/pdf/{stem}/
    5. 返回 ExtractedImage 列表
    """
    logger.info("开始解析 PDF: %s (stem=%s)", pdf_path, stem)

    # 1. 调用 MinerU API
    zip_url = _call_mineru_parser(pdf_path)
    logger.info("MinerU 解析完成，下载链接: %s", zip_url[:80])

    # 2. 下载并解压到临时目录
    extract_dir = storage_root / ".pdf_extract_tmp" / stem
    if extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
    _download_and_extract_zip(zip_url, extract_dir)
    logger.info("ZIP 解压完成: %s", extract_dir)

    # 3. 优先用 middle.json 匹配图文（MinerU 已将图片与图注配对）
    json_path = _find_middle_json(extract_dir)
    caption_map: dict[str, str] = {}
    image_name_to_page: dict[str, int] = {}

    if json_path:
        # middle.json 和 content_list.json 用不同解析函数
        if json_path.name.endswith("_middle.json"):
            parsed = _parse_middle_json(json_path)
        else:
            parsed = _parse_content_list(json_path)

        if parsed:
            caption_map = {name: cap for name, (cap, _) in parsed.items()}
            image_name_to_page = {name: page for name, (_, page) in parsed.items()}
            captioned_count = sum(1 for cap, _ in parsed.values() if cap)
            logger.info("JSON 匹配完成: %d 张图片，%d 张有 caption",
                        len(parsed), captioned_count)

            # 降级：对无 caption 的图片用坐标匹配（middle.json 含 bbox 信息）
            # MinerU 解析可能漏掉 image_footnote，此时用图片与文本块的坐标邻近度补充
            if json_path.name.endswith("_middle.json") and captioned_count < len(parsed):
                full = _parse_middle_json_full(json_path)
                uncaptioned = [img for img in full["images"] if not img["caption"]]
                if uncaptioned and full["text_blocks"]:
                    matched = _match_by_coordinates(uncaptioned, full["text_blocks"])
                    if matched:
                        caption_map.update(matched)
                        logger.info("坐标匹配降级补充: %d 张图片获得 caption", len(matched))
        else:
            logger.warning("JSON 中无 image 元素，降级用 Markdown 匹配")
    else:
        logger.warning("未找到 middle.json 或 content_list.json，降级用 Markdown 匹配")

    # 降级：JSON 不存在或无 image 时，用 Markdown 匹配
    if not caption_map:
        md_path = _find_markdown(extract_dir)
        if md_path:
            caption_map = _build_image_caption_map_from_markdown(md_path)
            logger.info("Markdown 降级匹配完成: %d 张图片有 caption", len(caption_map))

    # 4. 遍历图片，复制到存储目录并组装结果
    settings = get_settings()
    max_chars = settings.image_caption_max_chars
    results: list[ExtractedImage] = []
    for img_path in _find_images(extract_dir):
        try:
            rel_path = _copy_image_to_storage(img_path, stem, storage_root)
            caption = caption_map.get(img_path.name, "")
            if caption and len(caption) > max_chars:
                caption = caption[:max_chars]
            page_number = image_name_to_page.get(img_path.name)
            results.append(
                ExtractedImage(
                    file_path=rel_path,
                    caption=caption,
                    image_name=img_path.name,
                    page_number=page_number,
                )
            )
        except Exception:
            logger.exception("复制图片失败，跳过: %s", img_path)

    logger.info("PDF 提取完成: %s，共 %d 张图片", stem, len(results))
    return results


def extract_images_from_pdf(
    pdf_path: str,
    pdf_stem: Optional[str] = None,
) -> list[ExtractedImage]:
    """提取 PDF 中的所有图片及其对应文本（主入口，含大文件切分逻辑）。

    流程：
    1. 判断是否需要切分（超体积/页数阈值）
    2. 需要切分：按页拆分 → 并发调 _extract_single_pdf（pdf_concurrent_workers 线程）→ 合并结果 → 清理临时文件
    3. 无需切分：直接调 _extract_single_pdf

    pdf_stem 用于命名存储子目录，不传则用 PDF 文件名（去扩展名）。
    同名 PDF 重复提取会覆盖旧图片（相同文件名），适合重新解析场景。

    并发说明：
    - 切分后多个子PDF用 ThreadPoolExecutor 并发调用 MinerU API，大幅减少等待时间
    - 并发数由 settings.pdf_concurrent_workers 控制（默认 2，建议≤3 避免触发限流）
    - 单切片失败不影响其他切片（错误隔离）

    失败处理：
    - 整个 PDF 解析失败：抛异常（上层决定是否跳过）
    - 切分后单个子PDF失败：跳过该子文件，继续处理其他（记录 warning）
    - 单张图片复制失败：跳过该图片，继续处理其他
    - caption 匹配失败：图片仍保留，caption 为空（走 CLIP 向量检索）
    """
    settings = get_settings()
    storage_root = Path(settings.image_storage_dir)
    pdf_path_obj = Path(pdf_path)
    stem = pdf_stem or pdf_path_obj.stem

    # 判断是否需要切分
    if _should_split_pdf(pdf_path):
        logger.info("PDF 较大，按页切分后并发解析: %s（workers=%d）", stem, settings.pdf_concurrent_workers)
        parts = _split_pdf(pdf_path, stem)
        all_results: list[ExtractedImage] = []
        failed_parts: list[str] = []
        max_workers = min(settings.pdf_concurrent_workers, len(parts))

        def _process_part(args: tuple[int, Path]) -> tuple[int, list[ExtractedImage]]:
            idx, part_path = args
            part_stem = f"{stem}_part{idx:03d}"
            return idx, _extract_single_pdf(str(part_path), part_stem, storage_root)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_part, (idx, part_path)): idx
                for idx, part_path in enumerate(parts)
            }
            # 按完成顺序收集，但最终按 idx 排序保证图片顺序稳定
            results_by_idx: dict[int, list[ExtractedImage]] = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    idx, results = future.result()
                    results_by_idx[idx] = results
                    logger.info("切片 part%03d 解析完成: %d 张图片", idx, len(results))
                except Exception:
                    logger.exception("子PDF解析失败，跳过: %s", parts[idx])
                    failed_parts.append(parts[idx].name)

        # 按 idx 升序合并，保证图片顺序与 PDF 页码顺序一致
        for idx in sorted(results_by_idx.keys()):
            all_results.extend(results_by_idx[idx])

        _cleanup_split_tmp(stem)
        if failed_parts:
            logger.warning("失败的切片: %s", failed_parts)
        logger.info(
            "PDF 并发解析全部完成: %s，共 %d 张图片（失败 %d 切片）",
            stem, len(all_results), len(failed_parts),
        )
        return all_results

    # 无需切分，直接解析
    return _extract_single_pdf(pdf_path, stem, storage_root)


def cleanup_extract_temp(pdf_stem: Optional[str] = None) -> None:
    """清理解析临时目录（解压后的 ZIP 内容，已复制到正式存储，可删）。

    pdf_stem 为 None 时清理整个 .pdf_extract_tmp 目录。
    建议在索引完成后调用，避免磁盘占用。
    注意：切分场景下子PDF用 {stem}_partXXX 命名，需单独清理（由 _cleanup_split_tmp 处理）。
    """
    import shutil
    settings = get_settings()
    tmp_root = Path(settings.image_storage_dir) / ".pdf_extract_tmp"
    if not tmp_root.exists():
        return
    if pdf_stem:
        for sub in tmp_root.iterdir():
            if sub.name == pdf_stem or sub.name.startswith(f"{pdf_stem}_part"):
                shutil.rmtree(sub, ignore_errors=True)
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
