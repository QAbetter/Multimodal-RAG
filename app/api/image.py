"""
/image 路由：图片 RAG 的索引与检索接口。

对应书籍 RAG 的 api/chat.py，提供图片索引（上传+注册+向量化）和搜索功能。
接口清单：
- POST /image/index       ：上传图片并完成索引（注册→CLIP向量化→标签提取→入库）
- POST /image/batch_index ：批量索引已注册的图片（CLIP批量向量化+标签并发提取）
- POST /image/pdf/extract ：上传PDF，提取插图+对应文本（不索引，仅返回提取结果）
- POST /image/pdf/index   ：上传PDF，提取插图+对应文本并完成索引（提取→注册→CLIP→caption BM25）
- POST /image/search      ：文本搜图 / 以图搜图 / 混合检索
- GET  /image/stats       ：统计信息（图片数、向量数、标签索引规模）
- GET  /image/{image_id}  ：查询图片元数据
- DELETE /image/{image_id}：删除图片（清理注册表+向量库+标签索引+caption BM25）

图片静态文件服务：通过 main.py 挂载 StaticFiles，图片可通过 /images/{path} 访问。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.image_indexer import (
    batch_index_images,
    delete_image,
    get_image,
    index_image,
    load_registered_images,
    register_image,
)
from app.core.image_retriever import search
from app.core.image_vectorstore import count_images
from app.core.pdf_image_extractor import (
    ExtractedImage,
    cleanup_extract_temp,
    extract_images_from_pdf,
)
from app.core.tag_store import get_tag_stats
from app.models.image_schemas import (
    ImageIndexResponse,
    ImageSearchRequest,
    ImageSearchResponse,
)

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/index", response_model=ImageIndexResponse)
def index_uploaded_image(
    file: UploadFile = File(..., description="产品图片文件"),
    product_id: str = Form(..., description="所属产品 id"),
    category: str | None = Form(None, description="产品类别"),
) -> ImageIndexResponse:
    """上传图片并完成索引（注册 + CLIP 向量化 + 标签提取 + 写入向量库）。

    四步走第一步：能通过 API 上传/索引一张产品图片。
    """
    settings = get_settings()
    storage_dir = Path(settings.image_storage_dir)
    raw_dir = storage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传文件到 data/images/raw/
    # 用原始文件名（若冲突则覆盖，相同图片本就该是同一索引）
    file_path = f"raw/{file.filename}"
    dest = storage_dir / file_path
    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 注册 + 索引
    image = register_image(file_path, product_id, category)
    image = index_image(image.image_id)

    return ImageIndexResponse(
        image_id=image.image_id,
        status=image.status,
        tags=image.tags,
        message="索引成功" if image.status.value == "ready" else f"索引状态: {image.status.value}",
    )


class BatchIndexItem(BaseModel):
    """批量索引单条结果。"""

    image_id: str
    file_name: str = ""
    status: str
    tags: list[str] = Field(default_factory=list)


class BatchIndexResponse(BaseModel):
    """批量索引响应。"""

    total: int
    success: int
    failed: int
    skipped: int = 0  # 跳过的重复图片数（相同内容不同文件名）
    results: list[BatchIndexItem]


@router.post("/batch_index", response_model=BatchIndexResponse)
def batch_index_uploaded_images(
    files: list[UploadFile] = File(..., description="产品图片文件列表"),
    category: str | None = Form(None, description="统一产品类别"),
    batch_size: int = Form(32, description="CLIP 批量向量化的批大小"),
    tag_workers: int = Form(4, description="GLM-4V 标签提取并发数（智谱 API 限流建议 ≤8）"),
) -> BatchIndexResponse:
    """批量上传并索引图片（保存→批量注册→CLIP批量向量化→标签并发提取→入库）。

    比循环调用 /image/index 快很多：
    - CLIP 批量前向（一次处理 batch_size 张，而非逐张）
    - 标签并发提取（tag_workers 个线程并行调 GLM-4V）
    - Chroma 批量写入（减少事务次数）

    product_id 规则：用文件名（去扩展名），与 scripts/batch_index_images.py 一致。
    image_id 规则：文件内容 MD5 前 16 位，相同图片（不同文件名）会去重，只索引一次。
    """
    if not files:
        raise HTTPException(status_code=400, detail="files 不能为空")

    settings = get_settings()
    storage_dir = Path(settings.image_storage_dir)
    raw_dir = storage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存所有上传文件到 data/images/raw/，并注册
    # 同一图片（不同文件名或重复上传）会生成相同 image_id（基于文件内容 MD5），
    # 需去重避免 Chroma delete/add 重复 id 报 DuplicateIDError
    image_ids: list[str] = []
    name_map: dict[str, str] = {}  # image_id -> 首次出现的 file_name
    for file in files:
        if not file.filename:
            continue
        file_path = f"raw/{file.filename}"
        dest = storage_dir / file_path
        with dest.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # 注册（product_id 用文件名 stem，与脚本一致）
        product_id = Path(file.filename).stem
        image = register_image(file_path, product_id, category=category)
        # 去重：同一 image_id 只索引一次（基于文件内容 MD5，相同图片不重复索引）
        if image.image_id not in name_map:
            name_map[image.image_id] = file.filename
            image_ids.append(image.image_id)

    skipped = len(files) - len(image_ids)

    if not image_ids:
        raise HTTPException(status_code=400, detail="没有有效的图片文件")

    # 2. 批量索引（CLIP 批量向量化 + 标签并发提取）
    results = batch_index_images(
        image_ids,
        batch_size=batch_size,
        tag_workers=tag_workers,
    )

    items = [
        BatchIndexItem(
            image_id=r.image_id,
            file_name=name_map.get(r.image_id, ""),
            status=r.status.value,
            tags=r.tags,
        )
        for r in results
    ]
    success = sum(1 for r in results if r.status.value == "ready")
    failed = len(results) - success

    return BatchIndexResponse(
        total=len(results),
        success=success,
        failed=failed,
        skipped=skipped,
        results=items,
    )


class PdfExtractItem(BaseModel):
    """PDF 提取的单条图片结果。"""

    image_name: str = Field(..., description="图片文件名（如 img_001.png）")
    file_path: str = Field(..., description="图片相对路径（image_storage_dir 之下）")
    caption: str = Field("", description="图片对应的文本（PDF周边段落/图注）")


class PdfExtractResponse(BaseModel):
    """PDF 提取响应。"""

    pdf_name: str = Field(..., description="PDF 文件名")
    total: int = Field(..., description="提取的图片总数")
    with_caption: int = Field(..., description="有 caption 的图片数")
    images: list[PdfExtractItem] = Field(default_factory=list)


@router.post("/pdf/extract", response_model=PdfExtractResponse)
def pdf_extract_images(
    file: UploadFile = File(..., description="PDF 文件"),
) -> PdfExtractResponse:
    """上传 PDF，提取插图及对应文本（不索引，仅返回提取结果）。

    调用智谱同步文件解析 API（prime-sync），返回 ZIP 含图片+Markdown，
    解析 Markdown 建立「图片→周边文本」映射，图片落到 data/images/raw/pdf/{stem}/。

    适合预览提取效果：先看提取了哪些图片、caption 是否合理，再决定是否索引。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    settings = get_settings()
    # 保存上传 PDF 到专用原始文件目录（data/raw/pdf/），便于复测和管理
    pdf_dir = Path(settings.pdf_raw_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / file.filename
    with pdf_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        extracted = extract_images_from_pdf(str(pdf_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {e}")

    items = [
        PdfExtractItem(
            image_name=img.image_name,
            file_path=img.file_path,
            caption=img.caption,
        )
        for img in extracted
    ]
    with_caption = sum(1 for it in items if it.caption)

    return PdfExtractResponse(
        pdf_name=file.filename,
        total=len(items),
        with_caption=with_caption,
        images=items,
    )


class PdfIndexItem(BaseModel):
    """PDF 索引的单条结果。"""

    image_id: str
    image_name: str = ""
    status: str
    tags: list[str] = Field(default_factory=list)
    caption: str = Field("", description="图片对应的文本（PDF提取）")


class PdfIndexResponse(BaseModel):
    """PDF 索引响应。"""

    pdf_name: str
    total: int
    success: int
    failed: int
    results: list[PdfIndexItem]


@router.post("/pdf/index", response_model=PdfIndexResponse)
def pdf_index_images(
    file: UploadFile = File(..., description="PDF 文件"),
    category: str | None = Form(None, description="统一产品类别"),
    batch_size: int = Form(32, description="CLIP 批量向量化的批大小"),
    tag_workers: int = Form(4, description="GLM-4V 标签提取并发数"),
) -> PdfIndexResponse:
    """上传 PDF，提取插图+对应文本并完成索引（提取→注册→CLIP→caption BM25→标签）。

    一站式接口：调一次即可把 PDF 里的所有图片纳入检索。
    - 智谱解析提取图片 + caption
    - 注册到 ImageMetadata（含 caption/pdf_source）
    - 批量索引：CLIP 向量化 + GLM-4V 标签 + caption BM25 索引
    - 索引后可通过 /image/search 用文本查图（caption BM25 自动启用混合检索）
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    settings = get_settings()
    pdf_dir = Path(settings.image_storage_dir) / ".pdf_uploads"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / file.filename
    with pdf_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 1. 提取图片 + caption
    try:
        extracted: list[ExtractedImage] = extract_images_from_pdf(str(pdf_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {e}")

    if not extracted:
        return PdfIndexResponse(pdf_name=file.filename, total=0, success=0, failed=0, results=[])

    # 2. 注册（带 caption/pdf_source）+ 去重
    image_ids: list[str] = []
    name_map: dict[str, str] = {}
    for img in extracted:
        product_id = Path(img.image_name).stem  # 用图片文件名 stem 作为 product_id
        image = register_image(
            file_path=img.file_path,
            product_id=product_id,
            category=category,
            caption=img.caption or None,
            pdf_source=file.filename,
        )
        if image.image_id not in name_map:
            name_map[image.image_id] = img.image_name
            image_ids.append(image.image_id)

    # 3. 批量索引（CLIP 向量化 + 标签提取 + caption BM25 入库）
    results = batch_index_images(
        image_ids,
        batch_size=batch_size,
        tag_workers=tag_workers,
    )

    # 4. 清理解析临时目录（图片已复制到正式存储，ZIP 解压内容可删）
    cleanup_extract_temp(Path(file.filename).stem)

    items = [
        PdfIndexItem(
            image_id=r.image_id,
            image_name=name_map.get(r.image_id, ""),
            status=r.status.value,
            tags=r.tags,
            caption=r.caption or "",
        )
        for r in results
    ]
    success = sum(1 for r in results if r.status.value == "ready")
    failed = len(results) - success

    return PdfIndexResponse(
        pdf_name=file.filename,
        total=len(results),
        success=success,
        failed=failed,
        results=items,
    )


@router.post("/search", response_model=ImageSearchResponse)
def search_images(request: ImageSearchRequest) -> ImageSearchResponse:
    """图片搜索：文本搜图 / 以图搜图 / 标签过滤。

    四步走第二、三、四步：
    - 文本搜图（query 非空）：CLIP 文本向量 → 相似图片；有 caption 索引时自动启用混合检索
    - 以图搜图（image_base64 非空）：CLIP 图像向量 → 相似图片
    - 结果含标签、类别、相似度分数、caption（PDF 提取的图片才有）
    """
    if not request.query and not request.image_base64:
        raise HTTPException(status_code=400, detail="query 和 image_base64 至少传一个")

    return search(
        query=request.query,
        image_base64=request.image_base64,
        category=request.category,
        tags=request.tags,
        top_k=request.top_k,
        base_url="",  # 同源访问，相对路径即可
    )


@router.get("/stats")
def image_stats() -> dict:
    """图片 RAG 系统统计信息（用于健康检查 / 监控）。

    返回：
    - registered：注册表中的图片总数
    - ready：已成功索引（status=ready）的图片数
    - vectors：向量库中的向量数（应与 ready 一致，不一致说明有脏数据）
    - tags：标签倒排索引统计（标签数、标签-图片关系数、平均每标签图片数）
    - captions：caption BM25 索引统计（有caption的图片数，PDF提取的图片才有）
    """
    from app.core.image_bm25_store import get_caption_stats

    registry = load_registered_images()
    ready = sum(1 for img in registry.values() if img.status.value == "ready")
    return {
        "registered": len(registry),
        "ready": ready,
        "vectors": count_images(),
        "tags": get_tag_stats(),
        "captions": get_caption_stats(),
    }


@router.get("/{image_id}")
def get_image_info(image_id: str) -> dict:
    """查询图片元数据（索引状态、标签等）。"""
    image = get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")
    return image.model_dump()


@router.delete("/{image_id}")
def delete_image_api(image_id: str) -> dict:
    """删除图片：清理注册表 + 向量库 + 标签倒排索引。

    删除是幂等的：图片不存在时返回 404，存在时清理所有关联数据。
    """
    deleted = delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")
    return {"image_id": image_id, "message": "删除成功"}
