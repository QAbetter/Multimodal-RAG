"""
RAGFlow / Dify 外部知识库适配层：把本项目的图片 RAG 检索结果转成 Dify 外部知识库 API 规范。

RAGFlow 作为编排/评测前端，调用本服务的 /api/v1/dify/retrieval 接口获取检索结果，
本服务作为检索后端，复用已有的 image_retriever.search（CLIP 向量 + 标签 + caption BM25 + RRF 融合）。

Dify 外部知识库 API 规范（事实标准）：
    POST /api/v1/dify/retrieval
    Headers: Authorization: Bearer <api_key>
    Body: {"query": "...", "retrieval_setting": {"top_k": 10, "score_threshold": 0.5},
           "knowledge_id": "...", "metadata_condition": null}
    Response: {"records": [{"content": "...", "title": "...", "score": 0.8, "metadata": {...}}]}

字段映射（图片检索结果 → Dify record）：
- content: caption（图片对应的文本），无 caption 时用 tags 拼接，确保 RAGFlow 有文本可喂给 LLM
- title: product_id 或 image_id（图片标识）
- score: 纯向量检索为余弦相似度（0~1）；混合检索为 RRF 融合分数（量纲不同，不与 score_threshold 比较）
- metadata: image_id / image_url / tags / category / pdf_source / page_number 等
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.image_retriever import search

router = APIRouter(prefix="/api/v1/dify", tags=["dify"])


class RetrievalSetting(BaseModel):
    """Dify 外部知识库检索设置。"""

    top_k: Optional[int] = Field(10, description="返回结果数")
    score_threshold: Optional[float] = Field(
        0.0, description="分数阈值，低于此值的结果被过滤"
    )


class DifyRetrievalRequest(BaseModel):
    """Dify 外部知识库检索请求。"""

    query: str = Field(..., description="查询文本")
    retrieval_setting: RetrievalSetting = Field(default_factory=RetrievalSetting)
    knowledge_id: Optional[str] = Field(None, description="知识库标识（本服务忽略，单知识库）")
    metadata_condition: Optional[dict] = Field(
        None, description="元数据过滤条件（本服务当前忽略）"
    )


class DifyRecord(BaseModel):
    """Dify 外部知识库单条检索结果。"""

    content: str = Field(..., description="片段内容（RAGFlow 喂给 LLM 的文本）")
    title: str = Field("", description="片段标题")
    score: float = Field(0.0, description="相似度分数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class DifyRetrievalResponse(BaseModel):
    """Dify 外部知识库检索响应。"""

    records: list[DifyRecord] = Field(default_factory=list)


def _check_api_key(authorization: str | None) -> None:
    """校验 Dify 外部知识库 API Key。

    RAGFlow 调用时会在 Authorization 头带 Bearer <api_key>。
    若服务端未配置 dify_api_key（本地调试），跳过校验。
    """
    settings = get_settings()
    if not settings.dify_api_key:
        return  # 本地调试模式，不校验

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.dify_api_key:
        raise HTTPException(status_code=401, detail="API Key 无效")


def _build_record(
    result: Any,
    base_url: str,
    score_threshold: float,
    is_hybrid: bool = False,
) -> Optional[DifyRecord]:
    """把 ImageResult 转成 Dify record，低于阈值的返回 None。

    混合检索（is_hybrid=True）的 score 是 RRF 融合分数（量纲与余弦相似度不同），
    不与 score_threshold 比较，与 image_retriever._check_quality 保持一致。
    纯向量检索的 score 是余弦相似度（0~1），正常应用阈值过滤。
    """
    if not is_hybrid and result.score < score_threshold:
        return None

    # content：优先用 caption（PDF 图注或 GLM-4V 著录描述），无则用 tags 拼接
    content = result.caption or ""
    if not content and result.tags:
        content = " ".join(result.tags)

    # title：用 product_id（图片标识），无则用 image_id
    title = result.product_id or result.image_id

    # image_url：拼绝对路径，让 RAGFlow 能访问图片
    # retriever 返回的 image_url 是相对 file_path（如 raw/xxx.jpg），
    # 静态文件服务挂载在 /images 下（见 main.py），完整路径为 /images/raw/xxx.jpg
    image_url = result.image_url
    if image_url and not image_url.startswith("http"):
        # 即使无 base_url 也补 /images 前缀（本地测试时作为相对路径）
        image_url = f"/images/{image_url.lstrip('/')}"
        if base_url:
            image_url = f"{base_url.rstrip('/')}{image_url}"
    # 对 URL 中的非 ASCII 字符（中文文件名/特殊符号）做 percent-encoding
    if image_url:
        image_url = quote(image_url, safe="/:#?=&%")

    metadata: dict[str, Any] = {
        "image_id": result.image_id,
        "image_url": image_url,
        "tags": result.tags,
        "category": result.category,
        "pdf_source": result.pdf_source,
    }
    # 补充文博结构化字段（若有）
    for field in (
        "dynasty",
        "material",
        "category_top",
        "category_sub",
        "caption_standard",
        "caption_public",
    ):
        val = getattr(result, field, None)
        if val:
            metadata[field] = val

    return DifyRecord(
        content=content or "(无文本描述)",
        title=title,
        score=round(result.score, 4),
        metadata=metadata,
    )


@router.post("/retrieval", response_model=DifyRetrievalResponse)
def retrieval(
    request: DifyRetrievalRequest,
    authorization: str | None = Header(None),
) -> DifyRetrievalResponse:
    """Dify / RAGFlow 外部知识库检索入口。

    RAGFlow 调用此接口，传入 query，返回本项目图片 RAG 的检索结果。
    本服务复用 image_retriever.search 的全部检索逻辑（CLIP + 标签 + caption BM25 + RRF）。
    """
    _check_api_key(authorization)

    settings = get_settings()
    top_k = request.retrieval_setting.top_k or settings.image_retrieval_top_k
    score_threshold = request.retrieval_setting.score_threshold or 0.0

    # 调用本项目已有的图片检索（文本搜图模式）
    search_response = search(
        query=request.query,
        image_base64=None,
        category=None,
        tags=None,
        top_k=top_k,
        base_url="",  # image_url 用相对路径，下面再拼绝对路径
    )

    # 混合检索（route 以 _hybrid 结尾）的 score 是 RRF 分数，不应用阈值过滤
    is_hybrid = search_response.route.endswith("_hybrid")

    # 转成 Dify record 格式
    records: list[DifyRecord] = []
    for result in search_response.results:
        record = _build_record(
            result, settings.external_base_url, score_threshold, is_hybrid=is_hybrid
        )
        if record:
            records.append(record)

    # 纯向量检索回退：CLIP 文-图跨模态相似度普遍偏低（0.2~0.4），
    # RAGFlow 默认 score_threshold=0.5 会过滤掉所有结果，导致接口返回空。
    # 若阈值过滤后无结果且原本有召回，回退到不过滤，保证 RAGFlow 拿到 top_k 候选。
    if not is_hybrid and not records and search_response.results:
        for result in search_response.results:
            record = _build_record(
                result, settings.external_base_url, 0.0, is_hybrid=False
            )
            if record:
                records.append(record)

    return DifyRetrievalResponse(records=records)
