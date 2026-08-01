"""
图片 RAG 系统的核心数据模型。

对应书籍 RAG 的 schemas.py，保持风格一致：
- ImageMetadata：单张产品图片的元数据，索引时作为 payload 附加到向量上
- ImageStatus：索引状态机（PENDING → EXTRACTING_TAGS → INDEXING → READY / FAILED）
- Image*Request/Response：FastAPI 层的请求/响应模型

设计要点：
- to_payload() 去掉 None 字段：Chroma 不接受 None 类型的 metadata 值，
  tags 为空列表时保留（列表可作为 metadata 值，下游用 .get() 读取）。
- 标签字段用 list[str]：GLM-4V 提取的标签列表，检索结果原样返回。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ImageStatus(str, Enum):
    """图片索引状态机。"""
    PENDING = "pending"            # 已注册，尚未索引
    EXTRACTING_TAGS = "extracting"  # 标签提取中
    INDEXING = "indexing"          # CLIP 向量化中
    READY = "ready"                # 可检索
    FAILED = "failed"              # 索引失败


class ImageMetadata(BaseModel):
    """单张产品图片的元数据。"""

    image_id: str = Field(..., description="图片唯一标识，建议用文件名 hash")
    product_id: str = Field(..., description="所属产品 id")
    category: Optional[str] = Field(None, description="产品类别，如 clothing/electronics")
    file_path: str = Field(..., description="图片相对路径（image_storage_dir 之下）")
    thumbnail_path: Optional[str] = Field(None, description="缩略图相对路径")
    tags: list[str] = Field(default_factory=list, description="GLM-4V 提取的标签")
    width: Optional[int] = Field(None, description="原图宽度")
    height: Optional[int] = Field(None, description="原图高度")
    status: ImageStatus = Field(default=ImageStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # PDF 插图提取新增字段：图片对应的文本（PDF周边段落/图注），用于基于文本查图
    caption: Optional[str] = Field(None, description="图片对应的文本（PDF提取的周边段落/图注），用于基于文本查图")
    pdf_source: Optional[str] = Field(None, description="图片来源 PDF 文件名（溯源用）")
    page_number: Optional[int] = Field(None, description="图片所在 PDF 页码（若可知）")

    # ===== 文博藏品结构化字段（GLM-4V 一次调用同时产出） =====
    caption_standard: Optional[str] = Field(None, description="标准文物著录描述（客观、形制、纹饰、材质、工艺、完整状态，50字）")
    caption_public: Optional[str] = Field(None, description="大众科普通俗描述（简洁易懂、讲清用途与看点，20字）")
    category_top: Optional[str] = Field(None, description="一级文物分类：陶瓷器/青铜器/玉器/书画/金银器/石刻/漆器/织绣/杂项")
    category_sub: Optional[str] = Field(None, description="二级具体器型名称")
    dynasty: Optional[str] = Field(None, description="年代/朝代/文化")
    material: Optional[str] = Field(None, description="材质/质地")
    color_feature: Optional[str] = Field(None, description="色彩、釉色、沁色特征")
    craft: Optional[str] = Field(None, description="核心工艺技法")
    pattern_theme: Optional[list[str]] = Field(default_factory=list, description="纹饰题材列表")
    function_usage: Optional[str] = Field(None, description="器物原始功用")
    relic_condition: Optional[str] = Field(None, description="完残状态")

    def to_payload(self) -> dict:
        """转换为向量库 payload（扁平字典，便于 metadata 过滤）。

        Chroma metadata 只支持 str/int/float/bool，不接受 list 和 None：
        - tags（list）转为逗号分隔字符串存储，检索时再 split 回 list
        - pattern_theme（list）同样转为逗号分隔字符串
        - datetime 转为 ISO 字符串（model_dump(mode="json") 自动处理）
        - None 字段直接去掉
        - 空字符串也去掉（文博字段未识别时为空字符串，不写入 metadata 节省空间）
        """
        data = self.model_dump(mode="json")
        # tags list → 逗号分隔字符串（Chroma 不支持 list 类型 metadata）
        if data.get("tags"):
            data["tags"] = ",".join(data["tags"])
        else:
            data.pop("tags", None)
        # pattern_theme list → 逗号分隔字符串
        if data.get("pattern_theme"):
            data["pattern_theme"] = ",".join(data["pattern_theme"])
        else:
            data.pop("pattern_theme", None)
        # 去掉 None 和空字符串（文博字段未识别时为空字符串，不入库）
        return {k: v for k, v in data.items() if v not in (None, "")}


class ImageRegisterRequest(BaseModel):
    """注册图片的请求（索引前先注册元数据）。"""

    product_id: str = Field(..., description="所属产品 id")
    category: Optional[str] = Field(None, description="产品类别")
    file_path: str = Field(..., description="图片相对路径（image_storage_dir 之下）")
    tags: Optional[list[str]] = Field(None, description="可选：手动指定标签，跳过 GLM-4V 提取")
    caption: Optional[str] = Field(None, description="可选：图片对应的文本（如PDF提取的周边段落），用于基于文本查图")
    pdf_source: Optional[str] = Field(None, description="可选：图片来源 PDF 文件名")
    page_number: Optional[int] = Field(None, description="可选：图片所在 PDF 页码")


class ImageSearchRequest(BaseModel):
    """图片搜索请求。query 和 image_base64 二选一。"""

    query: Optional[str] = Field(None, description="文本查询，文本搜图模式")
    image_base64: Optional[str] = Field(None, description="图片 base64，以图搜图模式")
    category: Optional[str] = Field(None, description="类别过滤")
    tags: Optional[list[str]] = Field(None, description="标签过滤")
    top_k: Optional[int] = Field(None, description="返回结果数，默认用配置 image_retrieval_top_k")


class ImageResult(BaseModel):
    """单条图片搜索结果。"""

    image_id: str
    product_id: str
    image_url: str = Field(..., description="图片访问 URL（由 API 层拼绝对路径）")
    thumbnail_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    category: Optional[str] = None
    score: float = Field(..., description="相似度分数，0~1")
    caption: Optional[str] = Field(None, description="图片对应的文本（PDF提取的周边段落/图注）")
    pdf_source: Optional[str] = Field(None, description="图片来源 PDF 文件名")


class ImageSearchResponse(BaseModel):
    """图片搜索响应。"""

    results: list[ImageResult] = Field(default_factory=list)
    route: str = Field(..., description="text_to_image | image_to_image | tag_filter")
    total: int
    answer_quality: str = Field(
        default="ok",
        description="ok=正常；low_confidence=最高分低于阈值；no_result=无结果",
    )


class ImageIndexResponse(BaseModel):
    """图片索引（注册+向量化）的响应。"""

    image_id: str
    status: ImageStatus
    tags: list[str] = Field(default_factory=list)
    message: str = ""
