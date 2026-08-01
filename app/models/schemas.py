"""
书籍 RAG 系统的核心数据模型。

对应改造方案中的 BookMetadata / ChunkMetadata 设计：
- BookMetadata：一本书的基础信息，索引时作为公共元数据附加到每个 chunk 上
- ChunkMetadata：单个文本块的元数据，用于检索时的过滤（book_id / chapter / page）
- Book*Request/Response：FastAPI 层的请求/响应模型
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BookFormat(str, Enum):
    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"


class BookStatus(str, Enum):
    PENDING = "pending"       # 已注册，尚未索引
    INDEXING = "indexing"     # 索引中
    READY = "ready"           # 可检索
    FAILED = "failed"         # 索引失败


class BookMetadata(BaseModel):
    """一本书的元数据，索引前需先完成注册。"""

    book_id: str = Field(..., description="书籍唯一标识，建议使用 slug，如 sanguo-yanyi")
    title: str = Field(..., description="书名")
    author: Optional[str] = Field(None, description="作者")
    format: BookFormat = Field(..., description="源文件格式")
    source_path: str = Field(..., description="源文件在 data/raw 下的相对路径")
    total_chapters: Optional[int] = Field(None, description="总章节数，索引完成后回填")
    status: BookStatus = Field(default=BookStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkMetadata(BaseModel):
    """单个文本块的元数据，写入向量库时作为 payload。"""

    book_id: str
    book_title: str
    chapter_index: int = Field(..., description="章节序号，从 0 开始；解析不到章节结构时为 0")
    chapter_title: Optional[str] = Field(None, description="章节标题")
    page: Optional[int] = Field(None, description="页码（仅 PDF 有效）")
    chunk_index: int = Field(..., description="该 chunk 在全书中的序号，用于定位上下文")

    def to_payload(self) -> dict:
        """转换为向量库 payload（扁平字典，便于 metadata 过滤）。

        去掉值为 None 的字段：Chroma 不接受 None 类型的 metadata 值，
        而 chapter_title/page 在 txt/epub 等格式下可能为 None。
        下游检索统一用 dict.get() 读取，缺失字段会回退为 None，行为一致。
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}


class BookRegisterRequest(BaseModel):
    title: str
    author: Optional[str] = None
    format: BookFormat
    source_path: str


class BookResponse(BaseModel):
    book_id: str
    title: str
    author: Optional[str]
    format: BookFormat
    status: BookStatus
    total_chapters: Optional[int]
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    book_id: Optional[str] = Field(
        None, description="指定书籍 id 则为单书精读模式，为空则走跨书问答路由"
    )
    session_id: Optional[str] = Field(
        None, description="多轮对话的会话 id，用于关联 LangGraph 的记忆状态"
    )


class SourceChunk(BaseModel):
    book_id: str
    book_title: str
    chapter_title: Optional[str]
    page: Optional[int]
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    route: str = Field(..., description="single_book | multi_book，标识本次实际走的检索路由")
    answer_quality: str = Field(
        default="ok",
        description="答案质量标记：ok=正常回答；no_answer=原文中无法找到答案",
    )
