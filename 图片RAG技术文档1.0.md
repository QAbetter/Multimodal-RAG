# 图片 RAG 系统技术文档

> 版本：v1.0
> 编写日期：2026-07-22
> 项目基线：基于现有 `book-rag-exe`（书籍 RAG）项目的工程化设计模式进行改造

---

## 一、项目背景与任务说明

### 1.1 任务来源

老师整理了一个 RAG 项目（即本仓库 `book-rag-exe`），要求同学们在现有代码基础上进行索引调优，完成一个**图片 RAG** 子系统。

### 1.2 任务定义

| 项目 | 说明 |
|------|------|
| **输入** | 图片（以图搜图） 或 文本 query（文本搜图） |
| **输出** | 相关的产品图片 + 该图片的标签等信息（类别、标签、相似度分数、元数据等） |
| **核心能力** | 多模态检索（图文互检）、产品图片管理、标签过滤 |

### 1.3 三周开发计划

| 阶段 | 时间 | 目标 | 交付物 |
|------|------|------|--------|
| 第一周 | 7.20 ~ 7.26 | 安装环境、撰写技术文档 | 技术文档 + 可运行的环境 |
| 第二周 | 7.27 ~ 8.2 | 熟悉现有数据库、写代码接入 RAG 系统 | 图片 RAG 核心代码（索引 + 检索） |
| 第三周 | 8.3 ~ 8.9 | 批量索引、部署 | 可用的图片 RAG 服务 |

> 若对任务和方案已非常清晰，可跳过第一周文档环节直接进入下一阶段。

---

## 二、现有 RAG 项目分析

### 2.1 项目架构概览

现有 `book-rag-exe` 是一个**书籍 RAG 系统**，支持 PDF/EPUB/TXT/DOCX/PPTX/HTML 多格式文档的索引与问答。其分层架构如下：

```
app/
├── main.py                  # FastAPI 入口（lifespan + 路由挂载）
├── api/                     # API 路由层（HTTP 边界）
│   └── chat.py              # /chat 问答接口
├── core/                    # 核心业务逻辑
│   ├── config.py            # 配置收口（pydantic-settings）
│   ├── indexer.py           # 索引主流程（注册→分块→入库）
│   ├── loader.py            # 文档加载 + 章节感知分块
│   ├── vectorstore.py       # Chroma 向量库封装
│   ├── bm25_store.py        # BM25 关键词检索
│   ├── hybrid_retriever.py  # 向量+BM25 混合检索
│   ├── fusion.py            # RRF 多路融合
│   ├── rerank.py            # Flashrank 精排
│   ├── router.py            # 语义路由分发
│   ├── query_rewriter.py    # query 改写
│   └── query_cache.py       # 查询缓存
├── db/
│   └── mysql_store.py       # MySQL 持久化（父子索引）
└── models/
    └── schemas.py           # Pydantic 数据模型
```

### 2.2 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM | GLM-4-flash（智谱） | 通过 OpenAI 兼容协议调用 |
| Embedding | embedding-3（智谱） | 文本向量，1024 维 |
| 向量库 | Chroma | 本地持久化，无需独立服务 |
| 关键词检索 | rank-bm25 + jieba | 进程内 BM25 倒排索引 |
| Rerank | Flashrank | 本地 cross-encoder |
| Web 框架 | FastAPI + Uvicorn | 自动生成 OpenAPI 文档 |
| 多轮对话 | LangGraph + MemorySaver | 状态图 + 内存检查点 |
| 配置管理 | pydantic-settings | 从 .env 读取 |

### 2.3 可复用的设计模式

项目中沉淀了 20 个工程化设计模式，详见 [design_patterns_for_reuse.md](file:///d:/AAAproject/01RAG/book-rag-exe/design_patterns_for_reuse.md)。

**对图片 RAG 最有价值的 10 个模式**：

| 模式 | 出处 | 图片 RAG 借鉴点 |
|------|------|----------------|
| 配置收口 | [config.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/config.py) | CLIP 模型路径、图片目录、top_k 等收口到 Settings |
| 单例模式（lru_cache） | [vectorstore.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/vectorstore.py) | CLIP 模型、向量库连接复用 |
| 生命周期管理 | [main.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/main.py) | 启动时 warm_up 重建内存索引 |
| 租户隔离 | [vectorstore.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/vectorstore.py) | 单 collection + metadata 过滤（按 product_id/category） |
| 错误降级 | [rerank.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/rerank.py) | 标签提取失败降级为纯向量检索 |
| 并发优化 | [retriever.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/retriever.py) | 批量图片 embedding 并发处理 |
| 缓存策略 | [query_cache.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/query_cache.py) | 标签提取结果缓存（LLM 调用昂贵） |
| 多路融合（RRF） | [fusion.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/fusion.py) | CLIP 向量 + 标签文本 + 元数据多路融合 |
| 适配器模式 | [hybrid_retriever.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/hybrid_retriever.py) | 统一多模态 retriever 接口 |
| 状态机 | [indexer.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/indexer.py) | 图片索引状态机（PENDING→EXTRACTING→READY） |

### 2.4 需要改造/新增的部分

| 现有模块 | 图片 RAG 改造方向 |
|---------|------------------|
| `loader.py`（文本分块） | 新增 `image_loader.py`：图片加载、预处理、特征提取 |
| `vectorstore.py`（文本 embedding） | 新增 `image_vectorstore.py`：CLIP 多模态 embedding |
| `bm25_store.py`（文本关键词） | 新增 `tag_store.py`：标签倒排索引（基于产品标签做精确过滤） |
| `rerank.py`（文本 cross-encoder） | 新增 `image_rerank.py`：跨模态 rerank（可选） |
| `schemas.py`（BookMetadata） | 新增 `ImageMetadata` / `ImageSearchRequest` 等模型 |
| `indexer.py`（书籍索引） | 新增 `image_indexer.py`：图片索引主流程 |
| `retriever.py`（文本问答） | 新增 `image_retriever.py`：图片检索主流程 |
| `api/chat.py`（问答接口） | 新增 `api/image.py`：图片搜索接口 |

**建议**：不直接修改原书籍 RAG 代码，而是在同仓库内新增 `app/core/image_*.py` 与 `app/api/image.py`，保持书籍 RAG 与图片 RAG 解耦，便于独立维护与回归。

---

## 三、图片 RAG 系统总体设计

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端 / 前端                              │
│              （文本 query / 图片上传 / 过滤条件）                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                     FastAPI (app/api/image.py)                   │
│   POST /image/search   POST /image/index   GET /image/{id}       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                  语义路由 (router.py 复用扩展)                     │
│     文本搜图 │ 以图搜图 │ 标签过滤 │ 多模态融合                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    多模态检索层 (image_retriever.py)              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ CLIP 向量检索 │  │ 标签文本检索  │  │ 元数据过滤(category) │   │
│  │ (图片+文本)   │  │ (BM25/精确)   │  │                      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         └──────────────────┴─────────────────────┘               │
│                        │ RRF 融合                                │
│                        ▼                                         │
│              ┌──────────────────┐                                │
│              │  跨模态 Rerank    │ (可选，第二期)                  │
│              └──────────────────┘                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                         存储层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Chroma 向量库 │  │ 图片元数据    │  │ 图片文件存储          │   │
│  │ (CLIP 向量)   │  │ (JSON/MySQL) │  │ (本地磁盘/对象存储)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心流程

#### 索引流程（写入）

```
图片文件
   │
   ▼
[1] 图片预处理（缩放、归一化、EXIF 方向修正）
   │
   ▼
[2] CLIP 图像 embedding（提取 512/768 维视觉向量）
   │
   ▼
[3] 标签提取（GLM-4V 多模态 LLM 生成产品标签）──┐
   │                                          │ 失败降级：只用向量，无标签
   ▼                                          │
[4] 标签文本 embedding（CLIP 文本编码器）        │
   │                                          │
   ▼                                          │
[5] 构造 ImageMetadata（product_id/category/tags/...）
   │
   ▼
[6] 写入 Chroma（图像向量 + metadata payload）
   │
   ▼
[7] 写入标签倒排索引（tag_store）+ 元数据注册表
   │
   ▼
[8] 状态机迁移：PENDING → EXTRACTING → READY
```

#### 检索流程（查询）

```
用户输入（文本 query 或 图片）
   │
   ▼
[1] 路由判定（文本搜图 / 以图搜图 / 标签过滤）
   │
   ├─ 文本 query ──→ CLIP 文本 embedding
   ├─ 图片输入  ──→ CLIP 图像 embedding
   └─ 标签过滤  ──→ 直接走 tag_store 精确匹配
   │
   ▼
[2] 多路并发检索
   ├─ 路A：CLIP 向量相似度检索（Chroma）
   ├─ 路B：标签文本检索（如有标签信息）
   └─ 路C：元数据过滤（category/product_id）
   │
   ▼
[3] RRF 融合（排名融合，避免不同模态分数量纲不一致）
   │
   ▼
[4] （可选）跨模态 Rerank 精排
   │
   ▼
[5] 截断 top_k，返回 ImageSearchResponse
       └─ 每条结果含：image_url / tags / category / score / metadata
```

---

## 四、技术选型

### 4.1 多模态 Embedding 模型

| 候选模型 | 维度 | 中文支持 | 说明 |
|---------|------|---------|------|
| **Chinese-CLIP（推荐）** | 512/768 | 优秀 | 针对中文优化，产品图片场景（含中文标签/描述）效果更好 |
| OpenAI CLIP | 512/768 | 一般 | 英文场景成熟，中文较弱 |
| 智谱 embedding-3 | 1024 | 优秀 | 但仅支持文本，不支持图片，不适用于图文互检 |

**选型决策**：采用 **Chinese-CLIP**（`OFA-Sys/chinese-clip-vit-base-patch16`），原因：
1. 图片 RAG 需要**图文互检**（文本搜图 + 以图搜图），必须用支持双模态的模型
2. 产品图片场景多为中文标签/描述，Chinese-CLIP 中文表现优于原版 CLIP
3. 开源可本地部署，无 API 调用成本，适合批量索引
4. 通过 HuggingFace transformers 可直接加载

> 若显存充足（≥8GB），可升级到 `chinese-clip-vit-large-patch14`（768 维，精度更高）。

### 4.2 标签提取 LLM

| 候选 | 说明 |
|------|------|
| **GLM-4V（推荐）** | 智谱多模态模型，支持图片理解，OpenAI 兼容协议，复用现有 API Key |
| GPT-4V | 效果好但成本高 |
| Qwen-VL | 阿里多模态，开源可自部署 |

**选型决策**：采用 **GLM-4V**，原因：
1. 复用现有 `.env` 中的智谱 API Key 与 base_url，零额外配置
2. 通过 OpenAI 兼容协议调用，与现有 `get_llm()` 模式一致
3. 中文产品标签识别效果好

### 4.3 向量库

**复用 Chroma**，原因：
1. 已在项目中验证可用，无需引入新依赖
2. 本地持久化，无需独立服务进程
3. 支持 metadata 过滤（按 category/product_id 过滤）
4. 新建独立 collection（`images`），与书籍 collection（`books`）隔离

### 4.4 图片存储

| 方案 | 适用场景 | 选型 |
|------|---------|------|
| 本地磁盘 + 路径引用 | 小规模（<10万张） | **第一期采用** |
| 对象存储（MinIO/OSS） | 大规模 + 高并发 | 第三期可选升级 |

**选型决策**：第一期采用**本地磁盘存储**，路径写入 metadata，理由：简单、零依赖，第三周部署时若规模增长再升级。

### 4.5 技术栈总览

| 组件 | 选型 | 复用/新增 |
|------|------|----------|
| 多模态 Embedding | Chinese-CLIP | 新增 |
| 标签提取 LLM | GLM-4V（智谱） | 新增（复用 API Key） |
| 向量库 | Chroma | 复用 |
| Web 框架 | FastAPI | 复用 |
| 图片处理 | Pillow | 已在 .venv |
| 深度学习 | PyTorch + transformers | 新增 |
| 配置管理 | pydantic-settings | 复用 |

---

## 五、模块设计

### 5.1 目录结构（建议）

在现有项目结构上新增图片 RAG 相关模块，保持与书籍 RAG 解耦：

```
app/
├── api/
│   ├── chat.py              # 原有书籍问答接口
│   └── image.py             # 【新增】图片搜索接口
├── core/
│   ├── ...                  # 原有书籍 RAG 模块
│   ├── image_config.py      # 【新增】图片 RAG 配置（可合并到 config.py）
│   ├── image_loader.py      # 【新增】图片加载与预处理
│   ├── image_embedder.py    # 【新增】CLIP 多模态 embedding
│   ├── image_vectorstore.py # 【新增】图片向量库封装
│   ├── tag_store.py         # 【新增】标签倒排索引
│   ├── image_indexer.py     # 【新增】图片索引主流程
│   ├── image_retriever.py   # 【新增】图片检索主流程
│   ├── image_fusion.py      # 【新增】多模态 RRF 融合
│   └── tag_extractor.py     # 【新增】GLM-4V 标签提取
└── models/
    └── image_schemas.py     # 【新增】图片相关数据模型

data/
├── images/                  # 【新增】图片文件存储目录
│   ├── raw/                 # 原始图片
│   └── thumbnails/          # 缩略图
├── chroma/                  # 向量库持久化（已有）
└── processed/
    └── images.json          # 【新增】图片元数据注册表
```

### 5.2 配置层设计

在 [config.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/config.py) 的 `Settings` 类中新增图片 RAG 相关配置：

```python
# 图片 RAG 配置（新增到 Settings 类）
# 多模态 embedding
clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"
clip_device: str = "cpu"  # 或 "cuda"，有 GPU 时改为 cuda
image_embedding_dim: int = 512

# 图片存储
image_storage_dir: str = "data/images"
image_thumbnail_size: tuple = (224, 224)  # CLIP 输入尺寸

# 标签提取
image_tag_llm_model: str = "glm-4v"  # 智谱多模态模型
image_tag_max_count: int = 5  # 每张图最多提取 5 个标签

# 图片向量库（独立 collection，与书籍隔离）
chroma_image_collection: str = "images"

# 图片检索参数
image_retrieval_top_k: int = 10
image_score_threshold: float = 0.2  # 相似度低于此值视为低相关
```

### 5.3 数据模型设计

在 `app/models/image_schemas.py` 中定义：

```python
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ImageStatus(str, Enum):
    PENDING = "pending"           # 已注册，尚未索引
    EXTRACTING_TAGS = "extracting"  # 标签提取中
    INDEXING = "indexing"         # 向量化中
    READY = "ready"               # 可检索
    FAILED = "failed"             # 索引失败


class ImageMetadata(BaseModel):
    """单张产品图片的元数据。"""
    image_id: str = Field(..., description="图片唯一标识，建议用文件名 hash")
    product_id: str = Field(..., description="所属产品 id")
    category: Optional[str] = Field(None, description="产品类别，如 clothing/electronics")
    file_path: str = Field(..., description="图片在 data/images/raw 下的相对路径")
    thumbnail_path: Optional[str] = Field(None, description="缩略图路径")
    tags: list[str] = Field(default_factory=list, description="GLM-4V 提取的标签")
    width: Optional[int] = None
    height: Optional[int] = None
    status: ImageStatus = ImageStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_payload(self) -> dict:
        """转换为向量库 payload，去掉 None 字段（Chroma 不接受 None）。"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ImageSearchRequest(BaseModel):
    """图片搜索请求。"""
    query: Optional[str] = Field(None, description="文本查询，与 image 二选一")
    image_base64: Optional[str] = Field(None, description="图片 base64，用于以图搜图")
    category: Optional[str] = Field(None, description="类别过滤")
    tags: Optional[list[str]] = Field(None, description="标签过滤")
    top_k: Optional[int] = Field(None, description="返回结果数，默认用配置")


class ImageResult(BaseModel):
    """单条图片搜索结果。"""
    image_id: str
    product_id: str
    image_url: str
    thumbnail_url: Optional[str]
    tags: list[str]
    category: Optional[str]
    score: float = Field(..., description="相似度分数，0~1")


class ImageSearchResponse(BaseModel):
    """图片搜索响应。"""
    results: list[ImageResult]
    route: str = Field(..., description="text_to_image | image_to_image | tag_filter")
    total: int
    answer_quality: str = Field(default="ok", description="ok | low_confidence | no_result")
```

### 5.4 核心模块代码骨架

#### 5.4.1 图片加载与预处理（`image_loader.py`）

```python
"""图片加载与预处理：统一尺寸、EXIF 方向修正、缩略图生成。"""
from pathlib import Path
from PIL import Image, ImageOps
from app.core.config import get_settings


def load_and_preprocess(file_path: str) -> Image.Image:
    """加载图片并预处理：EXIF 方向修正 + 转 RGB + 缩放到 CLIP 输入尺寸。"""
    settings = get_settings()
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)  # 修正手机拍摄方向
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(settings.image_thumbnail_size)
    return img


def generate_thumbnail(file_path: str, thumb_dir: str) -> str:
    """生成缩略图，返回缩略图相对路径。"""
    # 生成 256x256 缩略图用于前端展示，减小带宽
    ...
```

#### 5.4.2 CLIP 多模态 Embedding（`image_embedder.py`）

```python
"""Chinese-CLIP 多模态 embedding：同时支持图片和文本编码。"""
from functools import lru_cache
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from app.core.config import get_settings


@lru_cache
def get_clip_model():
    """单例加载 CLIP 模型（模型加载昂贵，进程内复用）。"""
    settings = get_settings()
    model = ChineseCLIPModel.from_pretrained(settings.clip_model)
    processor = ChineseCLIPProcessor.from_pretrained(settings.clip_model)
    device = torch.device(settings.clip_device)
    model = model.to(device).eval()
    return model, processor, device


def embed_image(img: Image.Image) -> list[float]:
    """提取图片的 CLIP 视觉向量。"""
    model, processor, device = get_clip_model()
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 归一化
    return features[0].cpu().tolist()


def embed_text(text: str) -> list[float]:
    """提取文本的 CLIP 文本向量（与图片向量同一空间，可直接相似度匹配）。"""
    model, processor, device = get_clip_model()
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()
```

#### 5.4.3 标签提取（`tag_extractor.py`）

```python
"""用 GLM-4V 多模态 LLM 提取产品图片标签，失败时降级为空标签列表。"""
import logging
import base64
from langchain_core.messages import HumanMessage
from app.core.retriever import get_llm  # 复用现有 LLM 单例模式
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_TAG_PROMPT = """请分析这张产品图片，提取最多 {max_count} 个核心标签。
要求：
1. 每行一个标签，不要编号
2. 标签应描述产品类别、颜色、材质、风格等关键属性
3. 只输出标签，不要解释"""


def extract_tags(image_path: str) -> list[str]:
    """调用 GLM-4V 提取图片标签，失败时降级为空列表（不影响索引主流程）。"""
    settings = get_settings()
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()

        llm = get_llm()  # 复用现有 LLM 单例
        message = HumanMessage(content=[
            {"type": "text", "text": _TAG_PROMPT.format(max_count=settings.image_tag_max_count)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ])
        response = llm.invoke([message])
        tags = [line.strip() for line in response.content.splitlines() if line.strip()]
        return tags[:settings.image_tag_max_count]
    except Exception:
        logger.exception("标签提取失败，降级为空标签列表: %s", image_path)
        return []
```

#### 5.4.4 图片索引主流程（`image_indexer.py`）

复用 [indexer.py](file:///d:/AAAproject/01RAG/book-rag-exe/app/core/indexer.py) 的状态机模式：

```python
"""图片索引主流程：注册 → 预处理 → CLIP 向量化 → 标签提取 → 写入向量库。"""
from app.core.image_loader import load_and_preprocess, generate_thumbnail
from app.core.image_embedder import embed_image
from app.core.tag_extractor import extract_tags
from app.core.image_vectorstore import get_image_vectorstore, delete_image_vectors
from app.models.image_schemas import ImageMetadata, ImageStatus


def index_image(image_id: str) -> ImageMetadata:
    """对已注册的图片执行索引，状态机迁移：PENDING → EXTRACTING → READY/FAILED。"""
    images = load_registered_images()
    image = images[image_id]

    image.status = ImageStatus.EXTRACTING_TAGS
    save_registered_images(images)

    try:
        # 1. 预处理 + 生成缩略图
        img = load_and_preprocess(image.file_path)
        image.thumbnail_path = generate_thumbnail(image.file_path, ...)

        # 2. CLIP 图像 embedding
        image.status = ImageStatus.INDEXING
        save_registered_images(images)
        vector = embed_image(img)

        # 3. 标签提取（失败降级，不中断主流程）
        image.tags = extract_tags(image.file_path)

        # 4. 写入向量库
        delete_image_vectors(image_id)  # 重新索引前清理旧向量
        vectorstore = get_image_vectorstore()
        vectorstore.add_embeddings([{
            "embedding": vector,
            "id": image_id,
            "metadata": image.to_payload(),
        }])

        image.status = ImageStatus.READY
    except Exception:
        image.status = ImageStatus.FAILED
        raise
    finally:
        images[image_id] = image
        save_registered_images(images)

    return image
```

#### 5.4.5 图片检索主流程（`image_retriever.py`）

```python
"""图片检索主流程：支持文本搜图、以图搜图、标签过滤，多路 RRF 融合。"""
from app.core.image_embedder import embed_text, embed_image
from app.core.image_vectorstore import get_image_vectorstore
from app.core.fusion import _RRF_K  # 复用现有 RRF 常量


def search_by_text(query: str, category: str | None = None, top_k: int = 10) -> list:
    """文本搜图：CLIP 文本向量 → Chroma 相似度检索。"""
    query_vector = embed_text(query)
    where_filter = {"category": category} if category else None
    vectorstore = get_image_vectorstore()
    results = vectorstore.similarity_search_by_vector(
        embedding=query_vector, k=top_k, filter=where_filter
    )
    return _format_results(results)


def search_by_image(image_base64: str, top_k: int = 10) -> list:
    """以图搜图：CLIP 图像向量 → Chroma 相似度检索。"""
    # base64 解码 → PIL Image → embed_image → similarity_search_by_vector
    ...


def search_with_fusion(query: str, tags: list[str], top_k: int = 10) -> list:
    """多路融合：CLIP 向量检索 + 标签精确检索，RRF 融合。"""
    # 路 A：CLIP 向量检索
    # 路 B：标签倒排检索（tag_store）
    # RRF 融合后截断 top_k
    ...
```

#### 5.4.6 API 接口（`api/image.py`）

```python
"""图片搜索 API 路由。"""
from fastapi import APIRouter, HTTPException
from app.models.image_schemas import ImageSearchRequest, ImageSearchResponse

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/search", response_model=ImageSearchResponse)
def search(request: ImageSearchRequest) -> ImageSearchResponse:
    """图片搜索：文本搜图 / 以图搜图 / 标签过滤。"""
    if not request.query and not request.image_base64:
        raise HTTPException(400, "query 和 image_base64 至少传一个")

    if request.image_base64:
        results = search_by_image(request.image_base64, request.top_k)
        route = "image_to_image"
    elif request.tags:
        results = search_with_fusion(request.query, request.tags, request.top_k)
        route = "tag_filter"
    else:
        results = search_by_text(request.query, request.category, request.top_k)
        route = "text_to_image"

    return ImageSearchResponse(
        results=results, route=route, total=len(results),
        answer_quality="no_result" if not results else "ok",
    )
```

---

## 六、第一周交付清单（7.20 ~ 7.26）

### 6.1 环境配置

#### 6.1.1 系统要求

- Python 3.10+（项目已用 3.10）
- pip 包管理器
- 建议 8GB+ 内存（CLIP 模型推理）
- 可选：NVIDIA GPU + CUDA（加速 CLIP 推理，CPU 也可运行）

#### 6.1.2 安装步骤

```bash
# 1. 进入项目目录
cd d:\AAAproject\01RAG\book-rag-exe

# 2. 激活现有虚拟环境（已存在 .venv）
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. 安装原有依赖
pip install -r requirements.txt

# 4. 安装图片 RAG 新增依赖
pip install torch torchvision transformers  # CLIP 模型
# Pillow 已在 .venv 中，无需重装

# 5. 验证 CLIP 模型可加载（首次会自动下载模型权重，约 1GB）
python -c "from transformers import ChineseCLIPModel; print('CLIP 加载成功')"
```

#### 6.1.3 配置文件

在 `.env` 中新增图片 RAG 配置（原有书籍 RAG 配置保留）：

```env
# === 图片 RAG 配置（新增） ===
# 多模态 Embedding
CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16
CLIP_DEVICE=cpu
IMAGE_EMBEDDING_DIM=512

# 图片存储
IMAGE_STORAGE_DIR=data/images
IMAGE_THUMBNAIL_SIZE=224,224

# 标签提取 LLM
IMAGE_TAG_LLM_MODEL=glm-4v
IMAGE_TAG_MAX_COUNT=5

# 图片向量库（独立 collection）
CHROMA_IMAGE_COLLECTION=images

# 图片检索参数
IMAGE_RETRIEVAL_TOP_K=10
IMAGE_SCORE_THRESHOLD=0.2
```

### 6.2 技术文档

即本文档（`图片RAG技术文档.md`）。

### 6.3 环境验证脚本

第一周末交付一个最小验证脚本 `scripts/verify_clip_env.py`，验证：
1. Chinese-CLIP 模型可正常加载
2. 能对一张测试图片提取向量
3. 能对一段文本提取向量
4. 图文向量可计算相似度

```python
# scripts/verify_clip_env.py 骨架
"""验证 CLIP 环境可用：图片向量 + 文本向量 + 相似度计算。"""
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

def main():
    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    # 1. 文本向量
    inputs = processor(text=["一张红色连衣裙", "蓝色牛仔裤"], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # 2. 图片向量（用一张测试图）
    img = Image.new("RGB", (224, 224), "red")  # 占位，实际用真实图片
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # 3. 相似度
    sim = torch.cosine_similarity(image_features, text_features[0:1])
    print(f"图文相似度: {sim.item():.4f}")
    print("环境验证通过！")

if __name__ == "__main__":
    main()
```

---

## 七、第二周开发计划（7.27 ~ 8.2）

### 7.1 目标

熟悉现有数据库结构，写代码接入 RAG 系统，跑通"单张图片索引 → 单次检索"的最小闭环。

### 7.2 任务分解

| 序号 | 任务 | 对应模块 | 验收标准 |
|------|------|---------|---------|
| 1 | 扩展配置层 | `config.py` 新增图片配置 | Settings 可读取图片相关参数 |
| 2 | 定义数据模型 | `image_schemas.py` | ImageMetadata/ImageSearchRequest 等定义完成 |
| 3 | 实现图片加载 | `image_loader.py` | 能加载 + 预处理一张图片 |
| 4 | 实现 CLIP embedding | `image_embedder.py` | 图片/文本都能出向量 |
| 5 | 实现标签提取 | `tag_extractor.py` | 调用 GLM-4V 提取标签，失败可降级 |
| 6 | 实现图片向量库 | `image_vectorstore.py` | Chroma 新建 images collection |
| 7 | 实现索引主流程 | `image_indexer.py` | 单张图片能完成索引（状态机迁移） |
| 8 | 实现检索主流程 | `image_retriever.py` | 文本搜图 + 以图搜图跑通 |
| 9 | 实现 API 接口 | `api/image.py` | /image/search 接口可调用 |
| 10 | 编写验证脚本 | `scripts/verify_image_rag.py` | 单图索引 + 检索闭环跑通 |

### 7.3 验收标准

- 能通过 API 上传/索引一张产品图片
- 能用文本 query 搜到该图片
- 能用图片搜到相似图片
- 检索结果含标签、类别、相似度分数

---

## 八、第三周开发计划（8.3 ~ 8.9）

### 8.1 目标

批量索引产品图片集，部署为可用的图片 RAG 服务。

### 8.2 任务分解

| 序号 | 任务 | 说明 |
|------|------|------|
| 1 | 批量索引脚本 | `scripts/batch_index_images.py`，支持从目录批量索引 |
| 2 | 并发优化 | 用 ThreadPoolExecutor 并发提取向量（复用现有并发模式） |
| 3 | 标签倒排索引 | `tag_store.py`，支持标签精确过滤 |
| 4 | 多路融合检索 | CLIP 向量 + 标签文本 RRF 融合 |
| 5 | 缓存层 | 复用 query_cache 模式，缓存检索结果 |
| 6 | 服务部署 | Uvicorn 启动 FastAPI 服务，提供 HTTP 接口 |
| 7 | 评测脚本 | `scripts/run_image_eval.py`，计算 Recall@K / MRR |
| 8 | 接口文档 | FastAPI 自动生成 OpenAPI，前端可对接 |

### 8.3 部署方案

```bash
# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 接口示例
# 文本搜图
curl -X POST http://localhost:8000/image/search \
  -H "Content-Type: application/json" \
  -d '{"query": "红色连衣裙", "top_k": 10}'

# 以图搜图（base64）
curl -X POST http://localhost:8000/image/search \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64>", "top_k": 10}'
```

### 8.4 验收标准

- 批量索引 ≥100 张产品图片
- 文本搜图 Recall@10 ≥ 0.7（标注评测集）
- 以图搜图 Recall@10 ≥ 0.8
- 服务可通过 HTTP 接口稳定访问
- 接口响应时间 P95 ≤ 500ms（单次检索）

---

## 九、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| CLIP 模型下载慢/失败 | 阻塞环境搭建 | 使用 modelscope 镜像源，或提前下载权重到本地 |
| GPU 不可用，CPU 推理慢 | 批量索引耗时 | 第一期用 CPU + 小模型（base），第三期视情况升级；批量索引可离线过夜跑 |
| GLM-4V 标签提取限流 | 批量索引受阻 | 加重试 + 限流控制；标签提取失败降级为空标签，不阻塞索引 |
| 产品图片数据未明确 | 第二周无法开工 | 第一周末需向老师确认数据集来源、格式、规模 |
| Chroma 规模上限 | 第三期数据量大时性能下降 | 10 万级以内 Chroma 足够；超 50 万考虑迁移 Qdrant/Milvus |
| 中英文混合标签 | CLIP 匹配精度下降 | 标签归一化（同义词合并）；Chinese-CLIP 已对中文优化 |

---

## 十、参考资料

- 现有项目设计模式文档：[design_patterns_for_reuse.md](file:///d:/AAAproject/01RAG/book-rag-exe/design_patterns_for_reuse.md)
- Chinese-CLIP 模型：https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16
- CLIP 论文：*Learning Transferable Visual Models From Natural Language Supervision*
- RRF 融合算法：*Reciprocal Rank Fusion* 论文
- 智谱 GLM-4V 文档：https://open.bigmodel.cn/
- LangChain Chroma 集成：https://python.langchain.com/docs/integrations/vectorstores/chroma

---

## 附录 A：与现有书籍 RAG 的关系

图片 RAG 与书籍 RAG **共享基础设施**（配置、向量库、Web 框架、设计模式），但**业务逻辑独立**：

```
共享层：config.py / main.py / FastAPI / Chroma 持久化目录
        ├─ 书籍 RAG：api/chat.py + core/[indexer|retriever|...].py + collection=books
        └─ 图片 RAG：api/image.py + core/image_*.py + collection=images
```

两个 RAG 子系统通过**独立 collection** 隔离数据，通过**独立 API 路由**对外提供服务，互不影响。原有书籍 RAG 功能保持不变。

## 附录 B：关键概念速查（RAG 新手向）

| 概念 | 通俗解释 |
|------|---------|
| **RAG**（Retrieval-Augmented Generation） | 检索增强生成：先从知识库检索相关内容，再交给 LLM 生成答案 |
| **Embedding** | 把文本/图片转成向量（一串数字），相似内容向量也相似 |
| **CLIP** | OpenAI 提出的多模态模型，能把图片和文本映射到同一向量空间，使图文可以直接算相似度 |
| **向量库** | 专门存储和检索向量的数据库，能快速找到"最相似的 K 个向量" |
| **RRF 融合** | 多路检索结果的排名融合算法，不依赖各路分数绝对值，更稳健 |
| **Rerank** | 对初步检索的结果做二次精排，提升精度 |
| **top_k** | 检索时返回的最相似结果数量 |
| **Recall@K** | 前 K 个结果中包含正确答案的比例，衡量召回质量 |
