"""
应用全局配置，基于 pydantic-settings 从环境变量 / .env 读取。

对应改造方案：所有可变参数（模型名、向量库地址、chunk 大小等）统一收口到这里，
禁止在业务代码里写死 magic number。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Embedding（暂沿用 OpenAI 协议，embedding 来源待确认后再切换）
    openai_api_key: str = ""
    openai_base_url: str | None = None
    embedding_model: str = "embedding-3"

    # LLM
    llm_model: str = "glm-4-flash"  #

    # 向量库（Chroma，本地持久化，无需额外起服务）
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "books"

    # 索引参数
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 检索参数
    retrieval_top_k: int = 6
    multi_query_count: int = 3

    # 第五步：RAG-Fusion（多路 query 融合）+ Rerank（跨书问答场景增强）
    fusion_retrieval_k: int = 10  # 每路 query 的初始检索条数，融合后再由 rerank 截断到 retrieval_top_k
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"  # flashrank 内置轻量 cross-encoder 模型

    # 父子索引（第四步：子块向量存 Chroma，父块存 MySQL）
    mysql_host: str = ""
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = ""
    chroma_child_collection: str = "book_child_chunks"
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # 数据目录
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    # PDF 原始文件专用目录（待解析的 PDF 存放处，类比图片的 image_storage_dir）
    # 脚本批量扫描和 API 上传的 PDF 都存到这里
    pdf_raw_dir: str = "data/pdf"

    # 工程性优化：查询结果缓存（同一 route+book_id+query 命中缓存则跳过检索与 LLM 调用）
    query_cache_enabled: bool = True
    query_cache_max_size: int = 256
    query_cache_ttl_seconds: int = 3600

    # 工程性优化：评测集目录
    eval_dataset_dir: str = "data/eval"

    # 预检索优化：查询意图分类 top_k 调整
    # 事实型问题（精度优先）用较小 top_k；归纳推理型问题（召回优先）用较大 top_k
    intent_fact_top_k: int = 4
    intent_reasoning_top_k: int = 10

    # ===== 图片 RAG 配置（与书籍 RAG 共享基础设施，业务逻辑独立） =====
    # 多模态 Embedding（Chinese-CLIP，图文同一向量空间，支持图文互检）
    clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    clip_device: str = "cpu"  # 有 GPU 改为 cuda
    image_embedding_dim: int = 512

    # 图片存储
    image_storage_dir: str = "data/images"
    image_thumbnail_size: int = 224  # CLIP 模型输入尺寸（正方形边长，模型固定）

    # 标签提取 LLM（复用智谱 API Key，用多模态模型 glm-4v 识别图片标签）
    image_tag_llm_model: str = "glm-4v"
    image_tag_max_count: int = 5

    # 图片向量库（独立 collection，与书籍 books 隔离）
    chroma_image_collection: str = "images"

    # 图片检索参数
    image_retrieval_top_k: int = 10
    image_score_threshold: float = 0.2  # 相似度低于此值视为低相关，结果标记 low_confidence

    # ===== PDF 插图提取配置（MinerU 精准解析 API） =====
    # MinerU API 基础地址
    mineru_api_base: str = "https://mineru.net/api/v4"
    # MinerU API Token（在 API 管理页面自定创建，与智谱 API Key 独立）
    mineru_token: str = ""
    # 解析模型版本：pipeline（默认）/ vlm（推荐，精度更高）
    mineru_model_version: str = "vlm"
    # 是否启用 OCR（扫描件需要）
    mineru_is_ocr: bool = False
    # 轮询间隔（秒）
    mineru_poll_interval: int = 5
    # 轮询总超时（秒）
    mineru_poll_timeout: int = 600
    # PDF 提取的图片存储子目录（image_storage_dir 之下）
    pdf_image_subdir: str = "raw/pdf"
    # 提取的 caption 最大字符数（截断过长的周边文本，避免 metadata 膨胀）
    image_caption_max_chars: int = 500
    # 解析结果 ZIP 下载超时（秒）
    mineru_download_timeout: int = 180
    # PDF 切分阈值（MinerU 限制单文件≤200MB/200页，留余量）
    # 超过任一阈值则按页切分后分批解析，最后合并结果
    pdf_split_size_mb: int = 150  # 文件体积超过此值则切分
    pdf_split_page_threshold: int = 150  # 页数超过此值则切分
    pdf_split_chunk_pages: int = 100  # 每个子PDF包含的页数（切分粒度）

    # ===== RAGFlow / Dify 外部知识库对接 =====
    # 外部知识库回调本服务时的基础 URL（RAGFlow 拿到 image_url 后需要能访问图片）
    # 本地开发留空则用相对路径；服务器部署填 http://<服务器IP>:<端口>
    external_base_url: str = ""
    # 外部知识库 API Key（RAGFlow 调用 /api/v1/dify/retrieval 时带在 Authorization 头里）
    # 留空则不校验（仅本地调试用，生产环境务必设置）
    dify_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
