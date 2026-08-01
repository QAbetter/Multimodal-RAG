"""
Chinese-CLIP 多模态 embedding：同时支持图片和文本编码。

对应书籍 RAG 的 vectorstore.py 中的 get_embeddings()，区别在于：
- 书籍 RAG 用 OpenAIEmbeddings（仅文本，调智谱 embedding-3 API）
- 图片 RAG 用 Chinese-CLIP（本地模型，图文同一向量空间，支持图文互检）

设计要点：
- @lru_cache 单例：CLIP 模型加载昂贵（188M 参数），进程内复用，与 get_llm() / get_embeddings() 一致
- L2 归一化：CLIP 原始向量未归一化，余弦相似度需要单位向量，这里统一归一化后再返回
- 图文同一空间：CLIP 的图像编码器和文本编码器输出在同一向量空间，
  文本向量可直接与图片向量算相似度，无需额外对齐
"""
from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from app.core.config import get_settings


@lru_cache
def get_clip_model():
    """单例加载 Chinese-CLIP 模型（模型加载昂贵，进程内复用）。

    返回 (model, processor, device) 三元组：
    - model：ChineseCLIPModel，含图像编码器和文本编码器
    - processor：ChineseCLIPProcessor，负责图片/文本的预处理（resize、normalize、tokenize）
    - device：推理设备（cpu / cuda）
    """
    settings = get_settings()
    model = ChineseCLIPModel.from_pretrained(settings.clip_model)
    processor = ChineseCLIPProcessor.from_pretrained(settings.clip_model)
    device = torch.device(settings.clip_device)
    model = model.to(device).eval()
    return model, processor, device


def embed_image(img: Image.Image) -> list[float]:
    """提取图片的 CLIP 视觉向量。

    输入 PIL Image，输出 L2 归一化后的 512 维向量。
    """
    model, processor, device = get_clip_model()
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 归一化
    return features[0].cpu().tolist()


def embed_text(text: str) -> list[float]:
    """提取文本的 CLIP 文本向量。

    与图片向量在同一空间，可直接用余弦相似度匹配（图文互检的核心）。
    """
    model, processor, device = get_clip_model()
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 归一化
    return features[0].cpu().tolist()


def embed_images(imgs: list[Image.Image]) -> list[list[float]]:
    """批量提取图片向量（批量索引时用，减少模型调用次数）。

    CLIP 支持批量输入，比循环单张调用快很多。
    """
    model, processor, device = get_clip_model()
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().tolist()
