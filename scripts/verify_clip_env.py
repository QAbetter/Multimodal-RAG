"""
验证 Chinese-CLIP 环境可用：模型加载 + 图片向量 + 文本向量 + 图文相似度计算。

用法：
    # 使用 HuggingFace 国内镜像加速（推荐）
    set HF_ENDPOINT=https://hf-mirror.com
    .venv\Scripts\python.exe scripts\verify_clip_env.py

    # 或直接运行（首次会从 HuggingFace 下载模型，约 400MB~1GB）
    .venv\Scripts\python.exe scripts\verify_clip_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"


def main() -> None:
    print("=" * 60)
    print("Chinese-CLIP 环境验证")
    print("=" * 60)

    # 1. 打印环境信息
    print(f"\n[1/5] 环境信息")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA可用:  {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  推理设备:  {device}")

    # 2. 加载模型（首次会下载权重）
    print(f"\n[2/5] 加载 Chinese-CLIP 模型: {MODEL_NAME}")
    print("  （首次运行需下载模型权重，请耐心等待...）")
    model = ChineseCLIPModel.from_pretrained(MODEL_NAME)
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device).eval()
    print(f"  模型加载成功，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 3. 文本向量提取
    print(f"\n[3/5] 文本向量提取")
    texts = ["蓝色的汽车", "红色连衣裙", "猫咪", "蓝色的自行车", "蓝色天空"]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # L2 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"  输入文本: {texts}")
    print(f"  向量维度: {text_features.shape[1]}")
    print(f"  向量范数: {text_features.norm(dim=-1).tolist()}")

    # 4. 图片向量提取（用纯色图片模拟）
    print(f"\n[4/5] 图片向量提取")
    # 创建一张红色图片模拟"红色连衣裙"
    red_img = Image.open("data/raw/test.jpg")
    inputs = processor(images=red_img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    print(f"  输入图片: 224x224 纯红色图片（模拟红色物体）")
    print(f"  向量维度: {image_features.shape[1]}")
    print(f"  向量范数: {image_features.norm(dim=-1).item():.4f}")

    # 5. 图文相似度计算
    print(f"\n[5/5] 图文相似度计算")
    similarities = torch.cosine_similarity(image_features, text_features)
    print(f"  红色图片与各文本的相似度:")
    for text, sim in zip(texts, similarities.tolist()):
        bar = "█" * int(sim * 50) if sim > 0 else ""
        print(f"    {text:20s} : {sim:.4f} {bar}")

    print("\n" + "=" * 60)
    print("✓ 环境验证通过！Chinese-CLIP 可正常用于图片 RAG 系统。")
    print("=" * 60)


if __name__ == "__main__":
    main()
