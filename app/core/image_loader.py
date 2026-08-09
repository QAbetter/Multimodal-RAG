"""
图片加载与预处理：统一尺寸、EXIF 方向修正、缩略图生成。

对应书籍 RAG 的 loader.py，但图片 RAG 不做文本分块，而是：
- 加载原图并修正方向（手机拍摄的图片可能带 EXIF 旋转标记）
- 缩放到 CLIP 模型输入尺寸（默认 224×224）
- 生成缩略图供前端展示（减小带宽，原图保留用于标签提取）
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import get_settings


def load_and_preprocess(file_path: str) -> Image.Image:
    """加载图片并预处理：EXIF 方向修正 + 转 RGB + 缩放到 CLIP 输入尺寸。

    用于 CLIP embedding 提取。CLIP 模型对输入尺寸有严格要求（224×224），
    这里统一缩放，processor 内部还会再做一次 normalize，不需要手动归一化。
    """
    settings = get_settings()
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)  # 修正手机拍摄方向（EXIF orientation）
    if img.mode != "RGB":
        img = img.convert("RGB")
    size = settings.image_thumbnail_size
    img = img.resize((size, size))
    return img


def get_image_size(file_path: str) -> tuple[int, int]:
    """获取原图宽高（EXIF 修正后），用于元数据记录。

    损坏图片返回 (0, 0)，不抛异常，避免批量索引中断。
    """
    try:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        return img.size  # (width, height)
    except Exception as e:
        print(f"[!] 无法读取图片尺寸 {file_path}: {e}")
        return (0, 0)


def generate_thumbnail(file_path: str, thumb_dir: str, image_id: str) -> str:
    """生成 256×256 缩略图，返回相对路径。

    缩略图用于前端展示，原图用于标签提取。两者分开存储避免重复处理。
    """
    thumb_path = Path(thumb_dir) / f"{image_id}.jpg"
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((256, 256))
    img.save(thumb_path, "JPEG", quality=85)
    return f"thumbnails/{image_id}.jpg"


def compute_image_id(file_path: str) -> str:
    """用文件内容 MD5 作为 image_id，相同图片（不同文件名）不会重复索引。"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
