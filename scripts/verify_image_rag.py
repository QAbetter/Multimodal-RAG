"""
图片 RAG 四步走闭环验证脚本。

支持两种模式：
1. 真实图片模式（默认）：自动扫描 data/images/raw/ 下的所有图片进行测试。
   用法：把真实图片放到 data/images/raw/，然后直接运行。
2. 测试图模式（--demo）：用代码生成几张彩色几何图模拟产品图片。
   用法：加 --demo 参数。

验证四步走闭环：
1. 索引：注册 + CLIP 向量化 + 标签提取 + 写入向量库
2. 文本搜图：文本 query → CLIP 文本向量 → 检索相似图片
3. 以图搜图：图片 → CLIP 图像向量 → 检索相似图片
4. 结果校验：含标签、类别、相似度分数

用法：
    # 真实图片模式（推荐）
    $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
    .venv\Scripts\python.exe scripts\verify_image_rag.py

    # 自定义文本搜图 query（默认用第一张图片的文件名）
    .venv\Scripts\python.exe scripts\verify_image_rag.py --query "红色连衣裙"

    # 无真实图片时，用代码生成测试图
    .venv\Scripts\python.exe scripts\verify_image_rag.py --demo

注：标签提取会调用 GLM-4V API（需联网），若失败会自动降级为空标签列表，不中断验证。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from app.core.image_indexer import index_image, register_image
from app.core.image_retriever import search
from app.core.config import get_settings
from app.core.image_vectorstore import reset_image_collection

# 支持的图片格式
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _scan_real_images(raw_dir: Path) -> list[Path]:
    """扫描 data/images/raw/ 目录下的所有图片文件，按文件名排序。"""
    if not raw_dir.exists():
        return []
    return sorted(
        [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS],
        key=lambda p: p.name,
    )


def _make_test_image(color: tuple[int, int, int], shape: str, name: str, raw_dir: Path) -> Path:
    """生成一张测试图片并保存到 raw_dir，返回 Path（--demo 模式用）。"""
    file_path = raw_dir / name
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([50, 50, 250, 250], fill=color)
    elif shape == "square":
        draw.rectangle([50, 50, 250, 250], fill=color)
    else:  # triangle
        draw.polygon([(150, 50), (250, 250), (50, 250)], fill=color)
    img.save(file_path, "JPEG", quality=90)
    return file_path


def _generate_demo_images(raw_dir: Path) -> list[Path]:
    """生成 4 张彩色几何图作为测试图片（--demo 模式用）。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    demos = [
        ("red_circle.jpg", (200, 30, 30), "circle"),
        ("blue_square.jpg", (30, 30, 200), "square"),
        ("green_triangle.jpg", (30, 180, 30), "triangle"),
        ("yellow_circle.jpg", (220, 220, 30), "circle"),
    ]
    return [_make_test_image(color, shape, name, raw_dir) for name, color, shape in demos]


def main() -> None:
    parser = argparse.ArgumentParser(description="图片 RAG 四步走闭环验证")
    parser.add_argument("--demo", action="store_true", help="用代码生成测试图（无真实图片时用）")
    parser.add_argument("--query", type=str, default=None, help="文本搜图的 query（默认用第一张图片文件名）")
    parser.add_argument("--clean", action="store_true", help="运行前清空向量库（删除残留的旧向量）")
    args = parser.parse_args()

    print("=" * 60)
    print("图片 RAG 四步走闭环验证")
    print("=" * 60)

    settings = get_settings()
    raw_dir = Path(settings.image_storage_dir) / "raw"

    # 清理残留向量（删除文件后向量库仍有旧数据）
    if args.clean:
        print("\n[清理] 清空 images collection 中的所有旧向量...")
        reset_image_collection()
        print("  ✓ 已清空，将重新索引当前目录下的图片")

    # 收集要索引的图片
    if args.demo:
        print("\n[模式] --demo：生成测试图")
        image_paths = _generate_demo_images(raw_dir)
    else:
        image_paths = _scan_real_images(raw_dir)
        if not image_paths:
            print(f"\n[!] data/images/raw/ 目录下没有图片")
            print(f"    请把真实图片放到: {raw_dir}")
            print(f"    或用 --demo 生成测试图: python scripts/verify_image_rag.py --demo")
            return
        print(f"\n[模式] 真实图片：扫描到 {len(image_paths)} 张图片")
        for p in image_paths:
            print(f"    - {p.name}")

    # ===== 第 1 步：索引 =====
    print(f"\n[第1步] 索引 {len(image_paths)} 张图片（含 CLIP 向量化 + 标签提取）")
    indexed = []  # (image_id, file_path, tags, category)
    for img_path in image_paths:
        rel_path = f"raw/{img_path.name}"
        product_id = img_path.stem  # 文件名去扩展名作为 product_id
        image = register_image(rel_path, product_id, category=None)
        image = index_image(image.image_id)
        indexed.append((image.image_id, img_path, image.tags, image.category))
        print(f"  ✓ {img_path.name} → image_id={image.image_id[:8]}... "
              f"状态={image.status.value} 标签={image.tags[:3]}")

    if not indexed:
        print("  ✗ 没有成功索引的图片，终止验证")
        return

    # ===== 第 2 步：文本搜图 =====
    print(f"\n[第2步] 文本搜图")
    # 默认用第一张图片的文件名作为 query，可通过 --query 自定义
    default_query = indexed[0][1].stem  # 第一张图片的文件名（去扩展名）
    text_queries = [args.query] if args.query else [default_query]
    for q in text_queries:
        resp = search(query=q, top_k=3)
        if resp.results:
            for i, r in enumerate(resp.results):
                print(f"  query='{q}' → top{i+1}: {r.image_id[:8]}... "
                      f"score={r.score:.4f} 标签={r.tags[:2]}")
        else:
            print(f"  query='{q}' → 无结果")

    # ===== 第 3 步：以图搜图 =====
    print(f"\n[第3步] 以图搜图")
    import base64
    query_img_path = indexed[0][1]  # 用第一张图片作为查询图
    with open(query_img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = search(image_base64=img_b64, top_k=min(4, len(indexed)))
    print(f"  查询图: {query_img_path.name}")
    for i, r in enumerate(resp.results):
        print(f"    top{i+1}: {r.image_id[:8]}... score={r.score:.4f} "
              f"标签={r.tags[:2]}")

    # ===== 第 4 步：结果字段校验 =====
    print(f"\n[第4步] 结果字段校验（标签、类别、相似度分数）")
    # 用以图搜图的第一条结果做字段校验（一定有结果，因为查询图自己也在库里）
    sample = resp.results[0] if resp.results else None
    if sample:
        checks = {
            "含 image_id": bool(sample.image_id),
            "含 product_id": bool(sample.product_id),
            "含 image_url": bool(sample.image_url),
            "含 tags（列表）": isinstance(sample.tags, list),
            "含 score（数值）": isinstance(sample.score, float),
        }
        for label, ok in checks.items():
            print(f"  {'✓' if ok else '✗'} {label}")
        print(f"\n  完整结果示例:")
        print(f"    {sample.model_dump_json(indent=2)}")
        all_pass = all(checks.values())
    else:
        all_pass = False
        print("  ✗ 无结果，无法校验")

    # ===== 总结 =====
    print("\n" + "=" * 60)
    if all_pass:
        print("✓ 四步走闭环验证通过！图片 RAG 核心功能可用。")
    else:
        print("✗ 部分验证未通过，请检查上方输出。")
    print("=" * 60)


if __name__ == "__main__":
    main()
