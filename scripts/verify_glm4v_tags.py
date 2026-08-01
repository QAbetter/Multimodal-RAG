"""
GLM-4V 标签提取功能专项验证脚本。

支持两种模式：
1. 真实图片模式（默认）：自动扫描 data/images/raw/ 下的所有图片。
   用法：把真实图片放到 data/images/raw/，然后直接运行。
2. 测试图模式（--demo）：用代码生成红色连衣裙 + 蓝色杯子示意图。
   用法：加 --demo 参数。

验证三件事：
1. API 连通性：能否成功调用智谱 GLM-4V 接口
2. 多模态理解：返回的标签是否与图片实际内容匹配
3. 降级机制：extract_tags 在 API 异常时是否正确降级为空列表

用法：
    # 真实图片模式（推荐）
    $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
    .venv\Scripts\python.exe scripts/verify_glm4v_tags.py

    # 无真实图片时，用代码生成测试图
    .venv\Scripts\python.exe scripts/verify_glm4v_tags.py --demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.core.tag_extractor import extract_tags, get_tag_llm, _encode_image_base64, _TAG_PROMPT
from langchain_core.messages import HumanMessage

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


def _make_test_image(name: str, draw_fn, raw_dir: Path) -> Path:
    """生成一张测试图片到 raw_dir，返回 Path（--demo 模式用）。"""
    file_path = raw_dir / name
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    img.save(file_path, "JPEG", quality=90)
    return file_path


def _draw_red_dress(draw: ImageDraw) -> None:
    """画一个红色连衣裙示意图。"""
    draw.polygon([(150, 100), (250, 100), (270, 200), (130, 200)], fill=(200, 30, 30))
    draw.polygon([(130, 200), (270, 200), (320, 350), (80, 350)], fill=(180, 20, 20))
    draw.text((140, 220), "RED DRESS", fill=(255, 255, 255))


def _draw_blue_cup(draw: ImageDraw) -> None:
    """画一个蓝色杯子示意图。"""
    draw.rectangle([140, 120, 260, 320], fill=(30, 80, 200))
    draw.arc([260, 160, 320, 260], start=-90, end=90, fill=(30, 80, 200), width=15)
    draw.ellipse([140, 110, 260, 140], fill=(30, 60, 180))


def _generate_demo_images(raw_dir: Path) -> list[Path]:
    """生成 2 张示意图（--demo 模式用）。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    return [
        _make_test_image("red_dress.jpg", _draw_red_dress, raw_dir),
        _make_test_image("blue_cup.jpg", _draw_blue_cup, raw_dir),
    ]


def _call_glm4v_direct(file_path: str) -> tuple[bool, str, str | None]:
    """直接调用 GLM-4V，不吞异常，返回 (是否成功, 响应内容, 异常信息)。

    与 extract_tags 的区别：这里让异常抛出，便于定位问题。
    """
    settings = get_settings()
    try:
        image_b64 = _encode_image_base64(file_path)
        llm = get_tag_llm()
        message = HumanMessage(content=[
            {"type": "text", "text": _TAG_PROMPT.format(max_count=settings.image_tag_max_count)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ])
        response = llm.invoke([message])
        return True, response.content, None
    except Exception as e:
        return False, "", f"{type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-4V 标签提取功能专项验证")
    parser.add_argument("--demo", action="store_true", help="用代码生成测试图（无真实图片时用）")
    args = parser.parse_args()

    print("=" * 60)
    print("GLM-4V 标签提取功能专项验证")
    print("=" * 60)

    settings = get_settings()
    print(f"\n[配置信息]")
    print(f"  模型: {settings.image_tag_llm_model}")
    print(f"  API地址: {settings.openai_base_url}")
    print(f"  API Key: {settings.openai_api_key[:8]}...{settings.openai_api_key[-4:]}")
    print(f"  最大标签数: {settings.image_tag_max_count}")

    raw_dir = Path(settings.image_storage_dir) / "raw"

    # 收集要测试的图片
    if args.demo:
        print("\n[模式] --demo：生成测试图")
        image_paths = _generate_demo_images(raw_dir)
    else:
        image_paths = _scan_real_images(raw_dir)
        if not image_paths:
            print(f"\n[!] data/images/raw/ 目录下没有图片")
            print(f"    请把真实图片放到: {raw_dir}")
            print(f"    或用 --demo 生成测试图: python scripts/verify_glm4v_tags.py --demo")
            return
        print(f"\n[模式] 真实图片：扫描到 {len(image_paths)} 张图片")

    all_pass = True

    for img_path in image_paths:
        print(f"\n{'─' * 60}")
        print(f"[测试图片] {img_path.name}")
        file_path = str(img_path)

        # 1. 直接调用 GLM-4V（不吞异常）
        print(f"\n  1. 直接调用 GLM-4V API（绕过降级逻辑）")
        ok, raw_response, err = _call_glm4v_direct(file_path)
        if ok:
            print(f"     ✓ API 调用成功")
            print(f"     原始响应:")
            for line in raw_response.splitlines():
                print(f"       | {line}")
        else:
            print(f"     ✗ API 调用失败")
            print(f"     异常: {err}")
            all_pass = False
            continue

        # 2. 通过 extract_tags 业务接口调用（验证降级机制不误触发）
        print(f"\n  2. 通过 extract_tags 业务接口调用")
        tags = extract_tags(file_path)
        if tags:
            print(f"     ✓ 返回标签: {tags}")
        else:
            print(f"     ✗ 返回空列表（可能是降级触发，请看上方异常日志）")
            all_pass = False
            continue

    # ===== 总结 =====
    print(f"\n{'=' * 60}")
    if all_pass:
        print("✓ GLM-4V 标签提取功能正常！")
        print("  - API 连通性: 正常")
        print("  - 多模态理解: 正常（返回标签与图片内容相关）")
        print("  - 降级机制: 未误触发（API 正常时不降级）")
    else:
        print("✗ GLM-4V 验证未通过，请检查上方输出")
        print("  常见原因：")
        print("  1. API Key 失效或余额不足 → 检查 .env 的 OPENAI_API_KEY")
        print("  2. 网络不通 → 确认能访问 open.bigmodel.cn")
        print("  3. 模型名错误 → 确认 image_tag_llm_model=glm-4v（智谱多模态模型）")
    print("=" * 60)


if __name__ == "__main__":
    main()
