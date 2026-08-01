"""
PDF 提取端到端验证脚本：从MinerU API连通性到完整提取+索引流程。

分层验证（从简单到完整，可单独运行某一层）：
1. --check-api       ：MinerU API连通性测试（上传1页测试PDF，验证返回ZIP）
2. --check-extract   ：完整提取流程（指定PDF，验证图片+caption提取效果）
3. --check-index     ：提取+索引全流程（指定PDF，验证可基于文本查图）
4. --check-all       ：全部执行（需指定 --pdf）

用法：
    # 第1层：API连通性（无需PDF，自动生成1页测试PDF）
    .venv\\Scripts\\python.exe scripts\\verify_pdf_extract.py --check-api

    # 第2层：提取效果（需指定PDF文件）
    .venv\\Scripts\\python.exe scripts\\verify_pdf_extract.py --check-extract --pdf "xxx.pdf"

    # 第3层：完整索引流程（需指定PDF文件）
    .venv\\Scripts\\python.exe scripts\\verify_pdf_extract.py --check-index --pdf "xxx.pdf"

    # 全部执行
    .venv\\Scripts\\python.exe scripts\\verify_pdf_extract.py --check-all --pdf "xxx.pdf"

前提：
    .env 已配置 MINERU_TOKEN（MinerU API Token，在 API 管理页面创建）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.pdf_image_extractor import (
    ExtractedImage,
    cleanup_extract_temp,
    extract_images_from_pdf,
    _call_mineru_parser,
    _download_and_extract_zip,
)


def _make_test_pdf(pdf_path: Path) -> None:
    """生成1页测试PDF（用于API连通性测试，无需用户准备PDF）。

    用 reportlab 生成含一张图片和一段文字的简单PDF。
    若 reportlab 未安装，用 pypdf 生成空白页PDF。
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        # 画一个矩形作为"图片"占位
        c.rect(100, height - 400, 200, 150, fill=0)
        c.drawString(120, height - 430, "测试图注文字")
        c.drawString(100, 100, "测试PDF正文内容")
        c.showPage()
        c.save()
        print(f"  [✓] 生成测试PDF（reportlab）: {pdf_path}")
    except ImportError:
        # reportlab 未安装，用 pypdf 生成空白页
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)  # A4
        with pdf_path.open("wb") as f:
            writer.write(f)
        print(f"  [✓] 生成测试PDF（pypdf空白页）: {pdf_path}")


def check_api() -> bool:
    """第1层：MinerU API连通性测试。

    生成1页测试PDF，上传到MinerU，验证返回ZIP下载链接并能下载解压。
    不做图文匹配，仅验证API链路畅通。
    """
    print("\n" + "=" * 60)
    print("[第1层] MinerU API连通性测试")
    print("=" * 60)

    settings = get_settings()
    if not settings.mineru_token:
        print("  [✗] 未配置 MINERU_TOKEN，请在 .env 中设置 MinerU Token")
        return False

    # 生成测试PDF
    test_pdf = Path(settings.image_storage_dir) / ".pdf_test" / "api_test.pdf"
    test_pdf.parent.mkdir(parents=True, exist_ok=True)
    _make_test_pdf(test_pdf)

    # 调用API
    print("  [2/4] 调用MinerU解析API（上传+轮询，请耐心等待）...")
    start = time.time()
    try:
        zip_url = _call_mineru_parser(str(test_pdf))
    except Exception as e:
        print(f"  [✗] API调用失败: {e}")
        return False
    elapsed = time.time() - start
    print(f"  [✓] API响应成功（耗时 {elapsed:.1f}s）")
    print(f"      下载链接: {zip_url[:80]}...")

    # 下载ZIP
    print("  [3/4] 下载并解压ZIP...")
    extract_dir = test_pdf.parent / "extracted"
    try:
        _download_and_extract_zip(zip_url, extract_dir)
    except Exception as e:
        print(f"  [✗] ZIP下载失败: {e}")
        return False

    # 检查ZIP内容
    print("  [4/4] 检查ZIP内容...")
    md_files = list(extract_dir.rglob("*.md"))
    json_files = list(extract_dir.rglob("*.json"))
    img_files = [p for p in extract_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    print(f"      Markdown文件: {len(md_files)} 个")
    print(f"      JSON布局文件: {len(json_files)} 个")
    print(f"      图片文件: {len(img_files)} 个")

    if md_files:
        print(f"      Markdown预览: {md_files[0].read_text(encoding='utf-8')[:100]}...")

    # 清理
    import shutil
    shutil.rmtree(test_pdf.parent, ignore_errors=True)

    print("\n  [结论] API连通性: " + ("✓ 正常" if zip_url else "✗ 异常"))
    return True


def check_extract(pdf_path: str) -> bool:
    """第2层：完整提取流程测试。

    指定PDF文件，调用完整提取流程，输出：
    - 提取了多少张图片
    - 每张图片的caption（验证图文对应是否正确）
    - 图片存储路径
    """
    print("\n" + "=" * 60)
    print("[第2层] PDF提取效果测试")
    print("=" * 60)

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"  [✗] PDF文件不存在: {pdf}")
        return False

    print(f"  PDF文件: {pdf.name}")
    print(f"  文件大小: {pdf.stat().st_size / 1024 / 1024:.2f} MB")

    # 调用完整提取流程
    print("  [1/2] 调用MinerU API提取图片+caption...")
    start = time.time()
    try:
        results: list[ExtractedImage] = extract_images_from_pdf(str(pdf))
    except Exception as e:
        print(f"  [✗] 提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    elapsed = time.time() - start

    print(f"  [✓] 提取完成（耗时 {elapsed:.1f}s）")
    print(f"      共提取 {len(results)} 张图片")

    if not results:
        print("  [!] 未提取到图片，请检查PDF是否含插图")
        return True  # 不算失败

    # 展示提取结果
    print("\n  [2/2] 提取结果详情:")
    print("-" * 60)
    with_caption = 0
    for i, img in enumerate(results, 1):
        cap_status = "有caption" if img.caption else "无caption"
        if img.caption:
            with_caption += 1
        page_info = f"页{img.page_number}" if img.page_number is not None else "页码未知"
        print(f"  [{i:3d}] {img.image_name}")
        print(f"        路径: {img.file_path}")
        print(f"        位置: {page_info} | {cap_status}")
        if img.caption:
            # 截断显示过长的caption
            display_cap = img.caption[:100] + "..." if len(img.caption) > 100 else img.caption
            print(f"        caption: {display_cap}")
        print()

    print(f"  [结论] 提取效果: {len(results)} 张图片，{with_caption} 张有caption")
    print(f"         caption覆盖率: {with_caption / len(results) * 100:.0f}%")

    # 清理临时文件
    cleanup_extract_temp(pdf.stem)
    return True


def check_index(pdf_path: str, query: str | None = None) -> bool:
    """第3层：提取+索引+检索全流程测试。

    指定PDF文件，完成提取+索引后，用query验证能否基于文本查到图片。
    """
    print("\n" + "=" * 60)
    print("[第3层] 提取+索引+检索全流程测试")
    print("=" * 60)

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"  [✗] PDF文件不存在: {pdf}")
        return False

    # 1. 提取
    print("  [1/4] 提取图片+caption...")
    try:
        results = extract_images_from_pdf(str(pdf))
    except Exception as e:
        print(f"  [✗] 提取失败: {e}")
        return False
    print(f"  [✓] 提取 {len(results)} 张图片")

    if not results:
        print("  [!] 未提取到图片，跳过索引")
        return True

    # 2. 注册+索引
    print("  [2/4] 注册并索引图片（CLIP向量化+标签+caption BM25）...")
    from app.core.image_indexer import batch_index_images, register_image

    image_ids: list[str] = []
    name_map: dict[str, str] = {}
    for img in results:
        image = register_image(
            file_path=img.file_path,
            product_id=Path(img.image_name).stem,
            caption=img.caption or None,
            pdf_source=pdf.name,
        )
        if image.image_id not in name_map:
            name_map[image.image_id] = img.image_name
            image_ids.append(image.image_id)

    index_results = batch_index_images(image_ids, batch_size=32, tag_workers=2)
    success = sum(1 for r in index_results if r.status.value == "ready")
    print(f"  [✓] 索引完成: 成功 {success}/{len(index_results)}")

    # 3. 清理临时文件
    cleanup_extract_temp(pdf.stem)

    # 4. 检索验证
    if not query:
        # 用第一张图片的caption作为查询
        query = results[0].caption or "测试查询"
        print(f"\n  [3/4] 自动用第一个caption作为查询: {query[:50]}...")
    else:
        print(f"\n  [3/4] 用指定query检索: {query}")

    from app.core.image_retriever import search_by_text

    response = search_by_text(query, top_k=5)
    print(f"  [✓] 检索返回 {response.total} 条结果，路由: {response.route}")

    print("\n  [4/4] 检索结果:")
    print("-" * 60)
    for i, r in enumerate(response.results, 1):
        print(f"  [{i}] image_id: {r.image_id}")
        print(f"      分数: {r.score}")
        print(f"      标签: {r.tags}")
        if r.caption:
            print(f"      caption: {r.caption[:80]}...")
        print()

    print(f"  [结论] 索引+检索: {'✓ 正常' if response.total > 0 else '✗ 无结果'}")
    return response.total > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF提取功能验证脚本")
    parser.add_argument("--check-api", action="store_true", help="第1层：MinerU API连通性测试")
    parser.add_argument("--check-extract", action="store_true", help="第2层：PDF提取效果测试")
    parser.add_argument("--check-index", action="store_true", help="第3层：提取+索引+检索全流程")
    parser.add_argument("--check-all", action="store_true", help="全部执行")
    parser.add_argument("--pdf", type=str, default=None, help="测试PDF文件路径（第2、3层需要）")
    parser.add_argument("--query", type=str, default=None, help="检索query（第3层可选）")
    args = parser.parse_args()

    print("=" * 60)
    print("PDF 提取功能验证")
    print("=" * 60)

    results = []

    if args.check_all or args.check_api:
        results.append(("API连通性", check_api()))

    if args.check_all or args.check_extract:
        if not args.pdf:
            print("\n[!] --check-extract 需要 --pdf 参数")
        else:
            results.append(("提取效果", check_extract(args.pdf)))

    if args.check_all or args.check_index:
        if not args.pdf:
            print("\n[!] --check-index 需要 --pdf 参数")
        else:
            results.append(("索引检索", check_index(args.pdf, args.query)))

    if not results:
        print("\n[!] 请指定要执行的测试层，例如:")
        print("    python scripts/verify_pdf_extract.py --check-api")
        print("    python scripts/verify_pdf_extract.py --check-extract --pdf xxx.pdf")
        return

    # 汇总
    print("\n" + "=" * 60)
    print("验证汇总")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name}: {'✓ 通过' if ok else '✗ 失败'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
