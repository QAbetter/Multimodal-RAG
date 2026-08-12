"""批量修复 images.json 中存在问题的文件名（首尾空格、连续点号）。

问题背景：
  早期导入器未清洗文件名，导致部分图片文件名含首尾空格或连续点号（如
  " 元龙泉窑粉青釉划花....jpeg"），RAGFlow 静态文件服务对这些 URL 返回 404。

修复范围（三处必须同步，否则会导致图片无法访问或元数据错乱）：
  1. 磁盘文件：data/images/raw/<museum>/<product_id>/<filename>  重命名
  2. 注册表：   data/processed/images.json 中 file_path 字段
  3. 向量库：   Chroma collection 'images' 中 metadata.file_path 字段

安全设计：
  - 默认 dry-run，仅打印将要修复的文件清单，不实际改动
  - 显式 --apply 才执行修复，全程加 registry_lock 保护注册表读改写
  - image_id 由文件内容 MD5 计算，重命名不影响 image_id（向量库无需重建）
  - 同一目录下若清洗后文件名冲突，自动追加 "_dup<N>" 后缀避免覆盖

用法：
  # 1. dry-run 预览（不改动任何文件）
  python scripts/fix_bad_filenames.py

  # 2. 实际执行修复
  python scripts/fix_bad_filenames.py --apply

  # 3. 只看统计，不打印明细
  python scripts/fix_bad_filenames.py --quiet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.image_indexer import load_registered_images, save_registered_images
from app.core.image_vectorstore import get_image_collection
from app.core.locks import registry_lock
from scripts.importers import clean_filename


def _collect_problems(images: dict, storage_dir: Path) -> list[dict]:
    """扫描注册表，收集所有需要修复的文件名问题。

    返回每条记录包含：
      - image_id
      - old_path (相对路径，如 'raw/内蒙古博物院/内蒙古_036/ 元....jpeg')
      - new_path (清洗后相对路径)
      - abs_old  (磁盘绝对路径)
      - abs_new  (修复后绝对路径)
    """
    problems: list[dict] = []
    for image_id, image in images.items():
        old_rel = image.file_path
        old_name = Path(old_rel).name
        new_name = clean_filename(old_name)
        if new_name == old_name:
            continue  # 无需修复
        new_rel = str(Path(old_rel).parent / new_name)
        problems.append({
            "image_id": image_id,
            "old_path": old_rel,
            "new_path": new_rel,
            "abs_old": storage_dir / old_rel,
            "abs_new": storage_dir / new_rel,
        })
    return problems


def _resolve_conflicts(problems: list[dict]) -> list[dict]:
    """处理清洗后文件名冲突（同一目录下不同原图清洗后重名）。

    例如 "a ....jpg" 和 "a.. .jpg" 都清洗成 "a.jpg"，需给第二个加 _dup1 后缀。
    原磁盘已存在的目标文件名也要避开（避免覆盖无关文件）。
    """
    # 按目标目录分组，统计同目录下相同 new_path 出现次数
    seen: dict[str, int] = {}
    for p in problems:
        key = str(p["abs_new"])
        seen[key] = seen.get(key, 0) + 1

    # 第一遍：若 abs_new 已存在但不是当前要改名的源文件，加 _dup<N>
    # 第二遍：同批内多个源文件清洗到同一 new_path，按出现顺序加 _dup1, _dup2
    used: dict[str, int] = {}
    for p in problems:
        abs_new = p["abs_new"]
        # 如果目标已存在，且不是当前源文件，需要改名
        if abs_new.exists() and abs_new.resolve() != p["abs_old"].resolve():
            stem = abs_new.stem
            suffix = abs_new.suffix
            n = used.get(str(abs_new), 0) + 1
            used[str(abs_new)] = n
            new_name = f"{stem}_dup{n}{suffix}"
            p["new_path"] = str(Path(p["new_path"]).parent / new_name)
            p["abs_new"] = abs_new.parent / new_name
        elif seen[str(abs_new)] > 1:
            # 同批冲突：第一个不动，后续加 _dup<N>
            n = used.get(str(abs_new), 0)
            if n > 0:
                stem = abs_new.stem
                suffix = abs_new.suffix
                new_name = f"{stem}_dup{n}{suffix}"
                p["new_path"] = str(Path(p["new_path"]).parent / new_name)
                p["abs_new"] = abs_new.parent / new_name
            used[str(abs_new)] = n + 1
    return problems


def _print_report(problems: list[dict], quiet: bool) -> None:
    """打印修复报告。"""
    total = len(problems)
    if total == 0:
        print("[OK] 未发现需要修复的文件名，所有图片路径都正常。")
        return

    # 按问题类型分类统计
    leading_space = sum(1 for p in problems if p["old_path"] != p["old_path"].lstrip())
    trailing_space = sum(1 for p in problems if p["old_path"] != p["old_path"].rstrip())
    # 连续点号：basename 中存在 ".."
    consecutive_dots = sum(
        1 for p in problems if ".." in Path(p["old_path"]).name
    )

    print(f"[ISSUE] 发现 {total} 个需要修复的文件名：")
    print(f"  - 首部空格: {leading_space} 个")
    print(f"  - 尾部空格: {trailing_space} 个")
    print(f"  - 连续点号: {consecutive_dots} 个")

    if quiet:
        return

    # 明细：最多打印 50 条，避免刷屏
    show = problems[:50]
    print(f"\n明细（前 {len(show)} 条，共 {total} 条）：")
    for p in show:
        old_name = Path(p["old_path"]).name
        new_name = Path(p["new_path"]).name
        print(f"  [{p['image_id'][:8]}] '{old_name}' → '{new_name}'")
    if total > len(show):
        print(f"  ... 还有 {total - len(show)} 条未显示，使用 --apply 执行修复")


def _apply_fixes(problems: list[dict], images: dict) -> tuple[int, int]:
    """实际执行修复：重命名磁盘文件 + 更新注册表 + 更新 Chroma metadata。

    全程在 registry_lock 内执行，保证 images.json 读改写的原子性。

    Returns:
        (renamed_count, failed_count)
    """
    collection = get_image_collection()
    renamed = 0
    failed = 0
    failed_ids: list[str] = []

    for p in problems:
        abs_old: Path = p["abs_old"]
        abs_new: Path = p["abs_new"]
        image_id = p["image_id"]
        new_path = p["new_path"]

        # 1. 重命名磁盘文件
        if not abs_old.exists():
            print(f"[SKIP] 磁盘文件不存在: {abs_old}")
            failed += 1
            failed_ids.append(image_id)
            continue

        try:
            abs_old.rename(abs_new)
        except OSError as e:
            print(f"[FAIL] 重命名失败 {abs_old.name} → {abs_new.name}: {e}")
            failed += 1
            failed_ids.append(image_id)
            continue

        # 2. 更新注册表中的 file_path
        image = images.get(image_id)
        if image is not None:
            image.file_path = new_path

        # 3. 更新 Chroma metadata 中的 file_path
        try:
            collection.update(
                ids=[image_id],
                metadatas=[{"file_path": new_path}],
            )
        except Exception as e:
            print(f"[WARN] Chroma metadata 更新失败 {image_id}: {e}（注册表已更新，可能需重建向量库）")

        renamed += 1

    # 注册表落盘（一次写）
    if renamed > 0:
        save_registered_images(images)

    return renamed, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量修复 images.json 中存在问题的文件名（首尾空格、连续点号）"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行修复；不带此参数则 dry-run 仅预览",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只打印统计，不显示明细",
    )
    args = parser.parse_args()

    settings = get_settings()
    storage_dir = Path(settings.image_storage_dir).resolve()
    print(f"[*] 图片存储目录: {storage_dir}")
    print(f"[*] 模式: {'APPLY（实际修复）' if args.apply else 'DRY-RUN（仅预览）'}")

    # 加载注册表（不加锁，因为 dry-run 不写）
    images = load_registered_images()
    print(f"[*] 注册表图片数: {len(images)}")

    # 扫描问题
    problems = _collect_problems(images, storage_dir)
    # 处理冲突
    problems = _resolve_conflicts(problems)

    # 打印报告
    _print_report(problems, args.quiet)

    if not args.apply:
        if problems:
            print("\n[*] 这是 DRY-RUN，未改动任何文件。使用 --apply 执行实际修复。")
        return 0

    if not problems:
        return 0

    # 实际修复：加锁保护注册表读改写
    print(f"\n[*] 开始修复 {len(problems)} 个文件名...")
    with registry_lock:
        # 锁内重新加载最新注册表（避免 dry-run 到 apply 之间数据被改）
        images = load_registered_images()
        # 重新收集一遍（image_id 集合可能与初次加载略有不同）
        problems = _collect_problems(images, storage_dir)
        problems = _resolve_conflicts(problems)
        renamed, failed = _apply_fixes(problems, images)

    print(f"\n[完成] 成功修复 {renamed} 个，失败 {failed} 个")
    if failed > 0:
        print("[提示] 失败的图片可能磁盘文件已被移动或删除，建议检查 data/images/raw/ 目录")
    print("[下一步] 请在 RAGFlow 中重新检索，验证图片 URL 是否可正常访问")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
