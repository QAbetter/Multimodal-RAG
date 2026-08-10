"""
数据集导入脚本（插件式）：自动发现 scripts/importers/ 下的博物馆导入器并调用。

新增博物馆流程：
1. 复制 scripts/importers/_template.py 为 xxx.py
2. 修改 MUSEUM_NAME、XLSX_FILENAME、字段映射、图片复制逻辑
3. 运行 python scripts/import_dataset.py（自动发现新导入器）

输出：
1. 图片复制到 data/images/raw/{博物馆}/{藏品编号}/
2. 元数据 JSON：data/processed/dataset_metadata.json

用法：
    python scripts/import_dataset.py                    # 增量导入（跳过未变化的博物馆）
    python scripts/import_dataset.py --force            # 强制全量重跑（忽略增量状态）
    python scripts/import_dataset.py --museum 包头博物馆  # 只导入指定博物馆
    python scripts/import_dataset.py --dry-run         # 只预览不实际复制
    python scripts/import_dataset.py --list            # 列出所有可用导入器
"""
import argparse
import importlib
import json
import pkgutil
from pathlib import Path

# 导入器包目录
IMPORTERS_PACKAGE = "scripts.importers"


def discover_importers() -> dict[str, object]:
    """自动发现 scripts/importers/ 下的所有导入器模块。

    两个来源：
    1. importers/*.py（手写导入器，优先级高）
    2. importers/museums_config.py（配置驱动，简单型博物馆）

    约定：每个导入器需定义 MUSEUM_NAME（博物馆名，对应 DataSet/ 下的目录名）
    和 import_museum(src_dir, dst_raw, dry_run) 函数。
    以 _ 开头的模块（如 _template）会被跳过。

    Returns: {MUSEUM_NAME: module}
    """
    import importlib.util

    importers_dir = Path(__file__).parent / "importers"
    importers = {}
    # 先确保 importers 包能被导入（加载 __init__.py 里的公共工具）
    init_path = importers_dir / "__init__.py"
    if init_path.exists():
        spec = importlib.util.spec_from_file_location("scripts.importers", init_path)
        if spec and spec.loader:
            pkg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pkg)
            import sys
            sys.modules["scripts.importers"] = pkg
    # 扫描 importers/ 下的 .py 文件（手写导入器，优先级高）
    for py_file in sorted(importers_dir.glob("*.py")):
        name = py_file.stem
        if name.startswith("_") or name == "museums_config":
            continue  # 跳过 _template / museums_config / detect_unconfigured
        spec = importlib.util.spec_from_file_location(f"scripts.importers.{name}", py_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            museum_name = getattr(module, "MUSEUM_NAME", None)
            if museum_name and hasattr(module, "import_museum"):
                importers[museum_name] = module
    # 加载配置驱动的导入器（同名时 .py 优先，不覆盖）
    from scripts.importers import load_config_importers
    for name, mod in load_config_importers().items():
        if name not in importers:
            importers[name] = mod
    return importers


def main():
    parser = argparse.ArgumentParser(description="导入 DataSet 真实数据（插件式，支持增量）")
    parser.add_argument("--museum", type=str, default=None, help="只导入指定博物馆")
    parser.add_argument("--dry-run", action="store_true", help="只预览不实际复制")
    parser.add_argument("--list", action="store_true", help="列出所有可用导入器")
    parser.add_argument("--force", action="store_true", help="强制全量重跑（忽略增量状态，所有博物馆都重新导入）")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="跳过 GLM-4 文本模型补全结构化字段（默认导入后自动补全 dynasty/material 等）")
    parser.add_argument("--enrich-workers", type=int, default=4, help="enrich 阶段并发数（默认 4）")
    args = parser.parse_args()

    # 加载导入器
    importers = discover_importers()

    if args.list:
        print(f"可用导入器（{len(importers)} 个）：")
        for name, mod in importers.items():
            mod_name = getattr(mod, "__name__", "配置")
            xlsx_files = getattr(mod, "XLSX_FILES", None)
            if xlsx_files:
                xlsx_str = ", ".join(xlsx_files)
            else:
                xlsx_str = getattr(mod, "XLSX_FILENAME", None) or "无"
            print(f"  {name}  ←  {mod_name}  (xlsx: {xlsx_str})")
        return

    # 导入增量支持函数
    from scripts.importers import (
        load_import_state, save_import_state,
        is_xlsx_changed, record_import_state,
    )

    dataset_dir = Path("DataSet")
    dst_raw = Path("data/images/raw")
    if not args.dry_run:
        dst_raw.mkdir(parents=True, exist_ok=True)

    # 加载上次导入的状态（记录每个博物馆 xlsx 的 mtime+size）
    state = load_import_state() if not args.force else {}
    if args.force:
        print("[强制模式] 忽略增量状态，所有博物馆重新导入")

    # 加载已有的 metadata.json（用于合并：跳过的博物馆保留旧数据）
    metadata_path = Path("data/processed/dataset_metadata.json")
    all_metadata = {}
    if metadata_path.exists():
        try:
            all_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            all_metadata = {}

    skipped = 0
    imported = 0

    for museum_name, module in importers.items():
        if args.museum and args.museum != museum_name:
            continue
        src = dataset_dir / museum_name
        if not src.exists():
            print(f"[跳过] {museum_name} 目录不存在（DataSet/{museum_name}/）")
            continue

        # xlsx 路径
        xlsx_filename = getattr(module, "XLSX_FILENAME", None)
        xlsx_files = getattr(module, "XLSX_FILES", None)  # 多 xlsx 列表

        # 增量判断：xlsx 未变化且非 --force 则跳过
        if not args.force and not args.dry_run:
            if xlsx_files:
                # 多 xlsx：任一变化则重新导入
                changed = any(
                    is_xlsx_changed(f"{museum_name}/{xf}", src / xf, state)
                    for xf in xlsx_files if (src / xf).exists()
                )
                if not changed:
                    print(f"[跳过] {museum_name}（所有 xlsx 未变化，保留已有数据）")
                    skipped += 1
                    continue
            elif xlsx_filename:
                xlsx_path = src / xlsx_filename
                if not is_xlsx_changed(museum_name, xlsx_path, state):
                    print(f"[跳过] {museum_name}（xlsx 未变化，保留已有数据）")
                    skipped += 1
                    continue

        # 删除旧 metadata 中该博物馆的记录（重新导入会覆盖）
        # 用 museum 字段过滤，避免旧 product_id 残留
        old_count = sum(1 for m in all_metadata.values() if m.get("museum") == museum_name)
        if old_count > 0:
            all_metadata = {pid: m for pid, m in all_metadata.items()
                           if m.get("museum") != museum_name}

        print(f"\n--- 导入 {museum_name} ---")
        meta = module.import_museum(src, dst_raw, dry_run=args.dry_run)
        all_metadata.update(meta)
        imported += 1

        # 记录本次导入状态
        if not args.dry_run:
            if xlsx_files:
                for xf in xlsx_files:
                    xf_path = src / xf
                    if xf_path.exists():
                        record_import_state(f"{museum_name}/{xf}", xf_path, state)
            elif xlsx_filename:
                xlsx_path = src / xlsx_filename
                if xlsx_path.exists():
                    record_import_state(museum_name, xlsx_path, state)

    # 保存元数据 JSON（合并后的全量 metadata）
    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # 导入完成后，自动用 GLM-4 文本模型补全"只有名字"的文物结构化字段
        # 对 dynasty+material 为空的条目调 GLM-4 推断，已有字段的跳过（不重复调 API）
        # 补全的字段会生成命名空间标签（如"朝代:唐"），写入 tags 一起落盘
        if not args.skip_enrich:
            print("\n--- 元数据补全（GLM-4 文本模型）---")
            from scripts.enrich_name_only_metadata import enrich_metadata_batch
            enrich_metadata_batch(all_metadata, workers=args.enrich_workers, verbose=True)
        else:
            print("[跳过] 元数据补全（--skip-enrich）")

        metadata_path.write_text(json.dumps(all_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        save_import_state(state)

    print(f"\n[完成] 共 {len(all_metadata)} 条元数据 → {metadata_path}")
    if skipped > 0:
        print(f"  本次导入 {imported} 个博物馆，跳过 {skipped} 个未变化的博物馆")
    if args.dry_run:
        for pid, meta in list(all_metadata.items())[:3]:
            print(f"  {pid}: name={meta.get('name')} | dynasty={meta.get('dynasty')} | "
                  f"material={meta.get('material')} | category_sub={meta.get('category_sub')} | "
                  f"tags={meta.get('tags')}")


if __name__ == "__main__":
    main()
