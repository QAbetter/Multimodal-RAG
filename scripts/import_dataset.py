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

    约定：每个模块需定义 MUSEUM_NAME（博物馆名，对应 DataSet/ 下的目录名）
    和 import_museum(src_dir, dst_raw, dry_run) 函数。
    以 _ 开头的模块（如 _template）会被跳过。

    用文件系统路径直接加载模块，不依赖 scripts 是 Python 包。

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
    # 扫描 importers/ 下的 .py 文件
    for py_file in sorted(importers_dir.glob("*.py")):
        name = py_file.stem
        if name.startswith("_"):
            continue  # 跳过 _template 等
        spec = importlib.util.spec_from_file_location(f"scripts.importers.{name}", py_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            museum_name = getattr(module, "MUSEUM_NAME", None)
            if museum_name and hasattr(module, "import_museum"):
                importers[museum_name] = module
    return importers


def main():
    parser = argparse.ArgumentParser(description="导入 DataSet 真实数据（插件式，支持增量）")
    parser.add_argument("--museum", type=str, default=None, help="只导入指定博物馆")
    parser.add_argument("--dry-run", action="store_true", help="只预览不实际复制")
    parser.add_argument("--list", action="store_true", help="列出所有可用导入器")
    parser.add_argument("--force", action="store_true", help="强制全量重跑（忽略增量状态，所有博物馆都重新导入）")
    args = parser.parse_args()

    # 加载导入器
    importers = discover_importers()

    if args.list:
        print(f"可用导入器（{len(importers)} 个）：")
        for name, mod in importers.items():
            xlsx = getattr(mod, "XLSX_FILENAME", "?")
            print(f"  {name}  ←  {mod.__name__}  (xlsx: {xlsx})")
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
        xlsx_path = src / xlsx_filename if xlsx_filename else None

        # 增量判断：xlsx 未变化且非 --force 则跳过
        if not args.force and not args.dry_run and xlsx_path:
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
        if not args.dry_run and xlsx_path:
            record_import_state(museum_name, xlsx_path, state)

    # 保存元数据 JSON（合并后的全量 metadata）
    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
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
