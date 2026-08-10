"""检查博物馆数据导入状态：验证 dataset_metadata.json 和 images.json 的一致性。

用法：
    python scripts/check_import_status.py                # 检查所有博物馆
    python scripts/check_import_status.py 李白纪念馆      # 只检查指定博物馆
"""
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    # 1. 检查 dataset_metadata.json
    meta_path = Path("data/processed/dataset_metadata.json")
    if not meta_path.exists():
        print("[错误] dataset_metadata.json 不存在")
        return
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # 按博物馆分组统计
    museum_meta: dict[str, list] = {}
    for pid, meta in metadata.items():
        museum = meta.get("museum", "?")
        museum_meta.setdefault(museum, []).append(pid)

    # 2. 检查 images.json
    images_path = Path("data/processed/images.json")
    if not images_path.exists():
        print("[错误] images.json 不存在")
        return
    images = json.loads(images_path.read_text(encoding="utf-8"))

    # images.json 按 product_id 前缀分组（无 museum 字段，用 product_id 推断）
    museum_images: dict[str, list] = {}
    for iid, img in images.items():
        pid = img.get("product_id", "")
        # 从 product_id 提取博物馆前缀（如 "李白_0001" → "李白"）
        prefix = pid.split("_")[0] if "_" in pid else pid
        museum_images.setdefault(prefix, []).append(iid)

    # 3. 打印统计
    print("=" * 80)
    print(f"dataset_metadata.json: {len(metadata)} 条")
    print(f"images.json: {len(images)} 条（READY: {sum(1 for v in images.values() if v.get('status') == 'ready')}）")
    print("=" * 80)

    if target:
        museums_to_check = [target]
    else:
        museums_to_check = sorted(museum_meta.keys())

    for museum in museums_to_check:
        meta_pids = museum_meta.get(museum, [])
        # 在 images.json 中找对应的图片
        # 博物馆名 → product_id 前缀的映射不精确，这里用多种方式查找
        matched_images = []
        for iid, img in images.items():
            pid = img.get("product_id", "")
            # 方式1：product_id 以博物馆名相关前缀开头
            if museum in pid or pid.startswith(museum[:2]):
                matched_images.append(iid)

        # 统计有 caption 的
        caption_count = sum(1 for iid in matched_images if images[iid].get("caption"))

        # 统计 dynasty 分布
        dynasty_counter = Counter()
        for pid in meta_pids:
            d = metadata[pid].get("dynasty")
            if d:
                dynasty_counter[d] += 1

        print(f"\n【{museum}】")
        print(f"  metadata: {len(meta_pids)} 条")
        print(f"  images.json: {len(matched_images)} 张（有caption: {caption_count}）")
        if dynasty_counter:
            top_dynasties = dynasty_counter.most_common(5)
            print(f"  dynasty 分布: {dict(top_dynasties)}")

        # 显示前3条样例
        if meta_pids:
            print(f"  样例:")
            for pid in meta_pids[:3]:
                m = metadata[pid]
                print(f"    {pid}: name={m.get('name', '?')[:30]}  "
                      f"dynasty={m.get('dynasty', '?')}  "
                      f"material={m.get('material', '?')}  "
                      f"caption={str(m.get('caption', '?'))[:40]}")


if __name__ == "__main__":
    main()
