"""元数据补全脚本：用 GLM-4 文本模型从文物名字推断结构化字段。

针对"只有文物名字，没有朝代/材质/类别等结构化字段"的博物馆数据，
调用 GLM-4 文本模型（glm-4-flash，比 glm-4v 快且便宜），从名字推断结构化字段。

适用场景：
- 苏州丝绸博物馆、金华市博物馆等只有"标题"列的博物馆
- xlsx 只提供文物名，没有年代/材质/完残等著录信息

执行流程：
1. 读取 dataset_metadata.json
2. 筛选"只有 name 没有其他结构化字段"的文物
3. 批量调 GLM-4 文本模型，传 name，提取朝代/材质/类别等
4. 4 并发处理，结果写回 dataset_metadata.json
5. 后续 batch_index_images.py 照常索引（自动跳过 GLM-4V）

用法：
    python scripts/enrich_name_only_metadata.py              # 补全所有
    python scripts/enrich_name_only_metadata.py --dry-run   # 只预览不写入
    python scripts/enrich_name_only_metadata.py --force      # 强制重新补全（包括已补全的）
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 让脚本能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


# 结构化字段列表（判断"是否只有 name"用）
_STRUCT_FIELDS = ["dynasty", "material", "category_top", "category_sub",
                  "craft", "color_feature", "function_usage", "relic_condition"]

# GLM-4 文本提取 Prompt
_ENRICH_PROMPT = """你是资深文博藏品著录专家。根据文物名称，推断其结构化著录信息。

文物名称：{name}

请根据名称中的信息推断以下字段，无法确定的填空字符串：
1. category_top: 一级文物分类【陶瓷器、青铜器、玉器、书画、金银器、石刻、漆器、织绣、杂项】选其一
2. category_sub: 二级具体器型名称（如"鼎""碗""扇页""袍"等）
3. dynasty: 年代/朝代（如"明""清""唐""新石器时代"等）
4. material: 材质/质地（如"瓷""青铜""纸""丝绸"等）
5. color_feature: 色彩特征（如"青""黄""朱"等，无法判断填空）
6. craft: 核心工艺技法（如"釉""织""绣"等，无法判断填空）
7. function_usage: 器物原始功用（如"食器""服饰""陈设"等，无法判断填空）
8. relic_condition: 完残状态（无法判断填空）

仅返回纯 JSON，无任何多余内容。示例：
{{"category_top":"书画","category_sub":"扇页","dynasty":"明","material":"纸","color_feature":"","craft":"","function_usage":"陈设","relic_condition":""}}"""


def get_enrich_llm() -> ChatOpenAI:
    """元数据补全专用 LLM（glm-4-flash 文本模型，复用智谱 API Key）。"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,  # glm-4-flash
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )


def parse_json_response(content: str) -> dict:
    """解析 GLM-4 返回的 JSON，容错处理。"""
    import re
    content = content.strip()
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


def is_name_only(meta: dict) -> bool:
    """判断是否为"只有 name 没有其他结构化字段"的文物。

    有任一结构化字段（非空）即认为已有元数据。
    用于标签重生成阶段：有结构化字段的条目都需生成命名空间标签。
    """
    for field in _STRUCT_FIELDS:
        val = meta.get(field)
        if val and str(val).strip():
            return False
    return True


def needs_enrichment(meta: dict) -> bool:
    """判断是否需要用 GLM-4 文本模型补全结构化字段。

    判断依据：dynasty 和 material 同时为空（这两个是最核心的可从名称推断的字段）。

    与 is_name_only 的区别：is_name_only 要求所有字段都为空，
    而 needs_enrichment 只看 dynasty+material。
    这样即使 importer 已经从名称提取了 category_sub（如"鼎"），
    但 dynasty/material 仍为空的条目也会被纳入补全范围。
    """
    dynasty = meta.get("dynasty")
    material = meta.get("material")
    dynasty_empty = not dynasty or not str(dynasty).strip()
    material_empty = not material or not str(material).strip()
    return dynasty_empty and material_empty


def enrich_one(product_id: str, name: str, llm: ChatOpenAI) -> dict | None:
    """调 GLM-4 从文物名字提取结构化字段。

    返回补全的字段 dict（如 {"dynasty": "明", "material": "纸", ...}），
    失败返回 None。
    """
    try:
        prompt = _ENRICH_PROMPT.format(name=name)
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_json_response(response.content)
        if not parsed:
            print(f"  [!] {product_id} ({name}): 返回内容解析失败")
            return None
        # 只保留非空字段
        result = {k: v for k, v in parsed.items() if v and str(v).strip()}
        return result
    except Exception as e:
        print(f"  [!] {product_id} ({name}): 调用失败 - {e}")
        return None


def _regenerate_tags(metadata: dict) -> None:
    """对有结构化字段的条目重新生成 tags（名称标签 + 命名空间标签）。

    在 enrich 补全字段后调用：把补全的 dynasty/material/craft 等字段
    转为 "朝代:唐" 格式标签，追加到 tags 列表，让标签路能精确命中。

    只处理 is_name_only=False（已有结构化字段）的条目；
    补全失败的条目（仍为 is_name_only=True）保留原 tags 不动。
    """
    try:
        from scripts.importers import build_name_tags, extract_ware_type
    except ImportError:
        return

    from app.core.cultural_relic_aliases import structured_fields_to_tags
    for pid, meta in metadata.items():
        if is_name_only(meta):
            continue  # 没有结构化字段，跳过
        name = meta.get("name", "")
        new_tags = list(build_name_tags(name))
        struct_tags = structured_fields_to_tags(
            dynasty=meta.get("dynasty"),
            material=meta.get("material"),
            category_sub=meta.get("category_sub") or extract_ware_type(name),
            craft=meta.get("craft"),
            function_usage=meta.get("function_usage"),
            relic_condition=meta.get("relic_condition"),
            color_feature=meta.get("color_feature"),
        )
        new_tags.extend(struct_tags)
        meta["tags"] = new_tags
        if not meta.get("category_sub"):
            meta["category_sub"] = extract_ware_type(name)


def enrich_metadata_batch(
    metadata: dict,
    workers: int = 4,
    verbose: bool = True,
    force: bool = False,
) -> tuple[int, int]:
    """对 metadata dict 中 dynasty+material 为空的条目批量调 GLM-4 补全。

    原地修改 metadata dict：补全 dynasty/material/category_top 等字段，
    并重新生成 tags（含命名空间标签）。

    已有 dynasty 或 material 的条目会被 needs_enrichment 跳过，不重复调 API
    （force=True 时强制全部重新补全）。
    补全失败的条目保留原状（tags 不动）。

    供 import_dataset.py 在导入完成后自动调用，也可被 main() 复用。

    Args:
        metadata: {product_id: {name, dynasty, material, ...}}（原地修改）
        workers: 并发数
        verbose: 打印进度
        force: 强制补全所有条目（包括已有 dynasty/material 的）

    Returns: (success_count, fail_count)
    """
    if force:
        to_enrich = [(pid, meta) for pid, meta in metadata.items()]
    else:
        to_enrich = [(pid, meta) for pid, meta in metadata.items() if needs_enrichment(meta)]

    if not to_enrich:
        if verbose:
            print("[enrich] 无需补全（所有条目已有 dynasty/material）")
        return 0, 0

    if verbose:
        print(f"[enrich] 开始补全 {len(to_enrich)} 条（{workers} 并发）...")

    llm = get_enrich_llm()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_pid = {
            pool.submit(enrich_one, pid, meta.get("name", ""), llm): pid
            for pid, meta in to_enrich
        }
        for i, future in enumerate(as_completed(future_to_pid), 1):
            pid = future_to_pid[future]
            try:
                result = future.result()
                if result:
                    for k, v in result.items():
                        if v and str(v).strip():
                            metadata[pid][k] = str(v).strip()
                    success_count += 1
                    if verbose and (i <= 5 or i % 20 == 0):
                        name = metadata[pid].get("name", "")
                        print(f"  [{i}/{len(to_enrich)}] {pid} ({name}): "
                              f"朝代={result.get('dynasty', '?')}, "
                              f"材质={result.get('material', '?')}, "
                              f"类别={result.get('category_top', '?')}")
                else:
                    fail_count += 1
            except Exception as e:
                if verbose:
                    print(f"  [!] {pid}: 异常 - {e}")
                fail_count += 1

    # 重新生成 tags（含命名空间标签）
    _regenerate_tags(metadata)

    if verbose:
        print(f"[enrich] 完成: 成功 {success_count} 条, 失败 {fail_count} 条")
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description="用 GLM-4 文本模型从文物名字补全结构化字段")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写入 metadata.json")
    parser.add_argument("--force", action="store_true", help="强制重新补全（包括已补全的）")
    parser.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    args = parser.parse_args()

    metadata_path = Path("data/processed/dataset_metadata.json")
    if not metadata_path.exists():
        print(f"[!] metadata 文件不存在: {metadata_path}")
        print("    请先运行 python scripts/import_dataset.py")
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"元数据总计: {len(metadata)} 条")

    # 统计待补全数量
    if args.force:
        print(f"强制模式: 全部 {len(metadata)} 条需要补全")
    else:
        enrich_count = sum(1 for m in metadata.values() if needs_enrichment(m))
        print(f"  需补全（dynasty+material 为空）: {enrich_count} 条")
        print(f"  已有 dynasty/material: {len(metadata) - enrich_count} 条（跳过）")

    # 调用核心函数补全（原地修改 metadata）
    success, fail = enrich_metadata_batch(
        metadata, workers=args.workers, verbose=True, force=args.force
    )

    # 写回 metadata.json
    if not args.dry_run:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入: {metadata_path}")
    else:
        print("[dry-run] 未写入文件")

    # 预览前 5 条补全结果
    print("\n补全结果预览（前 5 条）:")
    shown = 0
    for pid, meta in metadata.items():
        if shown >= 5:
            break
        if meta.get("dynasty") or meta.get("material"):
            print(f"  {pid}: name={meta.get('name')}")
            print(f"    dynasty={meta.get('dynasty')}, material={meta.get('material')}, "
                  f"category_top={meta.get('category_top')}, category_sub={meta.get('category_sub')}")
            shown += 1


if __name__ == "__main__":
    main()
