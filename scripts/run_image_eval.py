"""
图片 RAG 检索质量评测脚本：计算 Recall@K 和 MRR。

对应书籍 RAG 的 run_eval.py，区别在于：
- 书籍 RAG：query → 文本检索 → 判断 top_k chunk 是否含预期关键词（子串匹配）
- 图片 RAG：query/image → 向量/混合检索 → 判断 top_k 是否含预期 image_id（精确匹配）

评测指标：
- Recall@K：top_k 结果中是否包含至少一个预期 image_id（命中=1，未命中=0）
- MRR（Mean Reciprocal Rank）：第一个命中的预期 image_id 在结果中的排名倒数
  （排名第1=1.0，第2=0.5，第3=0.33...，未命中=0）
  MRR 比 Recall@K 更严格：不仅要求命中，还要求命中位置靠前

三种评测模式（评测集中每条样本的 mode 字段）：
- text：文本搜图（纯向量检索），用 query 字段
- image：以图搜图（纯向量检索），用 image_file 字段（相对 image_storage_dir 的路径）
- hybrid：混合检索（向量+标签 RRF），用 query + tags 字段

用法：
    # 用默认评测集 data/eval/images.jsonl 评测（top_k=10）
    $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
    .venv\\Scripts\\python.exe scripts\\run_image_eval.py

    # 指定 top_k
    .venv\\Scripts\\python.exe scripts\\run_image_eval.py --top-k 5

    # 指定评测集文件
    .venv\\Scripts\\python.exe scripts\\run_image_eval.py data/eval/custom_eval.jsonl

评测集格式（JSONL，每行一条）：
    {"mode": "text", "query": "青铜剑", "expected_image_ids": ["7096fdcf1df96617"]}
    {"mode": "image", "image_file": "raw/剑.webp", "expected_image_ids": ["7096fdcf1df96617"]}
    {"mode": "hybrid", "query": "青铜器", "tags": ["青铜"], "expected_image_ids": ["7096fdcf1df96617", "8212d17d8858ce89"]}

字段说明：
- mode：text / image / hybrid
- query：文本查询（text 和 hybrid 模式必填）
- image_file：图片相对路径（image 模式必填，相对 image_storage_dir）
- tags：标签列表（hybrid 模式必填，用于标签召回+RRF 融合）
- expected_image_ids：期望命中的 image_id 列表（必填，top_k 中包含任意一个即视为命中）
"""
from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.image_retriever import search


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ImageEvalSample:
    """单条评测样本。"""

    mode: str  # text / image / hybrid
    query: str = ""
    image_file: str = ""
    tags: list[str] = field(default_factory=list)
    expected_image_ids: list[str] = field(default_factory=list)
    # 保留原始字段用于结果展示
    raw: dict = field(default_factory=dict)


@dataclass
class ImageEvalResult:
    """单条评测结果。"""

    sample: ImageEvalSample
    hit: bool = False  # Recall@K 是否命中
    first_hit_rank: int = -1  # 第一个命中的排名（从 0 开始，-1 表示未命中）
    retrieved_ids: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Recall@K：命中=1.0，未命中=0.0。"""
        return 1.0 if self.hit else 0.0

    @property
    def rr(self) -> float:
        """Reciprocal Rank：1/(rank+1)，未命中=0.0。"""
        if self.first_hit_rank < 0:
            return 0.0
        return 1.0 / (self.first_hit_rank + 1)


# ---------------------------------------------------------------------------
# 评测集加载
# ---------------------------------------------------------------------------

def _load_eval_samples(path: Path) -> list[ImageEvalSample]:
    """加载 JSONL 评测集。"""
    if not path.exists():
        print(f"评测集文件不存在: {path}")
        print("请参考脚本开头注释创建评测集，或运行 --init 生成示例评测集")
        sys.exit(1)

    samples = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[警告] 第 {line_no} 行 JSON 解析失败，跳过: {exc}")
            continue

        mode = raw.get("mode", "text")
        expected = raw.get("expected_image_ids", [])
        if not expected:
            print(f"[警告] 第 {line_no} 行缺少 expected_image_ids，跳过")
            continue

        samples.append(
            ImageEvalSample(
                mode=mode,
                query=raw.get("query", ""),
                image_file=raw.get("image_file", ""),
                tags=raw.get("tags", []),
                expected_image_ids=expected,
                raw=raw,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# 评测执行
# ---------------------------------------------------------------------------

def _load_image_base64(image_file: str) -> str:
    """读取图片文件并转为 base64（image 模式用）。"""
    settings = get_settings()
    abs_path = Path(settings.image_storage_dir) / image_file
    if not abs_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {abs_path}")
    data = abs_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _eval_one_sample(sample: ImageEvalSample, top_k: int) -> ImageEvalResult:
    """对单条样本执行检索并计算命中情况。"""
    result = ImageEvalResult(sample=sample)

    try:
        if sample.mode == "text":
            if not sample.query:
                raise ValueError("text 模式需要 query 字段")
            resp = search(query=sample.query, top_k=top_k)
        elif sample.mode == "image":
            if not sample.image_file:
                raise ValueError("image 模式需要 image_file 字段")
            image_b64 = _load_image_base64(sample.image_file)
            resp = search(image_base64=image_b64, top_k=top_k)
        elif sample.mode == "hybrid":
            if not sample.query:
                raise ValueError("hybrid 模式需要 query 字段")
            if not sample.tags:
                raise ValueError("hybrid 模式需要 tags 字段")
            resp = search(query=sample.query, tags=sample.tags, top_k=top_k)
        else:
            raise ValueError(f"未知 mode: {sample.mode}（支持 text/image/hybrid）")
    except Exception as exc:
        print(f"    [检索异常] {sample.mode} {sample.query or sample.image_file!r}: {exc}")
        return result

    result.retrieved_ids = [r.image_id for r in resp.results]

    # 计算 Recall@K 和第一个命中的排名
    for rank, rid in enumerate(result.retrieved_ids):
        if rid in sample.expected_image_ids:
            result.hit = True
            result.first_hit_rank = rank
            break

    return result


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

def _print_results(results: list[ImageEvalResult], top_k: int) -> None:
    """打印评测结果汇总 + 逐题明细。"""
    if not results:
        print("没有可评测的样本")
        return

    # 按模式分组统计
    modes = sorted({r.sample.mode for r in results})

    print("\n" + "=" * 70)
    print(f"图片 RAG 检索评测结果（top_k={top_k}）")
    print("=" * 70)

    # 分模式汇总
    print(f"\n{'模式':<10} {'样本数':>6} {'Recall@K':>10} {'MRR':>8}")
    print("-" * 40)
    for mode in modes:
        mode_results = [r for r in results if r.sample.mode == mode]
        n = len(mode_results)
        recall = sum(r.recall for r in mode_results) / n
        mrr = sum(r.rr for r in mode_results) / n
        print(f"{mode:<10} {n:>6} {recall:>10.2%} {mrr:>8.4f}")

    # 整体汇总
    total = len(results)
    overall_recall = sum(r.recall for r in results) / total
    overall_mrr = sum(r.rr for r in results) / total
    print("-" * 40)
    print(f"{'整体':<10} {total:>6} {overall_recall:>10.2%} {overall_mrr:>8.4f}")

    # 验收标准对照（技术文档 8.4 节）
    print("\n验收标准对照（技术文档 8.4 节）：")
    text_results = [r for r in results if r.sample.mode == "text"]
    image_results = [r for r in results if r.sample.mode == "image"]
    if text_results:
        text_recall = sum(r.recall for r in text_results) / len(text_results)
        status = "✓ 达标" if text_recall >= 0.7 else "✗ 未达标"
        print(f"  文本搜图 Recall@10 ≥ 0.7：{text_recall:.2%} {status}")
    if image_results:
        image_recall = sum(r.recall for r in image_results) / len(image_results)
        status = "✓ 达标" if image_recall >= 0.8 else "✗ 未达标"
        print(f"  以图搜图 Recall@10 ≥ 0.8：{image_recall:.2%} {status}")

    # 逐题明细
    print("\n" + "-" * 70)
    print("逐题明细：")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        s = r.sample
        hit_icon = "✓" if r.hit else "✗"
        if s.mode == "text":
            desc = f"query={s.query!r}"
        elif s.mode == "image":
            desc = f"image={s.image_file!r}"
        else:
            desc = f"query={s.query!r}, tags={s.tags}"
        rank_str = f"rank={r.first_hit_rank + 1}" if r.first_hit_rank >= 0 else "未命中"
        print(f"  [{i}] {hit_icon} {s.mode:<7} {desc}")
        print(f"       期望: {s.expected_image_ids}")
        print(f"       实际: {r.retrieved_ids[:top_k]}")
        print(f"       命中: {rank_str}, RR={r.rr:.4f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# 示例评测集生成
# ---------------------------------------------------------------------------

def _init_sample_eval(path: Path) -> None:
    """根据已索引的图片自动生成示例评测集（便于快速开始）。"""
    from app.core.image_indexer import load_registered_images

    registry = load_registered_images()
    if not registry:
        print("当前没有已索引的图片，请先索引图片后再生成评测集")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# 图片 RAG 评测集（自动生成，请根据实际需求调整 expected_image_ids）")
    lines.append("# 每行一条 JSON 样本，字段说明见 scripts/run_image_eval.py 开头注释")
    lines.append("")

    # 为每张图片生成 3 条样本（text / image / hybrid）
    for image_id, img in registry.items():
        file_path = img.file_path
        # 用第一个标签作为 hybrid 的 tags
        first_tag = img.tags[0] if img.tags else ""

        # text 模式：用第一个标签作为 query
        if first_tag:
            sample_text = {
                "mode": "text",
                "query": first_tag,
                "expected_image_ids": [image_id],
            }
            lines.append(json.dumps(sample_text, ensure_ascii=False))

        # image 模式：用图片自身搜自身
        sample_image = {
            "mode": "image",
            "image_file": file_path,
            "expected_image_ids": [image_id],
        }
        lines.append(json.dumps(sample_image, ensure_ascii=False))

        # hybrid 模式：query + tags
        if first_tag:
            sample_hybrid = {
                "mode": "hybrid",
                "query": first_tag,
                "tags": [first_tag],
                "expected_image_ids": [image_id],
            }
            lines.append(json.dumps(sample_hybrid, ensure_ascii=False))

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成示例评测集: {path}")
    print(f"共 {len(registry)} 张图片，每张生成 text/image/hybrid 各一条样本")
    print("请根据实际检索效果调整 expected_image_ids（例如查询'青铜'应命中青铜剑+编钟）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="图片 RAG 检索质量评测")
    parser.add_argument("eval_file", nargs="?", default=None, help="评测集 JSONL 文件路径（默认 data/eval/images.jsonl）")
    parser.add_argument("--top-k", type=int, default=10, help="评测的 top_k 值（默认 10）")
    parser.add_argument("--init", action="store_true", help="根据已索引图片自动生成示例评测集")
    args = parser.parse_args()

    settings = get_settings()
    eval_path = Path(args.eval_file) if args.eval_file else Path(settings.eval_dataset_dir) / "images.jsonl"

    if args.init:
        _init_sample_eval(eval_path)
        return

    print("=" * 70)
    print("图片 RAG 检索评测")
    print("=" * 70)
    print(f"评测集: {eval_path}")
    print(f"top_k: {args.top_k}")

    samples = _load_eval_samples(eval_path)
    print(f"加载样本: {len(samples)} 条")

    if not samples:
        print("没有可评测的样本，可用 --init 生成示例评测集")
        return

    # 逐条评测
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] 评测中: {sample.mode} {sample.query or sample.image_file!r}")
        result = _eval_one_sample(sample, args.top_k)
        results.append(result)

    _print_results(results, args.top_k)


if __name__ == "__main__":
    main()
