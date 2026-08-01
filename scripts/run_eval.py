"""
工程性优化第6项之三：评测体系（recall@k 自动化评测 + RAGAS 多维指标评测）。

问题背景：目前对"检索质量是否达标"没有任何量化指标，每次改动 chunk 策略、
embedding 模型、rerank 参数等都只能凭感觉判断效果好坏，缺乏回归基线。

方案：为每本已注册的书籍维护一份人工编写的小规模评测集（data/eval/<book_id>.jsonl），
每条样本包含 query + 预期命中的关键词列表；脚本对每条样本跑一次单书精读检索，
判断 top_k 个 chunk 中是否包含预期关键词，计算 recall@k 并按书籍/整体汇总输出。

RAGAS 评测（--ragas 模式）：需在评测集每条样本中补充 ground_truth 字段，
评测完成后自动在 data/eval/ 下生成 ragas_report_<timestamp>.md 评估文档。

用法：
    python scripts/run_eval.py                 # recall@k 评测所有书籍
    python scripts/run_eval.py <book_id>        # recall@k 评测指定书籍
    python scripts/run_eval.py --ragas          # RAGAS 5 指标评测所有书籍
    python scripts/run_eval.py --ragas <book_id>  # RAGAS 5 指标评测指定书籍

评测集格式说明见 data/eval/README.md。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.retriever import get_book_retriever


# ---------------------------------------------------------------------------
# recall@k 评测（原有逻辑不变）
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    query: str
    expected_book_id: str
    expected_keywords: list[str]


@dataclass
class EvalResult:
    book_id: str
    total: int = 0
    hit: int = 0
    misses: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.hit / self.total if self.total else 0.0


def _load_eval_samples(path: Path) -> list[EvalSample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        samples.append(
            EvalSample(
                query=raw["query"],
                expected_book_id=raw["expected_book_id"],
                expected_keywords=raw["expected_keywords"],
            )
        )
    return samples


def _hit_keywords(chunks_text: str, keywords: list[str]) -> bool:
    return any(keyword in chunks_text for keyword in keywords)


def _eval_one_book(book_id: str, samples: list[EvalSample], top_k: int) -> EvalResult:
    result = EvalResult(book_id=book_id)
    retriever = get_book_retriever(book_id, top_k=top_k)

    for sample in samples:
        result.total += 1
        try:
            docs = retriever.invoke(sample.query)
        except Exception as exc:
            result.misses.append(f"[检索异常] {sample.query!r}: {exc}")
            continue

        chunks_text = "\n".join(doc.page_content for doc in docs)
        if _hit_keywords(chunks_text, sample.expected_keywords):
            result.hit += 1
        else:
            result.misses.append(sample.query)

    return result


# ---------------------------------------------------------------------------
# RAGAS 5 指标评测
# ---------------------------------------------------------------------------

def _load_ragas_samples(eval_files: list[Path]) -> tuple[list[dict], int]:
    """
    从评测集文件中收集有 ground_truth 的样本，执行检索+问答，
    返回 (ragas_dataset_rows, skipped_count)。
    """
    from app.core.retriever import ask_single_book

    rows = []
    skipped = 0

    for eval_file in eval_files:
        book_id = eval_file.stem
        for line in eval_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not raw.get("ground_truth"):
                skipped += 1
                continue

            query = raw["query"]
            try:
                resp = ask_single_book(
                    query=query,
                    book_id=book_id,
                    session_id=None,
                )
                contexts = [chunk.content for chunk in resp.sources]
                answer = resp.answer
            except Exception as exc:
                print(f"    [跳过] 问答异常 {query!r}: {exc}")
                skipped += 1
                continue

            rows.append({
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": raw["ground_truth"],
                "_book_id": book_id,
            })

    return rows, skipped


def _build_ragas_report(
    df,
    eval_files: list[Path],
    top_k: int,
    run_time: str,
) -> str:
    """将 RAGAS evaluate 的 DataFrame 结果渲染为 Markdown 评估文档。"""
    metric_names = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    ]

    cp_mean = df["context_precision"].mean() if "context_precision" in df.columns else 0.0
    cr_mean = df["context_recall"].mean() if "context_recall" in df.columns else 0.0
    f1_mean = (2 * cp_mean * cr_mean / (cp_mean + cr_mean)) if (cp_mean + cr_mean) > 0 else 0.0

    book_ids = sorted(df["_book_id"].unique()) if "_book_id" in df.columns else ["—"]

    lines = []
    lines.append("# RAG 评估报告")
    lines.append("")
    lines.append(f"- **评测时间**：{run_time}")
    lines.append(f"- **评测书籍**：{', '.join(book_ids)}")
    lines.append(f"- **样本数量**：{len(df)} 条")
    lines.append(f"- **检索 top_k**：{top_k}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 一、总体指标")
    lines.append("")
    lines.append("| 指标 | 说明 | 均值 |")
    lines.append("|------|------|------|")

    metric_desc = {
        "faithfulness": "忠实度——答案是否忠于检索内容，不产生幻觉",
        "context_precision": "上下文精度——检索内容中相关片段占比",
        "context_recall": "上下文召回率——ground_truth 关键信息被检索覆盖程度",
        "answer_relevancy": "答案相关性——答案是否直接回应了问题",
    }

    for m in metric_names:
        if m in df.columns:
            val = df[m].mean()
            lines.append(f"| {m} | {metric_desc.get(m, '')} | **{val:.4f}** |")

    lines.append(f"| context_f1 _(派生)_ | Context Precision 与 Recall 的调和均值 | **{f1_mean:.4f}** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 二、逐题明细")
    lines.append("")

    col_header = "| # | 书籍 | 问题 | Faithfulness | Ctx Precision | Ctx Recall | Ans Relevancy | Ctx F1 |"
    col_sep    = "|---|------|------|:---:|:---:|:---:|:---:|:---:|"
    lines.append(col_header)
    lines.append(col_sep)

    for i, row in df.iterrows():
        fa  = f"{row['faithfulness']:.3f}"      if "faithfulness"       in df.columns else "—"
        cp  = f"{row['context_precision']:.3f}" if "context_precision"  in df.columns else "—"
        cr  = f"{row['context_recall']:.3f}"    if "context_recall"     in df.columns else "—"
        ar  = f"{row['answer_relevancy']:.3f}"  if "answer_relevancy"   in df.columns else "—"
        cp_v = row.get("context_precision", 0)
        cr_v = row.get("context_recall", 0)
        f1_v = (2 * cp_v * cr_v / (cp_v + cr_v)) if (cp_v + cr_v) > 0 else 0.0
        f1  = f"{f1_v:.3f}"
        bk  = row.get("_book_id", "—")
        q   = str(row.get("question", "")).replace("|", "｜")
        lines.append(f"| {i+1} | {bk} | {q} | {fa} | {cp} | {cr} | {ar} | {f1} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 三、指标说明")
    lines.append("")
    lines.append("| 指标 | 取值范围 | 评测方式 | 越高越好 |")
    lines.append("|------|----------|----------|----------|")
    lines.append("| Faithfulness | 0 ~ 1 | LLM 判断答案每句话是否有检索依据 | ✓ |")
    lines.append("| Context Precision | 0 ~ 1 | LLM 判断检索片段中有多少与 ground_truth 相关 | ✓ |")
    lines.append("| Context Recall | 0 ~ 1 | LLM 判断 ground_truth 的关键信息是否被检索覆盖 | ✓ |")
    lines.append("| Answer Relevancy | 0 ~ 1 | Embedding 相似度衡量答案与问题的匹配程度 | ✓ |")
    lines.append("| Context F1 | 0 ~ 1 | 2×Precision×Recall / (Precision+Recall)，无额外开销 | ✓ |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 四、评测集要求")
    lines.append("")
    lines.append("RAGAS 评测需要在 `data/eval/<book_id>.jsonl` 每条样本中补充 `ground_truth` 字段：")
    lines.append("")
    lines.append("```jsonl")
    lines.append('{"query": "问题", "expected_book_id": "book_id", "expected_keywords": ["关键词"], "ground_truth": "标准答案"}')
    lines.append("```")
    lines.append("")
    lines.append("> `ground_truth` 建议写 2~4 句覆盖核心事实的标准答案，过于简短会导致 Context Recall 虚低。")
    lines.append("")

    return "\n".join(lines)


def run_ragas_eval(eval_files: list[Path], top_k: int) -> None:
    """
    基于 RAGAS 评测 5 个指标并在 data/eval/ 下生成 Markdown 评估文档：
    - Faithfulness（忠实度）
    - Context Precision（上下文精度）
    - Context Recall（上下文召回率）
    - Answer Relevancy（答案相关性）
    - Context F1（派生，无额外开销）
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("请先安装依赖：pip install 'ragas>=0.2.0,<0.3.0' datasets")
        return

    print("\n=== RAGAS 评测开始 ===")
    rows, skipped = _load_ragas_samples(eval_files)

    if not rows:
        print(f"没有可用于 RAGAS 评测的样本（跳过 {skipped} 条，缺少 ground_truth 字段）")
        print("请在评测集 JSONL 中为每条样本补充 ground_truth 字段，详见 data/eval/README.md。")
        return

    if skipped:
        print(f"（跳过 {skipped} 条缺少 ground_truth 的旧样本）")

    # 构建 Dataset：去掉内部字段 _book_id
    book_id_col = [r["_book_id"] for r in rows]
    dataset_rows = [{k: v for k, v in r.items() if k != "_book_id"} for r in rows]
    dataset = Dataset.from_list(dataset_rows)

    settings = get_settings()
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import OpenAIEmbeddings

    judge_llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key or None,
        base_url=settings.anthropic_base_url,
    )
    judge_embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    df["_book_id"] = book_id_col

    # 计算 Context F1（逐行）
    df["context_f1"] = (
        2 * df["context_precision"] * df["context_recall"]
        / (df["context_precision"] + df["context_recall"]).replace(0, float("nan"))
    ).fillna(0.0)

    # 打印摘要
    metric_names = ["faithfulness", "context_precision", "context_recall", "answer_relevancy", "context_f1"]
    print(f"\n{'指标':<25} {'均值':>8}")
    print("-" * 35)
    for m in metric_names:
        if m in df.columns:
            print(f"{m:<25} {df[m].mean():>8.4f}")

    # 保存 CSV（供进一步分析）
    eval_dir = eval_files[0].parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = eval_dir / f"ragas_result_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 生成 Markdown 评估文档
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_md = _build_ragas_report(df, eval_files, top_k, run_time)
    report_path = eval_dir / f"ragas_report_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n评估文档：{report_path}")
    print(f"详细数据：{csv_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    eval_dir = Path(settings.eval_dataset_dir)
    top_k = settings.retrieval_top_k

    use_ragas = "--ragas" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--ragas"]
    target_book_id = args[0] if args else None

    eval_files = sorted(eval_dir.glob("*.jsonl"))
    if target_book_id:
        eval_files = [f for f in eval_files if f.stem == target_book_id]

    if not eval_files:
        print(f"未找到评测集文件（目录: {eval_dir}），请先按 data/eval/README.md 编写评测样本")
        sys.exit(1)

    if use_ragas:
        run_ragas_eval(eval_files, top_k)
        return

    # 原有 recall@k 逻辑
    results: list[EvalResult] = []
    for eval_file in eval_files:
        book_id = eval_file.stem
        samples = _load_eval_samples(eval_file)
        if not samples:
            print(f"[跳过] {book_id}: 评测集为空")
            continue
        result = _eval_one_book(book_id, samples, top_k)
        results.append(result)

        print(f"[{book_id}] recall@{top_k} = {result.recall:.2%} ({result.hit}/{result.total})")
        for miss in result.misses:
            print(f"    未命中: {miss}")

    if not results:
        print("没有可评测的样本")
        return

    total_hit = sum(r.hit for r in results)
    total_count = sum(r.total for r in results)
    overall_recall = total_hit / total_count if total_count else 0.0
    print(f"\n整体 recall@{top_k} = {overall_recall:.2%} ({total_hit}/{total_count})")


if __name__ == "__main__":
    main()
