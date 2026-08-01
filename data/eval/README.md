# 评测集说明

## 目录结构

每本书对应一个 `<book_id>.jsonl` 文件，每行一条评测样本（JSON Lines 格式）。

### recall@k 模式（基础字段）

```json
{"query": "用户问题", "expected_book_id": "book_id", "expected_keywords": ["关键词1", "关键词2"]}
```

字段说明：
- `query`：模拟用户会问的问题，尽量覆盖事实型/归纳型/跨章节型等不同问法
- `expected_book_id`：期望命中的书籍 id（跨书问答场景下用于判断路由是否正确）
- `expected_keywords`：期望出现在检索结果（top_k 个 chunk 的 page_content 拼接）中的关键词列表，
  只要命中其中任意一个关键词即视为该条命中（因为原文表述可能和关键词不完全一致，
  用"包含关键词"而不是"精确匹配句子"作为宽松但可自动化的相关性判定）

### RAGAS 模式（额外补充 ground_truth）

```json
{
  "query": "用户问题",
  "expected_book_id": "book_id",
  "expected_keywords": ["关键词1", "关键词2"],
  "ground_truth": "覆盖核心事实的标准答案，建议 2~4 句话"
}
```

新增字段说明：
- `ground_truth`：用于 Context Recall 和 Context Precision 评测的标准答案。
  建议写 2~4 句覆盖核心事实的完整答案，过于简短会导致 Context Recall 虚低。
  没有此字段的旧样本在 `--ragas` 模式下会被自动跳过，recall@k 评测不受影响。

## 命名与新增方式

- 文件名与 `book_id` 一致，例如 `book_id=sanguo-yanyi` 对应 `sanguo-yanyi.jsonl`
- 每本书建议积累 10-20 条问答对，覆盖：
  - 事实型问题（"XX 是谁"、"XX 发生在第几章"）
  - 归纳型问题（"XX 的主要观点是什么"）
  - 章节定位型问题（"哪一章讲了 XX"）

## 运行评测

```bash
# recall@k（原有评测，无需 ground_truth）
python scripts/run_eval.py
python scripts/run_eval.py sanguo-yanyi

# RAGAS 5 指标评测（需补充 ground_truth，评测完成后自动生成评估文档）
python scripts/run_eval.py --ragas
python scripts/run_eval.py --ragas sanguo-yanyi
```

### RAGAS 模式输出

评测完成后，自动在 `data/eval/` 目录下生成两个文件：
- `ragas_report_<timestamp>.md`：Markdown 评估文档，含总体指标表 + 逐题明细
- `ragas_result_<timestamp>.csv`：原始数据，供后续对比分析

评测的 5 个指标：

| 指标 | 说明 |
|------|------|
| Faithfulness（忠实度） | 答案是否忠于检索内容，不产生幻觉 |
| Context Precision（上下文精度） | 检索内容中有多少是真正相关的 |
| Context Recall（上下文召回率） | ground_truth 关键信息是否被检索覆盖 |
| Answer Relevancy（答案相关性） | 答案是否直接回应了问题 |
| Context F1（派生） | Precision 与 Recall 的调和均值，无额外 LLM 开销 |

### RAGAS 依赖安装

```bash
pip install 'ragas>=0.2.0,<0.3.0' datasets
```

## 已知局限

- recall@k 的关键词匹配是简单的子串包含判断，不做同义词/语义匹配，可能低估真实召回率
- 目前只评测单书精读检索（`get_book_retriever`），未覆盖跨书问答的 RAG-Fusion + Rerank 链路
- RAGAS 评测每条样本会调用多次 LLM 打分，10 条样本约产生 50~100 次 LLM 请求，建议先用少量样本验证
- 评测集需要人工编写，暂无自动生成脚本
