# RAG 检索质量评估报告

## 评估方法

五个指标全部使用 RAGAs 库原生实现，LLM 裁判使用 DeepSeek v4-pro，embedding 使用 BGE-small-zh-v1.5（本地部署）。

| 指标 | 说明 | 评估方式 |
|------|------|---------|
| Context Precision | 检索返回的 3 个 chunk 中真正相关的占比 | RAGAs (LLM-as-judge) |
| Context Recall | 关键事实被检索到的比例 | RAGAs (LLM-as-judge) |
| Faithfulness | LLM 答案每句话在检索文档中是否能找到依据 | RAGAs (LLM-as-judge) |
| Answer Relevancy | LLM 答案是否扣题 | RAGAs (LLM-as-judge + embedding) |
| Answer Correctness | LLM 答案与参考答案在事实上是否一致 | RAGAs (LLM-as-judge + embedding) |

## 评估结果（30 题全量，DeepSeek v4-pro）

| 指标 | 分数 | 说明 |
|------|------|------|
| Context Precision | **73.9%** | 检索返回 3 个 chunk，平均 2.2 个相关 |
| Context Recall | **74.2%** | 关键事实约 3/4 被检索覆盖 |
| Faithfulness | **94.6%** | 答案几乎完全基于检索文档，幻觉极少 |
| Answer Relevancy | **77.2%** | 大部分答案扣题，约 1/4 存在跑题或冗余 |
| Answer Correctness | **61.4%** | 答案与参考答案的事实一致性偏低 |

## 分析

### 检索层

**Precision 73.9% — 约 1/4 的检索结果不相关。** 根因是跨文档语义重叠：同一个关键词出现在多份文档中，无关文档的 chunk 被带进 top-3。典型 case：搜"机票可以选头等舱吗"，预订指南和平台指南的 chunk 排在 FAQ 前面。

**Recall 74.2% — 约 1/4 的关键事实漏检。** 主要发生在答案分散的场景（如"报销需要哪些材料"分散在正式报销政策和 FAQ 两个文档），一次检索 3 个 chunk 不够覆盖。

**改进方向：**
- metadata 类别过滤：检索时按文档类别预筛选，排除无关文档
- 调整 top_k 从 3 到 5：增加检索返回数量换取召回率

### 生成层

**Faithfulness 94.6% — 答案几乎不编造。** DeepSeek v4-pro 的幻觉率远低于之前测试的豆包 mini（64.5%），说明 Faithfulness 的大头取决于模型本身的可靠性。

**Answer Relevancy 77.2% — 约 1/4 答案不够扣题。** 部分答案附带过多无关的政策条款。改进方向：RAG prompt 加长度限制和扣题约束。

**Answer Correctness 61.4% — 最大的弱项。** 答案和参考答案在事实层面差异较大。这是 Faithfulness 和 Correctness 的关键差距——Faithfulness 保证答案"有根据"（94.6%），Correctness 暴露答案"不够准"（61.4%）。根因是 LLM 倾向于从检索文档里提取一段话而非精确回答，例如问"北京住宿标准"可能返回一段政策条文而不是简单的"500 元/晚"。

**改进方向：**
- RAG prompt 加"直接给出数字，不要大段引用条文"
- 对数值型问题考虑不走 LLM 生成，直接从检索结果提取数字

### 三个指标的关系

```
Faithfulness 94.6% → "每句话都有出处"（防幻觉）
Answer Relevancy 77.2% → "回答在讲正事"（防跑题）
Answer Correctness 61.4% → "回答是准确的"（防说错）

三者递进：Faithfulness 是底线（别瞎编），Relevancy 是体验（别说废话），
Correctness 是目标（别说不准）。当前系统底线扎实，体验良好，准确性仍需提升。
```

## 复现

```bash
pip install ragas datasets langchain-openai langchain-community
python evaluation/run_eval.py
```
