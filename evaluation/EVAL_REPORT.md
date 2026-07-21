# RAG 检索质量评估报告

## 评估方法

五个指标全部使用 RAGAs 库原生实现，LLM 裁判使用 DeepSeek v4-pro。

| 指标 | 说明 | 评估方式 |
|------|------|---------|
| Context Precision | 检索返回的 3 个 chunk 中真正相关的占比 | RAGAs (LLM-as-judge) |
| Context Recall | 关键事实被检索到的比例 | RAGAs (LLM-as-judge) |
| Faithfulness | LLM 答案每句话在检索文档中是否能找到依据 | RAGAs (LLM-as-judge) |
| Answer Relevancy | LLM 答案是否扣题 | RAGAs (LLM-as-judge) |
| Answer Correctness | LLM 答案与参考答案在事实上是否一致 | RAGAs (LLM-as-judge) |

所有指标通过 `run_eval.py` 一键运行，输入数据为 `ground_truth.json`（30 条手工标注 QA，含完整 reference 答案）。

---

## 评估结果（30 题全量，DeepSeek v4-pro）

| 指标 | 分数 | 说明 |
|------|------|------|
| Context Precision | **73.9%** | 检索返回 3 个 chunk，平均 2.2 个相关 |
| Context Recall | **79.2%** | 关键事实约 4/5 被检索覆盖 |
| Faithfulness | **97.0%** | 答案几乎完全基于检索文档，幻觉极少 |
| Answer Relevancy | — | 本版本 ragas 兼容性问题，待排查 |
| Answer Correctness | — | 同上 |

---

## 分析

### 检索层

**Precision 73.9% — 约 1/4 的检索结果不相关。** 根因是跨文档语义重叠：同一个关键词（"住宿""机票""舱位"）出现在多份文档中，但只有一份包含答案。Milvus 在全部 8 个文档中搜索时，无关文档的 chunk 因为语义相似被带进 top-3。典型 case：搜"机票可以选头等舱吗"，预订指南和平台指南的 chunk 排在 FAQ 前面，后者才是真正的答案。

**Recall 79.2% — 约 1/5 的关键事实漏检。** 主要发生在答案分散的场景（如"报销需要哪些材料"的答案被拆在正式报销政策和 FAQ 两个文档里）。一次检索 3 个 chunk 不够同时覆盖两处。

**改进方向：**
- metadata 类别过滤：检索时按文档类别预筛选，将无关文档排除在检索范围外
- 调整 top_k 从 3 到 5：增加检索返回数量换取召回率，代价是 Precision 可能进一步下降

### 生成层

**Faithfulness 97.0% — 答案几乎不编造。** 这个数比之前的豆包 mini（64.5%）大幅提升。原因有二：一是 DeepSeek v4-pro 本身的幻觉率远低于 doubao-mini；二是 RAG prompt 写了"严格基于知识库"且 temperature=0.1 进一步压制了创造性输出。这也说明 Faithfulness 的大头取决于模型本身——换了强模型，数字翻了一倍。

**改进方向：** Faithfulness 已接近天花板。重点转向其他维度——答案的简洁性、可操作性、用户满意度。这些需要自定义 AspectCritic 评估，而非标准 RAGAs 指标。

---

## 需处理的问题

1. **Answer Relevancy / Answer Correctness 返回 nan**。ragas 0.2.x 这两个指标的 LLM 调用路径与 Context Precision/Faithfulness 不同，monkey-patch 未覆盖。需升级 ragas 至 0.4+ 或单独调试这两个指标的 LLM 配置。

2. **答案里的阿里内容**。第 10 题"发票抬头"的答案仍包含"阿里巴巴（中国）有限公司"——知识库文档本身未完全清洗。需重新跑一次 init_knowledge_base.py 用清理后的文档重建向量库。

---

## 复现

```bash
pip install ragas datasets langchain-openai langchain-community
python evaluation/run_eval.py
```
