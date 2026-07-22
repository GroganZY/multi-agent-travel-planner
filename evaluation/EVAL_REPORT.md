# RAG 检索质量评估报告

## 评估方法

五个指标使用 RAGAs 库原生实现，外加一个自建 LLM Correctness 裁判。LLM 裁判使用 DeepSeek v4-pro，embedding 使用 BGE-small-zh-v1.5（本地部署）。

| 指标 | 说明 | 评估方式 |
|------|------|---------|
| Context Precision | 检索返回的 chunk 中真正相关的占比 | RAGAs (LLM-as-judge) |
| Context Recall | 关键事实被检索到的比例 | RAGAs (LLM-as-judge) |
| Faithfulness | 答案每句话在检索文档中是否能找到依据 | RAGAs (LLM-as-judge) |
| Answer Relevancy | 答案是否扣题 | RAGAs (LLM-as-judge + embedding) |
| Answer Correctness | 答案与参考答案在事实上是否一致 | RAGAs (embedding 语义相似度) |
| **LLM Correctness** | **同上，但由 LLM 逐事实判断，不受措辞/长度/格式影响** | **自建 (LLM-as-judge)** |

### 为什么加了 LLM Correctness

RAGAs 0.2.x 的 Answer Correctness 底层是 embedding 语义相似度——把 LLM 答案和 reference 做余弦距离。这导致"答案正确但比 reference 更长/更详细"时被系统性低估。LLM Correctness 用 LLM 直接判断事实一致性，排除了格式和长度的干扰。

### 测试数据

44 条手工标注的 ground truth QA，覆盖全部 8 个文档类别。每条包含问题、参考答案、关键事实和所属文档。类别分布：差旅规定 9 / 报销规定 9 / FAQ 7 / 预订指南 6 / 紧急处理 6 / 城市指南 3 / 环保倡议 3 / 平台指南 1。

---

## 评估结果（44 题全量，DeepSeek v4-pro）

| 指标 | 分数 | 说明 |
|------|------|------|
| Context Precision | **72.7%** | 检索返回 chunk 中约 3/4 真正相关 |
| Context Recall | **75.2%** | 关键事实约 3/4 被检索覆盖 |
| Faithfulness | **90.0%** | 答案高度基于检索文档，幻觉率低 |
| Answer Relevancy | **68.1%** | 约 2/3 答案扣题，部分附带冗余政策条文 |
| Answer Correctness (embedding) | 59.8% | RAGAs 原版，受措辞差异影响大 |
| **LLM Correctness** | **82.3%** | **逐事实判断，更贴近真实质量** |

### 关键对比：两种 Correctness 差 22.5 个百分点

```
Embedding Correctness: 59.8%  ← 认为"只有 6 成对"
LLM Correctness:       82.3%  ← 实际超过 8 成对
```

差距最大的 case：答案完全正确但格式更详细（如航班延误处理给了六步骤，reference 只需一句话），embedding 给了 0.36，LLM 裁判给了 1.0。Embedding Correctness 系统性低估了详细但正确的答案。

---

## 分析

### 检索层（Precision 72.7%，Recall 75.2%）

检索整体可用，但有明显的跨文档语义重叠问题。FAQ 和正式标准文档在语义空间里竞争——同一个关键词出现在多份文档中时，检索到的 chunk 可能来自无关文档。典型 case：#3 "航班舱位"的答案在差旅标准里，但预订指南的"如何订机票"占用了检索名额。

### 生成层（Faithfulness 90.0%，LLM Correctness 82.3%）

Faithfulness 90% 说明答案基本不编造——DeepSeek v4-pro 的幻觉率显著低于此前测试的 doubao-mini（64.5%）。LLM Correctness 82% 说明超过 8 成答案在事实上准确。剩余约 18% 的低分主要来自检索失败（答案不在检索结果中，LLM 诚实地说不知道）。

### Correctness 低分的两个根因

1. **检索失败（约 3-4 题）**：答案在知识库中但检索未命中。这些题 Faithfulness 也是 1.0（LLM 诚实地说不知道），但 Correctness 为 0。
2. **Embedding 偏见（约 5-6 题）**：答案完全正确但格式比 reference 详细，embedding 语义相似度被稀释。改用 LLM Correctness 后这些题从 0.2-0.4 提升到 0.8-1.0。

---

## 改进实验

### 实验一：结构化分块

**做法**：FAQ 文档按 QA 对切分（每对独立成 chunk），标准文档按章节标题切分。chunk 从 150+ 精简为 96 个。

**结果**：Precision 从 74% 提升到 85%（+11%），但 Faithfulness 从 95% 降到 88%（-7%）。更小的 chunk 让 LLM 缺少上下文，部分陈述失去原文支撑。

**决策**：不采纳。差旅合规场景下 Faithfulness 是底线指标，不能为检索精度牺牲。实验保留作为"检索-生成 tradeoff"的工程案例。

### 实验二：简洁 prompt 约束

**做法**：RAG 生成 prompt 从"请直接回答"改为"用一句话回答，不需要展开政策背景"。

**结果**：Correctness +3.4%，但 Faithfulness 暴跌 8 个点。简洁和完整之间存在明显 tradeoff。

**决策**：不采纳。还原 prompt 为原版。

### 实验三：LLM Correctness 裁判

**做法**：在 RAGAs embedding-based Correctness 之外，增加一个自建的 LLM 裁判——直接判断答案和 reference 在事实上是否一致，不受措辞/长度/格式影响。

**结果**：LLM Correctness 82.3% vs Embedding Correctness 59.8%，差距 22.5 个百分点。验证了 embedding 指标对详细答案有系统性偏见的假设。

**决策**：采纳。作为正式质量指标之一，和 RAGAs 五指标并列。

---

## 改进方向

### 检索层

- **metadata 类别过滤**：在地面真值已知类别的前提下，检索时按文档类别预筛选。Milvus 原生支持 filter 表达式，当前受限于 Milvus Lite 的 schema 限制。切换到完整版 Milvus 后一行代码可启用。
- **query 改写**：在检索前用 LLM 将用户口语改写为更接近文档措辞的表达。例如"可以订什么舱位"→"飞机标准舱位等级"。

### 生成层

- **Faithfulness 已接近天花板（90%），重点转向其他质量维度**：答案的简洁性、可操作性、用户满意度。这些需要自定义 AspectCritic 评估，而非标准 RAGAs 指标。
- **Correctness 的剩余 gap 全部来自检索失败**：修了检索层后 Correctness 自然上升。

### 评估体系

- **LLM Correctness 作为正式指标**：比 embedding-based 版本更真实反映系统质量
- **后续可加业务维度**：如"答案是否可直接用于行程规划""数值是否具体不模糊"

---

## 复现

```bash
pip install ragas datasets langchain-openai langchain-community
python evaluation/run_eval.py
```
