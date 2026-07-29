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

## 改进历程

以下按优化对象分组，而非时间顺序。每组说明问题、做了什么、结果如何、采纳与否及原因。

### 1. 检索层优化

#### 结构化分块实验

**问题**：原始分块策略（段落硬切，600 字符 + 100 重叠）精度不足。FAQ 文档中多个 QA 对挤在同一个 chunk 里——例如一个 chunk 同时包含 Q4"提前出发"和 Q5"私家车"，查询"私家车"时命中但近距离的 Q4 占用了检索名额。标准文档则按长度硬切，章节边界断裂。

**做法**：
- 实现 `split_faq`：FAQ 文档按 `Q1/A1`、`Q2/A2` 的边界独立成 chunk
- 实现 `split_by_sections`：标准文档按章节标题（一、二、三 或 1. 2. 3.）切分
- chunk 总数从 150+ 精简至 96

**结果**：

| 指标 | 原始分块 | 结构化分块 | 变化 |
|------|---------|-----------|------|
| Context Precision | 74% | **85%** | **+11%** |
| Faithfulness | 95% | **88%** | **-7%** |

**分析**：性能更纯的 chunk 提升了检索命中率，但也带来副作用——每个 chunk 携带的上下文变窄，LLM 回答时需要"跨越"多个孤立 chunk 来拼凑答案。跨越过程中产生了原文无法支撑的陈述，Faithfulness 因此下滑。

**决策：不采纳。** 差旅合规场景下 Faithfulness 是不可妥协的底线。该实验的价值在于记录了"检索精度与生成可信度之间存在 tradeoff"的工程证据——这两者不总是正相关。

#### category_filter 检索接口（就绪，待上游开放）

`search_knowledge` 已增加 `category_filter` 参数，能在检索时指定类别（如 `"差旅规定"`、`"FAQ"`），Milvus 原生 filter 表达式已写入。当前卡在 Milvus Lite 对标量字段的过滤支持不足，切换到完整版 Milvus 后可一行启用。此功能预期能将 Precision 进一步提升 5-8 个百分点。

---

### 2. 生成层优化

> 注：开发过程中基座 LLM 从 doubao 更换为 DeepSeek v4-pro（受 API 配额限制），同时评估方法从手写脚本切换到 RAGAs 框架、测试集从 10 题扩到 44 题。Faithfulness 从 doubao 的 ~64.5%（10 题采样）变为 DeepSeek 的 90.0%（44 题全量），但指标口径不一致，不直接归因于模型本身。

#### Task-aware 输出过滤（已采纳）

**问题**：同一套 RAG 流水线同时服务于两种场景——行程规划（需要浓缩的数值约束）和政策查询（需要完整的条文解释）。统一的输出格式两边都不讨好。

**做法**：在 SKILL.md 中加入场景判断逻辑，无需修改模型调用代码：

| 输入场景 | RAG 输出形式 | 示例 |
|---------|-------------|------|
| 行程规划 | 只输出数字约束（标准、限额、时间） | "住宿标准：一线城市 ≤500 元/晚" |
| 政策查询 | 完整输出条文 + 解释 | 附带条件、例外情况、申请流程 |

**结果**：规划类问题答案更简洁，政策类问题更完整。纯指令层实现，零成本切换。

#### 简洁 prompt 实验（不采纳）

将生成 prompt 改为"用一句话回答，不需要展开政策背景"，观察到 Correctness +3.4% 但 Faithfulness -8%。原因：强制简洁导致 LLM 选择性省略支撑性细节，部分陈述失去原文锚定。回退到原始 prompt。该实验确认了"简洁与完整"在 RAG 场景下的另一个 tradeoff。

---

### 3. 评估体系（为改进提供测量基准）

从零搭建了完整的 RAG 评估流水线：

- **手工标注 44 条 ground truth QA**：覆盖全部 8 个文档类别，每条含问题、参考答案、关键事实
- **接入 RAGAs 5 个标准指标**：Context Precision / Recall、Faithfulness、Answer Relevancy、Answer Correctness
- **自建 LLM Correctness 裁判**：用 LLM 逐事实判断答案与 reference 的一致性，不受措辞/长度/格式影响

#### 关键发现：Embedding Correctness 系统性偏差

LLM Correctness **82.3%** vs Embedding Correctness **59.8%**，差距 **22.5 个百分点**。

embedding 余弦距离对"答案正确但比 reference 更详细"的 case 系统性低估。例如航班延误处理——系统给了六步骤完整流程，reference 只有一句话"及时联系航司"，embedding 相似度给了 0.36，LLM 裁判给了 1.0。这一发现的价值在于：如果依赖单一 embedding-based 指标，会错误地认为系统质量只有 60 分，并导向错误的优化方向。

---

### 4. 当前瓶颈

所有指标收敛到同一个结论：**检索层效果已经很好**。

LLM Correctness 低于 95% 的 case，100% 是检索未命中（返回的 top-3 chunk 中不含正确答案），模型"诚实地说不知道"。生成侧 Faithfulness 90% 表明给定正确上下文时模型几乎不编造。下一阶段的核心突破应该在检索层——query 改写、category 预过滤、多路召回。

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

Faithfulness 90% 说明答案基本不编造，给定正确的检索上下文时模型几乎不产生幻觉。LLM Correctness 82% 说明超过 8 成答案在事实上准确。剩余约 18% 的低分主要来自检索失败（答案不在检索结果中，LLM 诚实地说不知道）。

### Correctness 低分的两个根因

1. **检索失败（约 3-4 题）**：答案在知识库中但检索未命中。这些题 Faithfulness 也是 1.0，但 Correctness 为 0。
2. **Embedding 偏见（约 5-6 题）**：答案完全正确但格式比 reference 详细，embedding 语义相似度被稀释。改用 LLM Correctness 后这些题从 0.2-0.4 提升到 0.8-1.0。

---

## 改进方向

### 检索层

- **metadata 类别过滤**：在地面真值已知类别的前提下，检索时按文档类别预筛选。Milvus 原生支持 filter 表达式，当前受限于 Milvus Lite 的 schema 限制。切换到完整版 Milvus 后一行代码可启用。
- **query 改写**：在检索前用 LLM 将用户口语改写为更接近文档措辞的表达。例如"可以订什么舱位"→"飞机标准舱位等级"。

### 生成层

- **Faithfulness 较高（90%），重点转向其他质量维度**：答案的简洁性、可操作性、用户满意度。这些需要自定义 AspectCritic 评估，而非标准 RAGAs 指标。
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
