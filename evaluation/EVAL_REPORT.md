# RAG 检索质量评估报告

## 评估方法

### 两个版本

| | `run_eval.py` | `run_eval_ragas.py` |
|---|---|---|
| 检索层 | 手写 LLM-as-judge batch prompt | **RAGAs 原版** `context_precision` / `context_recall` |
| 生成层 | 手写 LLM-as-judge | **RAGAs 原版** `faithfulness` / `answer_relevancy` |
| 依赖 | 无额外依赖 | ragas, datasets |
| 区别 | 检索层 Precision+Recall 合并为一次 LLM 调用，省 Token | 四个指标全部调用 RAGAs 库，和业界标准完全对齐 |

### 为什么 run_eval.py 保留了手写版

唯一的工程理由：**检索层的 batch prompt**。RAGAs 的 `context_precision` 和 `context_recall` 是两次独立的 LLM 调用，30 题就是 60 次。手写版将它们合并成一次调用（同一个 prompt 同时评估 chunk 相关性和事实覆盖），30 题只需 30 次。在免费 API 限额下这个差别有意义。

方法论上没有优劣——两个版本用同一个 LLM 判断同样的内容，结果应该一致。

---

## 评估设计

### 测试数据
- 30 条 ground truth QA，手工构建，覆盖知识库全部 8 个文档类别
- 每条包含：问题、预期文档、关键事实、所属类别
- 类别分布：差旅规定 8 / FAQ 6 / 报销规定 5 / 紧急处理 3 / 预订指南 3 / 环保倡议 2 / 城市指南 2 / 平台指南 1

### 评估指标

| 指标 | 说明 | 评估方式 |
|------|------|---------|
| Context Precision | 检索返回的 3 个 chunk 中真正相关的占比 | LLM 逐 chunk 判断 |
| Context Recall | 关键事实在检索结果中被覆盖的比例 | LLM 逐 fact 判断 |
| MRR | 第一个相关 chunk 的倒数排名均值 | 由 Precision 结果推导 |
| Faithfulness | LLM 答案每句话在检索文档中是否能找到依据 | LLM-as-judge |
| Answer Relevancy | LLM 答案是否扣题 | LLM-as-judge |

---

## 评估结果

> 注：以下为上一轮规则评估的基线结果，LLM 评估需重新运行（`python evaluation/run_eval.py`）后更新。

### 检索质量（30 题全量，规则评估 baseline）

| 指标 | 分数 |
|------|------|
| Context Precision | **80.00%** |
| Context Recall | **89.44%** |
| MRR | **0.944** |

### 生成质量（10 题采样，LLM-as-judge）

| 指标 | 分数 |
|------|------|
| Faithfulness | **64.5%** |
| Answer Relevancy | **74.0%** |

---

## 分析

### 检索层
- 规则评估中 Precision 80% 偏低集中在跨文档语义重叠场景——同一关键词出现在无关文档中也触发了子串匹配。LLM 评估预计 Precision 会更准确，因为语义判断能排除"提到500元但不是在讲住宿标准"的干扰
- Recall 89% 说明大部分关键事实能被检索到

### 生成层
- Faithfulness 64.5%：mini 模型补充了检索文档中没有的细节，约 1/3 的答案内容属于模型推测
- Relevancy 74%：大部分答案扣题，少数问题（舱位类型、报销材料）完全跑偏

### 优化方向
1. **metadata 类别过滤**：检索时按文档类别预过滤，减少无关文档干扰
2. **RAG prompt 引用约束**：要求 LLM 每句标注来源，压制幻觉
3. **数值型问题直接提取**：标准化数字（住宿上限、餐饮标准）的查询不走 LLM 生成，直接从检索 chunk 提取

---

## 复现

```bash
export LLM_API_KEY=your-key
python evaluation/run_eval.py
# 结果 → evaluation/results/eval_result.json
```
