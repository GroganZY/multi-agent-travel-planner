# RAG 检索质量评估报告

## 评估方法

### 两个版本

| | `run_eval.py` | `run_eval_ragas.py` |
|---|---|---|
| 检索层 | 手写 LLM-as-judge | **子类化 RAGAs**：自定义 `KeyFactsContextPrecisionRecall` 类，适配 key_facts 格式 + batch prompt 合并 |
| 生成层 | 手写 LLM-as-judge | **RAGAs 原版**：`faithfulness` + `answer_relevancy`，无需修改 |
| 依赖 | 无额外依赖 | ragas, datasets |
| 设计模式 | 独立实现 | 继承 + 重写 |

### 子类化的设计思路

```
RAGAs 原版 ContextPrecision / ContextRecall
  ├── 输入：question + contexts + reference（完整答案文本）
  └── 不符合：本项目测试集使用 key_facts 格式

KeyFactsContextPrecisionRecall（自定义子类）
  ├── 继承：RAGAs 的指标定义和 LLM-as-judge 方法论
  ├── 重写：输入从 reference 改为 key_facts + question
  ├── 优化：Precision + Recall 合并为一次 LLM 调用
  │        （RAGAs 原版分开两次调用，30 题多 30 次）
  └── 输出格式：与 RAGAs 原版对齐
```

### 为什么没有全部删掉用 RAGAs

检索层：RAGAs 的 `context_precision` / `context_recall` 需要完整参考答案文本，本项目测试集使用 key_facts 短词格式，直接使用不兼容。通过子类化保留了 RAGAs 的方法论，只替换了输入适配层和 prompt 批量化。

生成层：直接使用 RAGAs 原版 `faithfulness` 和 `answer_relevancy`，这两个指标仅需 `question + answer + contexts`，完全兼容。

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
