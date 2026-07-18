# RAG 检索质量评估报告

## 评估设计

### 测试数据
- 30 条 ground truth QA，手工构建，覆盖知识库全部 8 个文档类别
- 每条包含：问题、预期文档、关键事实、所属类别
- 类别分布：差旅规定 8 / FAQ 6 / 报销规定 5 / 紧急处理 3 / 预订指南 3 / 环保倡议 2 / 城市指南 2 / 平台指南 1

### 评估指标

| 指标 | 说明 | 评估方式 |
|------|------|---------|
| Context Precision | 检索返回的 3 个 chunk 中真正相关的占比 | 自动（metadata 匹配 + 关键事实命中） |
| Context Recall | ground truth 中的关键事实在检索结果中的覆盖率 | 自动（子串匹配） |
| Category Hit | 至少命中 1 个正确类别文档的比例 | 自动 |
| MRR | 第一个相关 chunk 的倒数排名均值 | 自动 |
| Faithfulness | LLM 答案每句话在检索文档中是否能找到依据 | LLM-as-judge（doubao-seed-2.0-mini） |
| Answer Relevancy | LLM 答案是否扣题 | LLM-as-judge |

### 技术栈
- 检索评估：本地运行，无需外部 API，3 秒完成 30 题
- 生成评估：调用豆包 API 生成答案 + LLM-as-judge 打分，10 题采样，每题间隔 15 秒适配免费 API 限额

---

## 评估结果

### 检索质量（30 题全量）

| 指标 | 分数 |
|------|------|
| Context Precision | **80.00%** |
| Context Recall | **89.44%** |
| Category Hit | **100.00%** |
| MRR | **0.944** |

### 按文档类别分拆

| 类别 | 题数 | Precision | Recall |
|------|------|-----------|--------|
| 环保倡议 | 2 | 100% | 100% |
| 紧急处理 | 3 | 100% | 100% |
| 城市指南 | 2 | 83% | 100% |
| 报销规定 | 5 | 80% | 67% |
| FAQ | 6 | 78% | 92% |
| 差旅规定 | 8 | 75% | 88% |
| 预订指南 | 3 | 67% | 100% |
| 平台指南 | 1 | 67% | 100% |

### 生成质量（10 题采样，doubao-seed-2.0-mini）

| 指标 | 分数 |
|------|------|
| Faithfulness | **64.5%** |
| Answer Relevancy | **74.0%** |

---

## 分析

### 检索层
- **100% Category Hit** 说明 BGE-small-zh-v1.5 嵌入 + Milvus 余弦相似度对 8 文档 150 chunk 的知识库完全胜任
- **Precision 偏低集中在跨文档语义重叠场景**：同一关键词（"住宿""机票""舱位"）出现在多份文档中，无关文档的 chunk 抢占检索位置
- **Recall 偏低集中在"报销规定"（67%）**：答案分散在正式报销政策和 FAQ 两份文档里，一次检索 3 个 chunk 不够覆盖

### 生成层
- mini 模型比 lite 更容易产生幻觉（Faithfulness 64.5% vs 预期 lite 的 ~80%）
- 数值型和列举型问题（"多少钱""有哪些材料"）Faithfulness 最低，模型倾向于补全未检索到的细节

### 优化方向
1. **metadata 类别过滤**：检索时只搜相关文档类别，消除跨文档干扰，预计 Precision 提升至 90%+
2. **RAG prompt 引用约束**：要求 LLM 每句标注来源，对无来源的陈述做后验证
3. **数值型问题直接提取**：对于"XX 标准是多少"类问题，直接从检索 chunk 中提取数字，不经过 LLM 生成

---

## 测试过程记录

### 踩坑记录
1. **免费 API 限流**：豆包 lite 模型额度用尽被暂停，切换至 mini 模型后继续
2. **mini 模型未开通**：需在火山引擎控制台手动激活 mini 模型
3. **流式响应解析**：豆包流式每个 chunk 是累积全文而非增量，`+=` 拼接导致内容重复，改为取最后一个完整 chunk
4. **thinking mode**：mini 模型默认开启 thinking token，需在 `generate_kwargs` 中传入 `{"thinking": {"type": "disabled"}}` 关闭

### 可复现性
```bash
# 设置 API Key
export LLM_API_KEY=your-key

# 运行评估
python evaluation/run_eval.py

# 结果保存至 evaluation/results/eval_result.json
```
