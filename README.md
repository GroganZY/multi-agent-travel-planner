# Multi-Agent Travel Planner

基于豆包大模型和 AgentScope 框架的多智能体差旅规划系统。采用 Plan-and-Execute 架构，实现意图识别、双层记忆（自动降级）、RAG 知识库、联网搜索、优先级并行调度及 Redis 缓存加速。

## 核心特性

### 智能意图识别
- 基于 LLM 语义理解的多意图识别，支持行程规划、记忆查询、偏好管理、知识问答、信息查询、事项收集
- 自然语言理解，自动改写口语化 query，上下文消歧

### 双层记忆体系（生产级存储）
- **短期记忆**：Redis List 滑动窗口（10 轮），TTL 1 小时自动过期；Redis 不可用时自动降级为内存 Python list
- **长期记忆**：PostgreSQL 四表持久化，preferences 采用 JSONB + GIN 索引；PostgreSQL 不可用时自动降级为本地 JSON 文件
- **Redis 缓存层**：偏好热数据 Hash 缓存（Cache-Aside，仅变更时失效）+ LLM 摘要 String 缓存（content-hash 键，同会话内复用）
- 智能识别偏好追加/覆盖（"我还喜欢如家"→ 追加，"我搬家到上海了"→ 覆盖）

### RAG 知识库
- Milvus Lite 向量数据库 + BGE-small-zh-v1.5 本地部署
- 滑动窗口分块（600 字/块，100 字重叠）+ 余弦相似度检索（Top-K=3）
- 8 类差旅文档：差旅标准、报销政策、预订指南、FAQ、紧急处理、平台指南、城市指南、环保倡议

### 优先级并行调度
- IntentionAgent → OrchestrationAgent → 6 个可插拔 Skill Agent
- 同优先级 Agent 并行执行（asyncio.gather），不同优先级串行依赖

### 插件化架构
- Skill Plugins 位于 `.claude/skills/`，LazyAgentRegistry 自动扫描动态加载
- Progressive Disclosure：意图识别阶段仅加载元数据，执行时按需加载

### 信息缺失自动追问
- Orchestration 结束后自动检测 EventCollection 的 missing_info，代码驱动追问决策（每会话最多 2 次）
- LLM 仅润色追问话术，确保不循环追问
- 超限后提示用户"将按常见默认值规划"，不再追问，避免死循环

### 稳定性保障
- 熔断器（三态 CLOSED/OPEN/HALF_OPEN）+ 指数退避重试 + 健康检查
- 六层 JSON 解析降级策略，覆盖 LLM 输出异常

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **如果不需要数据库**（仅用 JSON 文件 + 内存存储）：可以跳过 `asyncpg` 和 `redis` 两个可选依赖，也不用启动 Docker。系统会自动探测——没有就降级，不影响功能。

### 2. 配置 API Key

编辑 `config.py`：

```python
LLM_CONFIG = {
    "api_key": "your-api-key-here",
    "model_name": "doubao-seed-2-0-lite-260428",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
}
```

### 3. 初始化知识库

```bash
python .claude/skills/ask-question/script/init_knowledge_base.py
```

### 4. 启动

```bash
python cli.py
```

> 系统默认自动探测 PostgreSQL/Redis——Docker 开着就用，没开自动降级为本地 JSON 文件 + 内存存储，零配置。若想强制关闭 DB 模式，将 `config.py` 中 `DB_CONFIG["enabled"]` 或 `CACHE_CONFIG["enabled"]` 设为 `False`。

### 可选：启用生产级存储

```bash
docker compose up -d                           # 一键启动 PostgreSQL + Redis
pip install asyncpg redis                       # 安装数据库驱动
python scripts/migrate_to_db.py                 # （可选）迁移旧 JSON 数据
python cli.py                                   # 启动，自动走 DB 模式
```

---

## 系统架构

```
用户输入
   ↓
IntentionAgent (意图识别) → 多意图解析 + Query 改写
   ↓
OrchestrationAgent (协调调度)
   ├── Priority 1 并行: MemoryQuery / EventCollection / Preference / InfoQuery / RAGKnowledge
   └── Priority 2 串行: ItineraryPlanning
   ↓
结果聚合 + 记忆更新 → 缺失信息追问 → 人性化展示
```

---

## CLI 命令

| 命令 | 说明 |
|------|------|
| `help` | 帮助信息 |
| `status` | 当前状态和记忆统计 |
| `health` | LLM 服务健康检查 + 熔断器状态 |
| `clear` | 清空短期记忆 |
| `history` | 查看历史行程 |
| `preferences` | 查看用户偏好 |
| `exit` | 退出 |

---

## 数据库设计

### PostgreSQL（长期记忆）

| 表 | 说明 | 关键字段 |
|---|------|---------|
| `users` | 用户标识 | user_id (PK) |
| `preferences` | 用户偏好 | pref_value (JSONB), UNIQUE(user_id, type), GIN 索引 |
| `chat_history` | 聊天记录 | role (CHECK约束), 按 session_id 隔离 |
| `trip_history` | 行程历史 | origin/destination/start_date/purpose |

### Redis（短期记忆 + 缓存）

| Key | 结构 | 用途 | TTL |
|-----|------|------|-----|
| `stm:{user}:{session}` | List | 短期记忆滑动窗口 | 1h |
| `prefs:{user}` | Hash | 偏好热数据 (Cache-Aside) | 24h |
| `summary:{user}:{hash}` | String | LLM 摘要 (content-hash key) | 30min |

---

## 技术栈

- **AgentScope 1.0.16** — 多智能体框架
- **豆包大模型** (doubao-seed-2-0-lite) — LLM
- **PostgreSQL 16** — 长期记忆持久化（可选，默认 JSON 文件）
- **Redis 7** — 短期记忆 + 缓存加速（可选，默认 Python list）
- **Milvus Lite** — 向量数据库（本地嵌入式）
- **BGE-small-zh-v1.5** — 中文 Embedding 模型（本地部署）
- **DDGS** — DuckDuckGo 联网搜索
- **wttr.in** — 免费天气 API
- **Rich** — CLI 终端界面

---

## 项目结构

```
├── agents/                       # 核心编排层
│   ├── intention_agent.py        # 意图识别
│   ├── orchestration_agent.py    # 调度协调
│   └── lazy_agent_registry.py    # 插件注册器（懒加载）
├── .claude/skills/               # 6 个 Skill Plugin
│   ├── ask-question/             # RAG 知识库
│   ├── event-collection/         # 事项收集
│   ├── plan-trip/                # 行程规划
│   ├── preference/               # 偏好管理
│   ├── query-info/               # 信息查询（天气+搜索）
│   └── memory-query/             # 记忆查询
├── context/                      # 双层记忆体系
│   ├── memory_manager.py         # 统一入口 + 缓存层
│   ├── short_term_memory.py      # 短期记忆（Redis+降级）
│   └── long_term_memory.py       # 长期记忆（PostgreSQL+降级）
├── utils/                        # 工具
│   ├── circuit_breaker.py        # 熔断器
│   ├── llm_resilience.py         # 退避重试+健康检查
│   ├── json_parser.py            # 6 层 JSON 降级解析
│   └── skill_loader.py           # Skill 元数据加载
├── migrations/init.sql           # PostgreSQL 建表 DDL
├── docker-compose.yml            # PostgreSQL + Redis 一键启动
├── scripts/migrate_to_db.py      # JSON → PostgreSQL 数据迁移
├── cli.py                        # CLI 入口
├── config.py                     # 全局配置
└── requirements.txt
```

---

## 评估结果

### RAG 检索质量

44 题手工标注 ground truth，覆盖全部 8 个文档类别。DeepSeek v4-pro + BGE-small-zh-v1.5 评估结果：

| 指标 | 分数 | 评估方式 |
|------|------|---------|
| Context Precision | 72.7% | RAGAs (LLM-as-judge) |
| Context Recall | 75.2% | RAGAs (LLM-as-judge) |
| Faithfulness | 90.0% | RAGAs (LLM-as-judge) |
| LLM Correctness | **82.3%** | 自建 LLM 逐事实裁判 |
| Answer Relevancy | 68.1% | RAGAs (LLM+embedding) |
| Answer Correctness (embedding) | 59.8% | RAGAs (embedding 余弦距离) |

> **关键发现**：LLM Correctness 比 Embedding Correctness 高 22.5 个百分点。embedding 余弦距离对"答案正确但比 reference 更详细"的 case 系统性低估。后续以 LLM Correctness 为质量基准。

完整的实验记录（结构化分块 tradeoff、prompt 约束实验、LLM Correctness 裁判）详见 [evaluation/EVAL_REPORT.md](evaluation/EVAL_REPORT.md)。

### 意图分类准确率

30 题手动标注 ground truth，覆盖全部 6 个 Skill Agent 及组合场景。评估 IntentionAgent 的路由决策质量：

| 指标 | 分数 | 说明 |
|------|------|------|
| Agent 调度精确匹配率 | **70.0%** | 预测的 agent 集合与期望完全一致 |
| Agent 调度 Precision | 88.5% | 调度了的 agent 中正确的比例 |
| Agent 调度 Recall | 96.7% | 该调度的 agent 中被调度的比例 |
| 实体提取准确率 | 86.0% | key_entities 字段精确匹配率 |

**关键发现**：高 Recall（96.7%）说明系统几乎不会漏掉该调度的 agent；中 Precision（88.5%）说明存在一定程度的过度调度（主要在行程规划场景中错误附加 preference agent，以及 memory_query 被误判为 information_query 的边界情况）。

详见 [evaluation/results/intent_eval_*.md](evaluation/results/intent_eval_20260729_124723.md)。

---

## 注意事项

- **API Key**：`config.py` 中 `LLM_CONFIG["api_key"]` 从环境变量 `LLM_API_KEY` 读取，未设置时默认为空字符串。启动前需通过环境变量或直接修改 `config.py` 填入豆包 API Key。`DB_CONFIG["password"]` 为 Docker 容器内 PostgreSQL 的本地密码，不涉及生产密钥。
- **`data/memory/*.json`**：为本地测试产生的示例记忆数据（用户偏好、聊天记录、行程历史），仅作格式参考。生产环境启用 PostgreSQL 后不再使用 JSON 文件存储。
- **Embedding 模型**：BGE-small-zh-v1.5 需放置于 `data/models/bge-small-zh-v1.5/`。首次运行 RAG 知识库初始化脚本时会自动下载（需联网），或手动下载后放入该路径。

## 许可证

MIT License
