---
name: plan-trip
description: Use this skill when the user wants to plan a trip or asks for itinerary planning. Triggers when user says "规划行程", "安排路线", "我要去XX", "从XX到XX", or provides trip details like dates and destinations. This skill orchestrates IntentionAgent, EventCollectionAgent, and ItineraryPlanningAgent; all agents take model=model and are async.
---

# Plan Trip (行程规划)

为用户规划出行行程：意图识别 → 事项收集（出发地、目的地、日期等）→ 行程规划。所有 Agent 均使用 **model 对象**，且 **reply() 均为 async**。

## When to Use

- 用户说「规划行程」「从XX到XX」「X月X日去北京」等

## Agents（按顺序）

1. **IntentionAgent** — 识别意图与改写 query  
2. **EventCollectionAgent** — 提取出发地、目的地、日期、目的等  
3. **ItineraryPlanningAgent** — 生成行程（每日安排、交通、住宿建议等）

## 统一模型与异步

- 先创建 `OpenAIChatModel`（来自 `config.LLM_CONFIG`），再传给各 Agent 的 **model** 参数（本项目无 `model_config_name`）。
- 三个 Agent 的 `reply()` 都是 **async**，需 **await**。

## 调用示例（简化链式）

```python
import asyncio
import json
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from config import LLM_CONFIG
from agents.intention_agent import IntentionAgent
from agents.event_collection_agent import EventCollectionAgent
from agents.itinerary_planning_agent import ItineraryPlanningAgent

async def plan_trip(user_query: str):
    init_agentscope()
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": 60},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    user_msg = Msg(name="user", content=user_query, role="user")

    # 1. 意图识别
    intention_agent = IntentionAgent(name="IntentionAgent", model=model)
    intention_result = await intention_agent.reply(user_msg)
    intention_data = json.loads(intention_result.content)
    rewritten_query = intention_data.get("rewritten_query", user_query)

    # 2. 事项收集（传入 context 格式，与 OrchestrationAgent 一致）
    context = {"rewritten_query": rewritten_query, "user_preferences": {}}
    event_input = Msg(name="Orchestrator", content=json.dumps({"context": context}), role="user")
    event_agent = EventCollectionAgent(name="EventCollectionAgent", model=model)
    event_result = await event_agent.reply(event_input)
    event_data = json.loads(event_result.content) if isinstance(event_result.content, str) else event_result.content

    # 3. 行程规划（传入 previous_results，包含 event_collection 结果）
    previous_results = [{"agent_name": "event_collection", "data": event_data}]
    plan_input = Msg(
        name="Orchestrator",
        content=json.dumps({"context": context, "previous_results": previous_results}, ensure_ascii=False),
        role="user",
    )
    plan_agent = ItineraryPlanningAgent(name="ItineraryPlanningAgent", model=model)
    plan_result = await plan_agent.reply(plan_input)
    plan_data = json.loads(plan_result.content) if isinstance(plan_result.content, str) else plan_result.content
    return plan_data

# 使用
result = asyncio.run(plan_trip("规划一下2月27日从上海到北京的路程"))
# result: {"itinerary": {"title", "duration", "route", "daily_plans", "notes", ...}, "planning_complete": bool}
```

## EventCollectionAgent 输出字段（示例）

- `origin`, `destination`, `start_date`, `end_date`, `duration_days`, `trip_purpose`, `missing_info` 等

## ItineraryPlanningAgent 输出字段（示例）

- `itinerary`: `title`, `duration`, `route`, `daily_plans`, `notes`, `estimated_budget` 等
- `planning_complete`: bool

## 错误与缺失信息

- 若意图解析非 JSON，可提示用户重新描述。
- 若 `event_data` 含 `missing_info`，可提示用户补全再继续。


## 行程规划 Prompt 指南

【核心原则】
本项目为**企业差旅出行助手**，所有行程规划以商务出差为核心场景。
1. **永远提供有价值的行程规划**，即使信息不完整
2. **不要因为缺少天气、交通等细节信息就拒绝规划**
3. **根据行程目的（trip_purpose）调整规划风格**：出差聚焦工作安排，旅游可推荐景点
4. 缺失的信息可以在注意事项中提醒用户补充，但不影响主体规划

【出差规划策略】（trip_purpose 为"出差"/"商务"/"拜访"/"会议"/"培训"等）
- 白天工作时段以工作任务为核心：会议、客户拜访、商务对接
- 住宿选择：靠近工作地点、地铁沿线、预算内商务酒店
- 交通安排：往返交通 + 市内通勤（地铁/打车）到各工作点位
- 餐饮建议：工作简餐为主，商务宴请在费用标准内推荐
- **晚间及自由时段**：可以推荐当地特色街区、夜市、商圈、免费或收费景点作为可选活动，
  标注"自费"或"个人时间"。出差人员利用闲暇游览是正常的，不要禁止，只需说明
  门票等个人消费不在差旅报销范围内
- **周末/节假日若在差旅期间**：当天可规划更轻松的行程，推荐当地值得去的景点，
  住宿仍按标准报销，门票和个人消费自理
- 若调用了 rag_knowledge：提取其输出中的**具体数字**（住宿上限金额、餐饮每餐上限、交通舱位要求）
  作为刚性约束写入酒店选择和餐饮建议。**忽略**RAG 输出中的申请流程、审批规则、报销步骤、
  紧急处理、平台操作方法等内容——这些是给用户单独查询用的，不是行程规划该输出的
- 若未获得差旅标准，在注意事项中提醒用户查询目的地标准

【旅游规划策略】（trip_purpose 为"旅游"/"度假"/"探亲"等）
- 可以推荐目的地标志性景点和游览路线
- 根据季节给出户外/室内活动建议
- 一日游通常安排 2-3 个主要景点

【行程规划要点】
1. 根据行程目的和时间合理安排活动数量
2. 考虑各点位之间的交通时间和距离
3. 安排午餐、晚餐时间和推荐地点
4. 给出大致的时间安排（如 09:00-12:00, 14:00-17:00 等）
5. 提供交通方式建议（地铁、打车、步行等）

【任务】
基于已有信息生成实用的行程规划：
1. **活动安排必须匹配行程目的**，不能对所有场景都用旅游景点填充
2. 在 daily_plans 中给出详细的时间表和活动
3. 在 notes 中补充注意事项和需要确认的信息

【输出格式】(严格JSON)
{{
    "itinerary": {{
        "title": "南京往返杭州3日差旅行程",
        "duration": "3天",
        "route": "南京 -> 杭州 -> 南京",
        "daily_plans": [
            {{
                "day": 1,
                "date": "2026-07-22",
                "city": "杭州",
                "theme": "抵达与商务对接",
                "activities": [
                    {{
                        "time": "08:30-10:30",
                        "location": "南京南站→杭州东站",
                        "description": "乘坐G字头高铁前往杭州，车程约2小时",
                        "transport": "地铁至南京南站"
                    }},
                    {{
                        "time": "11:00-12:00",
                        "location": "商务酒店（地铁沿线，预算内）",
                        "description": "办理入住，稍作休整",
                        "transport": "杭州东站打车10分钟或地铁1号线"
                    }},
                    {{
                        "time": "14:00-17:30",
                        "location": "客户/合作方办公地点",
                        "description": "进行商务对接或会议（具体点位需用户确认）",
                        "transport": "根据实际点位安排地铁或打车"
                    }}
                ],
                "meals": {{ "lunch": "高铁上或杭州东站商圈简餐", "dinner": "入住酒店周边商务简餐" }}
            }}
        ],
        "notes": ["建议提前1-3天购买往返高铁票", "杭州市内通勤优先使用地铁避开拥堵"],
        "estimated_budget": "待确认（建议调用差旅标准查询）"
    }},
    "planning_complete": true
}}
