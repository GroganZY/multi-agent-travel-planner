#!/usr/bin/env python
"""
意图分类与 Agent 调度评估

评估 IntentionAgent 的三个维度：
1. Agent 调度准确率（主指标）——调度的 agent 是否和预期一致
2. 实体提取准确率（辅助指标）——key_entities 中的字段是否匹配
3. 场景级分析——按场景标签统计准确率

用法：
    python evaluation/eval_intent.py

输出：
    evaluation/results/intent_eval_<timestamp>.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def load_ground_truth(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_intention_agent():
    from config import LLM_CONFIG
    from config_agentscope import init_agentscope
    from agentscope.model import OpenAIChatModel
    from agents.intention_agent import IntentionAgent

    init_agentscope()
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={
            "base_url": LLM_CONFIG["base_url"],
        },
        generate_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        temperature=0.1,
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    return IntentionAgent(name="IntentionEval", model=model)


async def main():
    gt_path = Path(__file__).parent / "intent_ground_truth.json"
    gt = load_ground_truth(str(gt_path))
    print(f"Loaded {len(gt)} ground truth queries\n")

    agent = init_intention_agent()
    from agentscope.message import Msg

    results = []
    errors = []

    for idx, item in enumerate(gt):
        query = item["query"]
        print(f"  [{item['id']:2d}] {query[:30]}...", end=" ")

        try:
            import asyncio
            msg = Msg(name="User", content=query, role="user")
            raw = await agent(msg)
            pred = json.loads(raw.content)

            # 提取预测的 agent_name 列表
            predicted_agents = sorted(set(
                s["agent_name"] for s in pred.get("agent_schedule", [])
            ))
            expected_agents = sorted(item["expected"]["agents"])

            # Agent 调度匹配
            pred_set = set(predicted_agents)
            exp_set = set(expected_agents)
            agent_exact = pred_set == exp_set
            agent_precision = len(pred_set & exp_set) / len(pred_set) if pred_set else 0
            agent_recall = len(pred_set & exp_set) / len(exp_set) if exp_set else 0

            # 实体提取匹配
            pred_entities = pred.get("key_entities", {}) or {}
            exp_entities = item["expected"].get("entities", {}) or {}
            entity_matches = []
            entity_total = 0
            for key, exp_val in exp_entities.items():
                if exp_val:  # 只检查有期望值的字段
                    entity_total += 1
                    pred_val = pred_entities.get(key, "")
                    entity_matches.append(1 if pred_val == exp_val else 0)
            entity_acc = sum(entity_matches) / entity_total if entity_total > 0 else None

            results.append({
                "id": item["id"],
                "query": item["query"],
                "scenario": item["scenario"],
                "tags": item["tags"],
                "predicted_agents": predicted_agents,
                "expected_agents": expected_agents,
                "agent_exact": agent_exact,
                "agent_precision": agent_precision,
                "agent_recall": agent_recall,
                "entity_acc": entity_acc,
                "predicted_entities": pred_entities,
                "expected_entities": exp_entities,
                "error": None,
            })

            status = "✓" if agent_exact else "✗"
            print(f"{status}  {predicted_agents}")

            if idx < len(gt) - 1:
                await asyncio.sleep(3)  # rate limit 保护

        except Exception as e:
            errors.append({"id": item["id"], "query": query, "error": str(e)})
            results.append({
                "id": item["id"],
                "query": query,
                "scenario": item.get("scenario", ""),
                "tags": item.get("tags", []),
                "predicted_agents": [],
                "expected_agents": item["expected"]["agents"],
                "agent_exact": False,
                "agent_precision": 0,
                "agent_recall": 0,
                "entity_acc": None,
                "predicted_entities": {},
                "expected_entities": item["expected"].get("entities", {}),
                "error": str(e),
            })
            print(f"✗ ERROR: {e}")

    # ── 计算汇总指标 ──────────────────────────────────────
    total = len(results)
    exact_ok = sum(1 for r in results if r["agent_exact"])
    total_prec = sum(r["agent_precision"] for r in results) / total
    total_recall = sum(r["agent_recall"] for r in results) / total
    entity_scores = [r["entity_acc"] for r in results if r["entity_acc"] is not None]
    avg_entity = sum(entity_scores) / len(entity_scores) if entity_scores else 0

    # 按场景统计
    from collections import Counter
    scenario_stats = Counter()
    scenario_ok = Counter()
    for r in results:
        for tag in r["tags"]:
            scenario_stats[tag] += 1
            if r["agent_exact"]:
                scenario_ok[tag] += 1

    # ── 生成混淆分析 ──────────────────────────────────────
    all_agents = sorted(set(
        a for r in results for a in r["expected_agents"]
    ))
    confusion = {}
    for agent_name in all_agents:
        confusion[agent_name] = {"correct": 0, "total": 0, "missed": 0, "extra": 0}

    for r in results:
        for a in set(r["expected_agents"]):
            if a not in confusion:
                confusion[a] = {"correct": 0, "total": 0, "missed": 0, "extra": 0}
            confusion[a]["total"] += 1
            if a in r["predicted_agents"]:
                confusion[a]["correct"] += 1
            else:
                confusion[a]["missed"] += 1
        for a in set(r["predicted_agents"]):
            if a not in confusion:
                confusion[a] = {"correct": 0, "total": 0, "missed": 0, "extra": 0}
            if a not in r["expected_agents"]:
                confusion[a]["extra"] += 1

    # ── 错误详情 ──────────────────────────────────────────
    failures = [r for r in results if not r["agent_exact"]]

    # ── 生成报告 ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"intent_eval_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 意图分类与 Agent 调度评估报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 总体统计
        f.write("## 总体指标\n\n")
        f.write(f"| 指标 | 分数 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| Agent 调度精确匹配率 | **{exact_ok}/{total} ({exact_ok/total*100:.1f}%)** |\n")
        f.write(f"| Agent 调度 Precision（宏平均） | **{total_prec*100:.1f}%** |\n")
        f.write(f"| Agent 调度 Recall（宏平均） | **{total_recall*100:.1f}%** |\n")
        f.write(f"| 实体提取准确率（含实体字段） | **{avg_entity*100:.1f}%** （{len(entity_scores)} 题有实体） |\n")
        f.write(f"| 执行报错 | {len(errors)} 题 |\n\n")
        if errors:
            f.write("### 报错详情\n\n")
            for e in errors:
                f.write(f"- [{e['id']}] {e['query'][:40]}: `{e['error']}`\n")
            f.write("\n")

        # 每个 Agent 的调度准确率
        f.write("## 各 Agent 调度统计\n\n")
        f.write("| Agent | 应出现次数 | 正确调度 | 遗漏 | 多余调度 | 准确率 |\n")
        f.write("|------|-----------|---------|------|---------|--------|\n")
        for a in all_agents:
            if confusion[a]["total"] == 0:
                continue
            rate = confusion[a]["correct"] / confusion[a]["total"]
            f.write(f"| {a} | {confusion[a]['total']} | {confusion[a]['correct']} | {confusion[a]['missed']} | {confusion[a]['extra']} | {rate*100:.1f}% |\n")
        f.write("\n")

        # 场景级分析
        f.write("## 场景级分析\n\n")
        f.write("| 场景标签 | 题数 | 准确数 | 准确率 |\n")
        f.write("|---------|------|--------|--------|\n")
        for tag in sorted(scenario_stats.keys()):
            total_tag = scenario_stats[tag]
            ok_tag = scenario_ok[tag]
            f.write(f"| {tag} | {total_tag} | {ok_tag} | {ok_tag/total_tag*100:.1f}% |\n")
        f.write("\n")

        # 错误详情
        f.write("## 调度失败详情\n\n")
        if failures:
            for r in failures:
                f.write(f"### [{r['id']}] {r['query'][:50]}\n\n")
                f.write(f"- 场景: {r['scenario']}\n")
                f.write(f"- 预期 agent: `{r['expected_agents']}`\n")
                f.write(f"- 预测 agent: `{r['predicted_agents']}`\n")
                if r.get("error"):
                    f.write(f"- 报错: `{r['error']}`\n")
                expected_set = set(r["expected_agents"])
                predicted_set = set(r["predicted_agents"])
                missed = expected_set - predicted_set
                extra = predicted_set - expected_set
                if missed:
                    f.write(f"- 遗漏: `{sorted(missed)}`\n")
                if extra:
                    f.write(f"- 多余: `{sorted(extra)}`\n")
                f.write("\n")
        else:
            f.write("全部精确匹配，无调度错误。\n\n")

        # 每题详情
        f.write("## 每题详情\n\n")
        f.write("| ID | query | 预期 agent | 预测 agent | 匹配 |\n")
        f.write("|----|-------|-----------|-----------|------|\n")
        for r in results:
            icon = "✅" if r["agent_exact"] else "❌"
            f.write(f"| {r['id']} | {r['query'][:30]} | {r['expected_agents']} | {r['predicted_agents']} | {icon} |\n")

    print(f"\n报告已保存: {report_path}")
    print(f"\n{'='*60}")
    print(f"Agent 调度精确匹配率: {exact_ok}/{total} ({exact_ok/total*100:.1f}%)")
    print(f"Precision (宏平均):   {total_prec*100:.1f}%")
    print(f"Recall (宏平均):      {total_recall*100:.1f}%")
    print(f"实体提取准确率:       {avg_entity*100:.1f}%" if entity_scores else "实体提取准确率: N/A")
    print(f"{'='*60}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
