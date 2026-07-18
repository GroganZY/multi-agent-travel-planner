#!/usr/bin/env python
"""
RAG 检索质量量化评估

评估指标：
  Context Precision  — 检索返回的 chunk 中真正相关的占比
  Context Recall     — 预期关键事实被检索到的占比
  Category Hit Rate  — 检索到的 chunk 类别是否匹配预期文档
  MRR                — 第一个相关 chunk 的倒数排名
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ── 类别 → 预期文档映射 ──────────────────────────────────────────
CATEGORY_TO_DOC = {
    "差旅规定": "01_travel_standards.txt",
    "报销规定": "02_reimbursement_policy.txt",
    "预订指南": "03_booking_guide.txt",
    "FAQ":      "04_faq.txt",
    "紧急处理":  "05_emergency_procedures.txt",
    "平台指南":  "06_platform_guide.txt",
    "城市指南":  "07_city_specific_tips.txt",
    "环保倡议":  "08_environmental_initiatives.txt",
}


def load_ground_truth(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_rag_agent():
    """初始化 RAG Agent（不依赖 LLM，只做检索）"""
    skill_root = project_root / ".claude" / "skills" / "ask-question"
    sys.path.insert(0, str(skill_root / "script"))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_agent", skill_root / "script" / "agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kb_path = skill_root / "data" / "rag_knowledge"
    agent = module.RAGKnowledgeAgent(
        name="EvalAgent",
        model=None,  # 不需要 LLM，只测试检索
        knowledge_base_path=str(kb_path),
        collection_name="business_travel_knowledge",
        top_k=3,
    )
    return agent


def chunk_matches_expected(chunk: Dict, expected_doc: str) -> bool:
    """chunk 的 metadata 中 parent_doc 是否匹配预期"""
    meta = chunk.get("metadata", {})
    parent = meta.get("parent_doc", "")
    return expected_doc in parent


def chunk_contains_facts(chunk: Dict, facts: List[str]) -> int:
    """返回 chunk 内容中命中的关键事实数"""
    content = chunk.get("content", "")
    hits = 0
    for fact in facts:
        if fact in content:
            hits += 1
    return hits


def evaluate_retrieval(agent, ground_truth: List[Dict]) -> Dict[str, Any]:
    """对全部 ground truth 问题跑检索评估"""
    results = []
    per_category = defaultdict(lambda: {"precision": [], "recall": [], "mrr": [], "count": 0})

    total_precision = []
    total_recall = []
    total_mrr = []
    total_category_hit = []

    for item in ground_truth:
        qid = item["id"]
        question = item["question"]
        expected_doc = item["expected_doc"]
        key_facts = item["key_facts"]
        category = item["category"]

        # 检索
        docs = agent.search_knowledge(question, top_k=3)

        # ── Context Precision ──────────────────────────────
        relevant_count = 0
        for d in docs:
            if chunk_matches_expected(d, expected_doc) or chunk_contains_facts(d, key_facts) > 0:
                relevant_count += 1
        precision = relevant_count / len(docs) if docs else 0
        total_precision.append(precision)

        # ── Category Hit ───────────────────────────────────
        cat_hit = any(chunk_matches_expected(d, expected_doc) for d in docs)
        total_category_hit.append(1.0 if cat_hit else 0.0)

        # ── Context Recall (关键事实覆盖率) ──────────────────
        all_content = " ".join(d.get("content", "") for d in docs)
        fact_hits = sum(1 for f in key_facts if f in all_content)
        recall = fact_hits / len(key_facts) if key_facts else 0
        total_recall.append(recall)

        # ── MRR ────────────────────────────────────────────
        mrr = 0.0
        for rank, d in enumerate(docs, 1):
            if chunk_matches_expected(d, expected_doc) or chunk_contains_facts(d, key_facts) > 0:
                mrr = 1.0 / rank
                break
        total_mrr.append(mrr)

        # ── 按类别统计 ──────────────────────────────────────
        per_category[category]["precision"].append(precision)
        per_category[category]["recall"].append(recall)
        per_category[category]["mrr"].append(mrr)
        per_category[category]["count"] += 1

        results.append({
            "id": qid,
            "question": question,
            "category": category,
            "expected_doc": expected_doc,
            "retrieved_docs": [
                {
                    "doc": d.get("metadata", {}).get("parent_doc", "?"),
                    "category": d.get("metadata", {}).get("category", "?"),
                    "distance": round(d.get("distance", 0), 4),
                    "matched": chunk_matches_expected(d, expected_doc),
                    "fact_hits": chunk_contains_facts(d, key_facts),
                }
                for d in docs
            ],
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "mrr": round(mrr, 3),
            "key_facts_found": round(fact_hits, 0),
            "key_facts_total": len(key_facts),
        })

    # ── 汇总 ────────────────────────────────────────────────
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0

    summary = {
        "total_questions": len(ground_truth),
        "avg_context_precision": avg(total_precision),
        "avg_context_recall": avg(total_recall),
        "avg_mrr": avg(total_mrr),
        "category_hit_rate": avg(total_category_hit),
        "by_category": {
            cat: {
                "questions": info["count"],
                "avg_precision": avg(info["precision"]),
                "avg_recall": avg(info["recall"]),
                "avg_mrr": avg(info["mrr"]),
            }
            for cat, info in sorted(per_category.items())
        },
    }

    return {"summary": summary, "details": results}


def print_report(summary: Dict, details: List[Dict]):
    """打印评估报告"""
    s = summary
    print()
    print("=" * 65)
    print("  RAG 检索质量评估报告")
    print("=" * 65)
    print(f"  评估问题数:        {s['total_questions']}")
    print(f"  Context Precision:  {s['avg_context_precision']:.2%}  检索返回的chunk中相关占比")
    print(f"  Context Recall:     {s['avg_context_recall']:.2%}  关键事实被检索到的占比")
    print(f"  Category Hit Rate:  {s['category_hit_rate']:.2%}  至少命中1个正确类别chunk的比例")
    print(f"  MRR:               {s['avg_mrr']:.3f}   第一个相关chunk的倒数排名均值")
    print()
    print("─" * 65)
    print("  按文档类别分拆")
    print("─" * 65)
    print(f"  {'类别':<8} {'问题数':<8} {'Precision':<12} {'Recall':<10} {'MRR':<8}")
    for cat, info in s["by_category"].items():
        print(f"  {cat:<8} {info['questions']:<8} "
              f"{info['avg_precision']:.2%}         "
              f"{info['avg_recall']:.2%}       "
              f"{info['avg_mrr']:.3f}")
    print()

    # 低分 case
    low = [d for d in details if d["precision"] < 0.5 or d["recall"] < 0.5]
    if low:
        print("─" * 65)
        print("  需要关注的低分问题")
        print("─" * 65)
        for d in low:
            retrieved = ", ".join(
                f"{r['doc']}({r['distance']})" for r in d["retrieved_docs"]
            )
            print(f"  [{d['id']}] {d['question']}")
            print(f"       期望: {d['expected_doc']}  检索: {retrieved}")
            print(f"       P={d['precision']:.0%} R={d['recall']:.0%} MRR={d['mrr']:.3f}")
            print()

    print("─" * 65)
    print("  各问题详情")
    print("─" * 65)
    for d in details:
        status = "✓" if d["precision"] >= 0.5 and d["recall"] >= 0.5 else "⚠"
        print(f"  {status} [{d['id']:2d}] P={d['precision']:.0%} R={d['recall']:.0%} "
              f"MRR={d['mrr']:.3f}  {d['question']}")


def main():
    gt_path = Path(__file__).parent / "ground_truth.json"
    out_path = Path(__file__).parent / "results" / "eval_result.json"

    print("Loading ground truth...")
    gt = load_ground_truth(str(gt_path))
    print(f"  {len(gt)} questions loaded")

    print("Initializing RAG agent (Milvus + BGE)...")
    t0 = time.time()
    agent = init_rag_agent()
    print(f"  Ready ({time.time() - t0:.1f}s)")

    print("Running retrieval evaluation...")
    t0 = time.time()
    report = evaluate_retrieval(agent, gt)
    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s, {len(gt)/elapsed:.1f} q/s)")

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to {out_path}")

    print_report(report["summary"], report["details"])

    # 关闭连接
    agent.close()


if __name__ == "__main__":
    main()
