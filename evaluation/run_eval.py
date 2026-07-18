#!/usr/bin/env python
"""
RAG 全链路质量评估（检索 + 生成）

指标：
  Context Precision  — 返回的 chunk 中真正相关的占比
  Context Recall     — 关键事实被检索到的占比
  Category Hit       — 至少命中 1 个可接受文档的比例
  MRR                — 第一个相关 chunk 的倒数排名
  Faithfulness       — LLM 答案中每句话能否在检索结果里找到依据
  Answer Relevancy   — LLM 答案是否扣题
"""
from __future__ import annotations

import json
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

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

# FAQ 和正式标准内容重叠——FAQ 里的回答也是正确答案
FAQ_OVERLAP_CATEGORIES = {"差旅规定", "报销规定", "预订指南"}


def load_ground_truth(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_rag_agent():
    skill_root = project_root / ".claude" / "skills" / "ask-question"
    sys.path.insert(0, str(skill_root / "script"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_agent", skill_root / "script" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kb_path = skill_root / "data" / "rag_knowledge"
    return module.RAGKnowledgeAgent(
        name="EvalAgent", model=None,
        knowledge_base_path=str(kb_path),
        collection_name="business_travel_knowledge", top_k=3,
    )


def init_llm_model():
    from config import LLM_CONFIG, SYSTEM_CONFIG
    from config_agentscope import init_agentscope
    from agentscope.model import OpenAIChatModel
    init_agentscope()
    return OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": float(SYSTEM_CONFIG.get("timeout", 60))},
        generate_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        temperature=0.1,
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )


def chunk_is_relevant(chunk: Dict, expected_doc: str, acceptable_docs: List[str], key_facts: List[str]) -> tuple:
    """判断 chunk 是否相关，返回 (是否匹配, 命中事实数)"""
    meta = chunk.get("metadata", {})
    parent = meta.get("parent_doc", "")
    cat = meta.get("category", "")

    doc_match = any(d in parent for d in acceptable_docs)
    facts = chunk_contains_facts(chunk, key_facts)
    return (doc_match or facts > 0, facts)


def chunk_contains_facts(chunk: Dict, facts: List[str]) -> int:
    content = chunk.get("content", "")
    return sum(1 for f in facts if f in content)


# ── 检索评估 ──────────────────────────────────────────────────

def evaluate_retrieval(agent, ground_truth: List[Dict]) -> Dict:
    results = []
    per_cat = defaultdict(lambda: {"precision": [], "recall": [], "mrr": [], "count": 0})
    all_p, all_r, all_mrr, all_hit = [], [], [], []

    for item in ground_truth:
        expected = item["expected_doc"]
        acceptable = [expected]
        if item["category"] in FAQ_OVERLAP_CATEGORIES:
            acceptable.append("04_faq.txt")

        facts = item["key_facts"]
        docs = agent.search_knowledge(item["question"], top_k=3)

        relevant = 0
        for d in docs:
            rel, _ = chunk_is_relevant(d, expected, acceptable, facts)
            if rel:
                relevant += 1
        precision = relevant / len(docs) if docs else 0

        all_content = " ".join(d.get("content", "") for d in docs)
        fact_hits = sum(1 for f in facts if f in all_content)
        recall = fact_hits / len(facts) if facts else 0

        cat_hit = any(chunk_is_relevant(d, expected, acceptable, facts)[0] for d in docs)
        mrr = 0.0
        for rank, d in enumerate(docs, 1):
            if chunk_is_relevant(d, expected, acceptable, facts)[0]:
                mrr = 1.0 / rank
                break

        all_p.append(precision); all_r.append(recall)
        all_mrr.append(mrr); all_hit.append(1.0 if cat_hit else 0.0)
        per_cat[item["category"]]["precision"].append(precision)
        per_cat[item["category"]]["recall"].append(recall)
        per_cat[item["category"]]["mrr"].append(mrr)
        per_cat[item["category"]]["count"] += 1

        results.append({
            "id": item["id"], "question": item["question"],
            "category": item["category"],
            "acceptable_docs": acceptable,
            "retrieved": [
                {"doc": d.get("metadata",{}).get("parent_doc","?"),
                 "category": d.get("metadata",{}).get("category","?"),
                 "distance": round(d.get("distance",0), 4),
                 "relevant": chunk_is_relevant(d, expected, acceptable, facts)[0]}
                for d in docs
            ],
            "precision": round(precision, 2), "recall": round(recall, 2),
            "mrr": round(mrr, 3), "facts_found": fact_hits, "facts_total": len(facts),
        })

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0
    return {
        "summary": {
            "total": len(ground_truth),
            "context_precision": avg(all_p),
            "context_recall": avg(all_r),
            "category_hit": avg(all_hit),
            "mrr": avg(all_mrr),
            "by_category": {
                cat: {"count": i["count"], "precision": avg(i["precision"]),
                      "recall": avg(i["recall"]), "mrr": avg(i["mrr"])}
                for cat, i in sorted(per_cat.items())
            }
        }, "details": results,
    }


# ── Faithfulness & Answer Relevancy（需 LLM）───────────────────

async def evaluate_faithfulness(model, item: Dict, docs: List[Dict], answer: str) -> float:
    """逐句检查答案是否能在检索结果里找到依据。返回忠实度比例。"""
    context = "\n".join(d.get("content", "")[:300] for d in docs)
    prompt = (
        "你的任务：判断以下「答案」中的每句话是否能在「检索到的文档」里找到依据。\n\n"
        f"【问题】\n{item['question']}\n\n"
        f"【检索到的文档】\n{context}\n\n"
        f"【LLM 生成的答案】\n{answer}\n\n"
        "请逐句分析答案中的每个陈述。如果一个陈述能在文档中找到明确依据，标记为 SUPPORTED。\n"
        "如果文档中没有提到、或是 LLM 自己编的，标记为 UNSUPPORTED。\n"
        "如果答案就是'知识库中没有相关信息'，直接输出 ALL_SUPPORTED。\n\n"
        "输出格式（严格 JSON）：\n"
        '{"claims": [{"text": "陈述内容", "verdict": "SUPPORTED/UNSUPPORTED"}], '
        '"supported_count": N, "total_count": N}'
    )
    try:
        resp = await model([{"role": "user", "content": prompt}])
        text = await _extract_response(resp)
        data = _parse_json(text)
        if data and "claims" in data:
            return data.get("supported_count", 0) / max(data.get("total_count", 1), 1)
    except Exception:
        pass
    return -1.0  # 评估失败


async def evaluate_relevancy(model, item: Dict, answer: str) -> float:
    """判断答案是否扣题。0=完全跑题，1=完全扣题。"""
    prompt = (
        f"【用户问题】{item['question']}\n\n"
        f"【系统回答】{answer}\n\n"
        "给这个回答的相关性打分（0.0-1.0）。\n"
        "- 1.0: 完全扣题，直接回答了用户问题\n"
        "- 0.5: 部分相关但偏题或啰嗦\n"
        "- 0.0: 完全不相关或答非所问\n"
        "只输出一个数字，如 0.85"
    )
    try:
        resp = await model([{"role": "user", "content": prompt}])
        text = (await _extract_response(resp)).strip()
        score = float(text)
        return max(0.0, min(1.0, score))
    except Exception:
        return -1.0


async def _extract_response(response) -> str:
    text = ""
    if hasattr(response, '__aiter__'):
        async for chunk in response:
            if isinstance(chunk, str):
                text = chunk  # doubao streaming: each chunk is full accumulated text
            elif hasattr(chunk, 'content'):
                c = chunk.content
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text = item.get('text', '')  # last text item wins
    elif hasattr(response, 'content'):
        c = response.content
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '')
        else:
            text = str(c)
    else:
        text = str(response)
    return text.strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text)
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {}


# ── LLM 生成答案 ──────────────────────────────────────────────

async def generate_answer(model, question: str, docs: List[Dict]) -> str:
    context = "\n\n".join(
        f"【知识片段{i+1}】\n{d['content']}" for i, d in enumerate(docs)
    )
    prompt = (
        "你是一个商旅知识专家。请严格基于以下知识库中的信息回答问题。\n"
        "如果知识库中没有相关信息，就说不知道，不要编造。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【知识库信息】\n{context}\n\n"
        "请直接给出简洁准确的答案："
    )
    resp = await model([{"role": "user", "content": prompt}])
    return await _extract_response(resp)


# ── Main ──────────────────────────────────────────────────────

async def main():
    import asyncio

    # 免费 API 限流控制：每次 LLM 调用间隔（秒）
    API_DELAY_SEC = 15.0

    gt_path = Path(__file__).parent / "ground_truth.json"
    out_path = Path(__file__).parent / "results" / "eval_result.json"

    print("Loading ground truth...")
    gt = load_ground_truth(str(gt_path))
    print(f"  {len(gt)} questions")

    print("Initializing RAG agent...")
    agent = init_rag_agent()
    print("  Ready")

    # ── Step 1: 检索评估 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: 检索质量评估")
    print("=" * 60)
    t0 = time.time()
    retrieval = evaluate_retrieval(agent, gt)
    print(f"  Done ({time.time()-t0:.1f}s)")
    s = retrieval["summary"]
    print(f"  Context Precision:  {s['context_precision']:.2%}")
    print(f"  Context Recall:     {s['context_recall']:.2%}")
    print(f"  Category Hit:       {s['category_hit']:.2%}")
    print(f"  MRR:                {s['mrr']:.3f}")

    # ── Step 2: LLM 生成 + Faithfulness + Relevancy ────────────
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("\n⚠ LLM_API_KEY 未设置，跳过 Faithfulness/Relevancy 评估")
        report = {"retrieval": retrieval["summary"], "details": retrieval["details"]}
    else:
        print("\n" + "=" * 60)
        print("Step 2: Faithfulness + Answer Relevancy（需 LLM）")
        print("=" * 60)
        model = init_llm_model()
        t0 = time.time()

        faith_scores, relev_scores = [], []
        for idx, item in enumerate(gt[:10]):  # 采样覆盖各类别
            if idx > 0:
                await asyncio.sleep(API_DELAY_SEC)  # 免费 API 限流
            docs = agent.search_knowledge(item["question"], top_k=3)
            answer = await generate_answer(model, item["question"], docs)
            faith = await evaluate_faithfulness(model, item, docs, answer)
            await asyncio.sleep(API_DELAY_SEC)
            relev = await evaluate_relevancy(model, item, answer)

            faith_scores.append(faith if faith >= 0 else None)
            relev_scores.append(relev if relev >= 0 else None)
            print(f"  [{item['id']:2d}] {item['question'][:20]:<20} "
                  f"F={faith:.2f} R={relev:.2f}")
            await asyncio.sleep(API_DELAY_SEC)

        valid_f = [f for f in faith_scores if f is not None]
        valid_r = [r for r in relev_scores if r is not None]

        print(f"\n  Avg Faithfulness:     {sum(valid_f)/len(valid_f):.2%}" if valid_f else "  N/A")
        print(f"  Avg Answer Relevancy: {sum(valid_r)/len(valid_r):.2%}" if valid_r else "  N/A")
        print(f"  Done ({time.time()-t0:.1f}s)")

        retrieval["summary"]["avg_faithfulness"] = round(sum(valid_f)/len(valid_f), 4) if valid_f else None
        retrieval["summary"]["avg_answer_relevancy"] = round(sum(valid_r)/len(valid_r), 4) if valid_r else None

        report = {"retrieval": retrieval["summary"], "details": retrieval["details"]}

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    agent.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
