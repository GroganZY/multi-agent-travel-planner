#!/usr/bin/env python
"""
RAG 全链路质量评估（检索 + 生成）

检索层指标（LLM-as-judge，同 RAGAs 口径）：
  Context Precision  — 返回的 chunk 中真正相关的占比
  Context Recall     — 关键事实被检索到的占比
  MRR                — 第一个相关 chunk 的倒数排名

生成层指标（LLM-as-judge）：
  Faithfulness       — LLM 答案中每句话能否在检索结果里找到依据
  Answer Relevancy   — LLM 答案是否扣题

评估方法：参考 RAGAs 论文指标定义，检索层和生成层均使用 LLM
进行语义判断。检索层对每道题发一次 LLM 调用同时评估 Precision
和 Recall，避免规则匹配（子串/文档名）的漏判和误判。
"""
from __future__ import annotations

import json
import sys
import time
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass


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


# ── 响应提取 ──────────────────────────────────────────────────

async def _extract_response(response) -> str:
    text = ""
    if hasattr(response, '__aiter__'):
        async for chunk in response:
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, 'content'):
                c = chunk.content
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text = item.get('text', '')
    elif hasattr(response, 'content'):
        c = response.content
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '')
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


# ── 检索评估（LLM-as-judge，同 RAGAs 口径）──────────────────────

async def evaluate_retrieval_llm(
    model, agent, ground_truth: List[Dict], *, delay_sec: float = 15.0
) -> Dict:
    """
    Context Precision 和 Context Recall 各一次独立 LLM 调用，
    与 RAGAs 原版同口径。不合并 prompt，保证每个指标专注单一任务。
    """
    results = []
    per_cat = defaultdict(lambda: {"precision": [], "recall": [], "mrr": [], "count": 0})
    all_p, all_r, all_mrr = [], [], []

    for idx, item in enumerate(ground_truth):
        if idx > 0:
            await asyncio.sleep(delay_sec)

        question = item["question"]
        facts = item["key_facts"]
        category = item["category"]
        docs = agent.search_knowledge(question, top_k=3)

        # ── Context Precision：逐 chunk 判断相关性 ────────────
        prec_vals = []
        for d in docs:
            prompt = (
                "判断以下文档片段是否包含回答该问题所需的信息。\n\n"
                f"问题：{question}\n\n"
                f"文档片段：\n{d.get('content', '')[:500]}\n\n"
                "这个片段是否包含回答该问题的相关信息？只回答 YES 或 NO。"
            )
            try:
                resp = await model([{"role": "user", "content": prompt}])
                text = await _extract_response(resp)
                prec_vals.append("YES" in text.upper())
            except Exception:
                prec_vals.append(True)
            await asyncio.sleep(delay_sec)

        precision = sum(1 for v in prec_vals if v) / len(prec_vals)

        # ── Context Recall：检查关键事实覆盖 ──────────────────
        chunks_text = "\n\n".join(
            f"[{i+1}] {d.get('content', '')[:400]}" for i, d in enumerate(docs)
        )
        rec_vals = []
        for f in facts:
            prompt = (
                "判断以下检索到的文档是否明确提到了给定的关键事实。\n\n"
                f"问题：{question}\n\n"
                f"检索到的文档：\n{chunks_text}\n\n"
                f"关键事实：{f}\n\n"
                "文档中是否明确提到了该事实？只回答 YES 或 NO。"
            )
            try:
                resp = await model([{"role": "user", "content": prompt}])
                text = await _extract_response(resp)
                rec_vals.append("YES" in text.upper())
            except Exception:
                rec_vals.append(True)
            await asyncio.sleep(delay_sec)

        recall = sum(1 for v in rec_vals if v) / len(rec_vals) if rec_vals else 0

        # ── MRR ───────────────────────────────────────────────
        mrr = 0.0
        for rank, v in enumerate(prec_vals, 1):
            if v:
                mrr = 1.0 / rank
                break

        all_p.append(precision); all_r.append(recall); all_mrr.append(mrr)
        per_cat[category]["precision"].append(precision)
        per_cat[category]["recall"].append(recall)
        per_cat[category]["mrr"].append(mrr)
        per_cat[category]["count"] += 1

        results.append({
            "id": item["id"], "question": question,
            "category": category,
            "retrieved": [
                {"doc": d.get("metadata",{}).get("parent_doc","?"),
                 "category": d.get("metadata",{}).get("category","?"),
                 "distance": round(d.get("distance",0), 4),
                 "relevant": prec_vals[i] if i < len(prec_vals) else None}
                for i, d in enumerate(docs)
            ],
            "precision": round(precision, 2), "recall": round(recall, 2),
            "mrr": round(mrr, 3),
            "facts_found": sum(1 for v in rec_vals if v),
            "facts_total": len(facts),
        })

        print(f"  [{item['id']:2d}] {question[:20]:<20} P={precision:.0%} R={recall:.0%}")

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0
    return {
        "summary": {
            "total": len(ground_truth),
            "method": "LLM-as-judge (RAGAs-aligned)",
            "context_precision": avg(all_p),
            "context_recall": avg(all_r),
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
    return -1.0


async def evaluate_relevancy(model, item: Dict, answer: str) -> float:
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
    API_DELAY_SEC = 15.0

    gt_path = Path(__file__).parent / "ground_truth.json"
    out_path = Path(__file__).parent / "results" / "eval_result.json"

    print("Loading ground truth...")
    gt = load_ground_truth(str(gt_path))
    print(f"  {len(gt)} questions")

    print("Initializing RAG agent...")
    agent = init_rag_agent()
    print("  Ready")

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("\n⚠ LLM_API_KEY 未设置，评估需要 LLM 无法运行")
        agent.close()
        return

    model = init_llm_model()

    # ── Step 1: 检索评估（LLM-as-judge）────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: 检索质量评估（LLM-as-judge，同 RAGAs 口径）")
    print("=" * 60)
    t0 = time.time()
    retrieval = await evaluate_retrieval_llm(model, agent, gt, delay_sec=API_DELAY_SEC)
    print(f"  Done ({time.time()-t0:.1f}s)")
    s = retrieval["summary"]
    print(f"  Context Precision:  {s['context_precision']:.2%}")
    print(f"  Context Recall:     {s['context_recall']:.2%}")
    print(f"  MRR:                {s['mrr']:.3f}")

    # ── Step 2: LLM 生成 + Faithfulness + Relevancy ────────────
    print("\n" + "=" * 60)
    print("Step 2: Faithfulness + Answer Relevancy（LLM-as-judge）")
    print("=" * 60)
    t0 = time.time()

    faith_scores, relev_scores = [], []
    for idx, item in enumerate(gt[:10]):
        if idx > 0:
            await asyncio.sleep(API_DELAY_SEC)
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

    # 保存
    report = {"retrieval": retrieval["summary"], "details": retrieval["details"]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    agent.close()


if __name__ == "__main__":
    asyncio.run(main())
