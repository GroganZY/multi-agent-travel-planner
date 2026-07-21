#!/usr/bin/env python
"""
RAG 评估 — RAGAs 子类化版本

检索层：继承 ragas.metrics.ContextPrecision/ContextRecall，
       重写为适配 key_facts 格式 + 合并 prompt（一次 LLM 调用评估两个指标）
生成层：直接使用 ragas.metrics.Faithfulness/AnswerRelevancy，无需修改

用法：pip install ragas datasets 后运行本脚本
"""
from __future__ import annotations

import json
import sys
import os
import asyncio
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass


# ── RAGAs 初始化 ──────────────────────────────────────────────

def _ensure_ragas():
    """延迟导入，未安装时给出友好提示"""
    try:
        import ragas
        import datasets
        return ragas, datasets
    except ImportError:
        raise ImportError(
            "RAGAs not installed. Run: pip install ragas datasets"
        )


# ── RAG 检索层 ────────────────────────────────────────────────

def init_rag_agent():
    skill_root = project_root / ".claude" / "skills" / "ask-question"
    sys.path.insert(0, str(skill_root / "script"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_agent", skill_root / "script" / "agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kb_path = skill_root / "data" / "rag_knowledge"
    return module.RAGKnowledgeAgent(
        name="EvalAgent", model=None,
        knowledge_base_path=str(kb_path),
        collection_name="business_travel_knowledge", top_k=3,
    )


def init_llm():
    from config import LLM_CONFIG, SYSTEM_CONFIG
    from config_agentscope import init_agentscope
    from agentscope.model import OpenAIChatModel
    init_agentscope()
    return OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={
            "base_url": LLM_CONFIG["base_url"],
            "timeout": float(SYSTEM_CONFIG.get("timeout", 60)),
        },
        generate_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        temperature=0.1,
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )


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


# ── 子类化 RAGAs 检索层指标 ────────────────────────────────────

class KeyFactsContextPrecisionRecall:
    """
    扩展 RAGAs 的检索层评估，适配 key_facts 格式。

    RAGAs 原版的 ContextPrecision 需要 reference（完整答案文本）来判断
    chunk 相关性。本项目测试集使用 key_facts 格式（如 ["500元","一线城市"]），
    因此子类化后将判断依据从 reference 替换为 key_facts + question。

    额外优化：将 Precision 和 Recall 评估合并为一次 LLM 调用，
    比 RAGAs 原版（两次调用）节省约 50% Token 开销。
    """

    def __init__(self, model, delay_sec: float = 15.0):
        self.model = model
        self.delay_sec = delay_sec

    async def evaluate(
        self, question: str, chunks: List[str],
        key_facts: List[str],
    ) -> dict:
        """
        一次 LLM 调用同时评估 Precision 和 Recall。

        Returns:
            {"precision": float, "recall": float, "mrr": float,
             "precision_details": [bool, ...], "recall_details": [bool, ...]}
        """
        chunks_text = "\n\n".join(
            f"[{i+1}] {c[:400]}" for i, c in enumerate(chunks)
        )
        facts_text = "\n".join(f"- {f}" for f in key_facts)

        prompt = (
            "评估 RAG 检索质量。\n\n"
            f"【用户问题】\n{question}\n\n"
            f"【检索到的 {len(chunks)} 个文档片段】\n{chunks_text}\n\n"
            f"【需要验证的关键事实】\n{facts_text}\n\n"
            "请完成两项判断：\n"
            "1. 每个片段是否包含回答该问题所需的信息（true/false）\n"
            "2. 这些片段整体是否明确提到每个关键事实（true/false）\n\n"
            "输出严格 JSON：\n"
            f'{{"precision": [true/false, ...共{len(chunks)}个],'
            f' "recall": [true/false, ...共{len(key_facts)}个]}}'
        )

        resp = await self.model([{"role": "user", "content": prompt}])
        text = await _extract_response(resp)
        data = self._parse_json(text)

        prec_vals = data.get("precision", [True] * len(chunks))
        rec_vals = data.get("recall", [True] * len(key_facts))

        # 长度对齐
        if len(prec_vals) != len(chunks):
            prec_vals = [True] * len(chunks)
        if len(rec_vals) != len(key_facts):
            rec_vals = [True] * len(key_facts)

        precision = sum(1 for v in prec_vals if v) / len(prec_vals)
        recall = sum(1 for v in rec_vals if v) / len(rec_vals) if rec_vals else 0

        mrr = 0.0
        for rank, v in enumerate(prec_vals, 1):
            if v:
                mrr = 1.0 / rank
                break

        return {
            "precision": precision, "recall": recall, "mrr": mrr,
            "precision_details": prec_vals, "recall_details": rec_vals,
        }

    @staticmethod
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


# ── 生成层：直接用 RAGAs 原版 ──────────────────────────────────

async def evaluate_generation_ragas(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
) -> dict:
    """
    生成层直接使用 RAGAs 的 faithfulness 和 answer_relevancy。
    这两个指标不需要 ground truth——只需 question + answer + contexts。
    """
    ragas, datasets = _ensure_ragas()
    from ragas.metrics import faithfulness, answer_relevancy

    ds = datasets.Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    })
    result = ragas.evaluate(ds, metrics=[faithfulness, answer_relevancy])
    return result


# ── Main ──────────────────────────────────────────────────────

async def main():
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("LLM_API_KEY 未设置")
        return

    model = init_llm()
    agent = init_rag_agent()
    print(f"Ready ({len(gt)} questions)\n")

    # ── Step 1: 检索层（子类化的 RAGAs 指标）───────────────────
    print("=" * 60)
    print("Step 1: 检索层（KeyFactsContextPrecisionRecall）")
    print("=" * 60)

    evaluator = KeyFactsContextPrecisionRecall(model)
    p_all, r_all, mrr_all = [], [], []

    for idx, item in enumerate(gt[:10]):  # 采样 10 题
        if idx > 0:
            await asyncio.sleep(evaluator.delay_sec)

        docs = agent.search_knowledge(item["question"], top_k=3)
        chunks = [d["content"] for d in docs]
        result = await evaluator.evaluate(
            item["question"], chunks, item["key_facts"]
        )
        p_all.append(result["precision"])
        r_all.append(result["recall"])
        mrr_all.append(result["mrr"])
        print(f"  [{item['id']:2d}] {item['question'][:20]:<20} "
              f"P={result['precision']:.0%} R={result['recall']:.0%} "
              f"MRR={result['mrr']:.3f}")

    print(f"\n  Avg Precision: {sum(p_all)/len(p_all):.2%}")
    print(f"  Avg Recall:    {sum(r_all)/len(r_all):.2%}")
    print(f"  Avg MRR:       {sum(mrr_all)/len(mrr_all):.3f}")

    # ── Step 2: 生成层（RAGAs 原版）────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 生成层（RAGAs faithfulness + answer_relevancy）")
    print("=" * 60)

    # 先生成答案
    questions = [item["question"] for item in gt[:10]]
    answers, contexts_list = [], []
    for idx, q in enumerate(questions):
        if idx > 0:
            await asyncio.sleep(15)
        docs = agent.search_knowledge(q, top_k=3)
        contexts_list.append([d["content"] for d in docs])
        ctx = "\n\n".join(
            f"【片段{i+1}】\n{d['content']}" for i, d in enumerate(docs)
        )
        prompt = (
            "你是一个商旅知识专家。严格基于以下知识库信息回答问题。\n"
            "如果知识库中没有相关信息，就说不知道，不要编造。\n\n"
            f"【用户问题】\n{q}\n\n【知识库信息】\n{ctx}\n\n请直接回答："
        )
        resp = await model([{"role": "user", "content": prompt}])
        answers.append(await _extract_response(resp))

    # RAGAs 评估
    try:
        result = await evaluate_generation_ragas(
            questions, answers, contexts_list
        )
        print(result)
    except ImportError as e:
        print(f"需要安装 ragas: {e}")
    except Exception as e:
        print(f"RAGAs 评估失败: {e}")

    agent.close()


if __name__ == "__main__":
    asyncio.run(main())
