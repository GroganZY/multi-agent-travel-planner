#!/usr/bin/env python
"""
RAG 评估 — RAGAs 版本（生成层）

使用 ragas 库的 faithfulness 和 answer_relevancy 指标。
检索层仍用手写版（见 run_eval.py），因为 RAGAs 的
context_precision/context_recall 需要完整参考答案文本，
而本项目的 ground truth 使用 key_facts 格式，不兼容。

用法：pip install ragas datasets 后运行本脚本
"""
from __future__ import annotations

import json
import sys
import time
import os
import asyncio
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass


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


async def generate_answers(model, agent, questions: list) -> list:
    """批量生成 LLM 答案"""
    answers, contexts_list = [], []
    for q in questions:
        docs = agent.search_knowledge(q, top_k=3)
        contexts_list.append([d["content"] for d in docs])
        context = "\n\n".join(f"【片段{i+1}】\n{d['content']}" for i, d in enumerate(docs))
        prompt = (
            "你是一个商旅知识专家。严格基于以下知识库信息回答问题。\n"
            "如果知识库中没有相关信息，就说不知道，不要编造。\n\n"
            f"【用户问题】\n{q}\n\n【知识库信息】\n{context}\n\n请直接回答："
        )
        resp = await model([{"role": "user", "content": prompt}])
        answers.append(await _extract_response(resp))
        await asyncio.sleep(15)  # 免费 API 限流
    return answers, contexts_list


async def main():
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("LLM_API_KEY 未设置"); return

    from config import LLM_CONFIG, SYSTEM_CONFIG
    from config_agentscope import init_agentscope
    from agentscope.model import OpenAIChatModel
    init_agentscope()
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"], api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": float(SYSTEM_CONFIG.get("timeout", 60))},
        generate_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        temperature=0.1, max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )

    print("Loading RAG agent...")
    agent = init_rag_agent()
    print(f"  Ready ({len(gt)} questions)")

    # 采样 10 题
    sample = gt[:10]
    questions = [item["question"] for item in sample]

    print("Generating answers...")
    answers, contexts_list = await generate_answers(model, agent, questions)

    print("Evaluating with RAGAs...")
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        })
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        print(result)
    except ImportError:
        print("RAGAs not installed. Run: pip install ragas datasets")
    except Exception as e:
        print(f"RAGAs evaluation failed: {e}")

    agent.close()


if __name__ == "__main__":
    asyncio.run(main())
