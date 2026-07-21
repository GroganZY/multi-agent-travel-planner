#!/usr/bin/env python
"""
RAG 全链路质量评估

五个指标全部使用 RAGAs 库：
  检索层 — context_precision / context_recall
  生成层 — faithfulness / answer_relevancy / answer_correctness

ground_truth.json 含 reference（完整参考答案），与 RAGAs 输入格式对齐。

用法：pip install ragas datasets 后运行本脚本
"""
from __future__ import annotations

import json
import sys
import os
import asyncio
from pathlib import Path
from typing import List

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


async def main():
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    from config import LLM_CONFIG
    api_key = LLM_CONFIG["api_key"]
    if not api_key:
        print(f"API Key 未设置（provider={LLM_CONFIG.get('provider','?')}）"); return

    model = init_llm()
    agent = init_rag_agent()
    print(f"Ready ({len(gt)} questions)\n")

    # 全量 30 题
    sample = gt

    # ── 生成答案 + 检索 ──────────────────────────────────────
    print("=" * 60)
    print("生成答案 + 检索...")
    print("=" * 60)

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    references: List[str] = []

    for idx, item in enumerate(sample):
        if idx > 0:
            await asyncio.sleep(5)

        docs = agent.search_knowledge(item["question"], top_k=3)
        contexts = [d["content"] for d in docs]
        contexts_list.append(contexts)

        ctx = "\n\n".join(
            f"【片段{i+1}】\n{d['content']}" for i, d in enumerate(docs)
        )
        prompt = (
            "你是一个商旅知识专家。严格基于以下知识库信息回答问题。\n"
            "如果知识库中没有相关信息，就说不知道，不要编造。\n\n"
            f"【用户问题】\n{item['question']}\n\n"
            f"【知识库信息】\n{ctx}\n\n请直接回答："
        )
        resp = await model([{"role": "user", "content": prompt}])
        answer = await _extract_response(resp)

        questions.append(item["question"])
        answers.append(answer)
        references.append(item.get("reference", ""))

        print(f"  [{item['id']:2d}] {item['question'][:25]} "
              f"→ {answer[:40]}...")

    # ── RAGAs 评估（四个指标，一次调用）──────────────────────
    print("\n" + "=" * 60)
    print("RAGAs 评估（context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness）")
    print("=" * 60)

    try:
        import ragas
        from ragas.metrics import (
            context_precision, context_recall,
            faithfulness, answer_relevancy, answer_correctness,
        )
        import datasets

        ds = datasets.Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": references,
        })

        result = ragas.evaluate(
            ds,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
            ],
        )
        print(result)

    except ImportError:
        print("请先安装: pip install ragas datasets")
    except Exception as e:
        print(f"评估失败: {e}")

    agent.close()


if __name__ == "__main__":
    asyncio.run(main())
