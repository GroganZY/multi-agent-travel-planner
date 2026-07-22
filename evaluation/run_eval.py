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
        temperature=0.1,  # RAG 事实性问答用低温保证稳定性
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
        # ragas 依赖 langchain_community.chat_models.vertexai.ChatVertexAI，
        # 该模块在较新版本的 langchain-community 中已移除。
        # 此处 mock 该模块以兼容 ragas 0.2-0.4 版本的导入。
        import sys as _sys, types as _types
        _stub = _types.ModuleType('langchain_community.chat_models.vertexai')
        _stub.ChatVertexAI = type('ChatVertexAI', (), {})
        _sys.modules.setdefault('langchain_community.chat_models', _types.ModuleType('langchain_community.chat_models'))
        _sys.modules['langchain_community.chat_models.vertexai'] = _stub

        # RAGAs 内部使用 OpenAI client。用 DeepSeek 的 endpoint 和 model 覆盖默认值。
        from config import LLM_CONFIG as _llm_cfg
        os.environ["OPENAI_API_KEY"] = _llm_cfg["api_key"]
        os.environ["OPENAI_BASE_URL"] = _llm_cfg["base_url"]

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

        # AnswerRelevancy / AnswerCorrectness 需要 embedding 模型
        from langchain_community.embeddings import HuggingFaceEmbeddings
        eval_embeddings = HuggingFaceEmbeddings(
            model_name=str(project_root / "data" / "models" / "bge-small-zh-v1.5"),
        )

        # RAGAs 0.2.x 内部硬编码了 gpt-4o-mini 作为默认模型。
        # monkey-patch langchain_openai.ChatOpenAI 将 model 参数默认值覆盖为 DeepSeek 模型。
        import langchain_openai
        _orig_init = langchain_openai.ChatOpenAI.__init__
        def _patched_init(self, *args, **kwargs):
            if 'model' not in kwargs or kwargs['model'] == 'gpt-4o-mini':
                kwargs['model'] = _llm_cfg['model_name']
            if 'temperature' not in kwargs:
                kwargs['temperature'] = 0.1
            _orig_init(self, *args, **kwargs)
        langchain_openai.ChatOpenAI.__init__ = _patched_init

        result = ragas.evaluate(
            ds,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
            ],
            embeddings=eval_embeddings,
        )
        print(result)

        # ── LLM Correctness（独立于 RAGAs embedding-based Correctness）──
        print("\n" + "=" * 60)
        print("LLM-based Correctness（逐事实判断，不受 embedding 相似度影响）")
        print("=" * 60)
        llm_c_scores = []
        for idx, item in enumerate(sample):
            if idx > 0:
                await asyncio.sleep(5)
            prompt = (
                f"问题：{item['question']}\n"
                f"参考答案：{item.get('reference','')}\n"
                f"系统答案：{answers[idx]}\n\n"
                "判断系统答案和参考答案在关键事实上是否一致。\n"
                "只关注事实准确性，不关注措辞、长度、格式差异。\n"
                "输出一个 0.0-1.0 的数字（1.0=完全一致，0.5=部分一致，0.0=完全不一致）："
            )
            try:
                resp = await model([{"role": "user", "content": prompt}])
                text = (await _extract_response(resp)).strip()
                score = float(text)
                score = max(0.0, min(1.0, score))
            except Exception:
                score = -1.0
            llm_c_scores.append(score)
            print(f"  [{item['id']:2d}] {item['question'][:20]:<20} LLM-C={score:.2f}")

        valid_llm_c = [s for s in llm_c_scores if s >= 0]
        avg_llm_c = sum(valid_llm_c)/len(valid_llm_c) if valid_llm_c else 0
        print(f"\n  Avg LLM Correctness: {avg_llm_c:.2%}")
        retrieval["summary"]["avg_llm_correctness"] = round(avg_llm_c, 4)

        # 保存每题得分，方便定位低分题
        df = result.to_pandas()
        df['id'] = range(1, len(df) + 1)
        df['question'] = questions
        df['answer'] = answers
        df['reference'] = references
        df['llm_correctness'] = llm_c_scores
        per_q_path = Path(__file__).parent / "results" / "per_question_scores.csv"
        df.to_csv(str(per_q_path), index=False, encoding='utf-8-sig')
        print(f"\n每题得分已保存至 {per_q_path}")

    except ImportError:
        print("请先安装: pip install ragas datasets")
    except Exception as e:
        print(f"评估失败: {e}")
    agent.close()


if __name__ == "__main__":
    asyncio.run(main())
