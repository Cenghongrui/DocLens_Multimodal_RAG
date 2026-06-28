"""
RAGAS 评估脚本（并行版）

优化：
  - 阶段一：30 个 RAG 链路调用  → asyncio.gather 并发（带信号量限流）
  - 阶段二：RAGAS 评分          → 调高内部 max_workers 并行度

用法：
控制台
  .venv\Scripts\python.exe -X utf8 -m eval.evaluate
"""

import json
import asyncio
import sys
import types
import re
import math
import time
from pathlib import Path
from typing import List, Tuple

# ─── 必须先做 monkey-patch，再导入 ragas ───
# RAGAS 0.4.3 无条件导入 ChatVertexAI，但它在最新 langchain-community 中已被移除
try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    ChatVertexAI = None
import langchain_community.chat_models
langchain_community.chat_models.vertexai = types.ModuleType("vertexai")
if ChatVertexAI:
    langchain_community.chat_models.vertexai.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = langchain_community.chat_models.vertexai

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings
from app.core.retriever import retrieve
from app.core.generator import generate

# ─── RAGAS ───
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
# ─── 配置 RAGAS 使用的 LLM ───
# 使用 DeepSeek 做 judge（与生成模型一致，用户指定）
from langchain_openai import ChatOpenAI

ragas_llm = ChatOpenAI(
    model=settings.llm_model,  # deepseek-v4-flash
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    temperature=0.1,
    max_retries=3,
    request_timeout=30,
)

for metric in [faithfulness, context_precision, context_recall]:
    metric.__setattr__("llm", ragas_llm)

EVAL_DIR = Path(__file__).parent
TEST_DATASET = EVAL_DIR / "test_dataset_v2.json"
RESULT_FILE = EVAL_DIR / "result_ragas_v2.json"

# ─── 并发控制 ───
# 同时跑太多 retrieve+generate 会打爆 API 限流，设信号量
RAG_CONCURRENCY = 3  # 阶段一并发数（DeepSeek + Qwen embedding 共用连接池，不宜过高）
RAGAS_MAX_WORKERS = 12  # 阶段二 RAGAS 内部并行度


# ─── 客观指标（关键词命中） ───

def _extract_keywords(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r'[\u4e00-\u9fa5]{2,6}|[A-Za-z][A-Za-z0-9\-]{2,}', text)
    stop = {
        "本文", "本文的", "我们", "通过", "使用", "采用", "基于", "方法",
        "研究", "进行", "可以", "能够", "一个", "这种", "这个", "这些",
        "以及", "并且", "对于", "根据", "由于", "从而",
        "the", "and", "for", "with", "that", "this", "are", "was",
    }
    return [t for t in tokens if t not in stop]


def compute_hit_rate_and_mrr(contexts: list[str], ground_truth: str) -> tuple[float, float]:
    keywords = _extract_keywords(ground_truth)
    if not keywords:
        return 0.0, 0.0
    for rank, ctx in enumerate(contexts, start=1):
        if any(kw.lower() in ctx.lower() for kw in keywords):
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def safe_float(v, default=0.0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


# ─── 单个用例的 RAG 链路（可并发执行） ───

async def _run_single_case(
    case: dict,
    idx: int,
    total: int,
    source: str,
    sem: asyncio.Semaphore,
) -> dict:
    """执行单个测试用例：检索 + 生成 + 客观指标，自动重试连接错误。"""
    q = case["question"].strip()
    gt = case.get("ground_truth", "").strip()

    async with sem:
        print(f"\n[{idx+1}/{total}] Q: {q[:80]}")

        # 检索 + 生成（重试连接错误）
        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                docs = await retrieve(q, source=source)
                contexts = [d.page_content for d in docs]
                print(f"  Retrieved {len(contexts)} chunks (attempt {attempt+1})")

                answer = await generate(q, docs)
                print(f"  Answer: {answer[:120]}...")
                break
            except Exception as e:
                last_error = str(e)
                is_conn_error = "Connection error" in last_error or "APIConnectionError" in last_error
                if attempt < max_attempts - 1 and is_conn_error:
                    wait = 2 ** attempt
                    print(f"  ⚠ 连接错误，{wait}s 后重试 ({attempt+2}/{max_attempts})...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ✗ 失败 ({max_attempts} 次): {last_error[:80]}")
                    contexts = []
                    answer = f"[ERROR] {last_error}"
                    break
        else:
            contexts = []
            answer = f"[ERROR] {last_error}"

        hit, rr = compute_hit_rate_and_mrr(contexts, gt) if contexts else (0.0, 0.0)
        print(f"  Hit={hit:.0f}  RR={rr:.3f}")

    return {
        "question": q,
        "ground_truth": gt,
        "answer": answer,
        "contexts": contexts,
        "hit_rate": hit,
        "reciprocal_rank": rr,
    }


async def run_evaluation(source: str = None):
    # 1. 加载测试集
    with open(TEST_DATASET, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    tag = f" [source={source}]" if source else ""
    print(f"Loaded {len(test_cases)} test cases{tag}")

    # ─────────────────────────────────────────────
    # 阶段一：并行跑 RAG 链路（并发限流）
    # ─────────────────────────────────────────────
    sem = asyncio.Semaphore(RAG_CONCURRENCY)
    t0 = time.time()

    tasks = [
        _run_single_case(case, i, len(test_cases), source, sem)
        for i, case in enumerate(test_cases)
        if case.get("question", "").strip()
    ]

    results = await asyncio.gather(*tasks)

    t1 = time.time()
    phase1_elapsed = t1 - t0
    n = len(results)

    questions = [r["question"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]
    answers = [r["answer"] for r in results]
    contexts_list = [r["contexts"] for r in results]
    hit_rates = [r["hit_rate"] for r in results]
    reciprocal_ranks = [r["reciprocal_rank"] for r in results]

    print(f"\n{'='*60}")
    print(f"阶段一完成: {n} 个用例, 耗时 {phase1_elapsed:.0f}s "
          f"(原顺序执行约 {n * 14:.0f}s, 并发度={RAG_CONCURRENCY})")

    # ─────────────────────────────────────────────
    # 阶段二：RAGAS 评分（调高内部并行度）
    # ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Running RAGAS evaluation on {n} cases (max_workers={RAGAS_MAX_WORKERS})...")
    print("  → 正在构建 HuggingFace Dataset...")

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    print(f"  → Dataset 构建完成，共 {len(eval_dataset)} 条")

    metrics = [context_precision, context_recall, faithfulness]
    print(f"  → 指标: {[getattr(m, 'name', str(m)) for m in metrics]}")

    run_config = RunConfig(
        max_workers=RAGAS_MAX_WORKERS,
        max_wait=120,
        max_retries=2,
    )
    print(f"  → RunConfig: max_workers={run_config.max_workers}")

    t2 = time.time()
    print(f"  → 开始评分（RAGAS 是同步调用，可能会卡住一会）...")
    sys.stdout.flush()

    # RAGAS evaluate() 是同步函数，用 asyncio.to_thread 避免阻塞事件循环
    result = await asyncio.to_thread(
        ragas_evaluate,
        eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        run_config=run_config,
    )
    t3 = time.time()
    phase2_elapsed = t3 - t2

    # ─────────────────────────────────────────────
    # 3. 提取结果
    # ─────────────────────────────────────────────
    ragas_raw = result.scores
    if isinstance(ragas_raw, dict):
        col_names = list(ragas_raw.keys())
        ragas_scores_list = []
        for i in range(n):
            row = {}
            for col in col_names:
                vals = ragas_raw[col]
                row[col] = safe_float(vals[i]) if i < len(vals) else 0.0
            ragas_scores_list.append(row)
    else:
        ragas_scores_list = [
            {k: safe_float(v) for k, v in row.items()}
            for row in ragas_raw
        ]

    detailed_results = []
    for i in range(n):
        row = ragas_scores_list[i] if i < len(ragas_scores_list) else {}
        detailed_results.append({
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "answer": answers[i],
            "contexts": contexts_list[i],
            "hit_rate": hit_rates[i],
            "reciprocal_rank": reciprocal_ranks[i],
            "scores": {
                "context_precision": row.get("context_precision", 0.0),
                "context_recall": row.get("context_recall", 0.0),
                "faithfulness": row.get("faithfulness", 0.0),
            },
        })

    # ─────────────────────────────────────────────
    # 4. 汇总统计
    # ─────────────────────────────────────────────
    avg_ragas = {}
    for key in ["context_precision", "context_recall", "faithfulness"]:
        vals = [r["scores"][key] for r in detailed_results]
        avg_ragas[key] = round(sum(vals) / n, 4)

    avg_hit = round(sum(hit_rates) / n, 4) if hit_rates else 0.0
    avg_mrr = round(sum(reciprocal_ranks) / n, 4) if reciprocal_ranks else 0.0

    total_elapsed = t3 - t0
    summary = {
        "total_cases": n,
        "evaluator": "RAGAS v0.4.3",
        "judge_llm": settings.judge_llm_model,
        "ragas_metrics_used": ["context_precision", "context_recall", "faithfulness"],
        "parallel_config": {
            "rag_concurrency": RAG_CONCURRENCY,
            "ragas_max_workers": RAGAS_MAX_WORKERS,
        },
        "timing_seconds": {
            "phase1_retrieve_generate": round(phase1_elapsed, 1),
            "phase2_ragas_scoring": round(phase2_elapsed, 1),
            "total": round(total_elapsed, 1),
        },
        "average_scores": avg_ragas,
        "objective_metrics": {
            "hit_rate": avg_hit,
            "mrr": avg_mrr,
        },
        "details": detailed_results,
    }

    # 保存
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印报告
    print(f"\n{'='*60}")
    print("[RAGAS Evaluation Report]")
    print(f"  Evaluator: RAGAS v0.4.3")
    print(f"  Judge LLM: {settings.judge_llm_model}")
    print(f"  Total cases: {n}")
    print(f"  ⏱  阶段一（检索+生成）: {phase1_elapsed:.0f}s (并发 {RAG_CONCURRENCY})")
    print(f"  ⏱  阶段二（RAGAS 评分）: {phase2_elapsed:.0f}s (max_workers={RAGAS_MAX_WORKERS})")
    print(f"  ⏱  总计: {total_elapsed:.0f}s")
    print(f"\n  ┌──────────────────────┬──────────┐")
    print(f"  │ Metric               │   Score  │")
    print(f"  ├──────────────────────┼──────────┤")
    for k, v in avg_ragas.items():
        print(f"  │ {k:<20s} │  {v:>6.2%} │")
    print(f"  ├──────────────────────┼──────────┤")
    print(f"  │ {'Hit Rate':<20s} │  {avg_hit:>6.2%} │")
    print(f"  │ {'MRR':<20s} │  {avg_mrr:>6.3f} │")
    print(f"  └──────────────────────┴──────────┘")
    print(f"\nResults saved to: {RESULT_FILE}")

    # 按类型分
    types_in_data = [c.get("type", "unknown") for c in test_cases[:n]]
    if any(t != "unknown" for t in types_in_data):
        print(f"\n--- Per-type breakdown ---")
        type_groups = {}
        for i, t in enumerate(types_in_data):
            type_groups.setdefault(t, []).append(detailed_results[i])

        for t, group in sorted(type_groups.items()):
            cnt = len(group)
            avg_t = {}
            for key in ["context_precision", "context_recall", "faithfulness"]:
                avg_t[key] = round(sum(r["scores"][key] for r in group) / cnt, 4)
            print(f"  [{t}] ({cnt} cases):")
            print(f"    CP={avg_t['context_precision']:.2%}  CR={avg_t['context_recall']:.2%}  "
                  f"Faith={avg_t['faithfulness']:.2%}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_evaluation(source=src))
