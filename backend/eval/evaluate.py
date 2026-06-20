"""
RAG 评估脚本  LLM-as-Judge

评估维度：
  Context Precision  检索到的文档上下文是否精确，有没有噪声
  Context Recall     是否检索到了回答所需的所有信息
  Faithfulness       回答是否忠实于检索到的文档（有无幻觉）
  Answer Relevancy   回答是否切题

用法：
  uv run python -m eval.evaluate
"""

import json
import asyncio
import sys
from pathlib import Path
from typing import List

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core.retriever import retrieve
from app.core.generator import generate
from app.config import settings

# Windows 下强制 stdout 用 UTF-8，避免 emoji/中文报错
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Judge LLM  用 DeepSeek ───
judge_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

EVAL_DIR = Path(__file__).parent
TEST_DATASET = EVAL_DIR / "test_dataset.json"
RESULT_FILE = EVAL_DIR / "result.json"


# ─── 结构化评分输出 ───
class MetricScores(BaseModel):
    context_precision: float      # 0-1，检索结果的相关性
    context_recall: float         # 0-1，检索的完整性
    faithfulness: float           # 0-1，回答是否无幻觉
    answer_relevancy: float       # 0-1，回答是否切题
    reasoning: str                # 评分理由


# ─── 评分 Prompt ───
JUDGE_PROMPT = """你是一个 RAG 系统的评估专家。请基于以下信息，对系统的输出进行评分。

【用户问题】
{question}

【参考标准答案（如果有）】
{ground_truth}

【检索到的文档上下文】（共 {context_count} 段）
{contexts}

【系统回答】
{answer}

请按以下标准逐项打分（每项 0-10 整数，10 为满分。最终 JSON 中各项值除以 10，即填 0.0 ~ 1.0 的小数）：

1. **Context Precision（上下文精确度）**：
   检索到的文档中有多少与问题相关？无关噪声多吗？
   10: 所有上下文都高度相关，无噪声 / 5: 部分相关，有噪声 / 0: 完全不相关

2. **Context Recall（上下文覆盖度）**：
   检索到的上下文是否包含了回答所需的所有信息？
   10: 包含全部所需信息 / 5: 只包含部分 / 0: 完全不包含

3. **Faithfulness（忠实度）**：
   回答中的每一条事实是否都能在检索到的文档中找到依据？
   10: 所有陈述都有依据，无编造 / 5: 部分有依据 / 0: 完全编造

4. **Answer Relevancy（回答切题度）**：
   回答是否直接回应了用户的问题？
   10: 完全切题 / 5: 部分切题 / 0: 完全不切题

严格按以下 JSON 格式输出，各项必须是 0.0 ~ 1.0 的小数：
{{"reasoning": "理由", "context_precision": 0.X, "context_recall": 0.X, "faithfulness": 0.X, "answer_relevancy": 0.X}}
只输出这一行 JSON，不要任何其他文字。"""

def _extract_json(raw: str):
    """多层回退提取 JSON"""
    # 尝试 1：直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    import re
    # 尝试 2：```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试 3：第一个 { 到最后一个 }（可能截断，补全）
    m = re.search(r'\{[\s\S]*', raw)
    if m:
        truncated = m.group(0)
        # 补全可能的截断 JSON
        for fix in [truncated, truncated + '"}', truncated + '"]}', truncated + '"]}' + '"}']:
            try:
                return json.loads(fix)
            except json.JSONDecodeError:
                pass

    return None


async def judge_one(question: str, ground_truth: str, answer: str, contexts: List[str]) -> MetricScores:
    """用 LLM 对单条结果打分"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth or "（未提供）",
        context_count=len(contexts),
        contexts="\n---\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)),
        answer=answer,
    )

    import time
    # 限速 + 重试，避免 API 限流返回空白
    raw = ""
    for attempt in range(3):
        time.sleep(1.5)
        response = await judge_client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,  # 增大，避免 reasoning + JSON 截断
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw:
            break
        print(f"  [RETRY] empty response, attempt {attempt+1}/3")

    print(f"  [DEBUG] judge raw ({len(raw)} chars): {raw[:200]}")

    data = _extract_json(raw)

    if data is None:
        print(f"  [WARN] parse failed after 3 attempts")
        return MetricScores(
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            reasoning=f"parse_failed: {raw[:200]}",
        )

    cp = float(data.get("context_precision", 0))
    cr = float(data.get("context_recall", 0))
    faith = float(data.get("faithfulness", 0))
    ar = float(data.get("answer_relevancy", 0))

    # 归一化：如果分数是 0-10 分制，统一除以 10 转为 0-1
    if cp > 1.0 or cr > 1.0 or faith > 1.0 or ar > 1.0:
        cp /= 10.0
        cr /= 10.0
        faith /= 10.0
        ar /= 10.0

    return MetricScores(
        context_precision=cp,
        context_recall=cr,
        faithfulness=faith,
        answer_relevancy=ar,
        reasoning=data.get("reasoning", ""),
    )


async def run_evaluation():
    """主评估流程"""
    # 1. 加载测试集
    with open(TEST_DATASET, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    # 2. 逐条跑 RAG 链路 + 评分
    results = []
    scores_sum = {"context_precision": 0, "context_recall": 0, "faithfulness": 0, "answer_relevancy": 0}

    for i, case in enumerate(test_cases):
        q = case["question"].strip()
        gt = case.get("ground_truth", "").strip()

        if not q:
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_cases)}] Q: {q}")

        # 检索
        docs = await retrieve(q)
        contexts = [d.page_content for d in docs]
        print(f"  Retrieved {len(contexts)} chunks")

        # 生成
        answer = await generate(q, docs)
        print(f"  Answer: {answer[:120]}...")

        # 评分
        scores = await judge_one(q, gt, answer, contexts)
        print(f"  CP={scores.context_precision:.2f}  CR={scores.context_recall:.2f}  "
              f"Faith={scores.faithfulness:.2f}  AR={scores.answer_relevancy:.2f}")

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "contexts": contexts,
            "scores": scores.model_dump(),
        })

        for k in scores_sum:
            scores_sum[k] += getattr(scores, k)

    # 3. 汇总
    n = len(results) or 1
    avg = {k: round(v / n, 3) for k, v in scores_sum.items()}
    summary = {
        "total_cases": n,
        "average_scores": avg,
        "details": results,
    }

    # 4. 保存
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 5. 输出报告
    print(f"\n{'='*60}")
    print("[Evaluation Report]")
    print(f"  Total cases: {n}")
    print(f"  Context Precision:   {avg['context_precision']:.2%}")
    print(f"  Context Recall:      {avg['context_recall']:.2%}")
    print(f"  Faithfulness:        {avg['faithfulness']:.2%}")
    print(f"  Answer Relevancy:    {avg['answer_relevancy']:.2%}")
    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
