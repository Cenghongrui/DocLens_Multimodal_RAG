"""HyDE 路由：规则层 → LLM 兜底层。"""

import re, json
from openai import AsyncOpenAI
from app.core.logger import logger
from app.config import settings

# 精确匹配模式（代码/编号/Hash）→ 不走 HyDE
_EXACT = [
    r"\b[A-Z]{2,}\d+[-.]?\d*\b", r"\b[0-9a-f]{8,}\b", r"\b\d{4,}\b",
    r"\b\d+[\\.-]\d+[\\.-]\d+\b", r"[A-Z][a-z]+[A-Z]\w*\b",
]
# 口语化标记 → 走 HyDE
_COLLOQUIAL = [
    "咋", "啥", "咋样", "咋办", "咋回事", "怎么搞", "怎么办",
    "行不行", "能不能", "有没有", "多少", "多久", "多大", "几个", "哪",
    "怎么样", "什么用", "干嘛", "怎么弄", "帮我看", "帮我", "看一下", "查一下", "找一下",
]
# 书面语标记 → 不走 HyDE
_FORMAL = [
    "方法", "模型", "算法", "实验", "分析", "提出", "基于",
    "设计", "实现", "评估", "优化", "架构", "机制", "框架", "理论",
]


def _match_count(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def _has_any(text: str, markers: list[str]) -> bool:
    return any(m in text for m in markers)


def rule_based_route(query: str) -> dict:
    """第一层：规则判断。返回 {"decision": "skip"|"use"|"uncertain", "reason": "..."}。"""
    if _match_count(query, _EXACT) >= 1:
        return {"decision": "skip", "reason": "含编号/代码，需精确匹配"}
    if re.match(r"^[\d\s\-_./]+$", query):
        return {"decision": "skip", "reason": "纯数字/代码"}
    if len(query) < 20 and _has_any(query, _COLLOQUIAL):
        return {"decision": "use", "reason": "口语化短 query"}
    if len(query) > 50 and _has_any(query, _FORMAL):
        return {"decision": "skip", "reason": "书面语，无需改写"}
    if len(query) < 15:
        return {"decision": "use", "reason": "短 query，HyDE 有帮助"}
    return {"decision": "uncertain", "reason": "规则不确定，交 LLM 判断"}


_ROUTE_PROMPT = """判断以下用户问题是否适合用 HyDE（假设性答案展开）来做查询改写。

HyDE 原理：先把问题展开成一段假设性答案，再用答案去检索。
- 适合 HyDE：口语化、模糊、语义不完整的问题
- 不适合 HyDE：精确查询、含专有名词/编号、已是专业表述

问题：{question}

回复 JSON：{{"suitable": true/false, "confidence": 0.0~1.0, "reason": "..."}}"""


async def llm_route_judge(query: str) -> dict:
    """第二层：LLM 兜底（仅规则层返回 uncertain 时调用）。"""
    client = AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)
    raw = ""
    try:
        resp = await client.chat.completions.create(
            model=settings.judge_llm_model,
            messages=[{"role": "user", "content": _ROUTE_PROMPT.format(question=query)}],
            temperature=0.0, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("\n```", 1)[0].strip()
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("LLM route judge failed: %s | raw=%s", e, raw[:200])
        return {"suitable": False, "confidence": 0.0, "reason": f"judge failed: {e}"}


async def should_use_hyde(query: str, threshold: float = 0.5) -> tuple:
    """两级路由：规则 → LLM 兜底。返回 (是否启用, 原因)。"""
    rule = rule_based_route(query)
    if rule["decision"] == "skip":
        logger.info("[HyDE] SKIP (rule): %s | query=%s", rule["reason"], query[:50])
        return False, rule["reason"]
    if rule["decision"] == "use":
        logger.info("[HyDE] USE (rule): %s | query=%s", rule["reason"], query[:50])
        return True, rule["reason"]

    logger.info("[HyDE] UNCERTAIN → LLM | query=%s", query[:50])
    llm = await llm_route_judge(query)
    use = llm.get("suitable", False) and llm.get("confidence", 0) >= threshold
    logger.info("[HyDE] %s (LLM): conf=%.2f | %s", "USE" if use else "SKIP", llm.get("confidence", 0), llm.get("reason", ""))
    return use, llm.get("reason", "")
