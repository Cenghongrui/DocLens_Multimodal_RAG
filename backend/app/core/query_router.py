"""HyDE 智能路由
"""
import re
import json
from openai import AsyncOpenAI

from app.core.logger import logger
from app.config import settings

# ─── 规则（patterns） ───
# 精确匹配模式（含编号、代码、hash、数字密集）→ 不走 HyDE
EXACT_PATTERNS = [
    r'\b[A-Z]{2,}\d+[-.]?\d*\b',        # 错误码/编号：E404-3, API_KEY_3f2a
    r'\b[0-9a-f]{8,}\b',                 # hash / id
    r'\b\d{4,}\b',                       # 长数字串（可能是配置值）
    r'\b\d+[\.-]\d+[\.-]\d+\b',         # 版本号：1.2.3
    r'[A-Z][a-z]+[A-Z]\w*\b',           # 驼峰命名（函数名/类名）
]

# 口语化特征（中文）→ 走 HyDE
COLLOQUIAL_MARKERS = [
    '咋', '啥', '咋样', '咋办', '咋回事', '咋整',
    '怎么搞', '怎么办', '行不行', '能不能', '有没有',
    '多少', '多久', '多大', '几个', '哪',
    '怎么样', '什么用', '干嘛', '怎么弄', '帮我看',
    '帮我', '看一下', '查一下', '找一下',
]

# 学术/书面语特征 → 不走HyDE
FORMAL_MARKERS = [
    '方法', '模型', '算法', '实验', '分析',
    '提出', '基于', '设计', '实现', '评估',
    '优化', '架构', '机制', '框架', '理论',
    '证明', '推导', '结论', '贡献', '改进',
]


def _count_patterns(text: str, patterns: list[str]) -> int:
    """统计文本中命中多少个模式。"""
    count = 0
    for p in patterns:
        if re.search(p, text):
            count += 1
    return count


def _has_any_marker(text: str, markers: list[str]) -> bool:
    """文本是否包含任意一个标记词。"""
    return any(m in text for m in markers)


def rule_based_route(query: str) -> dict:
    """规则层路由判断。

    返回 dict:
      - decision: "skip" | "use" | "uncertain"
      - reason: 判断依据（用于日志和调试）
    """
    if _count_patterns(query, EXACT_PATTERNS) >= 1:
        return {"decision": "skip", "reason": "query 含编号/代码/数字，需精确匹配"}

    if re.match(r'^[\d\s\-_.\/]+$', query):
        return {"decision": "skip", "reason": "query 是纯数字/代码，不需要改写"}

    if len(query) < 20 and _has_any_marker(query, COLLOQUIAL_MARKERS):
        return {"decision": "use", "reason": "口语化短 query，HyDE 能补全语义"}

    if len(query) > 50 and _has_any_marker(query, FORMAL_MARKERS):
        return {"decision": "skip", "reason": "query 已是书面语风格，无需改写"}

    if len(query) < 15 and not _count_patterns(query, EXACT_PATTERNS):
        return {"decision": "use", "reason": "短 query 语义不完整，HyDE 能展开"}

    return {"decision": "uncertain", "reason": "规则无法确定，交给 LLM 判断"}


# ─── 规则不确定时打分 ───

ROUTE_JUDGE_PROMPT = """你是一个检索系统路由器。判断下面的用户问题是否适合用 HyDE（假设性答案展开）来做查询改写。

HyDE 的原理：先把用户问题展开成一段假设性答案，再用答案去检索文档。
- 适合 HyDE：口语化、模糊、语义不完整的问题（如"这咋用的？""效果怎么样？"）
- 不适合 HyDE：精确查询、含专有名词/编号、已经是专业表述的问题

用户问题：{question}

请判断是否适合 HyDE，回复 JSON，不要带markdown代码块：{{"suitable": true/false, "confidence": 0.0~1.0, "reason": "..."}}"""


async def llm_route_judge(query: str) -> dict:
    """LLM 兜底判断（仅在规则层返回 uncertain 时调用）。"""
    client = AsyncOpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
    )
    raw = ""
    try:
        resp = await client.chat.completions.create(
            model=settings.judge_llm_model,
            messages=[{"role": "user", "content": ROUTE_JUDGE_PROMPT.format(question=query)}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        logger.debug(f"[HyDE Router] LLM raw response: {raw}")

        # LLM 经常包 markdown code block，先剥离
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("\n```", 1)[0]
            raw = raw.strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"[HyDE Router] LLM 返回非标准 JSON: {e} | raw={raw[:200]}")
        return {"suitable": False, "confidence": 0.0, "reason": f"JSON 解析失败: {e}"}
    except Exception as e:
        logger.warning(f"[HyDE Router] LLM 调用失败: {type(e).__name__}: {e}")
        return {"suitable": False, "confidence": 0.0, "reason": f"LLM 调用失败: {e}"}


# ─── 路由主函数 ───

async def should_use_hyde(query: str, threshold: float = 0.5) -> tuple[bool, str]:
    """判断一条 query 是否应该走 HyDE 通道。

    Args:
        query: 用户原始问题
        threshold: LLM 兜底判断的置信度阈值

    Returns:
        (should_use: bool, reason: str)
    """
    # 第一层：规则判断
    rule_result = rule_based_route(query)

    if rule_result["decision"] == "skip":
        logger.info(f"[HyDE Router] SKIP (rule): {rule_result['reason']} | query={query[:50]}")
        return False, rule_result["reason"]
    elif rule_result["decision"] == "use":
        logger.info(f"[HyDE Router] USE (rule): {rule_result['reason']} | query={query[:50]}")
        return True, rule_result["reason"]

    # 第二层：规则不确定 → LLM 打分
    logger.info(f"[HyDE Router] UNCERTAIN → LLM judge | query={query[:50]}")
    llm_result = await llm_route_judge(query)

    should_use = llm_result.get("suitable", False) and llm_result.get("confidence", 0) >= threshold
    reason = llm_result.get("reason", "LLM 兜底判断")

    logger.info(
        f"[HyDE Router] {'USE' if should_use else 'SKIP'} (LLM): "
        f"confidence={llm_result.get('confidence', 0):.2f} | {reason}"
    )
    return should_use, reason
