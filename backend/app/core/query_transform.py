"""HyDE 查询改写：问题 → 假设性回答。"""
from openai import AsyncOpenAI
from app.config import settings

_client = AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)

_PROMPT = """根据下面的问题写一段 100 字左右的假设性回答。
要求：
1. 用陈述句，假装已经知道答案
2. 语言风格接近论文陈述，不要像提问
3. 不编造具体数字
4. 直接输出回答内容

问题：{question}
回答："""


async def hyde_transform(query: str) -> str:
    """用 LLM 将问题改写为假设性答案。调用前应先通过 should_use_hyde() 判断。"""
    resp = await _client.chat.completions.create(
        model=settings.hyde_model,
        messages=[{"role": "user", "content": _PROMPT.format(question=query)}],
        temperature=0.3,
        max_tokens=settings.hyde_max_tokens,
    )
    return resp.choices[0].message.content.strip()
