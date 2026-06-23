"""查询改写模块。
"""
from openai import AsyncOpenAI

from app.config import settings

# 复用 qwen 客户端做查询改写（轻量任务，用便宜的模型）
transform_client = AsyncOpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)

HYDE_PROMPT = """你是一个HyDE转换系统，请根据下面的问题，写一段 100 字左右的假设性回答。
要求：
1. 假装你已经知道了答案，用陈述句写
2. 语言风格要像文档/论文里的陈述，不要像提问
3. 不要编造具体数字，只写类似"该研究提出了...""实验表明...提升..."这一类结构
4. 直接输出回答内容，不要加"假设答案:"等前缀

问题：{question}

假设性回答："""


async def hyde_transform(query: str) -> str:
    """用 LLM 把问题改写成假设性答案。

    注意：调用方应先用 query_router.should_use_hyde() 判断是否值得改写。
    此函数只做改写，不做路由判断。
    """
    response = await transform_client.chat.completions.create(
        model=settings.hyde_model,  # 用配置里的 HyDE 模型
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=query)}],
        temperature=0.3,
        max_tokens=settings.hyde_max_tokens,
    )
    return response.choices[0].message.content.strip()