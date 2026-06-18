import base64
from openai import AsyncOpenAI

from app.config import settings

vision_client = AsyncOpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)


async def describe_image(image_path: str) -> str:
    """调 Qwen-VL，返回图片中文描述"""

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "image/png")

    response = await vision_client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "请十分详细的描述这张图片的内容。图表请完整提取关键数据，架构图请描述结构和组件等，并加以总结。用中文回答。",
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content