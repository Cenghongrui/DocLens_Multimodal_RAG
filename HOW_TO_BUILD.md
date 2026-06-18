# DocLens 后端手动实现指南

> 跟着这篇文档，从零手写一个多模态 RAG 后端。
> 每个步骤都会解释**为什么这样做**，然后你照着写代码。
> 全部完成后，你得到一个能跑通的 FastAPI + RAG 后端，并且理解每一行代码。

---

## 目录

| 步骤 | 文件 | 做什么 |
|---|---|---|
| [Step 1](#1-step-1环境与依赖) | `pyproject.toml` | uv 初始化 + 安装依赖 |
| [Step 2](#2-step-2项目骨架) | 目录 + `__init__.py` | 建目录结构 |
| [Step 3](#3-step-3配置管理--configpy) | `config.py` | Pydantic Settings 加载 `.env` |
| [Step 4](#4-step-4数据模型--schemaspy) | `schemas.py` | 请求/响应 Pydantic 模型 |
| [Step 5](#5-step-5fastapi-入口--mainpy) | `main.py` | FastAPI 实例 + CORS + 路由骨架 |
| [Step 6](#6-step-6文档加载--loaderpy) | `loader.py` | PDF/MD/TXT/图片 → LangChain Document |
| [Step 7](#7-step-7文本分块--splitterpy) | `splitter.py` | 递归文本切分 |
| [Step 8](#8-step-8图片理解--visionpy) | `vision.py` | Qwen-VL 生成图片描述 |
| [Step 9](#9-step-9embedding--embedderpy) | `embedder.py` | 通义千问 Embedding + ChromaDB 连接 |
| [Step 10](#10-step-10检索器--retrieverpy) | `retriever.py` | 问题→向量→相似度检索 |
| [Step 11](#11-step-11rag-生成器--generatorpy) | `generator.py` | Prompt 拼接 + DeepSeek 生成 |
| [Step 12](#12-step-12上传-api--ingestpy) | `api/ingest.py` | **全链路串联**：上传→解析→入库 |
| [Step 13](#13-step-13对话-api--chatpy) | `api/chat.py` | 检索→生成→返回 |
| [Step 14](#14-step-14文档列表-api--documentspy) | `api/documents.py` | 已索引文档查询 |
| [Step 15](#15-step-15组装-mainpy) | `main.py` 最终版 | 注册所有路由 |
| [Step 16](#16-step-16联调测试) | — | 启动 + Swagger 测试 |

---

## 前置概念

写代码前，先搞清楚几个概念。不需要背，写代码的时候回来对照。

### FastAPI 是什么

```
浏览器请求 → FastAPI（路由匹配）→ 你的函数 → 返回 JSON
                │
         @app.get("/api/documents")  →  GET 请求走这里
         @app.post("/api/chat")      →  POST 请求走这里
```

FastAPI 自动生成 Swagger 文档（访问 `/docs`），可以在网页上直接测试 API。

### async / await 是什么

调外部 API 需要等网络返回。同步代码在这期间什么都做不了，异步代码可以去处理其他请求。

```python
# 同步：卡住等
response = client.chat.completions.create(model="deepseek-chat", ...)

# 异步：等的时候去干别的事
response = await async_client.chat.completions.create(model="deepseek-chat", ...)
```

你的代码里所有调外部 API 的地方都用 `AsyncOpenAI` + `await`。

### LangChain 在你项目里只做三件事

1. **Document 容器**：`Document(page_content="文本", metadata={"source":"a.pdf"})`
2. **Text Splitter**：`RecursiveCharacterTextSplitter` 切文本
3. **Chroma 封装**：`Chroma()` 包装 ChromaDB 操作

别的不用管，v1 不是来学 LangChain 的。

### Embedding 是什么

```
"苹果" → [0.12, 0.45, ..., 0.78]  ← 1024个浮点数
"香蕉" → [0.11, 0.43, ..., 0.80]  ← 和"苹果"的向量很近
"汽车" → [0.89, 0.02, ..., 0.15]  ← 和"苹果"的向量很远

向量夹角越小 = 语义越相似 = 余弦相似度越高
```

### ChromaDB 是什么

```
ChromaDB（本地文件夹 vectordb/）
  └── Collection "doclens"
      ├── doc_1: text="...", embedding=[0.1, 0.2, ...], metadata={source:"a.pdf"}
      ├── doc_2: text="...", embedding=[0.3, 0.4, ...], metadata={source:"a.pdf"}
      └── ...

存入：collection.add(documents, embeddings, metadatas)
查询：collection.query(query_embedding, n_results=5) → 返回最相似的 5 个
```

---

## 1. Step 1：环境与依赖

```bash
cd d:/桌面/agentRag/backend
uv init --name doclens-backend
```

`uv init` 会自动生成 `pyproject.toml`、`README.md`、`.python-version` 和一个示例 `main.py`。

把 uv 生成的示例文件删掉（我们用 `app/main.py` 而不是根目录的 `main.py`）：

```bash
rm main.py README.md
```

### 安装依赖

```bash
uv add fastapi "uvicorn[standard]" langchain langchain-community langchain-text-splitters chromadb PyMuPDF openai pydantic-settings python-multipart python-dotenv
```

一次装完，uv 会自动写入 `pyproject.toml` 并生成 `uv.lock`。

### 创建 `.env`

```bash
# 在 backend/ 目录下
echo 'DEEPSEEK_API_KEY=sk-你的key
QWEN_API_KEY=sk-你的key' > .env
```

### 创建 `.gitignore`

```bash
echo '__pycache__/
.venv/
.env
data/*
images/*
vectordb/*
!data/.gitkeep
!images/.gitkeep
!vectordb/.gitkeep
*.pyc
dist/' > .gitignore
```

### 验证

```bash
uv run python -c "import fastapi; print('ok')"
# 应该输出: ok
```

---

## 2. Step 2：项目骨架

确认当前在 `backend/` 目录下，建子目录：

```bash
mkdir -p app/api app/core app/models data images vectordb
```

创建 `.py` 文件（包括 `__init__.py`）：

```bash
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/models/__init__.py
touch app/main.py
touch app/config.py
touch app/api/chat.py
touch app/api/ingest.py
touch app/api/documents.py
touch app/core/loader.py
touch app/core/splitter.py
touch app/core/vision.py
touch app/core/embedder.py
touch app/core/retriever.py
touch app/core/generator.py
touch app/models/schemas.py
```

文件夹占位：

```bash
touch data/.gitkeep images/.gitkeep vectordb/.gitkeep
```

完成后目录结构：

```
backend/
├── .env
├── .gitignore
├── .python-version
├── pyproject.toml              ← uv 管理的依赖
├── uv.lock                     ← 锁死版本
├── app/
│   ├── __init__.py
│   ├── main.py                 ← FastAPI 入口（下一步写）
│   ├── config.py               ← 配置中心
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py             ← POST /api/chat
│   │   ├── ingest.py           ← POST /api/ingest
│   │   └── documents.py        ← GET  /api/documents
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loader.py           ← 文档加载
│   │   ├── splitter.py         ← 文本分块
│   │   ├── vision.py           ← 图片理解
│   │   ├── embedder.py         ← Embedding + ChromaDB 连接
│   │   ├── retriever.py        ← 检索
│   │   └── generator.py        ← LLM 生成
│   └── models/
│       ├── __init__.py
│       └── schemas.py          ← 数据模型
├── data/                       ← 上传的文档存这里
├── images/                     ← PDF 中提取的图片存这里
└── vectordb/                   ← ChromaDB 持久化数据
```

---

## 3. Step 3：配置管理 → `config.py`

### 要解决什么

API Key、模型名、chunk_size 这些不能写死在代码里。改一个值不应该翻遍所有文件。

### 怎么写

Pydantic Settings 自动从 `.env` 读配置，字段名自动匹配（`deepseek_api_key` → `DEEPSEEK_API_KEY`）。

打开 `app/config.py`，写：

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，自动从 .env 文件加载"""

    # ─── API Keys ───
    deepseek_api_key: str
    qwen_api_key: str

    # ─── API 地址 ───
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ─── 模型名 ───
    llm_model: str = "deepseek-chat"
    vision_model: str = "qwen-vl-plus"
    embedding_model: str = "text-embedding-v3"
    embedding_dimension: int = 1024

    # ─── 分块参数 ───
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ─── 检索参数 ───
    top_k: int = 5

    # ─── 本地路径 ───
    data_dir: str = "data/"
    image_dir: str = "images/"
    vectordb_dir: str = "vectordb/"

    class Config:
        env_file = ".env"


# 全局单例，其他文件 from app.config import settings 即可
settings = Settings()
```

### 为什么用 `class Config: env_file = ".env"`

告诉 Pydantic 去读 `.env`。字段名自动转换：`deepseek_api_key` → `DEEPSEEK_API_KEY`。带默认值的字段（如 `deepseek_base_url`）`.env` 里可以不写。

### 验证

```bash
cd backend
uv run python -c "from app.config import settings; print(settings.embedding_dimension)"
# 输出: 1024
```

---

## 4. Step 4：数据模型 → `schemas.py`

### 要解决什么

前后端用 JSON 通信，Python 用对象。Pydantic 自动互转 + 校验。

```
前端 JSON → Pydantic 验证 → Python 对象 → 你的业务逻辑
你的返回值 → Pydantic 序列化 → JSON → 前端
```

打开 `app/models/schemas.py`，写：

```python
from pydantic import BaseModel
from typing import List, Optional


# ─── 请求 ───

class ChatRequest(BaseModel):
    """POST /api/chat 的请求体"""
    message: str


# ─── 响应 ───

class SourceItem(BaseModel):
    """单个引用来源"""
    content: str
    source: str
    page: int = 0
    type: str = "text"          # "text" 或 "image"
    image_url: Optional[str] = None


class ChatResponse(BaseModel):
    """POST /api/chat 的响应体"""
    answer: str
    sources: List[SourceItem] = []


class IngestResponse(BaseModel):
    """POST /api/ingest 的响应体"""
    status: str                 # "ok" 或 "error"
    file_name: str
    chunks_created: int = 0
    images_processed: int = 0
    message: str = ""


class DocumentItem(BaseModel):
    """GET /api/documents 返回的单条文档"""
    name: str
    chunks: int
    images: int
    ingested_at: str = ""


class DocumentListResponse(BaseModel):
    """GET /api/documents 的响应体"""
    documents: List[DocumentItem] = []
```

---

## 5. Step 5：FastAPI 入口 → `main.py`

打开 `app/main.py`，写：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DocLens API",
    version="0.1.0",
)

# ─── CORS ───
# 前端 localhost:5173，后端 localhost:8000，浏览器默认禁止跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "DocLens API is running"}
```

### 验证

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

浏览器访问 `http://localhost:8000` → 显示 `{"message":"DocLens API is running"}`  
访问 `http://localhost:8000/docs` → Swagger 文档页

`Ctrl+C` 停掉，继续写后面的。

---

## 6. Step 6：文档加载 → `loader.py`

### 要解决什么

PDF、Markdown、图片是三种完全不同的格式。loader 把它们统一变成 `langchain_core.documents.Document`，后续分块、embedding 只认这个类型。

### 多模态策略

PDF 的每一页：文字 → 单独 Document，图片 → 提取保存到 `images/` → 创建空的 Document（等 vision.py 填内容）。

打开 `app/core/loader.py`，写：

```python
import os
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document


def load_file(file_path: str) -> Tuple[List[Document], List[str]]:
    """
    加载单个文件 → (文档列表, 图片路径列表)
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in (".md", ".txt"):
        return _load_text(file_path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return _load_image(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _load_pdf(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    documents = []
    image_paths = []

    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        # ① 提取文字
        text = page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_num + 1,
                    "type": "text",
                }
            ))

        # ② 提取图片
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"{Path(file_path).stem}_p{page_num+1}_img{img_index+1}.{image_ext}"
            image_save_path = os.path.join("images", image_filename)

            with open(image_save_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_save_path)

            # 图片先占位，page_content 等 vision.py 填充
            documents.append(Document(
                page_content="",
                metadata={
                    "source": source,
                    "page": page_num + 1,
                    "type": "image",
                    "image_path": image_save_path,
                }
            ))

    doc.close()
    return documents, image_paths


def _load_text(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return [], []

    doc = Document(
        page_content=text,
        metadata={"source": source, "page": 0, "type": "text"},
    )
    return [doc], []


def _load_image(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    doc = Document(
        page_content="",
        metadata={
            "source": source,
            "page": 0,
            "type": "image",
            "image_path": file_path,
        },
    )
    return [doc], [file_path]
```

### 关键理解

- `page.get_images(full=True)` 返回该页所有内嵌图片的列表
- `xref` 是 PDF 内部图片的引用号，`doc.extract_image(xref)` 取出原始字节
- 图片 Document 的 `page_content` 初始为空字符串，等 vision 填充
- 返回 `Tuple[List[Document], List[str]]`：文字文档和图片路径分开，方便中间处理

---

## 7. Step 7：文本分块 → `splitter.py`

### 为什么要分块

一篇 PDF 50000 字，整个塞给 Embedding API 要么截断要么"模糊"。切成小段后，检索时能精确匹配到相关段落。

### 为什么 overlap

```
Chunk 1: "...系统核心包括 API 网关、"
Chunk 2: "API 网关、消息队列、数据库..."
               ↑ 重叠 200 字，确保跨边界的句子不丢失
```

打开 `app/core/splitter.py`，写：

```python
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def create_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # 优先级：段落 → 行 → 句子 → 空格 → 字符
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )


def split_documents(documents: List[Document]) -> List[Document]:
    """
    文本 Document 切块，图片 Document 原样保留
    """
    splitter = create_splitter()
    result = []

    for doc in documents:
        if doc.metadata.get("type") == "image":
            result.append(doc)   # 图片不切
        else:
            chunks = splitter.split_documents([doc])
            result.extend(chunks)

    for i, chunk in enumerate(result):
        chunk.metadata["chunk_id"] = i

    return result
```

### RecursiveCharacterTextSplitter 的工作方式

```
原文本（3000 字）
  ↓ 先试 \n\n 切 → 切成 3 段
  ↓ 第 2 段还是 1500 字 → 试 \n 切 → 切成 2 段
  ↓ 其中一段还是 900 字 → 试 。切 → 切成 2 段
  ↓ 全部 ≤ chunk_size → 完成
```

---

## 8. Step 8：图片理解 → `vision.py`

### 要做什么

```python
图片文件 → base64 → AsyncOpenAI → Qwen-VL → 中文描述
```

### 用 OpenAI SDK

通义千问百炼兼容 OpenAI SDK，直接用 `openai` 库无需 `httpx` 手写。

打开 `app/core/vision.py`，写：

```python
import base64
from openai import AsyncOpenAI

from app.config import settings

# 全局 client，一次创建到处复用
vision_client = AsyncOpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)


async def describe_image(image_path: str) -> str:
    """调 Qwen-VL，返回图片中文描述"""

    # ① 读图 + 转 base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # ② 猜 MIME 类型
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "image/png")

    # ③ 调 API
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
                        "text": "请详细描述这张图片的内容。图表请提取关键数据，架构图请描述结构和组件。用中文回答。",
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content
```

### 好处

比 `httpx` 手写少了一半代码，SDK 内部处理了重试、超时、连接复用。尤其图片的 base64 体积大，连接复用能省时间。

---

## 9. Step 9：Embedding → `embedder.py`

### 要做什么

两件事：
1. 文本 → 调通义千问 Embedding API → 1024 维向量
2. 提供 `get_or_create_collection()` 给其他地方拿 ChromaDB 连接

打开 `app/core/embedder.py`，写：

```python
import os
from typing import List

from openai import OpenAI, AsyncOpenAI
from langchain_chroma import Chroma
from chromadb import PersistentClient

from app.config import settings

# 全局 client（embedding 只同步调，因为 LangChain 接口是同步的）
embedding_client = OpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)

# 异步 client（retriever.py 用）
async_embedding_client = AsyncOpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)


async def embed_text(text: str) -> List[float]:
    """单段文本 → 1024 维向量（异步）"""
    response = await async_embedding_client.embeddings.create(
        model=settings.embedding_model,
        input=text,
        dimensions=settings.embedding_dimension,
    )
    return response.data[0].embedding


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """批量文本 → 批量向量（异步，比逐个调快）"""
    response = await async_embedding_client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimension,
    )
    return [item.embedding for item in response.data]


# ─── ChromaDB 连接 ───

def get_or_create_collection() -> Chroma:
    """获取 ChromaDB collection（不存在则自动创建）"""
    os.makedirs(settings.vectordb_dir, exist_ok=True)

    chroma_client = PersistentClient(path=settings.vectordb_dir)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name="doclens",
        embedding_function=_EmbeddingFunction(),
    )
    return vectorstore


# ─── LangChain Embedding 适配器 ───

class _EmbeddingFunction:
    """
    LangChain Chroma 要求 embedding function 实现两个方法。
    内部转调我们的 embedding 函数。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量入库用（同步）"""
        response = embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
            dimensions=settings.embedding_dimension,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """查询用（同步）"""
        response = embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimension,
        )
        return response.data[0].embedding
```

### 为什么分同步和异步两个 client

`_EmbeddingFunction` 是给 LangChain Chroma 用的适配器，它的接口要求同步。但 `retriever.py` 里我们自己调 `embed_text()` 时想要异步。所以：
- `embedding_client`（同步 OpenAI）→ 给适配器用
- `async_embedding_client`（异步 AsyncOpenAI）→ 给 `retriever.py` 用

这样彻底消灭了 `asyncio.run()`。

---

## 10. Step 10：检索器 → `retriever.py`

### 要做什么

```
用户问题 → embed_text() 转向量 → ChromaDB 查 top-K 相似 → 返回 Document 列表
```

打开 `app/core/retriever.py`，写：

```python
from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.embedder import get_or_create_collection, embed_text


async def retrieve(query: str, top_k: int = None) -> List[Document]:
    """检索与 query 最相关的文档片段"""
    if top_k is None:
        top_k = settings.top_k

    query_embedding = await embed_text(query)
    vectorstore = get_or_create_collection()

    docs = vectorstore.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k,
    )
    return docs
```

---

## 11. Step 11：RAG 生成器 → `generator.py`

### 要做什么

```
检索结果 + 用户问题 → Prompt 模板 → DeepSeek → 回答
```

DeepSeek 也兼容 OpenAI SDK，和 Vision/Embedding 用同样的调用方式。

打开 `app/core/generator.py`，写：

```python
from typing import List
from openai import AsyncOpenAI

from langchain_core.documents import Document

from app.config import settings

# 全局 client
llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)


def build_prompt(query: str, retrieved_docs: List[Document]) -> str:
    """组装 Prompt"""

    parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        doc_type = doc.metadata.get("type", "text")

        icon = "📷" if doc_type == "image" else "📄"
        parts.append(
            f"[来源 {i+1}] {icon}（{source}，第 {page} 页）：\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(parts)

    return f"""你是一个知识库助手。你的回答必须基于以下检索到的文档内容。

【规则】
1. 优先使用文档内容回答问题
2. 引用时注明来源编号，如"根据 [来源 1]..."
3. 文档内容不足以回答时，直接说"根据已有文档，我无法回答这个问题"，不要编造
4. 用中文回答，简洁准确

【文档内容】
{context}

【用户问题】
{query}"""


async def generate(query: str, retrieved_docs: List[Document]) -> str:
    """调 DeepSeek 生成回答"""
    prompt = build_prompt(query, retrieved_docs)

    response = await llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content
```

### 为什么 temperature=0.1

知识库问答不需要创造性，需要准确。低温度降低幻觉。

### 为什么不用 system role

DeepSeek 对 system role 支持不一致，直接全放 user message 最稳。

---

## 12. Step 12：上传 API → `ingest.py`

### 全链路

这是整个项目的核心。打开 `app/api/ingest.py`，写：

```
文件上传 → 存到 data/
  → loader.load_file()           提取文字 + 提取图片
  → vision.describe_image()      给每张图片生成描述
  → splitter.split_documents()   文本分块
  → vectorstore.add_documents()  自动 embedding + 入库
  → 返回结果
```

```python
import os

from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.models.schemas import IngestResponse
from app.core.loader import load_file
from app.core.vision import describe_image
from app.core.splitter import split_documents
from app.core.embedder import get_or_create_collection

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """上传文档 → 解析 → 图片理解 → 分块 → embedding → 入库"""
    try:
        # ① 保存文件
        os.makedirs(settings.data_dir, exist_ok=True)
        file_path = os.path.join(settings.data_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # ② 加载文档
        documents, image_paths = load_file(file_path)

        # ③ Vision：给图片 Document 填描述
        images_processed = 0
        for doc in documents:
            if doc.metadata.get("type") == "image":
                img_path = doc.metadata.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    doc.page_content = f"[图片描述] {await describe_image(img_path)}"
                    images_processed += 1

        # ④ 分块
        chunks = split_documents(documents)

        # ⑤ 过滤空内容
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            return IngestResponse(
                status="error",
                file_name=file.filename,
                message="文档没有可提取的内容",
            )

        # ⑥ 向量化 + 入库（LangChain Chroma 内部调 embedding）
        vectorstore = get_or_create_collection()
        vectorstore.add_documents(chunks)

        return IngestResponse(
            status="ok",
            file_name=file.filename,
            chunks_created=len(chunks),
            images_processed=images_processed,
        )

    except Exception as e:
        return IngestResponse(
            status="error",
            file_name=file.filename,
            message=str(e),
        )
```

### 为什么 `vectorstore.add_documents(chunks)` 一行就完成了 embedding + 入库

LangChain 的 `Chroma.add_documents()` 内部调用 `_EmbeddingFunction.embed_documents()` 做批量 embedding，然后调 ChromaDB 原生 API 入库。你不是没写 embedding 逻辑，是封装在 `embedder.py` 的适配器里了。

---

## 13. Step 13：对话 API → `chat.py`

打开 `app/api/chat.py`，写：

```python
from pathlib import Path

from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, SourceItem
from app.core.retriever import retrieve
from app.core.generator import generate

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话：检索 + 生成"""

    # ① 检索
    retrieved_docs = await retrieve(request.message)

    # ② 生成
    answer = await generate(request.message, retrieved_docs)

    # ③ 组装来源引用
    sources = []
    for doc in retrieved_docs:
        item = SourceItem(
            content=doc.page_content[:200],
            source=doc.metadata.get("source", "未知"),
            page=doc.metadata.get("page", 0),
            type=doc.metadata.get("type", "text"),
        )

        if doc.metadata.get("type") == "image":
            image_path = doc.metadata.get("image_path", "")
            if image_path:
                item.image_url = f"/images/{Path(image_path).name}"

        sources.append(item)

    return ChatResponse(answer=answer, sources=sources)
```

---

## 14. Step 14：文档列表 API → `documents.py`

打开 `app/api/documents.py`，写：

```python
from fastapi import APIRouter

from app.models.schemas import DocumentItem, DocumentListResponse
from app.core.embedder import get_or_create_collection

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """返回已索引的文档列表（按 source 聚合）"""
    vectorstore = get_or_create_collection()
    results = vectorstore.get(include=["metadatas"])

    if not results["metadatas"]:
        return DocumentListResponse(documents=[])

    # 按 source 聚合
    doc_map = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "未知文档")
        if source not in doc_map:
            doc_map[source] = {"chunks": 0, "images": 0}

        if meta.get("type") == "image":
            doc_map[source]["images"] += 1
        else:
            doc_map[source]["chunks"] += 1

    documents = [
        DocumentItem(name=name, chunks=stats["chunks"], images=stats["images"])
        for name, stats in doc_map.items()
    ]

    return DocumentListResponse(documents=documents)
```

---

## 15. Step 15：组装 → `main.py` 最终版

把 Step 5 写的骨架替换为完整版。打开 `app/main.py`，改成：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import chat, ingest, documents

app = FastAPI(
    title="DocLens API",
    description="多模态 RAG 知识库问答系统",
    version="0.1.0",
)

# CORS：允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件：前端能访问 images/ 下的图片
app.mount("/images", StaticFiles(directory="images"), name="images")

# 注册路由
app.include_router(ingest.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "DocLens API is running"}
```

---

## 16. Step 16：联调测试

### 启动

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

### 测试步骤

**① 健康检查**

浏览器打开 `http://localhost:8000/` → 显示 `{"message":"DocLens API is running"}`

**② 上传文档**

打开 `http://localhost:8000/docs` → 找到 `POST /api/ingest` → Try it out → Choose File 选一个 PDF → Execute

返回示例：
```json
{
  "status": "ok",
  "file_name": "产品手册.pdf",
  "chunks_created": 24,
  "images_processed": 3
}
```

**③ 查看文档列表**

`GET /api/documents` → Execute → 看到刚才上传的文档

**④ 对话**

`POST /api/chat` → Request body:
```json
{"message": "这个文档讲了什么？"}
```
→ 检查返回的 `answer` 和 `sources`

---

## 附：故障排查

| 现象 | 可能原因 | 怎么查 |
|---|---|---|
| 启动报 `deepseek_api_key` 找不到 | `.env` 文件名或路径不对 | `ls -la .env` 确认文件在 `backend/` 下 |
| 上传 PDF 后没返回 chunks | PDF 是纯图片扫描件 | 检查 PDF 是否有可选中的文字 |
| `/api/chat` 返回无关内容 | 检索没命中 | 调大 `top_k`，打 log 看检索结果 |
| ChromaDB 报错 | `vectordb/` 权限或被占用 | 删掉重试，确保没有其他进程在写 |

---

## 附：关键概念速查

| 概念 | 一句话 |
|---|---|
| `async/await` | 允许等网络时不阻塞其他请求 |
| Pydantic BaseModel | JSON ↔ Python 对象，字段不对自动报 422 |
| `APIRouter` | FastAPI 的路由分组器 |
| LangChain Document | `page_content` + `metadata` 统一容器 |
| RecursiveTextSplitter | 按段落→句子→字符逐级切分 |
| ChromaDB Collection | 一个知识库 = 一组向量 + 文本 + 元数据 |
| Embedding | 文本 → 1024 维浮点向量，语义近的向量也近 |
| Temperature | 控制 LLM 输出随机性，知识问答用 0.1 |
| CORS | 后端声明"允许前端跨域调我" |
