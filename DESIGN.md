# DocLens — 多模态 RAG 知识库问答系统

## 项目定位

通用的多模态文档检索增强生成（Multimodal RAG）系统。支持 PDF、图片、Markdown、TXT 等格式，能同时检索文档中的**文字**和**图片/图表信息**，实现"看得见图的 RAG"。

**Slogan**：不只是读文字，图表也看得见。

---

## v1 目标

跑通核心链路：**文档上传 → 图文解析 → 向量化存储 → 语义检索 → LLM 生成回答 → Web 对话界面**

- 功能层面：能用，链路完整
- 多模态层面：图片被"读懂"并参与检索
- 前端层面：Vue3 单页应用，类 ChatGPT 对话界面

---

## 技术栈

| 层级 | 选型 | 理由 |
|---|---|---|
| 后端框架 | **FastAPI** | 异步高性能，自动生成 Swagger 文档，Python 生态无缝 |
| RAG 编排 | **LangChain** | 统一的 Document Loader / Text Splitter / Chain 抽象 |
| 文档解析 | `PyMuPDF` (fitz) | PDF 解析 + 图片提取一把梭 |
| 图片理解 | Qwen-VL（通义千问视觉） | 与 Embedding 同一生态，中文图表理解能力强 |
| Embedding | **通义千问 text-embedding-v3** | 中文检索效果 top 级，支持 1024 维向量，便宜 |
| 向量库 | ChromaDB | 本地持久化，零配置 |
| LLM 生成 | **DeepSeek-V4 Pro** | 推理能力强，中文优秀，性价比高 |
| 前端框架 | **Vue 3** + Vite | 主流、生态好、上手快 |
| UI 组件库 | Element Plus 或 Naive UI | 快速搭建对话界面 |
| HTTP 客户端 | Axios | 前后端通信 |
| 状态管理 | Pinia | Vue3 官方推荐，轻量 |
| 前端路由 | Vue Router | 后续扩展多页面用，v1 就一个对话页 |

---

## 项目结构

```
agentRag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI 入口，路由注册，CORS
│   │   ├── config.py               # 配置中心 + .env 加载
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py             # POST /api/chat   对话接口
│   │   │   ├── ingest.py           # POST /api/ingest 文档上传+入库
│   │   │   └── documents.py        # GET  /api/documents  文档列表
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py           # 文档加载 + 图片提取（LangChain Document Loader）
│   │   │   ├── splitter.py         # 文本分块策略
│   │   │   ├── vision.py           # Vision LLM 图片描述
│   │   │   ├── embedder.py         # Embedding 封装
│   │   │   ├── retriever.py        # LangChain Retriever 封装
│   │   │   └── generator.py        # RAG Chain：检索 + Prompt + LLM 生成
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py          # Pydantic 请求/响应模型
│   ├── data/                       # 上传的文档存储
│   ├── images/                     # PDF 提取的图片缓存
│   ├── vectordb/                   # ChromaDB 持久化
│   ├── requirements.txt
│   └── .env                        # API Key 等敏感配置
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.vue                 # 根组件
│   │   ├── main.js                 # 入口：挂载路由、Pinia、UI 库
│   │   ├── router/
│   │   │   └── index.js            # 路由（v1 只有 /chat）
│   │   ├── views/
│   │   │   └── ChatView.vue        # 对话页：侧边栏 + 对话区
│   │   ├── components/
│   │   │   ├── ChatMessage.vue     # 单条消息（支持 Markdown + 来源引用）
│   │   │   ├── ChatInput.vue       # 输入框 + 发送按钮
│   │   │   ├── Sidebar.vue         # 知识库列表、文档管理入口
│   │   │   └── SourceCard.vue      # 来源引用卡片
│   │   ├── api/
│   │   │   └── index.js            # Axios 封装 + API 方法
│   │   └── stores/
│   │       └── chat.js             # Pinia 对话状态管理
│   ├── index.html
│   ├── package.json
│   └── vite.config.js              # 代理配置（转发 /api 到后端）
│
└── README.md
```

---

## 多模态策略（v1 方案）

### 处理流程

```
输入文档 (PDF / 图片)
    │
    ├──→ 提取文字 ──→ LangChain 分块 ──→ Embedding
    │                                              │
    └──→ 提取图片 ──→ Vision LLM 描述 ──→ Embedding
                                                   │
                                            ┌──────┘
                                            ▼
                                      ChromaDB（统一向量空间）
```

### 图片处理

1. PyMuPDF 从 PDF 中提取内嵌图片 → 存到 `backend/images/`
2. 单独上传的图片直接读取
3. 调用 Qwen-VL 生成图片文字描述（含图表数据解读）
4. 描述文本 + 源文件路径 + 页码作为 metadata 存入 ChromaDB
5. 检索时，文本 chunks 和图片描述 chunks 在同一 collection 中同时参与
6. 前端展示时，回答中附带图片 URL（`/images/xxx.png`），前端直接渲染

### 为什么这样做

- **最简单有效**：图片描述和文字在同一个向量空间，一次查询全部命中
- **不需要多向量模型**：不用引入 CLIP 等多模态 embedding 模型
- **Qwen-VL 中文图表理解强**：对中文文档的表格、流程图、架构图解读准确

---

## API 设计（RESTful）

### 1. 对话接口

```
POST /api/chat
```

**Request**：
```json
{
  "message": "这个系统的架构是什么样的？",
  "conversation_id": "conv_xxx"   // 可选，v1 先不用
}
```

**Response**：
```json
{
  "answer": "根据文档，系统架构包含三层...",
  "sources": [
    {
      "content": "...",
      "source": "产品手册.pdf",
      "page": 3,
      "type": "text"
    },
    {
      "content": "图片描述：这是一个系统架构图...",
      "source": "产品手册.pdf",
      "page": 5,
      "type": "image",
      "image_url": "/images/page_5_fig1.png"
    }
  ]
}
```

### 2. 文档上传 + 入库

```
POST /api/ingest
```

**Request**：`multipart/form-data`

| 字段 | 类型 | 说明 |
|---|---|---|
| `file` | File | 单个文件 (pdf / md / txt / png / jpg) |
| `files` | File[] | 或多个文件（批量上传） |

**Response**：
```json
{
  "status": "ok",
  "file_name": "产品手册.pdf",
  "chunks_created": 24,
  "images_processed": 3
}
```

### 3. 文档列表

```
GET /api/documents
```

**Response**：
```json
{
  "documents": [
    {
      "name": "产品手册.pdf",
      "chunks": 20,
      "images": 3,
      "ingested_at": "2026-06-18T10:30:00"
    }
  ]
}
```

---

## 前端页面设计（v1 极简版）

### 布局

```
┌──────────────────────────────────────────────┐
│  DocLens                        [知识库管理]  │  ← 顶栏
├────────────┬─────────────────────────────────┤
│            │                                 │
│  知识库列表 │     对话消息区                    │
│            │  ┌─────────────────────────┐    │
│  • 产品手册  │  │ 🤖 根据文档，架构包含...  │    │
│  • 技术文档  │  │ 📎 来源: 产品手册.pdf p3  │    │
│  • ...     │  │ 📷 来源: 产品手册.pdf p5  │    │
│            │  └─────────────────────────┘    │
│  [+ 上传]  │                                 │
│            │  ┌─────────────────────────┐    │
│            │  │ 输入你的问题...      [发送]│    │
│            │  └─────────────────────────┘    │
└────────────┴─────────────────────────────────┘
```

### 组件树

```
App.vue
└── ChatView.vue
    ├── Sidebar.vue          ← 文档列表 + 上传按钮
    ├── ChatMessage.vue × N  ← 对话消息列表（可滚动）
    │   └── SourceCard.vue   ← 来源引用卡片（可展开）
    └── ChatInput.vue        ← 输入框 + 发送
```

### 交互流程

1. **上传文档**：侧边栏点"上传"→ 选文件 → `POST /api/ingest` → 刷新文档列表
2. **发起对话**：输入框打字 → 点发送 → `POST /api/chat` → 流式返回或一次性返回 → 消息追加到列表
3. **查看来源**：回答中的引用 → 点击展开 SourceCard → 显示原文片段 / 图片

---

## 核心模块设计

### 1. Backend — `config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    llm_model: str = "deepseek-v4-pro"
    llm_base_url: str = "https://api.deepseek.com"  # DeepSeek 官方 API

    # Vision — 通义千问多模态模型
    vision_model: str = "qwen-vl-plus"
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Embedding — 通义千问
    embedding_model: str = "text-embedding-v3"
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_dimension: int = 1024

    # API Keys
    deepseek_api_key: str
    qwen_api_key: str

    # 分块
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 检索
    top_k: int = 5

    # 路径
    data_dir: str = "data/"
    image_dir: str = "images/"
    vectordb_dir: str = "vectordb/"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. Backend — API 层 (`api/`)

| 文件 | 路由 | 职责 |
|---|---|---|
| `chat.py` | `POST /api/chat` | 接收消息 → 调 `generator.rag_chain` → 返回答案+来源 |
| `ingest.py` | `POST /api/ingest` | 接收文件 → 保存 → 调 `loader` + `splitter` + `embedder` → 入库 |
| `documents.py` | `GET /api/documents` | 查询 ChromaDB metadata，返回已索引文档列表 |

### 3. Backend — Core 层

```python
# loader.py — 文档加载
def load_document(filepath: str) -> List[Document]:
    """根据文件类型选择 Loader，提取文字 + 图片"""

# splitter.py — 分块
def split_documents(docs: List[Document]) -> List[Document]:
    """LangChain RecursiveCharacterTextSplitter"""

# vision.py — 图片理解
def describe_image(image_path: str) -> str:
    """调用 Qwen-VL，返回图片文字描述"""

# embedder.py — 向量化
def embed_and_store(docs: List[Document], collection):
    """调用 embedding API → ChromaDB 入库"""

# retriever.py — 检索
def get_retriever():
    """返回 LangChain Chroma Retriever"""

# generator.py — 生成
def create_rag_chain():
    """LangChain: retriever | prompt | llm | StrOutputParser"""
```

### 4. Frontend — 核心组件

#### `ChatView.vue`
- 页面布局（侧边栏 + 对话区）
- 维护 messageList 状态
- 调用 API 发送消息

#### `ChatMessage.vue`
- Props：`{ role, content, sources }`
- 渲染 Markdown（`marked` 或 `markdown-it`）
- 底部展示来源引用

#### `ChatInput.vue`
- 输入框 + 发送按钮
- Enter 发送，Shift+Enter 换行
- 发送状态 loading

#### `Sidebar.vue`
- 知识库文档列表
- 上传按钮 → `el-upload` 组件
- 上传成功后刷新列表

---

## 执行计划（v1）

### Step 1：环境搭建
- 后端：创建虚拟环境 → 安装 FastAPI / LangChain / ChromaDB 等
- 前端：`npm create vite@latest frontend -- --template vue` → 安装依赖
- 配置 `.env`

### Step 2：后端 — 基础设施
- `config.py` + `.env`
- `models/schemas.py` Pydantic 模型
- `main.py` FastAPI app 骨架 + CORS

### Step 3：后端 — 文档摄入
- `loader.py`：PDF/MD/TXT/图片 多格式加载
- `vision.py`：图片 → Claude Vision 描述
- `splitter.py`：LangChain 分块
- `embedder.py`：Embedding + ChromaDB 入库
- `api/ingest.py`：上传接口

### Step 4：后端 — 检索与生成
- `retriever.py`：ChromaDB Retriever
- `generator.py`：LangChain RAG Chain
- `api/chat.py`：对话接口

### Step 5：后端 — 辅助接口
- `api/documents.py`：文档列表查询

### Step 6：前端 — 项目骨架
- Vite + Vue3 + Vue Router + Pinia + Element Plus 初始化
- `api/index.js` Axios 封装

### Step 7：前端 — 对话页面
- `ChatView.vue` 布局
- `ChatMessage.vue` 消息渲染
- `ChatInput.vue` 输入交互

### Step 8：前端 — 文档管理
- `Sidebar.vue` 文档列表 + 上传
- 对接后端 API

### Step 9：联调测试
- 启动后端 `uvicorn app.main:app --reload`
- 启动前端 `npm run dev`
- 上传文档 → 提问 → 验证回答 + 来源

---

## 后续迭代方向（v2+）

- [ ] 流式生成（SSE / WebSocket）
- [ ] 多轮对话上下文
- [ ] 支持更多格式（PPT、Word、Excel）
- [ ] Agentic RAG（多跳推理、自动重检索）
- [ ] GraphRAG（实体关系图谱）
- [ ] 用户管理 + 多知识库切换
- [ ] Web 爬虫输入源
- [ ] 本地模型部署（Ollama）降低使用成本

---

## 待定决策

| 问题 | 选项 | v1 暂定 |
|---|---|---|
| Embedding 模型 | OpenAI / bge-local / qwen | 通义千问 text-embedding-v3 |
| LLM | Claude / GPT-4o / DeepSeek | DeepSeek-V4 Pro |
| Vision 模型 | Claude Vision / GPT-4o / Qwen-VL | Qwen-VL Plus |
| 多轮对话 | 支持 / 暂不支持 | 暂不支持（单轮） |
| 流式生成 | SSE 流式 / 一次性返回 | 一次性返回（简单） |
| 图片返回 | 前端展示图片 / 仅文字描述 | 前端直接展示图片 |
| UI 组件库 | Element Plus / Naive UI | Element Plus |
