# DocLens — 多模态 RAG 文档问答系统

基于 FastAPI + LangChain + ChromaDB 构建的多模态 RAG 文档问答系统，支持 PDF 文档上传、自动解析、文本分块、图片理解（Qwen-VL）、向量化入库，以及基于混合检索（向量语义 + BM25 关键词 + Cross-Encoder 重排序）的智能问答。内置基于 RAGAS 的自动化评估管线。

**技术栈：** FastAPI / LangChain / ChromaDB / Qwen（Embedding + VL + Rerank）/ DeepSeek / Chonkie SemanticChunker / RAGAS

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        用户请求 (FastAPI)                                 │
│  POST /api/chat ──┐   POST /api/ingest ──┐   GET /api/documents ──┐    │
└───────────────────┼───────────────────────┼────────────────────────┼────┘
                    │                       │                        │
┌───────────────────▼─────────┐  ┌──────────▼──────────┐  ┌────────▼──────┐
│     检索 + 生成管道          │  │     文档摄入管道      │  │   文档列表    │
│                             │  │                     │  │              │
│  ┌─────────────────────┐   │  │  PDF/TXT/MD → 解析   │  │ ChromaDB 聚合│
│  │    HyDE 路由         │   │  │  ↓                   │  │              │
│  │  (规则+LLM 两级判断)  │   │  │  图片 → Qwen-VL 描述  │  │              │
│  └─────────┬───────────┘   │  │  ↓                   │  │              │
│            ▼               │  │  Chonkie 语义分片     │  │              │
│  ┌─────────────────────┐   │  │  ↓                   │  │              │
│  │  混合检索             │   │  │  Embedding→ChromaDB │  │              │
│  │  (向量 + BM25 加权融合)│   │  │  BM25 索引重建       │  │              │
│  └─────────┬───────────┘   │  └──────────────────────┘  │              │
│            ▼               │                           │              │
│  ┌─────────────────────┐   │                           │              │
│  │  Cross-Encoder 重排序 │   │                           │              │
│  └─────────┬───────────┘   │                           │              │
│            ▼               │                           │              │
│  ┌─────────────────────┐   │                           │              │
│  │  DeepSeek 生成回答    │   │                           │              │
│  └─────────────────────┘   │                           │              │
└─────────────────────────────┘                           └──────────────┘
```

**检索管道流程：**

```
用户问题 → HyDE 路由（规则层 80% + LLM 兜底 20%）
         → 是否启用 HyDE 查询改写
         → 混合检索（向量 65% + BM25 35%，Min-Max 归一化融合，召回 15 个候选）
         → Cross-Encoder 重排序（Qwen text-rerank，精排输出 Top-3）
         → DeepSeek V4 Flash 生成回答（t=0.05，来源引用标注）
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/Cenghongrui/DocLens_Multimodal_RAG.git
cd DocLens
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 配置

创建 `backend/.env`：

```env
DEEPSEEK_API_KEY=sk-xxx
QWEN_API_KEY=sk-xxx
```

### 启动

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/ingest` | 上传文档（PDF/TXT/MD/CSV/JPG/PNG） |
| `POST` | `/api/chat` | 对话问答 |
| `GET` | `/api/documents` | 已入库文档列表 |

### 运行评估

```bash
cd backend
python -X utf8 -m eval.evaluate
```

---

## 项目结构

```
DocLens/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 配置（Pydantic Settings）
│   │   ├── api/
│   │   │   ├── chat.py          # 对话接口
│   │   │   ├── ingest.py        # 文档上传接口
│   │   │   └── documents.py     # 文档列表接口
│   │   └── core/
│   │       ├── loader.py        # 文件解析（PDF/TXT/图片）
│   │       ├── splitter.py      # Chonkie 语义分片
│   │       ├── embedder.py      # Qwen Embedding + ChromaDB
│   │       ├── vision.py        # Qwen-VL 图片理解
│   │       ├── bm25_retriever.py    # BM25 关键词检索
│   │       ├── hybrid_retriever.py  # 混合检索融合
│   │       ├── reranker.py      # Cross-Encoder 重排序
│   │       ├── query_router.py  # HyDE 两级路由
│   │       ├── query_transform.py   # HyDE 查询改写
│   │       ├── retriever.py     # 检索入口
│   │       ├── generator.py     # DeepSeek 生成
│   │       └── logger.py        # 日志
│   ├── eval/
│   │   ├── evaluate.py          # RAGAS 并行评估
│   │   ├── test_dataset_v2.json # 测试集
│   │   └── result_ragas_v2.json # 评估结果
│   ├── data/                    # PDF 源文档
│   ├── images/                  # 提取的图片
│   ├── vectordb/                # ChromaDB 持久化
│   └── .venv/                   # 虚拟环境
├── frontend/                    # 前端（Vite）
└── ARCHITECTURE.md              # 架构设计文档
```
