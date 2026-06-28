# DocLens RAG 系统架构设计与优化策略

> 基于 LangChain + ChromaDB + Qwen/DeepSeek 的中文 PDF 文档问答系统  
> 本文档覆盖系统架构、各模块设计原理、已实施的优化策略及未来改进方向。

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          用户请求 (FastAPI)                                   │
│  POST /api/chat ──┐    POST /api/ingest ──┐   GET /api/documents ──┐       │
└───────────────────┼────────────────────────┼────────────────────────┼───────┘
                    │                        │                        │
┌───────────────────▼─────────┐  ┌───────────▼──────────┐  ┌─────────▼──────┐
│     检索 + 生成管道          │  │     文档摄入管道       │  │   文档列表     │
│                             │  │                      │  │               │
│  ┌─────────────────────┐   │  │  PDF/TXT/MD → 解析    │  │ ChromaDB 聚合 │
│  │    HyDE 路由         │   │  │  ↓                    │  │               │
│  │  (规则+LLM 两级判断)  │   │  │  图片 → Qwen-VL 描述   │  │               │
│  └─────────┬───────────┘   │  │  ↓                    │  │               │
│            ▼               │  │  文本分块 (重叠切片)    │  │               │
│  ┌─────────────────────┐   │  │  ↓                    │  │               │
│  │  混合检索             │   │  │  Embedding → ChromaDB │  │               │
│  │  (向量 75% + BM25 25%)│   │  │  BM25 索引刷新         │  │               │
│  └─────────┬───────────┘   │  └──────────────────────┘  │               │
│            ▼               │                           │               │
│  ┌─────────────────────┐   │                           │               │
│  │  Cross-Encoder 重排序 │   │                           │               │
│  └─────────┬───────────┘   │                           │               │
│            ▼               │                           │               │
│  ┌─────────────────────┐   │                           │               │
│  │  DeepSeek 生成回答    │   │                           │               │
│  │  (引用标注 + 幻觉控制) │   │                           │               │
│  └─────────────────────┘   │                           │               │
└─────────────────────────────┘                           └───────────────┘
```

### 技术栈

| 组件 | 选型 | 版本 |
|------|------|------|
| 框架 | FastAPI + LangChain | 1.3.11 |
| 向量数据库 | ChromaDB (PersistentClient) | 1.5.9 |
| Embedding | Qwen text-embedding-v3 | 1024 维 |
| 生成 LLM | DeepSeek V4 Flash | — |
| 路由/Judge LLM | Qwen3.5-Flash | — |
| 视觉 LLM | Qwen-VL-Flash | — |
| 关键词检索 | BM25 (rank-bm25) + jieba 中文分词 | — |
| 重排序 | Qwen text-rerank API | — |
| 评估 | RAGAS 0.4.3 | context_precision / recall / faithfulness |
| 运行环境 | Python 3.14 | Windows |

---

## 二、管道详解

### 2.1 文档摄入管道（Ingest Pipeline）

```
上传文件 → load_file() → 图片Qwen-VL描述 → split_documents() → embedding → ChromaDB
                                                                         ↓
                                                                   BM25 索引重建
```

#### 文件加载（`loader.py`）

- 支持 `.pdf` / `.txt` / `.md` / `.csv` / `.jpg` / `.png`
- PDF 使用 PyMuPDF 按页提取文字 + 内嵌图片
- 图片 Document 的 `page_content` 初始为空，由后续 Vision 模块填充

#### 图片理解（`vision.py`）

- 使用 Qwen-VL-Flash 模型（多模态视觉-语言模型）
- 对每张提取的图片生成详细中文描述
- 描述内容包括：图表数据提取、架构图结构说明、组件关系总结
- 描述文本作为 Document 内容存入向量库

#### 文本分片（`splitter.py`）— 语义分片

- 使用 **Chonkie `SemanticChunker`**（v1.6.8），基于 Embedding 相似度自适应检测语义边界
- **原理**：将文本拆为句子 → 计算相邻句子组的 Embedding 余弦距离 → Savitzky-Golay 滤波平滑 → 选取距离突变的百分位阈值点作为断点
- **参数**：`chunk_size=512`（**token 数**，非字符数），`threshold=0.5`（语义相似度阈值）
- **后处理 overlap**：`chunk_overlap=100` 字符 — 语义切分后，每个 chunk 尾部追加下一 chunk 的开头 100 字符，衔接跨边界上下文
- **中英文断句**：`['。', '！', '？', '\n\n', '\n', '. ', '! ', '? ', '；']`
- **自定义 Embedding 适配器**：`QwenEmbeddings` 封装 Qwen `text-embedding-v3` 为 Chonkie 可用的 `BaseEmbeddings`，与检索用同一套 Embedding 模型
- **优势**：段落内语义连贯的内容合在一起，话题切换时自动断开，无需固定 `chunk_overlap`
- **图片 Document**：不切分，整张保留
- 每个 chunk 打上 `chunk_id`（格式：`{source}_p{page}_{index}`），用于混合检索融合去重

#### Embedding 与入库（`embedder.py` → `ingest.py`）

- 使用 Qwen `text-embedding-v3`，1024 维
- 批量入库，每批最多 20 条

---

### 2.2 检索管道（Retrieval Pipeline）

```
用户问题
    │
    ▼
┌────────────────────┐
│  HyDE 路由 (2级)    │ ← 规则层 → LLM兜底层
│  USE?               │
└──────┬──────┬──────┘
  YES  │      │  NO
       ▼      ▼
  HyDE 改写   原 query
  (Qwen生成    直接检索
  假设回答)
       │      │
       └──┬───┘
          ▼
┌────────────────────┐
│  混合检索            │
│  向量 75% + BM25 25% │  ← 各归一化后加权融合
│  召回 20 个候选      │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  Cross-Encoder    │
│  重排序 (Top 5)    │ ← Qwen text-rerank API
└──────────┬─────────┘
           ▼
        Top-5 Documents
```

#### 2.2.1 HyDE 智能路由（`query_router.py`）

**设计目的**：HyDE（Hypothetical Document Embedding）对模糊简短的 query 有显著提升，但对精确专有名词 query 反而有害。路由模块解决"什么时候用 HyDE"的问题。

**两级路由**：

```
第一层：规则匹配（O(1)，无 API 调用）
├── 含编号/版本号/Hash → SKIP（精确匹配）
├── 口语化短 query（< 20 字 + 口语标记）→ USE
├── 书面语长 query（> 50 字 + 学术标记）→ SKIP
├── 超短 query（< 15 字）→ USE
└── 不确定 → 进入第二层

第二层：LLM 判断（仅 ~30% 情况触发）
├── Qwen3.5-Flash 打分
├── 判断 suitable + confidence
└── confidence ≥ 0.5 → USE，否则 SKIP
```

**优化策略**：
- ✅ **规则先行**：80% 请求在规则层完成，无需 LLM 调用
- ✅ **两级降本**：LLM 兜底仅在规则不确定时（实测约 30%）触发
- ✅ **中文口语词库**：覆盖"咋样、怎么办、帮我查一下"等常见口语表达
- ✅ **学术语料库**：覆盖"方法、模型、实验、结论"等学术书面语标记

#### 2.2.2 HyDE 查询改写（`query_transform.py`）

- 使用 Qwen3.6-Flash 将问题改写为 100 字左右的假设性答案
- **Prompt 约束**：用陈述句、模仿论文风格、不编造具体数字
- 改写后的文本替代原 query 进行 embedding 检索

#### 2.2.3 混合检索（`hybrid_retriever.py`）

**融合公式**：
```
融合分数 = 0.75 × 归一化(向量距离) + 0.25 × 归一化(BM25分数)
```

| 通道 | 权重 | 功能 | 召回数 |
|------|:----:|------|:------:|
| 向量检索 (ChromaDB) | 75% | 语义相似性，理解意图 | 20 |
| BM25 (jieba分词) | 25% | 关键词精确匹配，保证命中率 | 20 |

**优化策略**：
- ✅ **Min-Max 归一化**：两个通道分数尺度不同（向量是余弦距离转 similarity，BM25 是 TF-IDF 分数），归一化到 [0,1] 再融合
- ✅ **chunk_id 去重**：以 `chunk_id` 为唯一键合并两路结果，同一个 chunk 不会重复计分
- ✅ **大候选池**：召回阶段多召（20 个），留给后续 reranker 精排，避免信息损失
- ✅ **权重可调**：`vector_weight` / `bm25_weight` 参数通过函数参数暴露，可针对不同场景调优

#### 2.2.4 BM25 关键词检索（`bm25_retriever.py`）

- 使用 `rank_bm25`（Okapi BM25 算法）
- **中文分词**：使用 jieba 做中文切词
- **单例模式**：模块级全局缓存索引，首次构建后复用
- **增量更新**：`reset_bm25_index()` 在摄入新文档后调用，使新内容立即可检索
- 索引源：从 ChromaDB 拉取所有 chunks 重建

#### 2.2.5 Cross-Encoder 重排序（`reranker.py`）

- 使用 Qwen `text-rerank` API（专用的 cross-encoder 排序模型）
- 对混合检索召回的 20 个候选做精确排序
- 只返回 Top-5（默认 `top_k=5`）
- **降级策略**：rerank API 失败时退回到混合融合排序的 top_k，保证服务可用
- rerank score 写入 Document metadata，可供前端展示

**当前配置参数**：

| 参数 | 值 | 说明 |
|------|:--:|------|
| `top_k` | 5 | 最终返回结果数 |
| `candidate_k` | 20 | 召回阶段候选数 |
| `vector_weight` | 0.75 | 向量检索融合权重 |
| `bm25_weight` | 0.25 | BM25 关键词权重 |

---

### 2.3 生成管道（Generation Pipeline）

```
检索结果 (Top-5 Documents)
    │
    ▼
┌────────────────────┐
│  组装 System Prompt │
│  - 角色定义         │
│  - 规则约束         │
│  - 引文标注格式      │
│  - 文档内容          │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  DeepSeek V4 Flash │
│  生成回答           │
│  (temperature=0.1) │
└──────────┬─────────┘
           ▼
    最终回答 + 来源引用
```

#### 生成器（`generator.py`）

**System Prompt 结构**：
- **角色定义**：知识库助手
- **规则约束**：
  1. 必须基于文档内容回答
  2. 引用注明来源编号（如 `[来源 1]`）
  3. 文档不足时明确说"无法回答"（禁止编造）
  4. 用用户提问的语言回答
  5. 语言不一致时：先翻译问题 → 检索 → 翻译答案
- **上下文拼接**：每份文档标注 `[来源 N] {图标}（文件名，第 X 页）`

**优化策略**：
- ✅ **t=0.1**：低温采样减少随机性幻觉
- ✅ **来源编号**：标注引用来源，便于用户追溯验证
- ✅ **模型分离**：生成用 DeepSeek（质量高），Judge/Routing 用 Qwen（成本低）
- ✅ **"无法回答"机制**：明确规定了无法回答时的回应方式

---

### 2.4 评估体系（Evaluation Pipeline）

```
test_dataset_v2.json (30 用例)
    │
    ├── 30 个用例逐条跑 RAG 链路
    │   ├── retrieve() → contexts
    │   └── generate() → answers
    │
    └── RAGAS 评估
        ├── context_precision ── 检索精确度（LLM Judge）
        ├── context_recall ──── 检索覆盖率（LLM Judge）
        └── faithfulness ────── 回答忠实度（LLM Judge）
             ↑ 核心指标：是否基于文档回答？有无幻觉？

客观指标：
    ├── Hit Rate ── 是否命中相关文档
    └── MRR ────── 命中文档的平均排名
```

**测试集构成**（30 用例，2 份文档）：

| 类型 | 数量 | 说明 |
|------|:----:|------|
| Simple | 9 | 简单事实型 |
| Reasoning | 5 | 推理型 |
| Multi-Hop | 4 | 多跳型（跨段落） |
| Conditional | 4 | 条件型 |
| Comparative | 3 | 对比型 |
| Listing | 4 | 列举型 |
| Numerical | 1 | 数值型 |

**最新评估结果**（`eval/result_ragas_v2.json`）：

| 指标 | 得分 | 状态 |
|------|:----:|:----:|
| Context Precision | 44.48% | ⚠️ 检索噪声偏高 |
| Context Recall | 41.67% | ⚠️ 信息覆盖不足 |
| Faithfulness | 9.22% | 🔴 核心问题（幻觉严重） |
| Hit Rate | 90.00% | ✅ 命中率良好 |
| MRR | 0.883 | ✅ 排名质量好 |

---

## 三、各模块优化策略总览

### ✅ 已实施的优化

| 模块 | 优化策略 | 效果 |
|------|---------|:----:|
| **HyDE 路由** | 规则层 + LLM 兜底两级判断 | 80% 请求零额外 API 调用 |
| **检索** | 向量 + BM25 混合加权融合 | Hit Rate 90% |
| **检索** | Min-Max 归一化 + chunk_id 去重 | 公平比较异源分数 |
| **检索** | 大候选池 (20) + Reranker (5) | 精排消除噪声 |
| **搜索** | Cross-Encoder 重排序 | 提升 top-k 相关性 |
| **分片** | 固定 `RecursiveCharacterTextSplitter` → **Chonkie SemanticChunker**（语义分片） | 按语义话题自动切分，替代固定字符切分 |
| **生成** | 低温采样 + 来源编号 + "无法回答"机制 | 可追溯性 |
| **评估** | 模型分离（DeepSeek 生成 / Qwen 评分） | 消除自评价偏差 |
| **BM25** | 单例缓存 + 增量重建 | 内存高效 |
| **Late Chunking** | 可选上下文增强嵌入 | 实验性，默认关闭 |

### 🔧 需进一步优化的方向

| 问题 | 根因分析 | 优化方案 |
|------|---------|---------|
| **Faithfulness 仅 9%** | DeepSeek 在信息不足时编造 | ① 降低 generation temperature <br>② 增加 **检索增强的 rejection**：如果检索结果置信度低，直接拒绝回答 <br>③ 优化 System Prompt，强调"只根据上下文回答" |
| **Context Precision 44%** | top-5 混入无关 chunk | ① 降低 `top_k` 到 3 <br>② 提高 reranker `candidate_k` 到 30，更多候选给精排筛选 <br>③ 加相关性阈值过滤（低于 0.3 的舍弃） |
| **Context Recall 42%** | 多跳/条件型问题信息分散 | ① `chunk_overlap` 从 200 提高到 300 <br>② 考虑 **Multi-Query** 策略：一个 query 生成多个变体分别检索后合并 <br>③ **Sentence-window 检索**：检索到 chunk 后，返回其上下文窗口 |
| **评估速度慢** | 90 次 Qwen API 串行评分 | ① 缓存 retrieve+generate 中间结果 <br>② 仅跑 Faithfulness 指标（最具代表性）<br>③ 换用更快的 Judge LLM |
| **无增量去重** | 重复 ingest 同一文档会重复入库 | ① ChromaDB 入库前按 chunk_id 查重 <br>② BM25 索引去重 |

---

## 四、配置文件说明

`backend/app/config.py` 集中管理所有可调参数。

```python
# ─── LLM 配置 ───
llm_model: str = "deepseek-v4-flash"        # 生成模型
judge_llm_model: str = "qwen3.5-flash"       # Judge/路由模型
vision_model: str = "qwen-vl-flash"          # 视觉模型
embedding_model: str = "text-embedding-v3"   # Embedding 模型

# ─── Embedding ───
embedding_dimension: int = 1024              # 向量维度
embedding_batch_size: int = 20               # 批量大小

# ─── 分块 ───
chunk_size: int = 512                        # 每块最大 token 数（语义分片，非字符）
chunk_overlap: int = 0                       # 语义分片无需固定重叠

# ─── 检索 ───
top_k: int = 5                               # 返回结果数

# ─── HyDE ───
hyde_enabled: bool = True                    # 全局开关
hyde_route_threshold: float = 0.5            # 路由置信度阈值
hyde_model: str = "qwen3.6-flash"            # 改写模型
hyde_max_tokens: int = 200                   # 改写长度
```

---

## 五、项目结构

```
DocLens/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口 + 路由挂载
│   │   ├── config.py            # 集中配置（Pydantic Settings）
│   │   ├── api/
│   │   │   ├── chat.py          # POST /api/chat 对话接口
│   │   │   ├── ingest.py        # POST /api/ingest 文档上传
│   │   │   └── documents.py     # GET  /api/documents 文档列表
│   │   ├── core/
│   │   │   ├── loader.py        # 文件解析（PDF/TXT/图片）
│   │   │   ├── splitter.py      # 文本分块
│   │   │   ├── embedder.py      # Embedding + ChromaDB 连接
│   │   │   ├── vision.py        # Qwen-VL 图片理解
│   │   │   ├── bm25_retriever.py    # BM25 关键词检索
│   │   │   ├── hybrid_retriever.py  # 混合检索融合
│   │   │   ├── reranker.py      # Cross-Encoder 重排序
│   │   │   ├── query_router.py  # HyDE 智能路由（规则 + LLM）
│   │   │   ├── query_transform.py  # HyDE 查询改写
│   │   │   ├── retriever.py     # 检索入口
│   │   │   ├── generator.py     # 回答生成（DeepSeek）
│   │   │   ├── logger.py        # 统一日志
│   │   │   └── __init__.py
│   │   ├── models/
│   │   │   ├── schemas.py       # Pydantic 请求/响应模型
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── eval/
│   │   ├── evaluate.py          # RAGAS 评估脚本
│   │   ├── __init__.py
│   │   ├── test_dataset_v2.json # 30 用例测试集
│   │   └── result_ragas_v2.json # 最新评估结果
│   ├── data/                    # PDF 源文档
│   ├── images/                  # 提取的内嵌图片
│   ├── vectordb/                # ChromaDB 持久化数据
│   ├── .venv313/                # Python 虚拟环境
│   └── pyproject.toml           # 项目配置 + 依赖
├── frontend/                    # 前端（Vite 开发服务器）
└── .claude/                     # Claude Code 配置
```

---

> 最后更新：2026-06-28  
> 评估基准：`result_ragas_v2.json`（30 用例，RAGAS v0.4.3）
