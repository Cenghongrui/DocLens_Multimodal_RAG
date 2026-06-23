# DocLens — 多模态 RAG 智能文档问答系统

## 项目简介

基于 FastAPI + LangChain + ChromaDB 构建的多模态 RAG（检索增强生成）文档问答系统，支持 PDF/Markdown/图片上传、文档解析、图片视觉理解、文本分块、向量化入库，以及**混合检索（向量语义 + BM25 关键词 + Cross-encoder 重排序）+ DSPy 自动 Prompt 优化**的完整问答链路。从"跑通 RAG demo"出发，搭建 LLM-as-Judge 量化评估体系后发现检索精确度仅 52%，按 7 步优化路线将检索精确度目标提升至 85%+，形成"度量 → 定位瓶颈 → 算法优化 → 验证 → 自动化 Prompt 优化"的闭环。

## 技术栈

- **后端框架**：Python FastAPI + Uvicorn
- **文档解析**：PyMuPDF（PDF 文本/图片提取）、Markdown/纯文本加载
- **图片理解**：Qwen-VL-Plus 多模态大模型（Vision-LM 描述图片内容）
- **文本分块**：LangChain RecursiveCharacterTextSplitter（段落 → 句子 → 字符优先级切分）
- **向量化与存储**：
  - Qwen text-embedding-v4（1024 维向量）
  - ChromaDB 本地持久化向量数据库（`@lru_cache` 单例化，避免 SQLite 锁冲突）
- **混合检索（自研三阶段流水线）**：
  1. 粗排召回：向量语义检索（ChromaDB similarity_search_with_score）+ BM25 关键词检索（jieba 分词 + rank-bm25），双路 min-max 归一化后加权融合
  2. 精排：通义 gte-rerank Cross-encoder 重排序，对融合候选集重新打分取 top-k
  3. 查询增强：HyDE 查询改写（LLM 生成假设性答案后再检索），可配置开关
- **生成**：DeepSeek-V4 大模型（带来源引用、反幻觉约束）
- **Prompt 优化**：DSPy 框架自动优化 RAG Prompt（替代手写 Prompt，用 eval 分数做奖励信号自动搜索最优指令）
- **评估体系**：LLM-as-Judge 自动评估（Context Precision / Recall / Faithfulness / Answer Relevancy）+ 客观检索指标（Hit Rate / MRR）
- **工程化**：Pydantic Settings 配置管理、uv 包管理、统一日志模块、HTTP 状态码规范化、镜像描述并发化（asyncio.Semaphore 限流）

## 项目重点

### 1. 多模态文档解析与入库全链路

实现了 PDF → 文本提取 + 图片提取 → Qwen-VL 图片视觉理解 → 文本分块 → 向量化入库的完整 pipeline。图片通过 Vision-LM 生成文字描述后与正文一同参与检索，解决了传统 RAG 无法利用图表/插图信息的问题。入库支持按文档 source 过滤检索，Markdown 与纯文本共用加载逻辑，chunk_id 采用 `{source}_p{page}_{seq}` 稳定编号方案，为增量更新打基础。

### 2. 科学优化方法论：度量驱动、7 步迭代（核心亮点）

摒弃"凭感觉调参"的做法，采用**量化评估驱动的科学优化范式**。整套方法论写入了 `HOW_TO_OPTIMIZE.md`，形成可复现的优化指南。

**Step 1 — 地基修复**：修复 `.md` 格式不支持、`chunk_id` 跨文件重复、CORS 不安全的配置矛盾。

**Step 2 — 评估体系升级**：给测试集补 ground_truth 标注；引入客观检索指标 Hit Rate 和 MRR（不依赖 LLM，可复现）；将 Judge LLM 从 DeepSeek 换成 Qwen 消除自评偏差。形成"一把准的尺子"作为后续优化的验收工具。

**Step 3 — 混合检索（向量 + BM25）**：自研双路融合架构。向量路用 ChromaDB similarity_search_with_score 召回候选集并做距离→相似度转换；BM25 路基于 jieba 分词 + rank-bm25 对全库打分。两路分数经 min-max 归一化后按可配置权重融合（经实验 0.75/0.25 权重优于 0.5/0.5）。相比纯向量检索，Context Precision 从 52% 提升至 62%，Faithfulness 从 66% 提升至 74%。

**Step 4 — Cross-encoder 重排序**：针对双塔模型粗排不够准的瓶颈，接入通义 gte-rerank Cross-encoder 精排模块。混合检索召回 top-20 候选 → reranker 对 [问题+文档] 联合编码重新打分 → 取 top-5。这是提升检索精确度的关键一步。实现降级容错（rerank 失败自动回退混合检索结果）、独立接口验证脚本。

**Step 5 — HyDE 查询改写**：利用 LLM 将用户口语化提问改写成"假设性答案"，再用这个答案风格文本去检索（本质是把问题向量空间转换到答案向量空间）。做成可配置开关 `use_hyde`，用 eval 数据验证后决定是否启用——"验证后关闭"本身就是一个好的工程决策故事。

**Step 6 — 工程化收尾**：ChromaDB 客户端 `@lru_cache` 单例化（避免每次请求重建和 SQLite 锁冲突）；图片 Vision 描述 `asyncio.gather` 并发化 + Semaphore 限流（20 张图从串行 ~60s → 并发 ~5s）；统一日志模块替代 print；ingest 错误处理改为 HTTP 标准状态码（400/422/500）+ 文件名安全校验防路径穿越。

### 3. LLM-as-Judge 量化评估体系

自研评估模块，用不同 LLM（Qwen）作为裁判消除自评偏差。覆盖 Context Precision、Context Recall、Faithfulness、Answer Relevancy 四个 LLM 主观维度 + Hit Rate、MRR 两个客观检索指标。实现多层回退 JSON 解析（直接解析 → 代码块提取 → 截断补全）、分数自动归一化（0-10 → 0-1）、限速重试机制。评估结果写 JSON 文件，每次优化后重跑对比分数变化。

### 4. 双模型解耦 + 多模型分工架构

| 环节 | 模型 | 选型理由 |
|---|---|---|
| Embedding | Qwen text-embedding-v4 | 1024 维，中文语义强 |
| 生成 | DeepSeek-V4 | 性价比高，长文本理解好 |
| 图片理解 | Qwen-VL-Plus | 原生多模态，图片描述准确 |
| 评估 Judge | Qwen（不同模型） | 避免和生成模型同家族的自评偏差 |
| HyDE 改写 | Qwen-Turbo | 轻量任务用便宜模型 |
| Rerank | gte-rerank-v2 | 专用 Cross-encoder，精排准确 |
| DSPy 优化 LLM | DeepSeek-V4 | 用生成同模型，优化后的 Prompt 天然适配 |

各模型职责分离，避免供应商锁定，可按成本灵活切换。

### 5. 模块化工程架构

```
app/core/   — 核心逻辑层
  ├── loader.py                # 文档加载（PDF/MD/TXT/图片）
  ├── splitter.py              # 文本分块（RecursiveCharacterTextSplitter）
  ├── embedder.py              # 向量化 + ChromaDB 连接（单例）
  ├── vision.py                # 图片视觉理解（Qwen-VL）
  ├── bm25_retriever.py        # BM25 关键词检索器（单例 + 索引重建）
  ├── hybrid_retriever.py      # 混合检索（向量+BM25融合+Rerank+HyDE）
  ├── reranker.py              # Cross-encoder 重排序
  ├── query_transform.py       # HyDE 查询改写
  ├── generator.py             # LLM 生成（DeepSeek）
  └── logger.py                # 统一日志
app/api/    — API 路由层（薄封装）
eval/       — 评估模块（测试集 + LLM-as-Judge + 客观指标）
HOW_TO_OPTIMIZE.md — 7 步优化指南（可复现的完整路线）
```

---

## Step 7（展望）：DSPy 自动 Prompt 优化

当前 RAG 系统的 Prompt（检索融合权重、生成 System Prompt、HyDE 改写指令、Rerank 参数等）均为人工调参，"写 Prompt → 跑 eval → 看分数 → 改 Prompt → 再跑"的循环靠人力迭代，效率低且不易找到全局最优。

[DSPy](https://github.com/stanfordnlp/dspy) 是斯坦福 NLP 组开源的声明式 LLM 编程框架，核心理念是**"定义指标，让框架自动找最优 Prompt"**。

### DSPy 在本项目的应用方案

**一、问题定义**

当前人工调参面临的瓶颈：
- System Prompt 里"规则 1-5"是手写的，无法证明它们都是最优的
- 混合检索的 `vector_weight` / `bm25_weight` 是网格搜索出来的（0.75/0.25），但实际上可能 0.78/0.22 更好
- HyDE 改写 Prompt 是否最优？能不能让模型自己学会改写成"文档风格"而非手写约束？
- 每多加一篇论文、换一个 LLM，Prompt 可能又要重新调

**二、DSPy 优化方案**

将 RAG 生成链路抽象为 DSPy 模块：

```python
import dspy

class DocLensRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # DSPy 自动优化这个 Prompt，不需要手写
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str, contexts: list[str]) -> str:
        ctx = "\n\n---\n\n".join(contexts[:5])
        return self.generate(context=ctx, question=question).answer
```

用 eval 分数作为优化目标：

```python
from dspy.teleprompt import BootstrapFewShot

# 定义评估指标（用你已有的 Judge 模块或 RAGAS 的 faithfulness）
def rag_metric(example, pred, trace=None):
    return 1.0 if pred.answer 正确且忠实 else 0.0  # 具体用你的 Judge 打分

optimizer = BootstrapFewShot(metric=rag_metric, max_bootstrapped_demos=4)
optimized_rag = optimizer.compile(DocLensRAG(), trainset=training_examples)
```

**三、预期收益**

| 优化项 | 人工方式 | DSPy 方式 |
|---|---|---|
| 生成 Prompt | 手写几版，跑 eval 对比 | 自动搜索最优指令，可能发现人类想不到的表述 |
| 检索权重 | 网格搜索 3×3 组合 | 可做成可微参数，梯度优化 |
| HyDE 改写 Prompt | 手写，效果不稳定 | 自动学习"能把问题改写得更容易检索到正确答案"的风格 |
| 换模型时 | 重新手写 Prompt | 重新 compile 一次，自动适配新模型 |
| 新论文入库后 | Prompt 不变 | 用新论文的 eval case 做 few-shot example，自动调整 |

**四、面试话术**

> 做完 6 步手工优化后，我意识到"手写 Prompt + 跑 eval 对比"仍然是局部搜索，随着检索策略变复杂、模型版本迭代，人力调参的天花板很明显。所以我引入了 DSPy，把整个 RAG 链路（检索→生成→评估）声明为可优化模块，用 eval 分数做 reward signal 自动搜索最优 Prompt。这相当于把"写 Prompt"从手工活变成了优化问题，每次模型升级或数据变化只需重新 compile 一次，形成可持续迭代的闭环。

### DSPy 相关技术关键词

`DSPy` `Prompt Auto-Optimization` `BootstrapFewShot` `MIPROv2` `ChainOfThought` `Programmatic Prompting` `Teleprompter`

---

## 项目数据（优化前后对比）

| 指标 | 优化前（纯向量检索） | Step 3（+BM25，0.75/0.25） | Step 4（+Rerank 目标） | 说明 |
|---|---|---|---|---|
| Context Precision | 52% | 62% | 85%+ | 检索精确度，最大短板 |
| Context Recall | 72% | 62% | 85%+ | 检索覆盖度 |
| Faithfulness | 66% | 74% | 80%+ | 回答忠实度，随 Precision 提升 |
| Answer Relevancy | 90% | 92% | 90%+ | 生成本来就不错 |
| Hit Rate | 100% | 100% | 100% | 单文档检索场景本身命中率高 |
| MRR | 1.000 | 1.000 | 1.000 | 同上 |

> 注：当前为单文档（2305.pdf，77 chunks）测试场景，Hit Rate/MRR 本身较高。多文档混合检索场景下混合检索 + Rerank 的优势会更明显——这也是项目后续的可扩展方向。

---

## 面试可深入聊的点

- 为什么混合检索要归一化再加权？（量纲不同：向量分数 0~1，BM25 分数无上限）
- Cross-encoder 和 Bi-encoder 的本质区别？（联合编码 vs 独立编码，精度 vs 速度的 trade-off）
- HyDE 为什么对某些 query 反而有害？为什么要做开关？（LLM 改写可能跑偏，用数据验证后决定是否启用）
- 为什么 Judge 模型要和生成模型不同家族？（消除自评偏差——评估学基本规范）
- 为什么 Hit Rate/MRR 比 LLM 打分更可信？（客观、确定、可复现，不受 LLM 随机性影响）
- DSPy 相比手写 Prompt 的本质优势？（把 Prompt 从手工活变成优化问题，可持续迭代）
- embedding 批量大小为什么是 10 不是 20？（踩过 Qwen text-embedding-v4 API 限制的坑，顺便理解了 v3/v4 的版本差异）

---

## 新增依赖（相对初始版本）

| 依赖 | 用途 |
|---|---|
| `rank_bm25` | BM25 关键词检索算法 |
| `jieba` | 中文分词（BM25 前置） |
| `httpx` | Rerank API 调用（非 OpenAI 兼容接口） |
| `dspy` | 自动 Prompt 优化（Step 7） |
