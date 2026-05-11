# MedMemoryBench：面向个性化医疗的 Agent 记忆能力基准测试

<div align="center">
  <img src="figs/4examples.png" alt="MedMemoryBench 概览 — 医疗记忆评测的四大核心挑战" width="800"/>
</div>

<p align="center">
  <em>解决医疗场景下 Agent 记忆能力评估的核心难题。</em>
</p>

<p align="center">
｜🤗 <a href="https://huggingface.co/datasets/Cyan27/MedMemoryBench" target="_blank">HuggingFace Dataset</a> ｜
📄 <a href="#-引用">Preprint (Coming Soon)</a> ｜
🌐 <a href="README.md">English</a> ｜
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="version"/>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="python"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="license"/>
</p>

---

**MedMemoryBench** 是一个面向 Agent 记忆方法评测的基准框架，专注于医疗对话场景下的记忆能力评估。该框架提供统一的评测接口、多种基线方法实现和灵活的配置管理系统，同时支持导入和评测其他数据集。

## 目录

- [最新动态](#-最新动态)
- [特性亮点](#-特性亮点)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [输出格式](#-输出格式)
- [引用](#-引用)

---

## 📰 最新动态

- **[2026.05]** MedMemoryBench v1.0 正式发布 — 包含数据集、评测框架与 16 种记忆方法基线。
- **[2026.05]** 数据集已上线 [HuggingFace](https://huggingface.co/datasets/Cyan27/MedMemoryBench)。

## ✨ 特性亮点

<table>
<tr>
<td width="50%">

**全面的医疗数据集**
- 20 个纵向患者画像，涵盖背景、生活事件与干扰事件
- 约 2,020 轮多会话医患对话
- 约 1,986 条评测查询，覆盖 6 种临床驱动的题型
- 双语支持：中文（~598 MB）+ 英文（~443 MB）

</td>
<td width="50%">

**丰富的基线覆盖**
- **3 种经典基线**：Long Context, Embedding RAG, BM25 RAG
- **7 种结构化 Agentic 记忆系统**：Mem0, Letta, MemOS, A-MEM, MIRIX, MemRL, LightMem
- **4 种基于图结构的记忆系统**：GraphRAG, HippoRAG-v2, ReMem, Zep

</td>
</tr>
<tr>
<td>

**统一评测框架**
- 通过 `BaseAgent` 即插即用集成新方法
- 多指标评测：字符串匹配 + LLM-as-a-Judge
- 断点续跑支持，适合长时间实验
- Dry-run 模式快速验证流水线

</td>
<td>

**灵活配置管理**
- YAML 驱动的方法与数据集配置
- 多供应商 LLM 支持（OpenAI / BigModel / Azure）
- 本地与远程 Embedding 模型
- 跨基准评测（MedMemoryBench + LoCoMo）

</td>
</tr>
</table>

### 查询类型

| 类型 | 数量 | 说明 |
|:-----|:-----:|:-----|
| `entity_exact_match` | 400 | 医疗实体的精确回忆 |
| `temporal_localization` | 400 | 时间相关的临床推理 |
| `multiple_choice` | 398 | 多选项医学问答 |
| `inference_generation` | 397 | 开放式临床推断 |
| `state_update` | 200 | 追踪患者状态演变 |
| `multi_hop_clinical_deduction` | 191 | 多跳临床推理 |

## 📁 项目结构

<details>
<summary>点击展开完整目录树</summary>

```
MedMemoryBench/
├── main.py                       # 评测入口
├── requirements.txt              # Python 依赖
├── LICENSE                       # Apache License 2.0
├── LEGAL.md                      # 注释语言法律声明
├── .env.example                  # 环境变量模板
│
├── configs/                      # 配置文件
│   ├── method_config/            # 各方法 YAML 配置（gpt-5.1 / qwen3 变体）
│   │   ├── long_context_gpt-5.1.yaml
│   │   ├── embedding_rag_gpt-5.1.yaml
│   │   ├── bm25_rag_gpt-5.1.yaml
│   │   ├── graph_rag_gpt-5.1.yaml
│   │   ├── mem0_gpt-5.1.yaml
│   │   ├── memos_gpt-5.1.yaml
│   │   ├── memrl_gpt-5.1.yaml
│   │   ├── amem_gpt-5.1.yaml
│   │   ├── hipporag_gpt-5.1.yaml
│   │   ├── lightmem_gpt-5.1.yaml
│   │   ├── letta_gpt-5.1.yaml
│   │   ├── mirix_gpt-5.1.yaml
│   │   ├── remem_gpt-5.1.yaml
│   │   ├── zep_gpt-5.1-chat.yaml
│   │   └── ...                   # + qwen3 变体
│   └── dataset_config/
│       ├── medmemorybench.yaml
│       └── locomo.yaml
│
├── methods/                      # 记忆方法实现
│   ├── base.py                   # BaseAgent 抽象基类
│   ├── long_context.py           # 长上下文基线
│   ├── embedding_rag.py          # 稠密向量 RAG
│   ├── bm25_rag.py               # BM25 稀疏 RAG
│   ├── graph_rag.py              # 图谱 RAG
│   ├── self_rag.py               # Self-RAG
│   ├── mem0_agent.py             # Mem0 适配器
│   ├── memos_agent.py            # MemOS 适配器
│   ├── memrl_agent.py            # MemRL 适配器
│   ├── amem_agent.py             # A-MEM 适配器
│   ├── hipporag_agent.py         # HippoRAG 适配器
│   ├── lightmem_agent.py         # LightMem 适配器
│   ├── letta_agent.py            # Letta 适配器
│   ├── mirix_agent.py            # MIRIX 适配器
│   ├── remem_agent.py            # ReMem 适配器
│   ├── zep_agent.py              # Zep Cloud 适配器
│   └── <第三方上游仓库>            # mem0/, memOS/, MemRL/, amem/, HippoRAG/,
│                                 # LightMem/, letta/, MIRIX/, REMem/, MEM1/,
│                                 # cognee/, memorag/（第三方源码）
│
├── benchmarks/                   # 数据集评测实现
│   ├── base.py                   # BaseDataset 抽象基类
│   ├── medmemorybench/           # MedMemoryBench 数据集
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   └── locomo/                   # LoCoMo 数据集
│       ├── dataset.py
│       └── evaluator.py
│
├── metrics/                      # 评测指标
│   ├── base.py                   # BaseMetric 抽象基类
│   ├── string_match.py           # 字符串匹配
│   ├── llm_judge.py              # LLM-as-a-Judge
│   └── locomo_metrics.py         # LoCoMo 专用指标
│
├── src/                          # 核心编排模块
│   ├── config.py                 # 配置加载器
│   ├── agent.py                  # AgentManager
│   ├── evaluator.py              # 评测调度器
│   └── result.py                 # 结果收集与报告
│
├── utils/                        # 工具模块
│   ├── llm_client.py             # 统一 LLM 客户端
│   ├── tokenizer.py              # 分词器
│   ├── templates.py              # Prompt 模板
│   ├── prompts_qa.py             # 问答 Prompt
│   ├── prompts_judge.py          # 评判 Prompt
│   ├── prompts_memorize.py       # 记忆化 Prompt
│   ├── langchain_callback.py     # LangChain 回调
│   └── logger.py                 # 日志
│
├── docker/                       # 可选的服务编排文件
│   ├── mirix-init.sql
│   └── mirix-services.yml
│
├── scripts/                      # 辅助脚本
│   ├── run_eval.sh
│   └── mirix-services.sh
│
├── data/                         # 数据集（Git LFS）
│   ├── MedMemoryBench/           # 中文版，~598 MB
│   ├── MedMemoryBench_EN/        # 英文版，~443 MB
│   └── locomo/                   # LoCoMo，~18 MB
│
├── generation/                   # 数据集生成流水线（子项目）
├── outputs/                      # 评测输出（gitignored）
├── exp_results/                  # 实验报告归档
├── logs/                         # 运行日志（gitignored）
└── results/                      # 方法侧缓存（gitignored）
```

</details>

## 🚀 快速开始

### 1. 克隆仓库

> **注意：** 本仓库通过 **Git LFS** 管理数据集，请在克隆前安装 Git LFS。

```bash
# 安装 Git LFS（已安装则跳过）
brew install git-lfs                  # macOS
sudo apt-get install git-lfs          # Ubuntu/Debian
# Windows: https://git-lfs.github.com/

git lfs install
git clone https://github.com/AQ-MedAI/MedMemoryBench.git
cd MedMemoryBench
```

### 2. 环境配置

<details open>
<summary><b>使用 uv（推荐）</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

</details>

<details>
<summary><b>使用 conda</b></summary>

```bash
conda create -n medmemorybench python=3.10
conda activate medmemorybench
pip install -r requirements.txt
```

</details>

> **方法特定依赖：** 部分记忆方法在 `methods/` 下集成了上游包（如 `methods/mem0/`、`methods/memOS/`）。若某方法自带 `requirements.txt` 或 `README`，请按其说明安装。

> **Embedding 模型：** 方法配置可引用本地 Embedding 模型或 API，若引用本地配置，运行前请先下载对应的模型。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写所需的 API 密钥：

```env
# BigModel（OpenAI 兼容接口，本项目主要使用）
BIGMODEL_API_KEY=your_bigmodel_api_key
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# OpenAI（可选）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI（可选）
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# Zep Cloud（可选，仅 Zep agent 需要）
ZEP_API_KEY=your_zep_api_key

# 默认模型选择
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai

# 可选：隔离 Letta 本地运行时数据（默认 ~/.letta）
LETTA_DIR=.tmp/letta_runtime
```

### 4. 运行评测

**通过脚本：**

```bash
./scripts/run_eval.sh bm25_rag_gpt-5.1 medmemorybench
```

**通过 Python：**

```bash
# 标准运行
python main.py -m bm25_rag_gpt-5.1 -d medmemorybench

# Dry run 模式（不实际调用 LLM API）
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --dry-run

# 断点续跑
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --resume

# 查看可用方法/数据集
python main.py --list-methods
python main.py --list-datasets
```

## 🔧 配置说明

### 方法配置

每个方法通过 `configs/method_config/` 下的 YAML 文件驱动：

```yaml
# configs/method_config/embedding_rag_gpt-5.1.yaml

method_name: "embedding_rag"
method_type: "rag"                  # baseline / rag / agentic_memory
description: "Embedding RAG Agent - Dense vector retrieval based RAG method"

model:
  provider: "openai"
  name: "gpt-5.1"
  temperature: 0.3
  max_completion_tokens: 100000

agent_params:
  top_k: 5                          # 检索文档数
  chunk_size: 512                   # 文本块大小
  chunk_overlap: 50                 # 块重叠大小

embedding:
  provider: "local"                 # openai / local / huggingface
  model: "/path/to/local/model"
```

### 数据集配置

数据集配置位于 `configs/dataset_config/`：

```yaml
# configs/dataset_config/medmemorybench.yaml

dataset_name: "medmemorybench"
description: "Medical dialogue memory evaluation dataset"
language: "zh"

data:
  root_dir: "data/MedMemoryBench"
  sessions_pattern: "persona_{id}/eval/generated_dialogues.json"
  queries_pattern: "persona_{id}/eval/generated_queries.json"

evaluation:
  mode: "independent"               # independent / merged
  evaluation_interval: 10           # 每 N 轮会话评测一次

query_types:
  - name: "entity_exact_match"
    metric: "string_contain"
  - name: "temporal_localization"
    metric: "llm_judge"
  # ... 更多类型
```

## 📄 输出格式

评测结果保存在 `outputs/<方法>_<模型>/` 目录下：

```
outputs/
└── bm25_rag_gpt-5.1/
    ├── eval_medmemorybench_20260330_181703.json    # 详细结果（JSON）
    ├── report_medmemorybench_20260330_181703.txt   # 可读报告
    └── memory_builds_20260330_181703.json          # 记忆构建日志
```

## 📝 引用

如果 MedMemoryBench 对您的研究有帮助，请考虑引用我们的工作：

```bibtex
@article{medmemorybench2026,
  title={MedMemoryBench: Benchmarking Agent Memory in Personalized Healthcare},
  author={TODO},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 📜 许可协议

- **代码** — [Apache License 2.0](LICENSE)
- **数据集**（`data/MedMemoryBench/`、`data/MedMemoryBench_EN/`）— [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- `methods/` 下的第三方源码保留其原始上游许可协议。
- 参见 [LEGAL.md](LEGAL.md) 了解源码注释语言条款。
