# MedMemoryBench

> A Benchmark Framework for Evaluating Agent Memory Methods on Medical Dialogue

**MedMemoryBench** is a benchmark framework for evaluating Agent memory methods, with a focus on memory capability assessment in medical dialogue scenarios. This framework provides unified evaluation interfaces, multiple baseline method implementations, and a flexible configuration management system, while also supporting the import and evaluation of other datasets.

## 📁 Project Structure

```
MedMemoryBench/
├── main.py                    # Evaluation entry point
├── requirements.txt           # Dependencies
├── .env                       # Environment variables (API Keys, etc.)
│
├── configs/                   # Configuration files
│   ├── method_config/         # Method configurations
│   │   ├── long_context_gpt-5.1-chat.yaml
│   │   ├── embedding_rag_gpt-5.1-chat.yaml
│   │   ├── bm25_rag_gpt-5.1-chat.yaml
│   │   ├── mem0_gpt-5.1-chat.yaml
│   │   ├── zep_gpt-5.1-chat.yaml
│   │   └── graph_rag_gpt-5.1-chat.yaml
│   └── dataset_config/        # Dataset configurations
│       ├── medmemorybench.yaml
│       └── locomo.yaml
│
├── methods/                   # Memory method implementations
│   ├── README.md             # Method extension guide
│   ├── base.py               # BaseAgent base class
│   ├── long_context.py       # Long Context method
│   ├── embedding_rag.py      # Embedding RAG method
│   ├── bm25_rag.py           # BM25 RAG method
│   ├── mem0_agent.py         # Mem0 Agent
│   ├── zep_agent.py          # Zep Agent
│   └── graph_rag_agent.py    # GraphRAG Agent
│
├── benchmarks/                # Dataset evaluation implementations
│   ├── base.py               # BaseDataset base class
│   └── medmemorybench/       # MedMemoryBench dataset
│       ├── dataset.py        # Data loading
│       ├── evaluator.py      # Evaluation logic
│       └── checkpoint.py     # Checkpoint resumption
│
├── metrics/                   # Evaluation metrics
│   ├── base.py               # BaseMetric base class
│   ├── string_match.py       # String matching metrics
│   └── llm_judge.py          # LLM Judge metrics
│
├── src/                       # Core modules
│   ├── config.py             # Configuration loader
│   ├── agent.py              # AgentManager
│   ├── evaluator.py          # Evaluation dispatcher
│   └── result.py             # Result collection and reporting
│
├── utils/                     # Utility modules
│   ├── llm_client.py         # LLM client
│   ├── templates.py          # Prompt templates
│   ├── tokenizer.py          # Tokenizer
│   └── logger.py             # Logger
│
├── scripts/                   # Scripts
│   └── run_eval.sh           # Evaluation run script
│
├── data/                      # Data directory
│   └── MedMemoryBench/       # MedMemoryBench dataset
│
└── outputs/                   # Evaluation results output
```

## 🚀 Quick Start

### 1. Environment Setup

**Clone with Git LFS:**
```bash
# Install Git LFS (if not already installed)
# macOS
brew install git-lfs
# Ubuntu/Debian
sudo apt-get install git-lfs
# Windows
# Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

git clone <repository-url>
```

**Using uv (Recommended):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv pip install -r requirements.txt
```

**Using conda:**
```bash
# Create conda environment
conda create -n medmemorybench python=3.10
conda activate medmemorybench

# Install dependencies
pip install -r requirements.txt
```

**2. Method-Specific Setup**

Some memory methods include their own packages and require additional installation. Refer to the respective method's README for setup instructions.

**Embedding models:** Method configs reference local embedding models under `models/` (e.g., `models/bge-small-zh-v1.5`). Download them before running:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5').save('models/bge-small-zh-v1.5')"
```

You can also set the `MODELS_DIR` environment variable to point to a custom models directory.

### 3. Configure Environment Variables

Copy and edit the `.env` file:

```bash
cp .env.example .env
```

Configure API Keys:

```env
# BigModel (OpenAI-compatible, recommended for this project)
BIGMODEL_API_KEY=your_bigmodel_api_key
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# (Optional) Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# (Optional) Zep Cloud
ZEP_API_KEY=your_zep_api_key

# Default model configuration
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai

# (Optional) isolate Letta local runtime data
# If unset, Letta may use ~/.letta
LETTA_DIR=.tmp/letta_runtime
```

Notes:
- For Letta with BigModel, set `BIGMODEL_API_KEY` / `BIGMODEL_BASE_URL` first. The framework will map them to OpenAI-compatible settings internally.
- `OPENAI_API_KEY` can still be kept for other methods, but Letta auth should be validated against the effective runtime key.
- `LETTA_DIR` is recommended to avoid stale local SQLite metadata from previous Letta runs.

### 4. Run Evaluation

**Using script:**
```bash
./scripts/run_eval.sh bm25_rag_gpt-5.1-chat medmemorybench
```

**Using Python:**
```bash
python main.py -m bm25_rag_gpt-5.1-chat -d medmemorybench

# Dry Run mode (no actual API calls)
python main.py -m embedding_rag_gpt-5.1-chat -d medmemorybench --dry-run
```

> 💡 **Adding a new method?** See [methods/README.md](methods/README.md) for how to extend new memory methods.

## Troubleshooting (Letta + BigModel)

### Symptom: `curl` works, but Letta reports `401` / `forbidden`

Common cause: Letta runtime reads a different credential source than the one used in `curl`.

Check in the same shell session used to run evaluation:

```bash
python -c "from src.config import load_env_config; c=load_env_config(); print(bool(c.bigmodel_api_key or c.openai_api_key), c.bigmodel_base_url or c.openai_base_url)"
```

Then verify both Chat and Embedding endpoints with the same key/base:

```bash
python -c "from openai import OpenAI; from src.config import load_env_config; c=load_env_config(); client=OpenAI(api_key=c.bigmodel_api_key or c.openai_api_key, base_url=c.bigmodel_base_url or c.openai_base_url); print('chat', bool(client.chat.completions.create(model='glm-4-plus', messages=[{'role':'user','content':'ping'}], max_tokens=8).choices)); print('emb', len(client.embeddings.create(model='embedding-3', input='ping').data[0].embedding))"
```

If this passes, your key is valid and Letta should authenticate correctly.

### Symptom: Letta fails with SQLite schema / migration errors

Cause: Letta default local DB (`~/.letta/sqlite.db`) contains old schema state.

Fix:
- Set `LETTA_DIR` to an isolated project-local path, e.g. `.tmp/letta_runtime`.
- Re-run evaluation so Letta creates a fresh local DB under that directory.

### Symptom: intermittent timeout / handshake timeout

This usually indicates transient network/proxy/TLS instability, not invalid API key.

Suggested actions:
- Retry the same command in a quiet period (avoid parallel test runs).
- Ensure proxy settings in the terminal are consistent with your successful `curl` environment.

## 🔧 Configuration

### Method Configuration (method_config)

Each method configuration file is located in `configs/method_config/`:

```yaml
# configs/method_config/embedding_rag_gpt-5.1-chat.yaml

# Method basic information
method_name: "embedding_rag"
method_type: "rag"  # baseline / rag / agentic_memory
description: "Embedding RAG Agent - Vector retrieval based RAG method"

# Model configuration
model:
  provider: "openai"
  name: "gpt-5.1-chat"
  temperature: 1.0
  max_completion_tokens: 100000

# Method hyperparameters
agent_params:
  top_k: 5              # Number of documents to retrieve
  chunk_size: 512       # Text chunk size
  chunk_overlap: 50     # Chunk overlap size

# Embedding configuration (optional)
embedding:
  provider: "local"     # openai / local / huggingface
  model: "/path/to/local/model"
```

### Dataset Configuration (dataset_config)

Dataset configuration files are located in `configs/dataset_config/`:

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
  mode: "independent"       # independent / merged
  evaluation_interval: 10   # Evaluate every N sessions

query_types:
  - name: "entity_exact_match"
    metric: "string_contain"
  - name: "temporal_localization"
    metric: "llm_judge"
  # ... more types
```

## 📄 Output

Evaluation results are saved in `outputs/<method>_<model>/`:

```
outputs/
└── bm25_rag_gpt-5.1-chat/
    ├── eval_medmemorybench_20260330_181703.json    # Detailed evaluation results (JSON)
    ├── report_medmemorybench_20260330_181703.txt   # Human-readable report (text)
    └── memory_builds_20260330_181703.json          # Memory build logs
```
