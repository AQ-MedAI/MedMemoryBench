#!/bin/bash
# MIRIX LoCoMo Evaluation Runner
# 使用说明：
#   0. 安装缺失依赖: bash evals/run_locomo_eval.sh deps
#   1. 先启动 MIRIX 后端服务: bash <BENCH_ROOT>/scripts/mirix-services.sh start
#   2. 在一个终端启动 embedding server: bash evals/run_locomo_eval.sh embedding
#   3. 在另一个终端启动 MIRIX API server: bash evals/run_locomo_eval.sh server
#   4. 在第三个终端运行评测: bash evals/run_locomo_eval.sh eval [--limit N]
#   5. 计算指标: bash evals/run_locomo_eval.sh judge

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
VENV="$BENCH_ROOT/.venv/bin/activate"
ENV_FILE="$BENCH_ROOT/.env"
DATA_PATH="$BENCH_ROOT/data/locomo/data/locomo10.json"
CONFIG_PATH="$SCRIPT_DIR/configs/locomo_eval.yaml"
OUTPUT_PATH="$SCRIPT_DIR/results/locomo_gpt5.1"

# Activate venv
source "$VENV"

# Load env
set -a
source "$ENV_FILE"
set +a

case "${1:-help}" in
    deps)
        echo "=== Installing missing dependencies ==="
        pip install pg8000 pgvector psycopg2-binary 2>/dev/null || true
        echo "Done."
        ;;
    embedding)
        echo "=== Starting Embedding Server (bge-small-en-v1.5 on port 6100) ==="
        python "$SCRIPT_DIR/embedding_server.py" \
            --model "$BENCH_ROOT/models/bge-small-en-v1.5" \
            --port 6100
        ;;
    server)
        echo "=== Starting MIRIX API Server (port 8531) ==="
        cd "$PROJECT_ROOT"
        python scripts/start_server.py --port 8531
        ;;
    eval)
        shift
        echo "=== Running LoCoMo Evaluation ==="
        echo "Data: $DATA_PATH"
        echo "Config: $CONFIG_PATH"
        echo "Output: $OUTPUT_PATH"
        echo ""
        cd "$SCRIPT_DIR"
        python main_eval.py \
            --data "$DATA_PATH" \
            --mirix_config_path "$CONFIG_PATH" \
            --output_path "$OUTPUT_PATH" \
            --run-llm \
            "$@"
        ;;
    judge)
        echo "=== Computing Metrics ==="
        cd "$SCRIPT_DIR"
        python organize_results.py "$OUTPUT_PATH"
        echo ""
        echo "Results saved to: $OUTPUT_PATH/metrics.json"
        ;;
    help|--help|-h)
        echo ""
        echo "MIRIX LoCoMo Evaluation Runner"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  deps        Install missing Python dependencies (pg8000, etc.)"
        echo "  embedding   Start local embedding server (port 6100)"
        echo "  server      Start MIRIX API server (port 8531)"
        echo "  eval        Run evaluation (add --limit N for quick test)"
        echo "  judge       Compute accuracy metrics from results"
        echo ""
        echo "Example (full run):"
        echo "  Terminal 1: $0 embedding"
        echo "  Terminal 2: $0 server"
        echo "  Terminal 3: $0 eval --limit 1    # smoke test"
        echo "  Terminal 3: $0 eval              # full run (10 samples)"
        echo "  Terminal 3: $0 judge             # compute metrics"
        echo ""
        ;;
    *)
        echo "Unknown command: $1"
        $0 help
        exit 1
        ;;
esac
