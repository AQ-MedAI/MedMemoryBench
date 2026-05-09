#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TARGET_PERSONA_ID=15

PERSONA_COUNT=1
PERSONA_CONCURRENCY=3
PERSONA_TEMPERATURE=1.0
PERSONA_MAX_TOKENS=10000

# Event config
EVENT_EVENTS_PER_PHASE=19
EVENT_MAX_TOTAL=101
EVENT_CONCURRENCY=3
EVENT_TEMPERATURE=1.0
EVENT_MAX_TOKENS=100000

# Dialogue config
DIALOGUE_SESSIONS=101
DIALOGUE_TURNS=8
DIALOGUE_CONCURRENCY=1
DIALOGUE_TEMPERATURE=1.0
DIALOGUE_MAX_TOKENS=100000

# Query config
QUERY_NUM_EEM=2
QUERY_NUM_TLA=2
QUERY_NUM_SUA=1
QUERY_NUM_MQ=2
QUERY_NUM_IG=2
QUERY_NUM_MCD=2
QUERY_GENERATE_EVERY=10
QUERY_MCD_GENERATE_EVERY=20
QUERY_TEMPERATURE=1.0
QUERY_MAX_TOKENS=100000

mkdir -p logs
LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Activate the project-level virtual environment
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
else
    echo "Warning: No .venv found. Using current Python environment."
fi

if [ -n "${TARGET_PERSONA_ID}" ]; then
    PERSONA_ID_ARG="--persona-ids ${TARGET_PERSONA_ID}"
    PERSONA_COUNT_ARG=""
    echo "Using TARGET_PERSONA_ID=${TARGET_PERSONA_ID} for all stages"
else
    PERSONA_ID_ARG=""
    PERSONA_COUNT_ARG="--count ${PERSONA_COUNT}"
    echo "Using PERSONA_COUNT=${PERSONA_COUNT} to generate new personas"
fi

echo "Running Pipeline, log file: ${LOG_FILE}"
echo ""

{
    echo "========== Stage 1/6: Import base personas =========="
    python -m pipeline.cli import-personas \
      --input data/base_personas.json

    echo ""
    echo "========== Stage 2/6: Generate user personas =========="
    if [ -n "${TARGET_PERSONA_ID}" ]; then
        python -m pipeline.cli generate-personas \
          --persona-ids ${TARGET_PERSONA_ID} \
          --concurrency ${PERSONA_CONCURRENCY} \
          --temperature ${PERSONA_TEMPERATURE} \
          --max-tokens ${PERSONA_MAX_TOKENS}
    else
        python -m pipeline.cli generate-personas \
          --count ${PERSONA_COUNT} \
          --concurrency ${PERSONA_CONCURRENCY} \
          --temperature ${PERSONA_TEMPERATURE} \
          --max-tokens ${PERSONA_MAX_TOKENS}
    fi

    echo ""
    echo "========== Stage 3/6: Generate trap events =========="
    python -m pipeline.cli generate-trap-events \
      ${PERSONA_ID_ARG} \
      --output data/generated_trap_events.json \
      --temperature ${EVENT_TEMPERATURE} \
      --max-tokens ${EVENT_MAX_TOKENS}

    echo ""
    echo "========== Stage 4/6: Generate regular events =========="
    python -m pipeline.cli generate-regular-events \
      ${PERSONA_ID_ARG} \
      --trap-events-input data/generated_trap_events.json \
      --events-per-phase ${EVENT_EVENTS_PER_PHASE} \
      --max-total-events ${EVENT_MAX_TOTAL} \
      --output data/generated_events.json \
      --temperature ${EVENT_TEMPERATURE} \
      --max-tokens ${EVENT_MAX_TOKENS}

    echo ""
    echo "========== Stage 5/6: Generate dialogue sessions =========="
    python -m pipeline.cli generate-dialogues \
      ${PERSONA_ID_ARG} \
      --sessions ${DIALOGUE_SESSIONS} \
      --turns ${DIALOGUE_TURNS} \
      --concurrency ${DIALOGUE_CONCURRENCY} \
      --temperature ${DIALOGUE_TEMPERATURE} \
      --max-tokens ${DIALOGUE_MAX_TOKENS}

    echo ""
    echo "========== Stage 6/6: Generate evaluation queries =========="
    python -m pipeline.cli generate-queries \
      --input data/generated_dialogues.json \
      --output data/generated_queries.json \
      --num-eem ${QUERY_NUM_EEM} \
      --num-tla ${QUERY_NUM_TLA} \
      --num-sua ${QUERY_NUM_SUA} \
      --num-mq ${QUERY_NUM_MQ} \
      --num-ig ${QUERY_NUM_IG} \
      --num-mcd ${QUERY_NUM_MCD} \
      --generate-every ${QUERY_GENERATE_EVERY} \
      --mcd-generate-every ${QUERY_MCD_GENERATE_EVERY} \
      --temperature ${QUERY_TEMPERATURE} \
      --max-tokens ${QUERY_MAX_TOKENS}

    echo ""
    echo "✓ Pipeline completed successfully."
    echo ""
    echo "Note: Token usage statistics have been printed after each stage above."
    echo "You can search for 'Token Usage Summary' in the log to find detailed statistics."
} 2>&1 | tee "${LOG_FILE}"
