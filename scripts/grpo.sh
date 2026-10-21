#!/bin/bash
# GRPO training and evaluation on one training GPU plus one vLLM/judge GPU.
set -euo pipefail

MODEL_KEY="ministral3-8b"
DATASET="math"
SEED=84
SEED_SET=false
TRAIN_GPU=0
VLLM_GPU=1
VLLM_PORT=8000
DRY_RUN=false
CHECK_SFT_ONLY=false

usage() {
    cat <<'EOF'
Usage: scripts/grpo.sh [options]

Model and dataset:
  --model NAME          Model shortcut: qwen3-8b, ministral3-8b, or mimo-7b.
                        Default: ministral3-8b
  --dataset NAME        Dataset: math, gsm8k, chemistryqa, or formulareasoning.
                        Default: math
  --seed VALUE          Training and evaluation seed. Default: 84

Runtime:
  --train-gpu ID        CUDA device for training. Default: 0
  --vllm-gpu ID         CUDA device for vLLM and the non-math judge. Default: 1
  --vllm-port PORT      TRL vLLM server port. Default: 8000
  --check-sft-checkpoint
                        Verify the resolved SFT checkpoint and exit.
  --dry-run             Print resolved paths and settings without training or evaluation.
  -h, --help            Show this help.

All GRPO runs start from the model/dataset-specific SFT checkpoint. Evaluation
uses one model-specific temperature: 0.8 for Qwen3-8B, 0.6 for Ministral3-8B,
and 0.7 for MiMo-7B.

Examples:
  scripts/grpo.sh --model qwen3-8b --dataset gsm8k --seed 84
  scripts/grpo.sh --model ministral3-8b --dataset chemistryqa --seed 84
  scripts/grpo.sh --model mimo-7b --dataset formulareasoning --seed 168
EOF
}

require_value() {
    if [[ $# -lt 2 ]]; then
        echo "$1 requires a value."
        usage
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            require_value "$@"
            MODEL_KEY="$2"
            shift 2
            ;;
        --dataset)
            require_value "$@"
            DATASET="$2"
            shift 2
            ;;
        --seed)
            require_value "$@"
            SEED="$2"
            SEED_SET=true
            shift 2
            ;;
        --train-gpu)
            require_value "$@"
            TRAIN_GPU="$2"
            shift 2
            ;;
        --vllm-gpu)
            require_value "$@"
            VLLM_GPU="$2"
            shift 2
            ;;
        --vllm-port)
            require_value "$@"
            VLLM_PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --check-sft-checkpoint)
            CHECK_SFT_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer."
    exit 1
fi
if ! [[ "${TRAIN_GPU}" =~ ^[0-9]+$ && "${VLLM_GPU}" =~ ^[0-9]+$ ]]; then
    echo "--train-gpu and --vllm-gpu must be non-negative integers."
    exit 1
fi
if ! [[ "${VLLM_PORT}" =~ ^[0-9]+$ ]]; then
    echo "--vllm-port must be a non-negative integer."
    exit 1
fi

case "${MODEL_KEY}" in
    qwen3-8b)
        MODEL="Qwen/Qwen3-8B-Base"
        MODEL_DIR="qwen3_8b"
        USE_TRANSFORMERS_BACKEND=false
        TRUST_REMOTE_CODE=false
        EVAL_TEMP=0.8
        ;;
    ministral3-8b)
        MODEL="Aratako/Ministral-3-8B-Base-2512-TextOnly"
        MODEL_DIR="ministral3_8b"
        USE_TRANSFORMERS_BACKEND=true
        TRUST_REMOTE_CODE=true
        EVAL_TEMP=0.6
        ;;
    mimo-7b)
        MODEL="XiaomiMiMo/MiMo-7B-Base"
        MODEL_DIR="mimo_7b"
        USE_TRANSFORMERS_BACKEND=false
        TRUST_REMOTE_CODE=true
        EVAL_TEMP=0.7
        ;;
    *)
        echo "--model must be qwen3-8b, ministral3-8b, or mimo-7b."
        exit 1
        ;;
esac

case "${DATASET}" in
    math)
        DATASET_DISPLAY="MATH"
        HAS_JUDGE=false
        VLLM_GPU_MEMORY_UTILIZATION=0.8
        MAX_MODEL_LEN=4096
        ;;
    gsm8k)
        DATASET_DISPLAY="GSM8K"
        HAS_JUDGE=false
        VLLM_GPU_MEMORY_UTILIZATION=0.8
        MAX_MODEL_LEN=2048
        ;;
    chemistryqa)
        DATASET_DISPLAY="ChemistryQA"
        HAS_JUDGE=true
        VLLM_GPU_MEMORY_UTILIZATION=0.45
        MAX_MODEL_LEN=4096
        ;;
    formulareasoning)
        DATASET_DISPLAY="FormulaReasoning"
        HAS_JUDGE=true
        VLLM_GPU_MEMORY_UTILIZATION=0.45
        MAX_MODEL_LEN=4096
        ;;
    *)
        echo "--dataset must be math, gsm8k, chemistryqa, or formulareasoning."
        exit 1
        ;;
esac

case "${MODEL_KEY}:${DATASET}" in
    qwen3-8b:gsm8k)
        SFT_CHECKPOINT="workdirs/qwen3_8b/gsm8k/baseline/cot/e-1/train/lora_checkpoint"
        INITIALIZATION_LABEL="from_cot_e1"
        ;;
    qwen3-8b:math)
        SFT_CHECKPOINT="workdirs/qwen3_8b/math/baseline/e-3/train/lora_checkpoint"
        INITIALIZATION_LABEL="from_e3"
        ;;
    qwen3-8b:chemistryqa)
        SFT_CHECKPOINT="workdirs/qwen3_8b/chemistryqa/baseline/e-1/train/lora_checkpoint"
        INITIALIZATION_LABEL="from_e1"
        ;;
    qwen3-8b:formulareasoning)
        SFT_CHECKPOINT="workdirs/qwen3_8b/formulareasoning/baseline/e-1/train/lora_checkpoint"
        INITIALIZATION_LABEL="from_e1"
        ;;
    ministral3-8b:*|mimo-7b:*)
        SFT_CHECKPOINT="workdirs/${MODEL_DIR}/${DATASET}/baseline/e-1/train/lora_checkpoint"
        INITIALIZATION_LABEL="from_e1"
        ;;
esac

CONFIG="configs/grpo/grpo_${DATASET}_${MODEL_KEY}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "GRPO config does not exist: ${CONFIG}"
    exit 1
fi

check_sft_checkpoint() {
    if [[ ! -d "${SFT_CHECKPOINT}" || -z "$(ls -A "${SFT_CHECKPOINT}" 2>/dev/null)" ]]; then
        echo "SFT checkpoint does not exist or is empty: ${SFT_CHECKPOINT}"
        return 1
    fi
    echo "SFT checkpoint exists and is non-empty: ${SFT_CHECKPOINT}"
}

if [[ "${CHECK_SFT_ONLY}" == "true" ]]; then
    check_sft_checkpoint
    exit 0
fi

# Only an explicit --seed adds a suffix, so reruns with other seeds do not reuse this run's outputs.
SEED_SUFFIX=""
if [[ "${SEED_SET}" == "true" ]]; then
    SEED_SUFFIX="_seed-${SEED}"
fi
RUN_KEY="${INITIALIZATION_LABEL}${SEED_SUFFIX}"
WORK_DIR="workdirs/${MODEL_DIR}/${DATASET}/grpo/${RUN_KEY}"
TRAIN_DIR="${WORK_DIR}/train"
CHECKPOINT="${TRAIN_DIR}/lora_checkpoint"
RUN_NAME="${MODEL_DIR}_grpo_${DATASET}_${RUN_KEY}"
ALL_GPUS="${TRAIN_GPU},${VLLM_GPU}"

CFG_OPTIONS=(
    "training.vllm_server_port=${VLLM_PORT}"
)
if [[ "${HAS_JUDGE}" == "true" ]]; then
    CFG_OPTIONS+=(
        "judge_engine.cuda_devices=${VLLM_GPU}"
        "judge_engine.gpu_memory_utilization=0.4"
    )
fi

VLLM_ARGS=(
    trl vllm-serve
    --model "${MODEL}"
    --port "${VLLM_PORT}"
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max_model_len "${MAX_MODEL_LEN}"
)
if [[ "${USE_TRANSFORMERS_BACKEND}" == "true" ]]; then
    VLLM_ARGS+=(--vllm_model_impl transformers)
fi
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    VLLM_ARGS+=(--trust_remote_code)
fi

EVAL_OPTIONS=(
    "vllm_sampling.temperature=${EVAL_TEMP}"
    "vllm_engine.tensor_parallel_size=2"
)
if [[ "${USE_TRANSFORMERS_BACKEND}" == "true" ]]; then
    EVAL_OPTIONS+=("vllm_engine.model_impl=transformers")
fi
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    EVAL_OPTIONS+=("vllm_engine.trust_remote_code=true")
fi
if [[ "${HAS_JUDGE}" == "true" ]]; then
    EVAL_OPTIONS+=(
        "judge_engine.cuda_devices=${ALL_GPUS}"
        "judge_engine.gpu_memory_utilization=0.85"
    )
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    cat <<EOF
MODEL_KEY=${MODEL_KEY}
MODEL=${MODEL}
MODEL_DIR=${MODEL_DIR}
DATASET=${DATASET}
CONFIG=${CONFIG}
SEED=${SEED}
SFT_CHECKPOINT=${SFT_CHECKPOINT}
INITIALIZATION_LABEL=${INITIALIZATION_LABEL}
WORK_DIR=${WORK_DIR}
RUN_NAME=${RUN_NAME}
USE_TRANSFORMERS_BACKEND=${USE_TRANSFORMERS_BACKEND}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
HAS_JUDGE=${HAS_JUDGE}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
EVAL_TEMP=${EVAL_TEMP}
EOF
    printf 'VLLM_ARGS='
    printf ' %q' "${VLLM_ARGS[@]}"
    printf '\n'
    printf 'CFG_OPTIONS='
    printf ' %q' "${CFG_OPTIONS[@]}"
    printf '\n'
    printf 'EVAL_OPTIONS='
    printf ' %q' "${EVAL_OPTIONS[@]}"
    printf '\n'
    exit 0
fi

# Fail before starting any experiment process, even when a GRPO checkpoint
# already exists and training would otherwise be skipped.
check_sft_checkpoint

VLLM_PID=""
cleanup_vllm() {
    if [[ -z "${VLLM_PID}" ]]; then
        return
    fi
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        PGID="$(ps -o pgid= -p "${VLLM_PID}" 2>/dev/null | tr -d ' ' || true)"
        if [[ -n "${PGID}" ]]; then
            kill -TERM -- "-${PGID}" 2>/dev/null || true
        else
            kill -TERM "${VLLM_PID}" 2>/dev/null || true
        fi
        for _ in {1..10}; do
            if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            if [[ -n "${PGID}" ]]; then
                kill -KILL -- "-${PGID}" 2>/dev/null || true
            else
                kill -KILL "${VLLM_PID}" 2>/dev/null || true
            fi
        fi
    fi
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
}
trap cleanup_vllm EXIT INT TERM

start_vllm() {
    local vllm_log="$1"

    if command -v lsof >/dev/null 2>&1 && lsof -ti:"${VLLM_PORT}" >/dev/null 2>&1; then
        echo "Port ${VLLM_PORT} is already in use. Stop that process or select another --vllm-port."
        exit 1
    fi

    echo "[$(date)] Starting ${MODEL_KEY} TRL vLLM server on GPU ${VLLM_GPU} (log: ${vllm_log})..."
    setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_ARGS[@]}" > "${vllm_log}" 2>&1 &
    VLLM_PID=$!

    echo "[$(date)] Waiting for vLLM server to start..."
    for ((attempt = 1; attempt <= 120; attempt++)); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[$(date)] vLLM server is ready."
            return
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[$(date)] ERROR: vLLM server process died. See ${vllm_log}."
            exit 1
        fi
        sleep 5
    done

    echo "[$(date)] ERROR: vLLM server failed to become ready. See ${vllm_log}."
    exit 1
}

# Training
if [[ -d "${CHECKPOINT}" && -n "$(ls -A "${CHECKPOINT}" 2>/dev/null)" ]]; then
    echo "[$(date)] Checkpoint exists, skipping training: ${CHECKPOINT}"
else
    mkdir -p "${TRAIN_DIR}"
    start_vllm "${TRAIN_DIR}/vllm_server.log"

    echo "[$(date)] Starting ${MODEL_KEY} GRPO training on ${DATASET_DISPLAY}, seed ${SEED}..."
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python tools/grpo_train.py \
        "${CONFIG}" \
        --seed "${SEED}" \
        --checkpoint "${SFT_CHECKPOINT}" \
        -w "${TRAIN_DIR}" \
        --run-name "${RUN_NAME}" \
        -o "${CFG_OPTIONS[@]}"

    echo "[$(date)] Training complete. Stopping vLLM server..."
    cleanup_vllm
fi

# Evaluation uses the pair of GPUs on this node at one model-specific temperature.
EVAL_DIR="${WORK_DIR}/eval/temp-${EVAL_TEMP}"
METRICS_FILE="${EVAL_DIR}/metrics_summary.json"
if [[ -f "${METRICS_FILE}" ]]; then
    echo "[$(date)] Metrics exist, skipping: ${METRICS_FILE}"
else
    echo "[$(date)] Evaluating ${DATASET_DISPLAY} at temperature=${EVAL_TEMP}, seed=${SEED}..."
    CUDA_VISIBLE_DEVICES="${ALL_GPUS}" python tools/evaluate.py \
        "${CONFIG}" \
        --seed "${SEED}" \
        --checkpoint "${CHECKPOINT}" \
        -w "${EVAL_DIR}" \
        -o "${EVAL_OPTIONS[@]}"
fi

echo "[$(date)] GRPO experiment complete: ${RUN_NAME}"
