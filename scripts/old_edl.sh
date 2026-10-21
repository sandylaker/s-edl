#!/bin/bash
# Old EDL training and evaluation on one training GPU plus one vLLM/judge GPU.
set -euo pipefail

MODEL_KEY="qwen3-8b"
DATASET="math"
SEED=84
SEED_SET=false
EDL_WEIGHT=5.0
EDL_BETA=1.2
SFT_CHECKPOINT_OVERRIDE=""
TRAIN_GPU=0
VLLM_GPU=1
VLLM_PORT=8000
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: scripts/old_edl.sh [options]

Model and dataset:
  --model NAME          Model shortcut: qwen3-8b, ministral3-8b, or mimo-7b.
                        Default: qwen3-8b
  --dataset NAME        Dataset: math, gsm8k, chemistryqa, or formulareasoning.
                        Default: math
  --seed VALUE          Training and evaluation seed. Default: 84

Old EDL settings:
  --edl-weight VALUE    EDL loss weight. Default: 5.0
  --edl-beta VALUE      EDL beta. Default: 1.2
  --sft-checkpoint PATH SFT LoRA checkpoint to continue from, e.g. one trained by
                        scripts/baseline_sft.sh --epochs 1. Default for Ministral3-8B
                        and MiMo-7B (except GSM8K):
                        workdirs/<model>/<dataset>/baseline/e-1/train/lora_checkpoint.
                        Other runs start from the base model unless this is set.

Runtime:
  --train-gpu ID        CUDA device for training. Default: 0
  --vllm-gpu ID         CUDA device for vLLM and the non-math judge. Default: 1
  --vllm-port PORT      vLLM HTTP server port. Default: 8000
  --dry-run             Print resolved paths and settings without training or evaluation.
  -h, --help            Show this help.

Evaluation uses one model-specific temperature: 0.8 for Qwen3-8B, 0.6 for
Ministral3-8B, and 0.7 for MiMo-7B. Importance sampling is enabled only for
Qwen3-8B on GSM8K. Qwen3-8B and every GSM8K run start from the base model;
Ministral3-8B and MiMo-7B otherwise continue from a 1-epoch SFT checkpoint
(see --sft-checkpoint).

Examples:
  scripts/old_edl.sh --model qwen3-8b --dataset gsm8k --seed 84
  scripts/old_edl.sh --model ministral3-8b --dataset chemistryqa --seed 21
  scripts/old_edl.sh --model mimo-7b --dataset formulareasoning --seed 42
  scripts/old_edl.sh --model mimo-7b --dataset chemistryqa \
      --sft-checkpoint workdirs/mimo_7b/chemistryqa/baseline/e-1_seed-21/train/lora_checkpoint
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
        --edl-weight)
            require_value "$@"
            EDL_WEIGHT="$2"
            shift 2
            ;;
        --edl-beta)
            require_value "$@"
            EDL_BETA="$2"
            shift 2
            ;;
        --sft-checkpoint)
            require_value "$@"
            SFT_CHECKPOINT_OVERRIDE="$2"
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
        TRUST_REMOTE_CODE=false
        EVAL_USE_TRANSFORMERS_BACKEND=false
        LOAD_SFT_CHECKPOINT=false
        EVAL_TEMP=0.8
        ;;
    ministral3-8b)
        MODEL="Aratako/Ministral-3-8B-Base-2512-TextOnly"
        MODEL_DIR="ministral3_8b"
        TRUST_REMOTE_CODE=true
        EVAL_USE_TRANSFORMERS_BACKEND=true
        LOAD_SFT_CHECKPOINT=true
        EVAL_TEMP=0.6
        ;;
    mimo-7b)
        MODEL="XiaomiMiMo/MiMo-7B-Base"
        MODEL_DIR="mimo_7b"
        TRUST_REMOTE_CODE=true
        EVAL_USE_TRANSFORMERS_BACKEND=false
        LOAD_SFT_CHECKPOINT=true
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
        VLLM_GPU_MEMORY_UTILIZATION=0.85
        ;;
    gsm8k)
        DATASET_DISPLAY="GSM8K"
        HAS_JUDGE=false
        VLLM_GPU_MEMORY_UTILIZATION=0.85
        LOAD_SFT_CHECKPOINT=false
        ;;
    chemistryqa)
        DATASET_DISPLAY="ChemistryQA"
        HAS_JUDGE=true
        VLLM_GPU_MEMORY_UTILIZATION=0.45
        ;;
    formulareasoning)
        DATASET_DISPLAY="FormulaReasoning"
        HAS_JUDGE=true
        VLLM_GPU_MEMORY_UTILIZATION=0.45
        ;;
    *)
        echo "--dataset must be math, gsm8k, chemistryqa, or formulareasoning."
        exit 1
        ;;
esac

IS_VALUE=false
IS_SUFFIX=""
if [[ "${MODEL_KEY}" == "qwen3-8b" && "${DATASET}" == "gsm8k" ]]; then
    IS_VALUE=true
    IS_SUFFIX="_is"
fi

CONFIG="configs/edl/edl_${DATASET}_${MODEL_KEY}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "Old EDL config does not exist: ${CONFIG}"
    exit 1
fi

SFT_CHECKPOINT=""
CHECKPOINT_ARGS=()
RUN_PREFIX=""
if [[ -n "${SFT_CHECKPOINT_OVERRIDE}" ]]; then
    LOAD_SFT_CHECKPOINT=true
    SFT_CHECKPOINT="${SFT_CHECKPOINT_OVERRIDE}"
    RUN_PREFIX="from_sft_"
elif [[ "${LOAD_SFT_CHECKPOINT}" == "true" ]]; then
    SFT_CHECKPOINT="workdirs/${MODEL_DIR}/${DATASET}/baseline/e-1/train/lora_checkpoint"
    RUN_PREFIX="from_e1_"
fi
if [[ "${LOAD_SFT_CHECKPOINT}" == "true" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "${SFT_CHECKPOINT}")
fi
# Only an explicit --seed adds a suffix, so reruns with other seeds do not reuse this run's outputs.
SEED_SUFFIX=""
if [[ "${SEED_SET}" == "true" ]]; then
    SEED_SUFFIX="_seed-${SEED}"
fi
RUN_KEY="${RUN_PREFIX}w-${EDL_WEIGHT}_b-${EDL_BETA}${IS_SUFFIX}${SEED_SUFFIX}"
WORK_DIR="workdirs/${MODEL_DIR}/${DATASET}/edl/${RUN_KEY}"
TRAIN_DIR="${WORK_DIR}/train"
CHECKPOINT="${TRAIN_DIR}/lora_checkpoint"
RUN_NAME="${MODEL_DIR}_edl_${DATASET}_${RUN_KEY}"
ALL_GPUS="${TRAIN_GPU},${VLLM_GPU}"
LORA_ADAPTER_DIR="/dev/shm/s_edl_adapter_port-${VLLM_PORT}"

CFG_OPTIONS=(
    "training.edl_weight=${EDL_WEIGHT}"
    "training.edl_beta=${EDL_BETA}"
    "training.vllm_base_url=http://localhost:${VLLM_PORT}"
    "training.lora_adapter_dir=${LORA_ADAPTER_DIR}"
    "training.vllm_importance_sampling=${IS_VALUE}"
)
if [[ "${HAS_JUDGE}" == "true" ]]; then
    CFG_OPTIONS+=(
        "judge_engine.cuda_devices=${VLLM_GPU}"
        "judge_engine.gpu_memory_utilization=0.4"
    )
fi

# --generation-config vllm: ignore the model's generation_config.json so request-time sampling parameters are honored.
# --no-enable-prefix-caching: this server does not invalidate cached KV states when the LoRA adapter is reloaded, so
#   prefix caching would reuse stale states after each weight update.
VLLM_ARGS=(
    python -m vllm.entrypoints.openai.api_server
    --model "${MODEL}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --served-model-name model
    --generation-config vllm
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max-num-seqs 128
    --max-model-len 4096
    --enable-lora
    --max-lora-rank 16
    --max-loras 1
    --no-enable-prefix-caching
)
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    VLLM_ARGS+=(--trust-remote-code)
fi

EVAL_OPTIONS=(
    "vllm_sampling.temperature=${EVAL_TEMP}"
    "vllm_engine.max_num_seqs=128"
    "vllm_engine.tensor_parallel_size=2"
)
if [[ "${EVAL_USE_TRANSFORMERS_BACKEND}" == "true" ]]; then
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
EDL_WEIGHT=${EDL_WEIGHT}
EDL_BETA=${EDL_BETA}
IS_VALUE=${IS_VALUE}
SFT_CHECKPOINT=${SFT_CHECKPOINT}
WORK_DIR=${WORK_DIR}
RUN_NAME=${RUN_NAME}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
LOAD_SFT_CHECKPOINT=${LOAD_SFT_CHECKPOINT}
HAS_JUDGE=${HAS_JUDGE}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}
LORA_ADAPTER_DIR=${LORA_ADAPTER_DIR}
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

    echo "[$(date)] Starting ${MODEL_KEY} vLLM server on GPU ${VLLM_GPU} (log: ${vllm_log})..."
    # VLLM_ALLOW_RUNTIME_LORA_UPDATING enables in-place adapter reloads through /v1/load_lora_adapter.
    setsid env \
        CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
        "${VLLM_ARGS[@]}" > "${vllm_log}" 2>&1 &
    VLLM_PID=$!

    echo "[$(date)] Waiting for vLLM server to start..."
    for ((attempt = 1; attempt <= 120; attempt++)); do
        if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
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
    if [[ "${LOAD_SFT_CHECKPOINT}" == "true" && ! -f "${SFT_CHECKPOINT}/adapter_config.json" ]]; then
        echo "ERROR: SFT checkpoint not found: ${SFT_CHECKPOINT}"
        echo "Train one with scripts/baseline_sft.sh --epochs 1 and pass it with --sft-checkpoint."
        exit 1
    fi
    mkdir -p "${TRAIN_DIR}"
    start_vllm "${TRAIN_DIR}/vllm_server.log"

    echo "[$(date)] Starting ${MODEL_KEY} old EDL training on ${DATASET_DISPLAY}, seed ${SEED}..."
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python tools/edl_train.py \
        "${CONFIG}" \
        --seed "${SEED}" \
        "${CHECKPOINT_ARGS[@]}" \
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

echo "[$(date)] Old EDL experiment complete: ${RUN_NAME}"
