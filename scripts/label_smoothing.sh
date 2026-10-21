#!/bin/bash
# Label-smoothing SFT training on one GPU and evaluation on two GPUs.
set -euo pipefail

MODEL_KEY="ministral3-8b"
DATASET="math"
SEED=84
SEED_SET=false
LABEL_SMOOTHING=0.1
EPOCHS=""
TRAIN_GPU=0
EVAL_GPU=1
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: scripts/label_smoothing.sh [options]

Model and dataset:
  --model NAME          Model shortcut: qwen3-8b, ministral3-8b, or mimo-7b.
                        Default: ministral3-8b
  --dataset NAME        Dataset: math, gsm8k, chemistryqa, or formulareasoning.
                        Default: math
  --seed VALUE          Training and evaluation seed. Default: 84

Label-smoothing settings:
  --label-smoothing V   Label-smoothing factor in [0, 1). Default: 0.1
  --epochs VALUE        Number of training epochs. Default: 1, or 2.5 for
                        Ministral3-8B on GSM8K.

Runtime:
  --train-gpu ID        CUDA device for training and the first evaluation rank. Default: 0
  --eval-gpu ID         CUDA device for the second evaluation rank. Default: 1
  --dry-run             Print resolved paths and settings without training or evaluation.
  -h, --help            Show this help.

Training starts from the base model with the baseline SFT config and a
per-device batch size of 4. Evaluation uses one model-specific
temperature: 0.8 for Qwen3-8B, 0.6 for Ministral3-8B, and 0.7 for MiMo-7B.
Evaluation uses tensor parallelism over both GPUs. ChemistryQA and
FormulaReasoning load the judge on the same two GPUs after the generation
engine is released.

Examples:
  scripts/label_smoothing.sh --model qwen3-8b --dataset gsm8k --seed 84
  scripts/label_smoothing.sh --model ministral3-8b --dataset chemistryqa --seed 21 --train-gpu 2 --eval-gpu 3
  scripts/label_smoothing.sh --model mimo-7b --dataset formulareasoning --seed 42
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
        --label-smoothing)
            require_value "$@"
            LABEL_SMOOTHING="$2"
            shift 2
            ;;
        --epochs)
            require_value "$@"
            EPOCHS="$2"
            shift 2
            ;;
        --train-gpu)
            require_value "$@"
            TRAIN_GPU="$2"
            shift 2
            ;;
        --eval-gpu)
            require_value "$@"
            EVAL_GPU="$2"
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
if ! [[ "${TRAIN_GPU}" =~ ^[0-9]+$ && "${EVAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "--train-gpu and --eval-gpu must be non-negative integers."
    exit 1
fi
if [[ "${TRAIN_GPU}" == "${EVAL_GPU}" ]]; then
    echo "--train-gpu and --eval-gpu must select different CUDA devices."
    exit 1
fi
if ! [[ "${LABEL_SMOOTHING}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || ! awk -v factor="${LABEL_SMOOTHING}" 'BEGIN { exit !(factor < 1) }'; then
    echo "--label-smoothing must be a number in [0, 1)."
    exit 1
fi

case "${MODEL_KEY}" in
    qwen3-8b)
        MODEL_DIR="qwen3_8b"
        EVAL_TEMP=0.8
        EVAL_USE_TRANSFORMERS_BACKEND=false
        EVAL_TRUST_REMOTE_CODE=false
        ;;
    ministral3-8b)
        MODEL_DIR="ministral3_8b"
        EVAL_TEMP=0.6
        EVAL_USE_TRANSFORMERS_BACKEND=true
        EVAL_TRUST_REMOTE_CODE=true
        ;;
    mimo-7b)
        MODEL_DIR="mimo_7b"
        EVAL_TEMP=0.7
        EVAL_USE_TRANSFORMERS_BACKEND=false
        EVAL_TRUST_REMOTE_CODE=true
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
        ;;
    gsm8k)
        DATASET_DISPLAY="GSM8K"
        HAS_JUDGE=false
        ;;
    chemistryqa)
        DATASET_DISPLAY="ChemistryQA"
        HAS_JUDGE=true
        ;;
    formulareasoning)
        DATASET_DISPLAY="FormulaReasoning"
        HAS_JUDGE=true
        ;;
    *)
        echo "--dataset must be math, gsm8k, chemistryqa, or formulareasoning."
        exit 1
        ;;
esac

if [[ -z "${EPOCHS}" ]]; then
    EPOCHS=1
    if [[ "${MODEL_KEY}" == "ministral3-8b" && "${DATASET}" == "gsm8k" ]]; then
        EPOCHS=2.5
    fi
fi
if ! [[ "${EPOCHS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v epochs="${EPOCHS}" 'BEGIN { exit !(epochs > 0) }'; then
    echo "--epochs must be a positive number."
    exit 1
fi

CONFIG="configs/baseline/baseline_${DATASET}_${MODEL_KEY}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "Baseline config does not exist: ${CONFIG}"
    exit 1
fi

# Only an explicit --seed adds a suffix, so reruns with other seeds do not reuse this run's outputs.
SEED_SUFFIX=""
if [[ "${SEED_SET}" == "true" ]]; then
    SEED_SUFFIX="_seed-${SEED}"
fi
RUN_KEY="ls-${LABEL_SMOOTHING}_e-${EPOCHS}${SEED_SUFFIX}"
WORK_DIR="workdirs/${MODEL_DIR}/${DATASET}/label_smoothing/${RUN_KEY}"
TRAIN_DIR="${WORK_DIR}/train"
CHECKPOINT="${TRAIN_DIR}/lora_checkpoint"
RUN_NAME="${MODEL_DIR}_ls_${DATASET}_${RUN_KEY}"
ALL_GPUS="${TRAIN_GPU},${EVAL_GPU}"

CFG_OPTIONS=(
    "training.per_device_train_batch_size=4"
    "training.label_smoothing_factor=${LABEL_SMOOTHING}"
    "training.num_train_epochs=${EPOCHS}"
)

EVAL_OPTIONS=(
    "vllm_sampling.temperature=${EVAL_TEMP}"
    "vllm_sampling.seed=${SEED}"
    "vllm_engine.seed=${SEED}"
    "vllm_engine.tensor_parallel_size=2"
)
if [[ "${EVAL_USE_TRANSFORMERS_BACKEND}" == "true" ]]; then
    EVAL_OPTIONS+=("vllm_engine.model_impl=transformers")
fi
if [[ "${EVAL_TRUST_REMOTE_CODE}" == "true" ]]; then
    EVAL_OPTIONS+=("vllm_engine.trust_remote_code=true")
fi
if [[ "${HAS_JUDGE}" == "true" ]]; then
    # evaluate.py exports judge_engine.cuda_devices as CUDA_VISIBLE_DEVICES, so it takes the physical ID.
    EVAL_OPTIONS+=(
        "judge_engine.cuda_devices=${ALL_GPUS}"
        "judge_engine.seed=${SEED}"
        "judge_sampling.seed=${SEED}"
    )
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    cat <<EOF
MODEL_KEY=${MODEL_KEY}
MODEL_DIR=${MODEL_DIR}
DATASET=${DATASET}
CONFIG=${CONFIG}
SEED=${SEED}
LABEL_SMOOTHING=${LABEL_SMOOTHING}
EPOCHS=${EPOCHS}
TRAIN_GPU=${TRAIN_GPU}
EVAL_GPU=${EVAL_GPU}
WORK_DIR=${WORK_DIR}
CHECKPOINT=${CHECKPOINT}
RUN_NAME=${RUN_NAME}
HAS_JUDGE=${HAS_JUDGE}
EVAL_TEMP=${EVAL_TEMP}
EOF
    printf 'CFG_OPTIONS='
    printf ' %q' "${CFG_OPTIONS[@]}"
    printf '\n'
    printf 'EVAL_OPTIONS='
    printf ' %q' "${EVAL_OPTIONS[@]}"
    printf '\n'
    exit 0
fi

# Training
if [[ -d "${CHECKPOINT}" && -n "$(ls -A "${CHECKPOINT}" 2>/dev/null)" ]]; then
    echo "[$(date)] Checkpoint exists, skipping training: ${CHECKPOINT}"
else
    mkdir -p "${TRAIN_DIR}"
    echo "[$(date)] Starting ${MODEL_KEY} LS training on ${DATASET_DISPLAY}, seed ${SEED}..."
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python tools/baseline_train.py \
        "${CONFIG}" \
        --seed "${SEED}" \
        -w "${TRAIN_DIR}" \
        --run-name "${RUN_NAME}" \
        -o "${CFG_OPTIONS[@]}"
    echo "[$(date)] Training complete."
fi

if [[ ! -d "${CHECKPOINT}" || -z "$(ls -A "${CHECKPOINT}" 2>/dev/null)" ]]; then
    echo "[$(date)] ERROR: missing LS checkpoint: ${CHECKPOINT}"
    exit 1
fi

# Evaluation runs generation and then the judge with TP=2 on both GPUs.
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

echo "[$(date)] LS experiment complete: ${RUN_NAME}"
