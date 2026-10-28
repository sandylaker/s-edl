#!/bin/bash
# NewEDL training and evaluation on one training GPU plus one vLLM/judge GPU.
set -e

MODEL_KEY="ministral3-8b"
DATASET="math"
EDL_BETA=1.2
# Note: unlike the old EDL which uses edl_weight=5.0,
# here we scale the reward (i.e. negative gradient w.r.t. log-likelihood of a sample) by 10.0.
SCALE=10.0
TRAIN_TEMP=""
LOSS_TYPE="grpo"
VLLM_IS=false
VLLM_IS_MODE="sequence_mask"
AUX_GT=false
AUX_GT_LOSS_WEIGHT=1.0
AUX_GT_NORMALIZATION="row_matched"
MISSING_GT_STRATEGY=""
PSEUDO_GT_LOGPROB_PENALTY=4.605
SFT_CHECKPOINT_OVERRIDE=""
TRAIN_GPU=0
VLLM_GPU=1
VLLM_PORT=8000
VLLM_GROUP_PORT=51216
MAX_MODEL_LEN=""
VLLM_GPU_MEMORY_UTILIZATION=""
EVAL_TEMP=""
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: scripts/new_edl.sh [options]

Model and dataset:
  --model NAME                   Model shortcut: qwen3-8b, ministral3-8b, or mimo-7b.
                                 Default: ministral3-8b
  --dataset NAME                 Dataset: math, gsm8k, chemistryqa, or formulareasoning. Default: math

NewEDL settings:
  --scale VALUE                  EDL reward scale. Default: 10.0
  --edl-beta VALUE               EDL beta. Default: 1.2
  --train-temp VALUE             Override rollout temperature. Empty by default.
  --loss-type VALUE              GRPO loss reduction. Default: grpo
  --vllm-is true|false           Enable TRL vLLM importance-sampling correction. Default: false
  --vllm-is-mode VALUE           sequence_mask, sequence_truncate, token_mask, or token_truncate.
                                 Default: sequence_mask (inactive while --vllm-is=false)
  --aux-gt true|false            Enable auxiliary teacher-forced GT loss. Default: false
  --aux-gt-loss-weight VALUE     Auxiliary GT loss multiplier. Default: 1.0
  --aux-gt-normalization VALUE   row_matched: scale by active-row fraction to match sampled-row weighting;
                                 active_mean: average only active auxiliary rows (legacy). Default: row_matched
  --missing-gt-strategy VALUE    pseudo or skip. Default: training.edl_missing_gt_strategy in the config
  --pseudo-gt-penalty VALUE      Penalty on pseudo-GT mean logprob. Default: 4.605
  --sft-checkpoint PATH          SFT LoRA checkpoint to continue from, e.g. one trained by
                                 scripts/baseline_sft.sh --epochs 1. Default:
                                 workdirs/<model>/<dataset>/baseline/e-1/train/lora_checkpoint, or
                                 workdirs/qwen3_8b/gsm8k/baseline/cot/e-1/train/lora_checkpoint
                                 for Qwen3-8B on GSM8K.

Runtime:
  --train-gpu ID                 CUDA device for training. Default: 0
  --vllm-gpu ID                  CUDA device for vLLM and the non-math judge. Default: 1
  --vllm-port PORT               vLLM HTTP server port. Default: 8000
  --vllm-group-port PORT         TRL weight-sync group port. Default: 51216
  --vllm-gpu-memory VALUE        Override vLLM GPU memory utilization. Defaults: 0.85 for MATH,
                                 0.45 when sharing with the non-math cluster judge.
  --max-model-len VALUE          Override vLLM server maximum model length. Defaults: 2048 for GSM8K,
                                 4096 for other datasets.
  --eval-temp VALUE              Override the model-specific evaluation temperature.
  --dry-run                      Print resolved paths/settings without training or evaluation.
  -h, --help                     Show this help.

Best-known NewEDL recipe (the defaults):
  beta=1.2, scale=10.0, loss_type=grpo, vLLM IS off, aux-GT off, missing-GT from the config.

Examples:
  scripts/new_edl.sh --model ministral3-8b --dataset chemistryqa
  scripts/new_edl.sh --model mimo-7b --dataset math
  scripts/new_edl.sh --model ministral3-8b --dataset gsm8k
  scripts/new_edl.sh --model mimo-7b --dataset gsm8k
  scripts/new_edl.sh --model qwen3-8b --dataset gsm8k \
      --sft-checkpoint workdirs/qwen3_8b/gsm8k/baseline/e-1/train/lora_checkpoint
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_KEY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --edl-beta)
            EDL_BETA="$2"
            shift 2
            ;;
        --train-temp)
            TRAIN_TEMP="$2"
            shift 2
            ;;
        --loss-type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --vllm-is)
            VLLM_IS="$2"
            shift 2
            ;;
        --vllm-is-mode)
            VLLM_IS_MODE="$2"
            shift 2
            ;;
        --aux-gt)
            AUX_GT="$2"
            shift 2
            ;;
        --aux-gt-loss-weight)
            AUX_GT_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --aux-gt-normalization)
            AUX_GT_NORMALIZATION="$2"
            shift 2
            ;;
        --missing-gt-strategy)
            MISSING_GT_STRATEGY="$2"
            shift 2
            ;;
        --pseudo-gt-penalty)
            PSEUDO_GT_LOGPROB_PENALTY="$2"
            shift 2
            ;;
        --sft-checkpoint)
            SFT_CHECKPOINT_OVERRIDE="$2"
            shift 2
            ;;
        --train-gpu)
            TRAIN_GPU="$2"
            shift 2
            ;;
        --vllm-gpu)
            VLLM_GPU="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --vllm-group-port)
            VLLM_GROUP_PORT="$2"
            shift 2
            ;;
        --vllm-gpu-memory)
            VLLM_GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --eval-temp)
            EVAL_TEMP="$2"
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

for BOOL_VALUE in "${VLLM_IS}" "${AUX_GT}"; do
    if [[ "${BOOL_VALUE}" != "true" && "${BOOL_VALUE}" != "false" ]]; then
        echo "--vllm-is and --aux-gt must be true or false."
        exit 1
    fi
done
if [[ "${AUX_GT_NORMALIZATION}" != "row_matched" && "${AUX_GT_NORMALIZATION}" != "active_mean" ]]; then
    echo "--aux-gt-normalization must be row_matched or active_mean."
    exit 1
fi
case "${VLLM_IS_MODE}" in
    sequence_mask|sequence_truncate|token_mask|token_truncate)
        ;;
    *)
        echo "--vllm-is-mode must be sequence_mask, sequence_truncate, token_mask, or token_truncate."
        exit 1
        ;;
esac

case "${MODEL_KEY}" in
    qwen3-8b)
        MODEL="Qwen/Qwen3-8B-Base"
        MODEL_DIR="qwen3_8b"
        USE_TRANSFORMERS_BACKEND=false
        TRUST_REMOTE_CODE=false
        DEFAULT_EVAL_TEMP=0.8
        ;;
    ministral3-8b)
        MODEL="Aratako/Ministral-3-8B-Base-2512-TextOnly"
        MODEL_DIR="ministral3_8b"
        USE_TRANSFORMERS_BACKEND=true
        TRUST_REMOTE_CODE=true
        DEFAULT_EVAL_TEMP=0.6
        ;;
    mimo-7b)
        MODEL="XiaomiMiMo/MiMo-7B-Base"
        MODEL_DIR="mimo_7b"
        USE_TRANSFORMERS_BACKEND=false
        TRUST_REMOTE_CODE=true
        DEFAULT_EVAL_TEMP=0.7
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
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.85
        DEFAULT_MAX_MODEL_LEN=4096
        ;;
    gsm8k)
        DATASET_DISPLAY="GSM8K"
        HAS_JUDGE=false
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.85
        DEFAULT_MAX_MODEL_LEN=2048
        ;;
    chemistryqa)
        DATASET_DISPLAY="ChemistryQA"
        HAS_JUDGE=true
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.45
        DEFAULT_MAX_MODEL_LEN=4096
        ;;
    formulareasoning)
        DATASET_DISPLAY="FormulaReasoning"
        HAS_JUDGE=true
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.45
        DEFAULT_MAX_MODEL_LEN=4096
        ;;
    *)
        echo "--dataset must be math, gsm8k, chemistryqa, or formulareasoning."
        exit 1
        ;;
esac

if [ -z "${VLLM_GPU_MEMORY_UTILIZATION}" ]; then
    VLLM_GPU_MEMORY_UTILIZATION="${DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}"
fi
if [ -z "${EVAL_TEMP}" ]; then
    EVAL_TEMP="${DEFAULT_EVAL_TEMP}"
fi
if [ -z "${MAX_MODEL_LEN}" ]; then
    MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
fi

if [ "${VLLM_IS}" = "true" ]; then
    VLLM_IS_LABEL="${VLLM_IS_MODE}"
else
    VLLM_IS_LABEL="off"
fi
if [ "${AUX_GT}" = "true" ]; then
    AUX_GT_LABEL="on-w-${AUX_GT_LOSS_WEIGHT}-${AUX_GT_NORMALIZATION}"
else
    AUX_GT_LABEL="off"
fi

CONFIG="configs/new_edl/new_edl_${DATASET}_${MODEL_KEY}.yaml"
if [ ! -f "${CONFIG}" ]; then
    echo "NewEDL config does not exist: ${CONFIG}"
    exit 1
fi

if [ -z "${MISSING_GT_STRATEGY}" ]; then
    MISSING_GT_STRATEGY=$(python -c \
        'import sys; from mmengine import Config; print(Config.fromfile(sys.argv[1]).training.edl_missing_gt_strategy)' \
        "${CONFIG}")
fi
if [[ "${MISSING_GT_STRATEGY}" != "pseudo" && "${MISSING_GT_STRATEGY}" != "skip" ]]; then
    echo "--missing-gt-strategy must be pseudo or skip."
    exit 1
fi

TRAIN_TEMP_SUFFIX=""
if [ -n "${TRAIN_TEMP}" ]; then
    TRAIN_TEMP_SUFFIX="_t-${TRAIN_TEMP}"
fi
ABLATION_SUFFIX="_${LOSS_TYPE}-loss"
ABLATION_SUFFIX+="_vllm-is-${VLLM_IS_LABEL}"
ABLATION_SUFFIX+="_aux-gt-${AUX_GT_LABEL}"
ABLATION_SUFFIX+="_missing-gt-${MISSING_GT_STRATEGY}"
if [ "${MISSING_GT_STRATEGY}" = "pseudo" ] || [ "${AUX_GT}" = "true" ]; then
    ABLATION_SUFFIX+="_pseudo-penalty-${PSEUDO_GT_LOGPROB_PENALTY}"
fi

if [ -n "${SFT_CHECKPOINT_OVERRIDE}" ]; then
    SFT_CHECKPOINT="${SFT_CHECKPOINT_OVERRIDE}"
    RUN_PREFIX="from_sft"
else
    case "${MODEL_KEY}:${DATASET}" in
        qwen3-8b:gsm8k)
            SFT_CHECKPOINT="workdirs/qwen3_8b/gsm8k/baseline/cot/e-1/train/lora_checkpoint"
            ;;
        *)
            SFT_CHECKPOINT="workdirs/${MODEL_DIR}/${DATASET}/baseline/e-1/train/lora_checkpoint"
            ;;
    esac
    RUN_PREFIX="from_e1"
fi
RUN_KEY="${RUN_PREFIX}_b-${EDL_BETA}_s-${SCALE}${TRAIN_TEMP_SUFFIX}${ABLATION_SUFFIX}"
WORK_DIR="workdirs/${MODEL_DIR}/${DATASET}/new_edl/${RUN_KEY}"
TRAIN_DIR="${WORK_DIR}/train"
CHECKPOINT="${TRAIN_DIR}/lora_checkpoint"
RUN_NAME="${MODEL_DIR}_new_edl_${DATASET}_${RUN_KEY}"

CFG_OPTIONS=(
    "training.edl_beta=${EDL_BETA}"
    "training.edl_reward_scale=${SCALE}"
    "training.loss_type=${LOSS_TYPE}"
    "training.scale_rewards=none"
    "training.vllm_importance_sampling_correction=${VLLM_IS}"
    "training.vllm_importance_sampling_mode=${VLLM_IS_MODE}"
    "training.edl_aux_gt_enabled=${AUX_GT}"
    "training.edl_aux_gt_loss_weight=${AUX_GT_LOSS_WEIGHT}"
    "training.edl_aux_gt_normalization=${AUX_GT_NORMALIZATION}"
    "training.edl_missing_gt_strategy=${MISSING_GT_STRATEGY}"
    "training.edl_gt_pseudo_logprob_penalty=${PSEUDO_GT_LOGPROB_PENALTY}"
    "training.vllm_server_port=${VLLM_PORT}"
    "training.vllm_group_port=${VLLM_GROUP_PORT}"
)
if [ -n "${TRAIN_TEMP}" ]; then
    CFG_OPTIONS+=("training.temperature=${TRAIN_TEMP}")
fi
if [ "${HAS_JUDGE}" = "true" ]; then
    CFG_OPTIONS+=(
        "judge_engine.cuda_devices=${VLLM_GPU}"
        "judge_engine.gpu_memory_utilization=0.4"
    )
fi

if [ "${DRY_RUN}" = "true" ]; then
    cat <<EOF
MODEL_KEY=${MODEL_KEY}
MODEL=${MODEL}
MODEL_DIR=${MODEL_DIR}
DATASET=${DATASET}
CONFIG=${CONFIG}
SFT_CHECKPOINT=${SFT_CHECKPOINT}
WORK_DIR=${WORK_DIR}
RUN_NAME=${RUN_NAME}
USE_TRANSFORMERS_BACKEND=${USE_TRANSFORMERS_BACKEND}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
HAS_JUDGE=${HAS_JUDGE}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
EVAL_TEMP=${EVAL_TEMP}
EOF
    printf 'CFG_OPTIONS='
    printf ' %q' "${CFG_OPTIONS[@]}"
    printf '\n'
    exit 0
fi

if [ ! -f "${SFT_CHECKPOINT}/adapter_config.json" ]; then
    echo "ERROR: SFT checkpoint is missing adapter_config.json: ${SFT_CHECKPOINT}"
    echo "Train one with scripts/baseline_sft.sh --epochs 1 and pass it with --sft-checkpoint."
    exit 1
fi
if [ ! -f "${SFT_CHECKPOINT}/adapter_model.safetensors" ] && \
   [ ! -f "${SFT_CHECKPOINT}/adapter_model.bin" ]; then
    echo "ERROR: SFT checkpoint is missing adapter weights: ${SFT_CHECKPOINT}"
    exit 1
fi

VLLM_PID=""
cleanup_vllm() {
    if [ -z "${VLLM_PID}" ]; then
        return
    fi
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        PGID=$(ps -o pgid= -p "${VLLM_PID}" | tr -d ' ')
        if [ -n "${PGID}" ]; then
            kill -9 -"${PGID}" 2>/dev/null || true
        fi
    fi
    lsof -ti:"${VLLM_PORT}" | xargs kill -9 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
}
trap cleanup_vllm EXIT INT TERM

# Training
if [ -d "${CHECKPOINT}" ] && [ -n "$(ls -A "${CHECKPOINT}" 2>/dev/null)" ]; then
    echo "[$(date)] Checkpoint exists, skipping training: ${CHECKPOINT}"
else
    if lsof -ti:"${VLLM_PORT}" > /dev/null 2>&1; then
        echo "[$(date)] Killing leftover process on port ${VLLM_PORT}..."
        lsof -ti:"${VLLM_PORT}" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    VLLM_ARGS=(
        trl vllm-serve
        --model "${MODEL}"
        --port "${VLLM_PORT}"
        --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --max_model_len "${MAX_MODEL_LEN}"
    )
    if [ "${USE_TRANSFORMERS_BACKEND}" = "true" ]; then
        VLLM_ARGS+=(--vllm_model_impl transformers)
    fi
    if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
        VLLM_ARGS+=(--trust_remote_code)
    fi

    VLLM_LOG="${TRAIN_DIR}/vllm_server.log"
    mkdir -p "${TRAIN_DIR}"
    echo "[$(date)] Starting ${MODEL_KEY} vLLM server on GPU ${VLLM_GPU}, port ${VLLM_PORT} (log: ${VLLM_LOG})..."
    setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_ARGS[@]}" > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    echo "[$(date)] Waiting for vLLM server to start..."
    READY=0
    for _ in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "[$(date)] vLLM server is ready."
            READY=1
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[$(date)] ERROR: vLLM server process died. See ${VLLM_LOG}."
            exit 1
        fi
        sleep 5
    done
    if [ "${READY}" -ne 1 ]; then
        echo "[$(date)] ERROR: vLLM server failed to become ready. See ${VLLM_LOG}."
        exit 1
    fi

    echo "[$(date)] Starting ${MODEL_KEY} NewEDL training on ${DATASET_DISPLAY}, GPU ${TRAIN_GPU}..."
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python tools/new_edl_train.py \
        "${CONFIG}" \
        -w "${TRAIN_DIR}" \
        --run-name "${RUN_NAME}" \
        --checkpoint "${SFT_CHECKPOINT}" \
        -o "${CFG_OPTIONS[@]}"

    echo "[$(date)] Training complete. Stopping vLLM server..."
    cleanup_vllm
    sleep 3
    echo "[$(date)] vLLM server stopped."
fi

# Evaluation runs generation and then the judge with TP=2 on the training and vLLM GPUs.
EVAL_DIR="${WORK_DIR}/eval/temp-${EVAL_TEMP}"
METRICS_FILE="${EVAL_DIR}/metrics_summary.json"
if [ -f "${METRICS_FILE}" ]; then
    echo "[$(date)] Metrics exist, skipping: ${METRICS_FILE}"
else
    EVAL_OPTIONS=(
        "vllm_sampling.temperature=${EVAL_TEMP}"
        "vllm_engine.tensor_parallel_size=2"
    )
    if [ "${USE_TRANSFORMERS_BACKEND}" = "true" ]; then
        EVAL_OPTIONS+=("vllm_engine.model_impl=transformers")
    fi
    if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
        EVAL_OPTIONS+=("vllm_engine.trust_remote_code=true")
    fi
    if [ "${HAS_JUDGE}" = "true" ]; then
        EVAL_OPTIONS+=(
            "judge_engine.cuda_devices=${TRAIN_GPU},${VLLM_GPU}"
            "judge_engine.gpu_memory_utilization=0.85"
        )
    fi
    echo "[$(date)] Evaluating ${DATASET_DISPLAY} at temperature=${EVAL_TEMP}..."
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU},${VLLM_GPU}" python tools/evaluate.py \
        "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        -w "${EVAL_DIR}" \
        -o "${EVAL_OPTIONS[@]}"
fi

echo "[$(date)] NewEDL experiment complete: ${RUN_NAME}"
