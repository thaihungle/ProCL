#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

METHOD=${1:?method: seq_lora|procl|deal}
BACKBONE=${2:?backbone: t5|decoder}
MODEL=${3:?model name or path}
SEED=${4:-42}
START_TASK=${START_TASK:-1}
SEQUENCE_ROOT=${SEQUENCE_ROOT:-${SCRIPT_DIR}/outputs/${BACKBONE}/base-${MODEL//\//_}/seed${SEED}}
TASK1_ROOT=${TASK1_ROOT:-${SEQUENCE_ROOT}/1-boolq}
METHOD_ROOT=${METHOD_ROOT:-${SEQUENCE_ROOT}/${METHOD}}

if [[ "${START_TASK}" -lt 1 || "${START_TASK}" -gt 3 ]]; then
  echo "START_TASK must be 1, 2, or 3." >&2
  exit 1
fi

DEFAULT_TRAIN_BS=8
DEFAULT_EVAL_BS=8
DEFAULT_GRAD_ACC=2
DEFAULT_LR=1e-5
DEFAULT_EPOCHS=1

if [[ "${BACKBONE}" == "decoder" && "${MODEL,,}" == *"llama-3.2-3b"* ]]; then
  DEFAULT_LR=2e-5
  DEFAULT_EPOCHS=2
fi

COMMON_ARGS=(
  --backbone "${BACKBONE}"
  --method "${METHOD}"
  --do_train
  --do_predict
  --seed "${SEED}"
  --learning_rate "${LR:-${DEFAULT_LR}}"
  --num_train_epochs "${EPOCHS:-${DEFAULT_EPOCHS}}"
  --per_device_train_batch_size "${TRAIN_BS:-${DEFAULT_TRAIN_BS}}"
  --per_device_eval_batch_size "${EVAL_BS:-${DEFAULT_EVAL_BS}}"
  --gradient_accumulation_steps "${GRAD_ACC:-${DEFAULT_GRAD_ACC}}"
  --max_source_length "${MAX_SOURCE:-512}"
  --max_target_length "${MAX_TARGET:-64}"
  --max_length "${MAX_LENGTH:-512}"
  --max_new_tokens "${MAX_NEW:-32}"
  --lora_dim "${LORA_DIM:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.1}"
  --N "${N:-4}"
  --update_period "${UPDATE_PERIOD:-1}"
  --d_key "${D_KEY:-16}"
  --lambda_consolidation "${LAMBDA_CONSOLIDATION:-0.9}"
  --gamma "${GAMMA:--1}"
)

if [[ "${BF16:-0}" == "1" ]]; then
  COMMON_ARGS+=(--bf16)
fi
if [[ "${FP16:-0}" == "1" ]]; then
  COMMON_ARGS+=(--fp16)
fi

if [[ "${START_TASK}" -le 1 ]]; then
  python -u "${SCRIPT_DIR}/run_qa.py" \
    --model_name_or_path "${MODEL}" \
    --task_config_dir "${SCRIPT_DIR}/configs/QA_configs/boolq" \
    --output_dir "${TASK1_ROOT}" \
    "${COMMON_ARGS[@]}"
fi

if [[ "${START_TASK}" -le 2 ]]; then
  TASK1_ADAPTER="${TASK1_ROOT}/adapter"
  if [[ ! -f "${TASK1_ADAPTER}/adapter_config.json" ]]; then
    echo "Missing adapter for START_TASK=${START_TASK}: ${TASK1_ADAPTER}" >&2
    echo "Run with START_TASK=1 first, or set TASK1_ROOT to an existing task-1 output." >&2
    exit 1
  fi
  python -u "${SCRIPT_DIR}/run_qa.py" \
    --model_name_or_path "${TASK1_ADAPTER}" \
    --task_config_dir "${SCRIPT_DIR}/configs/QA_configs/squad" \
    --output_dir "${METHOD_ROOT}/2-squad" \
    --adapter_out_name adapter \
    "${COMMON_ARGS[@]}"
fi

if [[ "${START_TASK}" -le 3 ]]; then
  TASK2_ADAPTER="${METHOD_ROOT}/2-squad/adapter"
  if [[ ! -f "${TASK2_ADAPTER}/adapter_config.json" ]]; then
    echo "Missing adapter for START_TASK=${START_TASK}: ${TASK2_ADAPTER}" >&2
    echo "Run with START_TASK=1 or START_TASK=2 first, or set METHOD_ROOT to an existing method output." >&2
    exit 1
  fi
  python -u "${SCRIPT_DIR}/run_qa.py" \
    --model_name_or_path "${TASK2_ADAPTER}" \
    --task_config_dir "${SCRIPT_DIR}/configs/QA_configs/adversarial_qa" \
    --output_dir "${METHOD_ROOT}/3-adversarial_qa" \
    --adapter_out_name adapter \
    "${COMMON_ARGS[@]}"
fi
