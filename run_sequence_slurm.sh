#!/bin/bash
#SBATCH --job-name=procl_qa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
if [[ ! -f "${SUBMIT_DIR}/run_sequence.sh" || ! -f "${SUBMIT_DIR}/run_qa.py" ]]; then
  echo "Submit with ProCL as the working directory so all paths stay local to this folder." >&2
  echo "Use either:" >&2
  echo "  cd ProCL && sbatch run_sequence_slurm.sh" >&2
  echo "or:" >&2
  echo "  bash ProCL/submit_slurm.sh" >&2
  exit 1
fi
SCRIPT_DIR="${SUBMIT_DIR}"
cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/outputs" "${SCRIPT_DIR}/.cache" "${SCRIPT_DIR}/tmp"

METHOD=${METHOD:-procl}
BACKBONE=${BACKBONE:-decoder}
MODEL=${MODEL:-Qwen/Qwen3-4B}
SEED=${SEED:-42}
export START_TASK=${START_TASK:-1}

if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python - <<'PY'
try:
    from packaging import version  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Broken Python dependency: cannot import 'packaging.version'.\n"
        "Repair this environment with:\n"
        "  python -m pip install --force-reinstall --no-cache-dir packaging\n"
        f"Original error: {exc}"
    )
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${SCRIPT_DIR}/.cache/huggingface/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${SCRIPT_DIR}/.cache/huggingface/transformers}
export TMPDIR=${TMPDIR:-${SCRIPT_DIR}/tmp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SEQUENCE_ROOT=${SEQUENCE_ROOT:-${SCRIPT_DIR}/outputs/${BACKBONE}/base-${MODEL//\//_}/seed${SEED}}
export TASK1_ROOT=${TASK1_ROOT:-${SEQUENCE_ROOT}/1-boolq}
export METHOD_ROOT=${METHOD_ROOT:-${SEQUENCE_ROOT}/${METHOD}}

if [[ -z "${BF16:-}" && -z "${FP16:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  if [[ "${GPU_CC}" -le 70 ]] 2>/dev/null; then
    export FP16=1
  else
    export BF16=1
  fi
fi

export TRAIN_BS=${TRAIN_BS:-8}
export EVAL_BS=${EVAL_BS:-8}
export GRAD_ACC=${GRAD_ACC:-2}
DEFAULT_EPOCHS=1
DEFAULT_LR=1e-5
if [[ "${BACKBONE}" == "decoder" && "${MODEL,,}" == *"llama-3.2-3b"* ]]; then
  DEFAULT_EPOCHS=2
  DEFAULT_LR=2e-5
fi
export EPOCHS=${EPOCHS:-${DEFAULT_EPOCHS}}
export LR=${LR:-${DEFAULT_LR}}
export MAX_LENGTH=${MAX_LENGTH:-512}
export MAX_SOURCE=${MAX_SOURCE:-512}
export MAX_TARGET=${MAX_TARGET:-64}
export MAX_NEW=${MAX_NEW:-32}
export LORA_DIM=${LORA_DIM:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.1}
export N=${N:-4}
export UPDATE_PERIOD=${UPDATE_PERIOD:-1}
export D_KEY=${D_KEY:-16}
export LAMBDA_CONSOLIDATION=${LAMBDA_CONSOLIDATION:-0.9}
export GAMMA=${GAMMA:--1}

echo "method=${METHOD}"
echo "backbone=${BACKBONE}"
echo "model=${MODEL}"
echo "seed=${SEED}"
echo "start_task=${START_TASK}"
echo "sequence_root=${SEQUENCE_ROOT}"
echo "task1_root=${TASK1_ROOT}"
echo "method_root=${METHOD_ROOT}"
echo "train_bs=${TRAIN_BS} eval_bs=${EVAL_BS} grad_acc=${GRAD_ACC}"
echo "bf16=${BF16:-0} fp16=${FP16:-0}"
echo "N=${N} d_key=${D_KEY} lambda_consolidation=${LAMBDA_CONSOLIDATION} gamma=${GAMMA}"

bash "${SCRIPT_DIR}/run_sequence.sh" "${METHOD}" "${BACKBONE}" "${MODEL}" "${SEED}"
