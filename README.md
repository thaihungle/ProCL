# 🚀 ProCL Paper Code

Source code for:

**Continual Fine-Tuning of Large Language Models via Program Memory**  
https://arxiv.org/abs/2605.13162

This repository focuses on reproducing the paper's continual fine-tuning method,
**ProCL**, together with two baselines: `seq_lora` and `deal`.

## ✅ Current Support

The current implementation supports the QA continual-learning sequence:

```text
BoolQ -> SQuAD -> AdversarialQA
```

More task families from the paper are planned in the TODO section.

## 📦 Setup

Create and activate a conda environment:

```bash
conda create -n procl python=3.10 -y
conda activate procl
```

Install dependencies and create the output folder:

```bash
cd ProCL
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
mkdir -p outputs
```

Install PyTorch first with a CUDA wheel supported by your cluster driver, then
install the remaining packages. If your driver is older or newer than CUDA 12.4,
choose the matching PyTorch wheel index from the official PyTorch install page.

For DEAL runs, verify the wavelet dependency:

```bash
python -c "from pytorch_wavelets import DWTForward, DWTInverse; print('pytorch-wavelets ok')"
```

If this fails with `No module named 'pkg_resources'`, downgrade `setuptools`.
Recent `setuptools` releases removed `pkg_resources`, which `pytorch-wavelets`
still imports:

```bash
python -m pip install "setuptools==80.9.0" --force-reinstall
python -m pip install PyWavelets pytorch-wavelets --force-reinstall
```

The `outputs/` folder is required because trained adapters and evaluation JSON
files are written there. 

The code downloads Hugging Face datasets and loads the selected base model at
runtime. Make sure you have access to gated models such as LLaMA before running
those experiments.

## ⚡ Quick Start

Run ProCL with the default QA setting:

```bash
bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

This runs the full sequence:

```text
Task 1: BoolQ
Task 2: SQuAD
Task 3: AdversarialQA
```

## 🧪 Baselines

Run sequential LoRA:

```bash
bash run_sequence.sh seq_lora decoder Qwen/Qwen3-4B 42
```

Run DEAL:

```bash
bash run_sequence.sh deal decoder Qwen/Qwen3-4B 42
```

## 🧠 Backbones

Run LLaMA:

```bash
bash run_sequence.sh procl decoder meta-llama/Llama-3.2-3B-Instruct 42
```

Run FLAN-T5:

```bash
bash run_sequence.sh procl t5 google/flan-t5-base 42
```

### Tested LLMs

| Backbone | Model | Command `BACKBONE` |
|---|---|---|
| Qwen3 4B | `Qwen/Qwen3-4B` | `decoder` |
| Qwen3 8B | `Qwen/Qwen3-8B` | `decoder` |
| LLaMA 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `decoder` |
| LLaMA 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `decoder` |
| FLAN-T5 Base | `google/flan-t5-base` | `t5` |
| FLAN-T5 Large | `google/flan-t5-large` | `t5` |

## 🎲 Run Three Seeds

```bash
for SEED in 42 420 4200; do
  bash run_sequence.sh procl decoder Qwen/Qwen3-4B "$SEED"
done
```

## ⏩ Resume

Resume from task 2:

```bash
START_TASK=2 bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

Resume from task 3:

```bash
START_TASK=3 bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

## ⚙️ Defaults

Supported methods:

```text
procl
seq_lora
deal
```

Default hyperparameters:

```text
LR=1e-5
EPOCHS=1
TRAIN_BS=8
EVAL_BS=8
GRAD_ACC=2
LORA_DIM=16
LORA_ALPHA=32
N=4
D_KEY=16
LAMBDA_CONSOLIDATION=0.9
GAMMA=-1
```

For `meta-llama/Llama-3.2-3B-*` decoder runs, the scripts use the tuned default:

```text
LR=2e-5
EPOCHS=2
```

Override any default with an environment variable:

```bash
LR=2e-5 EPOCHS=2 bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

## 📁 Outputs

Results are saved under:

```text
outputs/<backbone>/base-<model>/seed<seed>/
```

Task 1 is shared across methods:

```text
outputs/.../seed42/1-boolq
```

Tasks 2 and 3 are method-specific:

```text
outputs/.../seed42/procl/2-squad
outputs/.../seed42/procl/3-adversarial_qa
```

Each task folder contains:

```text
adapter/
eval_results_<seed>.json
```

## 📝 TODO

- Add the remaining paper task families beyond QA.
- Extend to additional QA benchmarks.
- Add checked configs for each reported backbone.
- Add a result aggregation script for paper-style tables.
