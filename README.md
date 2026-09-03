# ProCL Paper Reproduction

This folder contains reproduction code for the paper:

**Continual Fine-Tuning of Large Language Models via Program Memory**  
https://arxiv.org/abs/2605.13162

The current implementation supports the QA continual-learning sequence:

```text
BoolQ -> SQuAD -> AdversarialQA
```


## Setup

```bash
cd ProCL
pip install -r requirements.txt
```

The code downloads Hugging Face datasets and loads the selected base model at
runtime. Make sure you have access to gated models such as LLaMA before running
those experiments.

## Run Locally

Default ProCL run:

```bash
bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

Run the baselines:

```bash
bash run_sequence.sh seq_lora decoder Qwen/Qwen3-4B 42
bash run_sequence.sh deal decoder Qwen/Qwen3-4B 42
```

Run LLaMA backbone:

```bash
bash run_sequence.sh procl decoder meta-llama/Llama-3.2-3B-Instruct 42
```

Run FLAN-T5:

```bash
bash run_sequence.sh procl t5 google/flan-t5-base 42
```

## Run All Seeds

```bash
for SEED in 42 420 4200; do
  bash run_sequence.sh procl decoder Qwen/Qwen3-4B "$SEED"
done
```

## Resume

Resume from task 2 or task 3 after earlier adapters have been saved:

```bash
START_TASK=2 bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
START_TASK=3 bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

## SLURM

Submit the default sequence from this folder:

```bash
sbatch run_sequence_slurm.sh
```

Or submit from the parent directory:

```bash
METHOD=procl BACKBONE=decoder MODEL=Qwen/Qwen3-4B SEED=42 bash ProCL/submit_slurm.sh
```

## Defaults

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

Any default can be overridden with an environment variable:

```bash
LR=2e-5 EPOCHS=2 bash run_sequence.sh procl decoder Qwen/Qwen3-4B 42
```

## Outputs

Results are written under:

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

Each task folder contains an adapter and an evaluation JSON:

```text
adapter/
eval_results_<seed>.json
```

## TODO

- Add the remaining paper task families beyond QA.
- Extend to other QA tasks
