#!/usr/bin/env python
import argparse
import json
import math
import os
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = "" if text is None else text.strip().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def exact_match(preds: List[str], refs: List[str]) -> float:
    return sum(normalize_text(p) == normalize_text(r) for p, r in zip(preds, refs)) / max(len(preds), 1)


def contains_match(preds: List[str], refs: List[str]) -> float:
    return sum(normalize_text(r) in normalize_text(p) for p, r in zip(preds, refs)) / max(len(preds), 1)


def boolq_pred_to_bool(text: str) -> str:
    text = text.strip().lower()
    if text.startswith(("yes", "true")):
        return "True"
    if text.startswith(("no", "false")):
        return "False"
    return text


def load_task_config(config_dir: str, split: str) -> List[Tuple[str, str, Optional[int]]]:
    with open(Path(config_dir) / f"{split}_tasks.json", encoding="utf-8") as f:
        cfg = json.load(f)
    tasks = []
    for group in cfg.values():
        for item in group:
            tasks.append((item["dataset name"], item.get("sampling strategy", "full"), item.get("num_samples")))
    return tasks


def load_qa_dataset(dataset_name: str, split: str, strategy: str, num_samples: Optional[int]):
    if dataset_name == "adversarial_qa":
        ds = load_dataset("adversarial_qa", "adversarialQA", split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    if strategy == "subset" and num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    return ds


def answer_to_text(answer, dataset_name: str) -> str:
    if dataset_name == "google/boolq":
        return "True" if bool(answer) else "False"
    if isinstance(answer, dict):
        values = answer.get("text", [])
        return str(values[0]) if values else ""
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer)


def normalize_records(ds, dataset_name: str) -> List[Dict[str, str]]:
    if dataset_name == "google/boolq":
        q_col, c_col, a_col = "question", "passage", "answer"
    else:
        q_col, c_col, a_col = "question", "context", "answers"
    return [
        {
            "dataset_name": dataset_name,
            "question": str(ex.get(q_col, "")),
            "context": str(ex.get(c_col, "")),
            "answer": answer_to_text(ex.get(a_col, ""), dataset_name),
        }
        for ex in ds
    ]


def preprocess_seq2seq(examples, tokenizer, max_source_len: int, max_target_len: int):
    prompts = [f"Question: {q}\nContext: {c}\nAnswer:" for q, c in zip(examples["question"], examples["context"])]
    targets = examples["answer"]
    inputs = tokenizer(prompts, max_length=max_source_len, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_len, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    inputs["ref_answer"] = list(targets)
    return inputs


def preprocess_decoder(examples, tokenizer, max_length: int):
    input_ids, attention_mask, labels = [], [], []
    prompt_ids, prompt_masks, refs = [], [], []
    eos = tokenizer.eos_token or ""
    for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
        prompt = f"Question: {q}\nContext: {c}\nAnswer:"
        full = f"{prompt} {a}{eos}"
        full_tok = tokenizer(full, max_length=max_length, truncation=True, padding="max_length")
        prompt_tok = tokenizer(prompt, max_length=max_length, truncation=True, padding="max_length")
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        y = full_tok["input_ids"][:]
        for i in range(min(prompt_len, len(y))):
            y[i] = -100
        for i, mask in enumerate(full_tok["attention_mask"]):
            if mask == 0:
                y[i] = -100
        input_ids.append(full_tok["input_ids"])
        attention_mask.append(full_tok["attention_mask"])
        labels.append(y)
        prompt_ids.append(prompt_tok["input_ids"])
        prompt_masks.append(prompt_tok["attention_mask"])
        refs.append(a)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_ids_prompt": prompt_ids,
        "attention_mask_prompt": prompt_masks,
        "ref_answer": refs,
    }


def normalize_method(method: str) -> str:
    if method not in {"seq_lora", "procl", "deal"}:
        raise ValueError(f"Unsupported method: {method}")
    return method


def detect_backbone(backbone: str, model_name_or_path: str) -> str:
    if backbone != "auto":
        return backbone
    lower = model_name_or_path.lower()
    return "t5" if "t5" in lower or "flan" in lower else "decoder"


def decoder_lora_targets(model) -> List[str]:
    leaves = {name.split(".")[-1] for name, _ in model.named_modules()}
    if {"q_proj", "v_proj"}.issubset(leaves):
        return ["q_proj", "v_proj"]
    for leaf in ["query_key_value", "c_attn", "qkv_proj", "Wqkv"]:
        if leaf in leaves:
            return [leaf]
    candidates = [name for name in ["q_proj", "k_proj", "v_proj", "o_proj"] if name in leaves]
    if candidates:
        return candidates
    raise ValueError("Could not infer decoder LoRA target modules.")


class AdapterMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearNoFilter(nn.Module):
    def __init__(
        self,
        original_weight: torch.Tensor,
        original_bias: Optional[torch.Tensor] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(original_weight.detach().clone())
        if original_bias is not None:
            self.register_buffer("bias", original_bias.detach().clone())
        else:
            self.bias = None
        self.mlp = AdapterMLP(original_weight.shape[0], hidden_dim or max(128, original_weight.shape[0] // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.to(x.device)
        y = F.linear(x, self.weight, self.bias)
        return self.mlp(y.reshape(-1, y.shape[-1])).view(*y.shape)


class DEALLinear(nn.Module):
    def __init__(
        self,
        original_weight: torch.Tensor,
        original_bias: Optional[torch.Tensor] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as exc:
            raise ImportError("DEAL requires pytorch-wavelets. Install it with: pip install pytorch-wavelets") from exc

        self.weight = nn.Parameter(original_weight.detach().clone())
        if original_bias is not None:
            self.register_buffer("bias", original_bias.detach().clone())
        else:
            self.bias = None
        self.dwt = DWTForward(J=1, wave="haar", mode="zero")
        self.idwt = DWTInverse(wave="haar", mode="zero")
        cA, _ = self.dwt(self.weight.detach().float().unsqueeze(0).unsqueeze(0))
        self.theta = nn.Parameter(torch.ones_like(cA))
        self.mlp = AdapterMLP(original_weight.shape[0], hidden_dim or max(128, original_weight.shape[0] // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.to(x.device)
        dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            weight_4d = self.weight.float().unsqueeze(0).unsqueeze(0)
            cA, cD = self.dwt(weight_4d)
            filtered_weight = self.idwt((cA * self.theta.float(), cD)).squeeze(0).squeeze(0)
            bias = self.bias.float() if self.bias is not None else None
            y = F.linear(x.float(), filtered_weight, bias)
        y = y.to(dtype)
        return self.mlp(y.reshape(-1, y.shape[-1])).view(*y.shape)


class ProCLLinear(nn.Module):
    def __init__(
        self,
        original_weight: torch.Tensor,
        original_bias: Optional[torch.Tensor],
        N: int,
        d_key: int,
        lambda_consolidation: float,
        gamma: float,
        update_period: int,
        routing_temperature: float,
        program_key_init_scale: float,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.register_buffer("original_weight", original_weight.detach().clone())
        self.weight = nn.Parameter(original_weight.detach().clone())
        if original_bias is not None:
            self.register_buffer("bias", original_bias.detach().clone())
        else:
            self.bias = None
        self.rank, self.in_features = self.weight.shape
        if self.rank % N != 0 and self.rank != 1:
            raise ValueError("LoRA rank must be 1 or divisible by N.")
        self.N = N
        self.update_period = update_period
        self.routing_temperature = max(routing_temperature, 1e-6)
        self.forward_counter = 1
        self.program_scale = nn.Parameter(torch.zeros(N))
        self.program_keys = nn.Parameter(torch.randn(N, N, d_key) * program_key_init_scale)
        self.task_encoder = nn.Sequential(
            nn.Linear(self.in_features, d_key),
            nn.Tanh(),
            nn.Linear(d_key, d_key * N),
        )
        self.mlp = AdapterMLP(self.rank, hidden_dim or max(128, self.rank // 2))
        self.lambda_consolidation = self._init_gate(lambda_consolidation, original_weight)
        self.gamma = self._init_gate(gamma, original_weight)
        self._pending_ema_target = None

    @staticmethod
    def _init_gate(value: float, weight: torch.Tensor):
        if value >= 0:
            value = max(1e-6, min(1 - 1e-6, value))
            return float(value)
        rms = max(weight.norm().item() / max(weight.numel() ** 0.5, 1e-6), 1e-6)
        return nn.Parameter(torch.tensor(max(-10.0, min(10.0, -math.log(rms))), dtype=torch.float32))

    def _gate(self, value):
        return torch.sigmoid(value) if isinstance(value, nn.Parameter) else value

    def _programs(self) -> torch.Tensor:
        if self.rank == 1:
            return self.weight.unsqueeze(0).repeat(self.N, 1, 1)
        return self.weight.view(self.N, self.rank // self.N, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.to(x.device)
        if self._pending_ema_target is not None:
            with torch.no_grad():
                lambda_c = self._gate(self.lambda_consolidation)
                lambda_c = lambda_c.item() if isinstance(lambda_c, torch.Tensor) else lambda_c
                target = self._pending_ema_target.to(self.weight.device, self.weight.dtype)
                self.weight.mul_(1.0 - lambda_c).add_(lambda_c * target)
            self._pending_ema_target = None
        bsz, seq_len, _ = x.shape
        if not self.training:
            y = F.linear(x, self.weight, self.bias)
            return self.mlp(y.reshape(-1, y.shape[-1])).view(bsz, seq_len, -1)
        task_key = self.task_encoder(x.mean(dim=1)).view(bsz, self.N, -1)
        programs = self._programs()
        rank_per_program = programs.shape[1]
        heads = self.rank // rank_per_program
        attn = torch.softmax(
            torch.einsum("bhk,hnk->bhn", task_key[:, :heads, :], self.program_keys[:heads]) / self.routing_temperature,
            dim=-1,
        )
        alpha = attn.mean(dim=0) * torch.sigmoid(self.program_scale).unsqueeze(0)
        delta_w = torch.einsum("hn,nrd->hrd", alpha, programs).reshape(self.rank, self.in_features)
        w_exec = self.original_weight * self._gate(self.gamma) + delta_w
        if isinstance(self.lambda_consolidation, nn.Parameter):
            lambda_c = torch.sigmoid(self.lambda_consolidation)
            w_exec = lambda_c * w_exec + (1.0 - lambda_c) * self.weight
        if self.forward_counter % self.update_period == 0:
            self._pending_ema_target = w_exec.detach()
        self.forward_counter += 1
        y = F.linear(x, w_exec, self.bias)
        return self.mlp(y.reshape(-1, y.shape[-1])).view(bsz, seq_len, -1)


def replace_lora_modules(model, method: str, backbone: str, args):
    modules = dict(model.named_modules())
    for name, module in list(modules.items()):
        replace_lora_a = "lora_A.default" in name
        replace_lora_b = "lora_B.default" in name and (backbone == "decoder" or method == "deal")
        if not replace_lora_a and not replace_lora_b:
            continue
        parent_name, _, child_name = name.rpartition(".")
        original_weight = module.weight
        original_bias = getattr(module, "bias", None)
        if "lora_B.default" in name:
            hidden_dim = max(128, original_weight.shape[1] // 2)
        else:
            hidden_dim = max(128, original_weight.shape[0] // 2)
        if method == "procl":
            replacement = ProCLLinear(
                original_weight,
                original_bias,
                args.N,
                args.d_key,
                args.lambda_consolidation,
                args.gamma,
                args.update_period,
                args.routing_temperature,
                args.program_key_init_scale,
                hidden_dim,
            )
        elif method == "seq_lora":
            replacement = LinearNoFilter(original_weight, original_bias, hidden_dim)
        elif method == "deal":
            replacement = DEALLinear(original_weight, original_bias, hidden_dim)
        else:
            continue
        modules[parent_name]._modules[child_name] = replacement
    return model


class QATrainer(Trainer):
    pass


def build_train_dataset(records, tokenizer, model, backbone: str, args):
    dataset = Dataset.from_list(records)
    if backbone == "t5":
        dataset = dataset.map(
            lambda x: preprocess_seq2seq(x, tokenizer, args.max_source_length, args.max_target_length),
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.map(
            lambda x: {"labels": [[tok if tok != tokenizer.pad_token_id else -100 for tok in seq] for seq in x["labels"]]},
            batched=True,
        )
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    else:
        dataset = dataset.map(
            lambda x: preprocess_decoder(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=dataset.column_names,
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    keep = {"input_ids", "attention_mask", "labels"}
    drop = [col for col in dataset.column_names if col not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)
    return dataset, collator


def evaluate_seq2seq(model, tokenizer, dataset, batch_size: int, max_new_tokens: int, beams: int):
    model.eval()
    preds, refs = [], dataset["ref_answer"]
    for start in range(0, len(dataset), batch_size):
        ids = torch.tensor(dataset["input_ids"][start : start + batch_size], device=model.device)
        mask = torch.tensor(dataset["attention_mask"][start : start + batch_size], device=model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, num_beams=beams)
        preds.extend([text.strip() for text in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds, refs


def evaluate_decoder(model, tokenizer, dataset, batch_size: int, max_new_tokens: int):
    model.eval()
    preds, refs = [], dataset["ref_answer"]
    for start in range(0, len(dataset), batch_size):
        ids = torch.tensor(dataset["input_ids_prompt"][start : start + batch_size], device=model.device)
        mask = torch.tensor(dataset["attention_mask_prompt"][start : start + batch_size], device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        preds.extend([text.split("Answer:")[-1].strip() for text in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds, refs


def load_model_and_tokenizer(args, backbone: str):
    adapter_config = os.path.join(args.model_name_or_path, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config)
    base_name = PeftConfig.from_pretrained(args.model_name_or_path).base_model_name_or_path if is_adapter else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if backbone == "decoder":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    if backbone == "t5":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path, is_trainable=True) if is_adapter else get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=args.lora_dim,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q", "v"],
            ),
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path, is_trainable=True) if is_adapter else get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_dim,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=decoder_lora_targets(base_model),
            ),
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    if is_adapter and args.do_train:
        model = replace_lora_modules(model, args.method, backbone, args)
    return model, tokenizer, is_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--task_config_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_out_name", default="adapter")
    parser.add_argument("--backbone", choices=["auto", "t5", "decoder"], default="auto")
    parser.add_argument("--method", choices=["seq_lora", "procl", "deal"], default="procl")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--lora_dim", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--update_period", type=int, default=1)
    parser.add_argument("--d_key", type=int, default=16)
    parser.add_argument("--lambda_consolidation", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=-1)
    parser.add_argument("--routing_temperature", type=float, default=1.0)
    parser.add_argument("--program_key_init_scale", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    args.method = normalize_method(args.method)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_tasks = load_task_config(args.task_config_dir, "train")
    test_tasks = load_task_config(args.task_config_dir, "test")
    if len(train_tasks) != 1:
        raise ValueError("Each QA continual-learning round must contain exactly one train task.")

    backbone = detect_backbone(args.backbone, args.model_name_or_path)
    model, tokenizer, _ = load_model_and_tokenizer(args, backbone)

    train_name, train_strategy, train_num = train_tasks[0]
    raw_train = load_qa_dataset(train_name, "train", train_strategy, train_num)
    train_records = normalize_records(raw_train, train_name)
    train_dataset, collator = build_train_dataset(train_records, tokenizer, model, backbone, args)

    trainer = QATrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=True,
            bf16=args.bf16,
            fp16=args.fp16,
        ),
        train_dataset=train_dataset if args.do_train else None,
        data_collator=collator,
    )

    if args.do_train:
        trainer.train()
        adapter_dir = os.path.join(args.output_dir, args.adapter_out_name)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"saved_adapter={adapter_dir}")

    if args.do_predict:
        results = {}
        for ds_name, strategy, num in test_tasks:
            raw_eval = load_qa_dataset(ds_name, "validation", strategy, num)
            eval_records = normalize_records(raw_eval, ds_name)
            eval_ds = Dataset.from_list(eval_records)
            if backbone == "t5":
                eval_ds = eval_ds.map(
                    lambda x: preprocess_seq2seq(x, tokenizer, args.max_source_length, args.max_target_length),
                    batched=True,
                    remove_columns=eval_ds.column_names,
                )
                eval_ds = eval_ds.map(
                    lambda x: {"labels": [[tok if tok != tokenizer.pad_token_id else -100 for tok in seq] for seq in x["labels"]]},
                    batched=True,
                )
                preds, refs = evaluate_seq2seq(
                    model, tokenizer, eval_ds, args.per_device_eval_batch_size, args.max_new_tokens, args.generation_num_beams
                )
                if ds_name == "google/boolq":
                    preds = [boolq_pred_to_bool(p) for p in preds]
                score = exact_match(preds, refs)
            else:
                eval_ds = eval_ds.map(
                    lambda x: preprocess_decoder(x, tokenizer, args.max_length),
                    batched=True,
                    remove_columns=eval_ds.column_names,
                )
                preds, refs = evaluate_decoder(model, tokenizer, eval_ds, args.per_device_eval_batch_size, args.max_new_tokens)
                if ds_name == "google/boolq":
                    preds = [boolq_pred_to_bool(p) for p in preds]
                score = contains_match(preds, refs)
            results[f"EM_{ds_name}"] = round(score * 100.0, 2)
            print(f"{ds_name}: EM={results[f'EM_{ds_name}']}")
        out_path = os.path.join(args.output_dir, f"eval_results_{args.seed}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved_results={out_path}")


if __name__ == "__main__":
    main()
