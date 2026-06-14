"""HuggingFace Trainer-based training loop.

Lazy-imports torch/transformers so the dataset code stays torch-free.

Called by models/run.py when zero_shot=False.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from models.config import ModelConfig, RunConfig
from models.evaluate import compute_metrics, write_metrics, write_predictions


# ── Prompt → tokens ───────────────────────────────────────────────────────────

class _PromptDataset:
    """torch.utils.data.Dataset of tokenized (prompt + completion) pairs."""

    def __init__(self, items: List[dict], tokenizer, max_length: int = 512):
        import torch

        self.examples = []
        for item in items:
            full = item["x"] + item["y"]
            prompt_ids = tokenizer.encode(item["x"], add_special_tokens=False)
            full_ids = tokenizer.encode(full, add_special_tokens=True)

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            # Pad/truncate
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                labels = labels[:max_length]

            self.examples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── Val-accuracy callback ────────────────────────────────────────────────────

def _make_val_accuracy_callback(val_items: List[dict], tokenizer, config: ModelConfig, device: str):
    """
    Returns a HuggingFace TrainerCallback that computes exact-match accuracy
    on a random subsample of val items at each eval step and injects
    `eval_accuracy` into the metrics so EarlyStoppingCallback can read it.
    """
    from transformers import TrainerCallback
    from models.evaluate import evaluate

    n_eval = min(config.training.val_eval_items, len(val_items))
    rng = random.Random(0)
    subsample = rng.sample(val_items, n_eval)

    class ValAccCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if model is None:
                return
            preds = evaluate(
                model, tokenizer, subsample,
                max_new_tokens=config.max_new_tokens,
                batch_size=config.eval_batch_size,
                device=device,
            )
            acc = sum(p["correct"] for p in preds) / len(preds)
            if metrics is not None:
                metrics["eval_accuracy"] = acc

    return ValAccCallback()


# ── Training entry point ──────────────────────────────────────────────────────

def train_and_eval(
    run_config: RunConfig,
    train_items: List[dict],
    val_items: List[dict],
    test_items: List[dict],
    log_path: Path,
    device: str = "cuda",
) -> None:
    """
    Fine-tune (or train from scratch) the model specified by run_config,
    evaluate on val and test, and write all artifacts to run_config.output_dir.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    cfg = run_config.model_config
    tc = cfg.training
    out = Path(run_config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = cfg.resolve_model_path(run_config.models_root or None)

    # ── Load tokenizer + model ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name_or_path or model_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}.get(cfg.dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    ).to(device)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = _PromptDataset(train_items, tokenizer)
    val_ds = _PromptDataset(val_items, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True,
                                      label_pad_token_id=-100)

    # ── TrainingArguments ─────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=tc.max_epochs,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=tc.lr,
        warmup_steps=tc.warmup_steps,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        max_grad_norm=tc.max_grad_norm,
        weight_decay=tc.weight_decay,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.eval_steps,
        logging_steps=tc.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        seed=run_config.seed,
        report_to="none",
        log_level="warning",
    )

    val_acc_callback = _make_val_accuracy_callback(val_items, tokenizer, cfg, device)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=tc.patience)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[val_acc_callback, early_stopping],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))

    _run_eval_and_save(model, tokenizer, val_items, test_items, cfg, out, device)


# ── Shared post-training eval ─────────────────────────────────────────────────

def _run_eval_and_save(
    model,
    tokenizer,
    val_items: List[dict],
    test_items: List[dict],
    cfg: ModelConfig,
    out: Path,
    device: str,
) -> None:
    """Write predictions_val.jsonl, predictions_test.jsonl, metrics.json."""
    from models.evaluate import evaluate

    for split_name, items in (("val", val_items), ("test", test_items)):
        if not items:
            continue
        preds = evaluate(model, tokenizer, items,
                         max_new_tokens=cfg.max_new_tokens,
                         batch_size=cfg.eval_batch_size,
                         device=device)
        write_predictions(preds, out / f"predictions_{split_name}.jsonl")

    # Combined metrics
    all_preds = {}
    for split_name in ("val", "test"):
        path = out / f"predictions_{split_name}.jsonl"
        if path.exists():
            from models.evaluate import read_predictions
            all_preds[split_name] = compute_metrics(read_predictions(path))

    write_metrics(all_preds, out / "metrics.json")
