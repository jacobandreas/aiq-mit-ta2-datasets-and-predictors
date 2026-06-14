For each dataset in the `data` directory, we would like to train a family of
models whose held-out behavior we can evaluate. Eventually, we will train the
four model families:

- A GPT2-type transformer with 4 layers (trained from scratch)
- A GPT2-small sized model (trained from scratch, zero-shot/off-the-shelf, and fine-tuned)
- A Llama-3.2-1B-Instruct model (zero-shot and fine-tuned; initial weights in
  `/raid/lingo/models` on the server)
- A Qwen3-0.6B model (zero-shot and fine-tuned; in `/raid/lingo/models`)

IMPORTANT: for now, do not train or fine-tune any models, just set up
scaffolding and make sure we can run GPT2, Llama, and Qwen off the shelf.

## Running off-the-shelf evaluation (current state)

```bash
cd /raid/lingo/jda/code/darpa3
python -m models.run eval \
    --data-dir data \
    --families gpt2_small llama32_1b qwen3_0_6b \
    --seeds 0 \
    --output-dir runs \
    --models-root /raid/lingo/models
```

The orchestrator auto-discovers all leaf split directories under `data/`
(any directory containing `train.jsonl`). Results land in
`runs/<dataset_path>/<family>/seed<N>/`.

To aggregate results after eval:
```bash
python -m models.run aggregate --runs-dir runs
```

To re-run a completed triple: pass `--force`.

## Training details

Use configurations that are as standard as possible; do early stopping based on
val accuracy. Do three training runs for each model (for from-scratch models,
randomize both the initial weights and the dataset order, for fine-tuned models,
randomize only the dataset order).

## Evaluation details

The main thing we're interested in is whether there is interesting variability
across models (at the level of datasets or individual items). So once models are
trained, run them on the test set, and report both set-level accuracies and
item-level agreement statistics (e.g. on what fraction of items do different
model families, or different training runs from the same model family, disagree
on the answer).

## Implementation decisions

### Sweep scope

The orchestrator auto-discovers all *leaf* directories under `data/` that contain
`train.jsonl` — currently 9 split configs (arithmetic_fixed × 4 languages,
arithmetic_parametric × 2 configs, state_tracking × 2 configs, causal_inference × 1
config). World modeling adds more once generated. Off-the-shelf families are
`{gpt2_small, llama32_1b, qwen3_0_6b}` × `{1 seed}` = 27 runs initially.
Full sweep with training families and 3 seeds = ~63 runs.
Each run lands at `runs/<relative_dataset_path>/<family>/seed<N>/`.
Repeated runs are skipped unless `--force` is passed.

### Causal dataset

`causal_default` has no `val.jsonl`. The data loader concatenates
`train_obs.jsonl` + `train_struct.jsonl` for training, and `test_obs.jsonl` +
`test_struct.jsonl` for test. 10% of the merged train is held out (by `seed`)
as val so early stopping has something to score against. The same model
therefore handles both kinds of causal items; per-kind accuracy is recovered
post-hoc from the `features.kind` field in the per-item predictions.

### Tokenizers

* From-scratch GPT-2 variants reuse the public `gpt2` BPE tokenizer (byte-level
  fallback handles all four arithmetic languages including Chinese, though
  inefficiently).
* Fine-tunes use the upstream tokenizer that ships with each pretrained model.
* All tokenizers have `pad_token = eos_token` when not otherwise defined.

### Architecture choices

* `gpt2_4l_scratch`: 4 layers, 12 heads, 768 hidden (~52M params),
  trained from scratch on the task.
* `gpt2_small_ft`: 12 / 12 / 768 (~124M params) trained from scratch.
* `gpt2_small`: same architecture, fine-tuned from OpenAI's pretrained
  GPT-2 checkpoint.
* `llama32_1b`: `meta-llama/Llama-3.2-1B-Instruct` evaluated out of the
  box (`num_train_epochs=0`).
* `llama32_1b_ft`: full fine-tune of the same checkpoint on the task.
* `qwen3_0_6b`: `Qwen/Qwen3-0.6B` evaluated out of the box.
* `qwen3_0_6b_ft`: full fine-tune of the same checkpoint.

Per-family training hyperparameters (LR, batch size, max epochs, eval cadence)
live inside each model's YAML under a `training:` block and override
`TrainConfig` defaults.

### Training loop

Standard HuggingFace `Trainer`, lazy-imported so the dataset code stays
torch-free. Loss is causal-LM cross-entropy with `-100` masking over the
prompt tokens. The val-accuracy-based early-stopping signal is computed by a
custom callback (`_make_val_accuracy_callback`) that runs greedy decode on a
random val subsample (`val_eval_items`, default 200) at every eval step and
injects `eval_accuracy` into the metrics dict before HF's `EarlyStoppingCallback`
reads it. `metric_for_best_model="eval_accuracy"` together with
`load_best_model_at_end=True` ensures the best-val-accuracy checkpoint is the
one we save and evaluate.

### Run artifacts

Per `runs/<dataset>/<family>/seed<N>/`:

* `config.json` — fully-resolved `TrainConfig`.
* `final/` — saved best HF checkpoint (model + tokenizer).
* `train_log.jsonl` — append-only log of every Trainer log line plus the val
  accuracy at each eval.
* `predictions_val.jsonl`, `predictions_test.jsonl` — per-item `{input,
  target, prediction, correct, features}` rows.
* `metrics.json` — set-level accuracy on val/test plus per-feature breakdowns
  (e.g. accuracy by `length`, by `format`, by `dag_id`).

### Cross-model agreement

`darpa2-models aggregate` walks `runs/<dataset>/<family>/<seed>/` and computes:

* per-`(family, seed)` test accuracy,
* within-family disagreement (fraction of items on which two seeds within a
  family produce different predictions),
* family-consensus accuracy (majority vote across seeds within a family),
* between-family disagreement (mean over family pairs of the fraction of items
  on which the two families' consensus predictions disagree).

### Operational notes

* Develop locally; the actual run lives on `align-3` per CLAUDE.md.
* Llama / Qwen weights are expected under `/raid/lingo/models/<dir>/`; the
  trainer auto-detects them via `--models-root` and falls back to the HF Hub.
* The orchestrator is single-process. For 8× A100 parallelism, partition the
  job and launch one orchestrator per GPU via `CUDA_VISIBLE_DEVICES=N ...`.

