"""aiq-magnet InstancePredictor adapter for the darpa3 circuit overlap predictor.

This module bridges the aiq-magnet HELM-based evaluation framework and the
darpa3 PatchingCircuitOverlapPredictor (counterfactual attribution patching,
as in arithmetic-inconsistencies) and TokenProbPredictor.  It implements the InstancePredictor interface
so that circuit overlap scores can be submitted to the AIQ-MAGNET leaderboard.

Key design notes
----------------
* The model is identified from ``run_spec.adapter_spec.model`` in the eval
  run's run-spec DataFrame.  This is the HuggingFace Hub model ID (e.g.
  ``meta-llama/Llama-3.2-3B``).  A ``models_root`` constructor argument lets
  you redirect to a local weights directory (same convention as darpa3's
  ``--models-root`` flag).

* The train split is filtered to the same model as the eval run so that the
  reference circuit is built from that model's activations.  If no train rows
  share the same model, all available train data is used as a fallback (with a
  warning).

* Per-item prompts come from
  ``scenario_state.request_states.request.prompt``, and the per-item answer
  is the model's own greedy completion, the first entry of
  ``scenario_state.request_states.result.completions``.  Token prob scores
  that completion, and circuit overlap decodes as many tokens as it has.  No
  gold answer is used.  Prompts and completions are available even in the
  sequestered test split; only aggregate/per-instance *stats* are withheld.

* Counterfactual prompts come from a JSONL written by
  magnet_adapter/make_counterfactuals.py, keyed by instance id.

* Correctness labels for the train split are extracted from
  ``per_instance_stats`` by filtering to ``stat_name == "exact_match"`` and
  joining on ``magnet.instance_predict_id``.

* Train/eval runs are chosen by MAGNET's ``eval_run_selector``: a regex over
  ``run_spec.name`` picking the eval run (e.g. ``:split_config=cross_lingual,``);
  the remaining run (e.g. ``split_config=cross_lingual_train``) is the train
  split.  Defaults use one train run and every eval instance, rather than
  MAGNET's random 20-instance eval subsample.

Usage
-----
::

    from magnet_adapter.circuit_overlap_predictor import CircuitOverlapInstancePredictor
    import magnet

    outputs = magnet.HelmOutputs("/path/to/benchmark_output")
    suite_path = outputs.suites()[0].path
    predictor = CircuitOverlapInstancePredictor(
        counterfactuals_fpath="magnet_adapter/counterfactuals/arithmetic_fixed.jsonl",
        models_root="/raid/lingo/models",
        eval_run_selector=r":split_config=cross_lingual,",
    )
    predictor(helm_suites=suite_path)
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from magnet.instance_predictor import InstancePredictor, InstancePrediction
from magnet.predictor import Policy
from magnet.data_splits import TrainSplit, SequesteredTestSplit


# ---------------------------------------------------------------------------
# Column name constants (HELM → pandas after DotDict flattening)
# ---------------------------------------------------------------------------

_COL_RUN_SPEC_MODEL = "run_spec.adapter_spec.model"
_COL_RUN_SPEC_NAME  = "run_spec.name"
_COL_INSTANCE_ID    = "scenario_state.request_states.instance.id"
_COL_PROMPT         = "scenario_state.request_states.request.prompt"
_COL_COMPLETIONS    = "scenario_state.request_states.result.completions"
_COL_PREDICT_ID     = "magnet.instance_predict_id"
_COL_STAT_NAME      = "per_instance_stats.stats.name.name"
_COL_STAT_MEAN      = "per_instance_stats.stats.mean"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_model_name(run_specs_df: pd.DataFrame, models_root: str) -> str:
    """
    Return the HuggingFace model ID (or local path) for the run.

    Tries ``run_spec.adapter_spec.model`` first; falls back to parsing the
    model component out of ``run_spec.name`` (e.g. ``...,model=openai/gpt2``).
    """
    if _COL_RUN_SPEC_MODEL in run_specs_df.columns:
        model_id = run_specs_df[_COL_RUN_SPEC_MODEL].iloc[0]
    else:
        # Parse from run_spec.name: last component after "model="
        name = run_specs_df[_COL_RUN_SPEC_NAME].iloc[0]
        parts = {kv.split("=", 1)[0]: kv.split("=", 1)[1]
                 for kv in name.split(",") if "=" in kv}
        model_id = parts.get("model", name)

    # HELM replaces "/" with "_" in paths; try to restore it.
    # E.g. "meta-llama_Llama-3-2-3B" → probably won't match, but the raw
    # adapter_spec.model usually keeps the slash.
    if models_root:
        # Try <models_root>/<basename> first (darpa3 convention)
        local = Path(models_root) / Path(model_id).name
        if local.exists():
            return str(local)
    return model_id


def _load_model_and_tokenizer(model_path: str, dtype_str: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}.get(dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _scenario_state_to_items(
    scenario_df: pd.DataFrame,
    per_instance_stats_df: Optional[pd.DataFrame] = None,
) -> List[dict]:
    """
    Convert a magnet scenario_state DataFrame into the darpa3 item list format.

    Each item has keys: ``id``, ``x`` (prompt), ``y`` (the model's completion),
    and optionally ``correct`` (1/0 int from exact_match, if stats are
    provided).  ``y`` keeps the completion's leading space and drops trailing
    whitespace; an empty completion becomes a single space so there is still
    a token to score.
    """
    for col in (_COL_PROMPT, _COL_COMPLETIONS):
        if col not in scenario_df.columns:
            raise KeyError(f"scenario_state has no {col!r} column")

    items = []
    for _, row in scenario_df.iterrows():
        completions = row[_COL_COMPLETIONS]
        y = completions[0]["text"].rstrip() if len(completions) else ""
        items.append({
            "id":  row.get(_COL_INSTANCE_ID, row.get(_COL_PREDICT_ID, "")),
            "x":   row[_COL_PROMPT],
            "y":   y or " ",
            "split": "train",
            "features": {},
        })

    # Attach correctness labels from per_instance_stats (train split only)
    if per_instance_stats_df is not None and not per_instance_stats_df.empty:
        exact_match = per_instance_stats_df[
            per_instance_stats_df[_COL_STAT_NAME] == "exact_match"
        ]
        # Build a predict_id → correct mapping
        correct_map: dict = {}
        if _COL_PREDICT_ID in exact_match.columns:
            for _, srow in exact_match.iterrows():
                pid = srow[_COL_PREDICT_ID]
                correct_map[pid] = int(round(float(srow[_COL_STAT_MEAN])))

        # Attach to items via position index (predict_id == reset_index value)
        for i, item in enumerate(items):
            item["correct"] = correct_map.get(i, 0)

    return items


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class CircuitOverlapInstancePredictor(InstancePredictor):
    """
    Instance-level accuracy predictor using MLP circuit overlap attribution.

    Wraps :class:`predictors.circuit_overlap.patching_predictor.PatchingCircuitOverlapPredictor`
    (and the token prob baseline) and adapts it to the aiq-magnet
    ``InstancePredictor`` interface.

    Parameters
    ----------
    counterfactuals_fpath : str
        JSONL from magnet_adapter/make_counterfactuals.py giving each
        instance id's counterfactual prompt.
    models_root : str
        Optional local directory that mirrors the HuggingFace Hub layout.
        If ``<models_root>/<model_basename>`` exists it is used instead of
        downloading from the Hub (same convention as darpa3's
        ``--models-root`` flag).
    k_fraction : float
        Top-K fraction of neurons used to define the reference circuit "S".
        Default 0.01 (top 1%) matches the original paper's main result.
    batch_size : int
        Items per attribution forward+backward pass.
    dtype : str
        Model weight dtype: "float32", "float16", or "bfloat16".
    reference_id_regex : str
        Build the reference circuit from train items whose instance id
        matches this regex (default: all train items).
    fallback_max_draws : int
        Items without a usable counterfactual (none exists, or the two greedy
        answers are identical) are contrasted with random length-matched
        prompts from the same split, redrawn until the model's answer
        differs, for at most this many decodes per item.
    position : str
        Read attribution at each prompt's last token, requiring the two
        prompts to tokenize to the same length ("last", the reference) or
        not ("last_unaligned").
    eval_run_selector : str
        Regex over ``run_spec.name`` selecting the eval run; non-matching
        runs are train candidates.
    num_example_runs : int
        Number of train runs MAGNET samples (default 1: the single train run).
    num_eval_samples : int
        Eval instances MAGNET samples (default: all of them).
    **kwargs
        Forwarded to :class:`magnet.instance_predictor.InstancePredictor`
        (e.g. ``random_seed``).
    """

    def __init__(
        self,
        counterfactuals_fpath: str,
        models_root: str = "",
        k_fraction: float = 0.01,
        batch_size: int = 4,
        dtype: str = "bfloat16",
        reference_id_regex: str = "",
        position: str = "last",
        fallback_max_draws: int = 20,
        eval_run_selector: Optional[str] = None,
        num_example_runs: int = 1,
        num_eval_samples: int = sys.maxsize,
        insufficient_eval_sample_policy: Policy = Policy.IGNORE,
        **kwargs,
    ):
        super().__init__(
            eval_run_selector=eval_run_selector,
            num_example_runs=num_example_runs,
            num_eval_samples=num_eval_samples,
            insufficient_eval_sample_policy=insufficient_eval_sample_policy,
            **kwargs,
        )
        self.models_root = models_root
        self.k_fraction = k_fraction
        self.batch_size = batch_size
        self.dtype = dtype
        if not counterfactuals_fpath:
            raise ValueError("counterfactuals_fpath is required")
        self.counterfactuals_fpath = counterfactuals_fpath
        self.reference_id_regex = reference_id_regex
        if position not in ("last", "last_unaligned"):
            raise ValueError(f"position must be 'last' or 'last_unaligned', not {position!r}")
        self.position = position
        self.fallback_max_draws = fallback_max_draws

        # Set by predict(): per-test-item circuit_overlap / token_prob /
        # combined scores, keyed by magnet.instance_predict_id.
        self.component_scores_ = None

    # ------------------------------------------------------------------
    # InstancePredictor interface
    # ------------------------------------------------------------------

    def predict(
        self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit,
    ) -> List[InstancePrediction]:
        import torch
        # Import lazily so that the module is importable without torch/transformers
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        # ── Identify model ────────────────────────────────────────────────
        eval_run_specs = sequestered_test_split.run_specs
        model_path = _extract_model_name(eval_run_specs, self.models_root)
        eval_model_id = eval_run_specs[_COL_RUN_SPEC_MODEL].iloc[0] \
            if _COL_RUN_SPEC_MODEL in eval_run_specs.columns else model_path

        print(f"[circuit_overlap_magnet] eval model: {eval_model_id}")
        print(f"[circuit_overlap_magnet] loading weights from: {model_path}")

        # ── Filter train split to same model ──────────────────────────────
        train_scenario = train_split.scenario_state
        train_stats    = train_split.per_instance_stats

        if _COL_RUN_SPEC_MODEL in train_split.run_specs.columns:
            same_model_runs = train_split.run_specs[
                train_split.run_specs[_COL_RUN_SPEC_MODEL] == eval_model_id
            ][_COL_RUN_SPEC_NAME].tolist()

            if same_model_runs and _COL_RUN_SPEC_NAME in train_scenario.columns:
                filtered_scenario = train_scenario[
                    train_scenario[_COL_RUN_SPEC_NAME].isin(same_model_runs)
                ]
                filtered_stats = train_stats[
                    train_stats[_COL_RUN_SPEC_NAME].isin(same_model_runs)
                ] if train_stats is not None else train_stats
            else:
                warnings.warn(
                    f"No train runs found for model '{eval_model_id}'; "
                    "using all available train data for reference circuit.",
                    stacklevel=2,
                )
                filtered_scenario = train_scenario
                filtered_stats    = train_stats
        else:
            filtered_scenario = train_scenario
            filtered_stats    = train_stats

        # ── Build item lists ──────────────────────────────────────────────
        train_items = _scenario_state_to_items(filtered_scenario, filtered_stats)
        test_items  = _scenario_state_to_items(sequestered_test_split.scenario_state)

        if not train_items:
            raise RuntimeError("No train items found after filtering.")
        if not test_items:
            raise RuntimeError("No test items found.")

        n_correct = sum(item.get("correct", 0) for item in train_items)
        print(f"[circuit_overlap_magnet] train items: {len(train_items)} "
              f"({n_correct} correct)")
        print(f"[circuit_overlap_magnet] test items:  {len(test_items)}")

        # ── Load model ────────────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[circuit_overlap_magnet] device: {device}")
        model, tokenizer = _load_model_and_tokenizer(model_path, self.dtype)
        model = model.to(device)

        scores = self.score_items(train_items, test_items, model, tokenizer, device)

        # Free GPU memory
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ── Build InstancePrediction list ─────────────────────────────────
        # The MAGNET prediction is the circuit overlap score; the baseline and
        # combined scores are kept alongside for the runner to evaluate.
        test_df = sequestered_test_split.scenario_state.reset_index(drop=True)
        self.component_scores_ = pd.DataFrame({
            "prediction_id": test_df[_COL_PREDICT_ID].to_numpy(),
            **scores,
        })
        predictions: List[InstancePrediction] = []
        for i, score in enumerate(scores["circuit_overlap"]):
            row = test_df.iloc[i]
            predictions.append(InstancePrediction(
                run_spec_name=row[_COL_RUN_SPEC_NAME],
                instance_predict_id=row[_COL_PREDICT_ID],
                stat_name="exact_match",
                mean=float(score),
            ))
        return predictions

    def score_items(self, train_items, test_items, model, tokenizer, device) -> dict:
        """Fit on train items and score test items with circuit overlap, the
        token prob baseline, and their combination.

        The combination is the mean of the two scores' within-test-set rank
        percentiles: equal weight, scale-free, nothing fitted.  Only items
        scored with their *own* counterfactual contribute circuit overlap
        (its percentile is over those items); items scored with a fallback
        contrast, or not at all, take the token prob percentile alone.
        Unscored items get the median circuit overlap score.
        """
        from scipy.stats import rankdata
        from predictors.token_prob.predictor import TokenProbPredictor

        co_scores = self._patching_scores(train_items, test_items, model, tokenizer, device)
        contrast = self.contrast_
        co_kept = ~np.isnan(co_scores)
        if not co_kept.any():
            raise ValueError("no test items have a circuit overlap score")
        co_scores = np.where(co_kept, co_scores, np.median(co_scores[co_kept]))

        tp = TokenProbPredictor(batch_size=self.batch_size, device=device)
        tp.fit(train_items, model=model, tokenizer=tokenizer)
        tp_scores = np.asarray(tp.predict(test_items, model=model, tokenizer=tokenizer), dtype=float)

        def _rank(x):
            return (rankdata(np.clip(x, -1e9, 1e9)) - 1) / max(len(x) - 1, 1)

        own = np.array([c == "own" for c in contrast])
        tp_rank = _rank(tp_scores)
        combined = tp_rank.copy()
        if own.any():
            combined[own] = (_rank(co_scores[own]) + tp_rank[own]) / 2
        return {
            "circuit_overlap": co_scores,
            "circuit_overlap_kept": co_kept,
            # "own", "fallback" or None (unscored): the contrast each item used.
            "circuit_overlap_contrast": contrast,
            "token_prob": tp_scores,
            "combined": combined,
        }

    def _patching_scores(self, train_items, test_items, model, tokenizer, device) -> np.ndarray:
        """Attribution-patching circuit overlap scores (NaN for filtered items)."""
        from predictors.circuit_overlap.patching_predictor import PatchingCircuitOverlapPredictor

        counterfactuals = {}
        with open(self.counterfactuals_fpath) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    counterfactuals[row["id"]] = row

        def _patch_items(items):
            """One patching item per input item.  K is the token length of the
            model's own completion; items with no counterfactual (e.g. a
            swapped arithmetic problem that divides by zero) get
            ``x_prime=None`` and a fallback contrast."""
            out = []
            for item in items:
                cf = counterfactuals.get(item["id"])
                x_prime = None
                if cf is not None:
                    if not item["x"].endswith(cf["x"]):
                        raise ValueError(f"prompt of {item['id']!r} does not end with its "
                                         f"counterfactual file text {cf['x']!r}")
                    x_prime = item["x"][:len(item["x"]) - len(cf["x"])] + cf["x_prime"]
                k = max(1, len(tokenizer(item["y"], add_special_tokens=False)["input_ids"]))
                out.append({"x": item["x"], "x_prime": x_prime, "k": k})
            n_cf = sum(p["x_prime"] is not None for p in out)
            print(f"[circuit_overlap_magnet] counterfactuals for {n_cf} of {len(items)} items")
            return out

        ref_items = train_items
        if self.reference_id_regex:
            pattern = re.compile(self.reference_id_regex)
            ref_items = [it for it in train_items if pattern.search(it["id"])]
            print(f"[circuit_overlap_magnet] reference circuit from {len(ref_items)} train "
                  f"items matching {self.reference_id_regex!r}")
            if not ref_items:
                raise ValueError(f"no train items match {self.reference_id_regex!r}")

        co = PatchingCircuitOverlapPredictor(
            k_fraction=self.k_fraction, batch_size=self.batch_size, device=device,
            position=self.position, fallback_max_draws=self.fallback_max_draws,
        )
        co.fit(_patch_items(ref_items), model=model, tokenizer=tokenizer)
        scores = co.predict(_patch_items(test_items), model=model, tokenizer=tokenizer)
        self.contrast_ = np.array(co.predict_info_["contrast"], dtype=object)
        return scores
