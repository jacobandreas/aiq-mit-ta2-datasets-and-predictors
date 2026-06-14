"""aiq-magnet InstancePredictor adapter for the darpa3 circuit overlap predictor.

This module bridges the aiq-magnet HELM-based evaluation framework and the
darpa3 CircuitOverlapPredictor.  It implements the InstancePredictor interface
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
  ``scenario_state.request_states.request.prompt``.
  Per-item completions come from
  ``scenario_state.request_states.result.completions.0.text``
  (the first greedy-decoded completion stored by HELM).  Both are available
  even in the sequestered test split; only aggregate/per-instance *stats* are
  withheld.

* Correctness labels for the train split are extracted from
  ``per_instance_stats`` by filtering to ``stat_name == "exact_match"`` and
  joining on ``magnet.instance_predict_id``.

Usage
-----
::

    from magnet_adapter.circuit_overlap_predictor import CircuitOverlapInstancePredictor
    import magnet

    outputs = magnet.HelmOutputs("/path/to/benchmark_output")
    suite_path = outputs.suites()[0].path
    predictor = CircuitOverlapInstancePredictor(
        models_root="/raid/lingo/models",
        k_fraction=0.01,
        batch_size=4,
    )
    predictor(helm_suites=suite_path)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from magnet.instance_predictor import InstancePredictor, InstancePrediction, _reindex_split
from magnet.data_splits import TrainSplit, TestSplit, SequesteredTestSplit


# ---------------------------------------------------------------------------
# Column name constants (HELM → pandas after DotDict flattening)
# ---------------------------------------------------------------------------

_COL_RUN_SPEC_MODEL = "run_spec.adapter_spec.model"
_COL_RUN_SPEC_NAME  = "run_spec.name"
_COL_INSTANCE_ID    = "scenario_state.request_states.instance.id"
_COL_PROMPT         = "scenario_state.request_states.request.prompt"
_COL_COMPLETION     = "scenario_state.request_states.result.completions.0.text"
_COL_PREDICT_ID     = "magnet.instance_predict_id"
_COL_STAT_NAME      = "per_instance_stats.stats.name.name"
_COL_STAT_MEAN      = "per_instance_stats.stats.mean"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_completion_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column that holds the model's first completion text."""
    # Try the common flattened name first, then fall back to any column
    # containing 'completions' and 'text'.
    candidates = [
        _COL_COMPLETION,
        "scenario_state.request_states.result.completions.0.text",
    ]
    col = _first_present(df, candidates)
    if col:
        return col
    # Broader search
    for c in df.columns:
        if "completions" in c and "text" in c:
            return c
    return None


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

    Each item has keys: ``id``, ``x`` (prompt), ``y`` (completion), and
    optionally ``correct`` (1/0 int from exact_match, if stats are provided).

    The completion column is read from the scenario state (HELM stores the
    model's response there).  If no completion column is found, ``y`` defaults
    to a single space so that the attribution forward pass still has a target
    token.
    """
    completion_col = _get_completion_col(scenario_df)

    items = []
    for _, row in scenario_df.iterrows():
        x = row.get(_COL_PROMPT, "")
        y = row.get(completion_col, " ").strip() if completion_col else " "
        if not y:
            y = " "
        item = {
            "id":  row.get(_COL_INSTANCE_ID, row.get(_COL_PREDICT_ID, "")),
            "x":   x,
            "y":   y,
            "split": "train",
            "features": {},
        }
        items.append(item)

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

    Wraps :class:`predictors.circuit_overlap.predictor.CircuitOverlapPredictor`
    and adapts it to the aiq-magnet ``InstancePredictor`` interface.

    Parameters
    ----------
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
    **kwargs
        Forwarded to :class:`magnet.instance_predictor.InstancePredictor`
        (e.g. ``num_example_runs``, ``num_eval_samples``, ``random_seed``).
    """

    def __init__(
        self,
        models_root: str = "",
        k_fraction: float = 0.01,
        batch_size: int = 4,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models_root = models_root
        self.k_fraction = k_fraction
        self.batch_size = batch_size
        self.dtype = dtype

    # ------------------------------------------------------------------
    # InstancePredictor interface
    # ------------------------------------------------------------------

    def prepare_all_dataframes(self, helm_runs):
        """Split runs deterministically by name: runs whose directory name
        contains '_train' (i.e. split_config=cross_lingual_train) become the
        train split; all others become the eval split."""
        from magnet.backends.helm.helm_outputs import HelmRuns

        coerced_runs = list(HelmRuns.coerce(helm_runs))

        train_runs = [r for r in coerced_runs if "_train" in r.path.name]
        eval_runs  = [r for r in coerced_runs if "_train" not in r.path.name]

        if not train_runs or not eval_runs:
            names = [r.path.name for r in coerced_runs]
            raise RuntimeError(
                f"Expected runs with and without '_train' in name; got: {names}"
            )

        def _concat(dfs):
            return pd.concat(dfs, ignore_index=True)

        train_split = TrainSplit(
            run_specs=_concat([r.run_spec() for r in train_runs]),
            scenario_state=_concat([r.scenario_state() for r in train_runs]),
            stats=_concat([r.stats() for r in train_runs]),
            per_instance_stats=_concat([r.per_instance_stats() for r in train_runs]),
        )
        test_split = TestSplit(
            run_specs=_concat([r.run_spec() for r in eval_runs]),
            scenario_state=_concat([r.scenario_state() for r in eval_runs]),
            stats=_concat([r.stats() for r in eval_runs]),
            per_instance_stats=_concat([r.per_instance_stats() for r in eval_runs]),
        )

        _reindex_split(train_split)
        _reindex_split(test_split)
        return train_split, test_split

    def predict(
        self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit,
    ) -> List[InstancePrediction]:
        import torch
        # Import lazily so that the module is importable without torch/transformers
        import sys, os
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from predictors.circuit_overlap.predictor import CircuitOverlapPredictor

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

        # ── Run circuit overlap predictor ─────────────────────────────────
        predictor = CircuitOverlapPredictor(
            k_fraction=self.k_fraction,
            batch_size=self.batch_size,
            device=device,
        )
        predictor.fit(train_items, model=model, tokenizer=tokenizer)
        test_scores = predictor.predict(test_items, model=model, tokenizer=tokenizer)

        # Free GPU memory
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ── Build InstancePrediction list ─────────────────────────────────
        test_df = sequestered_test_split.scenario_state.reset_index(drop=True)
        predictions: List[InstancePrediction] = []
        for i, score in enumerate(test_scores):
            row = test_df.iloc[i]
            predictions.append(InstancePrediction(
                run_spec_name=row[_COL_RUN_SPEC_NAME],
                instance_predict_id=row[_COL_PREDICT_ID],
                stat_name="exact_match",
                mean=float(score),
            ))
        return predictions
