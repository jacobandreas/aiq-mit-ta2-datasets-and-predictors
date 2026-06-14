We would like to build a "meta-model" that predicts LM performance at the
instance level. We will also use these to derive task-level accuracy estimates.

## General scaffolding

At a high level, a predictor should take as input (1) a trained model, a set of
validation examples with (inputs, model predictions, ground truth answers), and
a set of test examples with (inputs, model predictions). For each test example,
it should output a scalar confidence estimate (in (-inf, inf)) that should
correlate with the instance-level accuracy.

Then there should be an analysis script that takes a (model, predictor, val set,
test set) group, fits and runs the predictor, then computes standard measures of
the prediction accuracy: fit an isotonic regression to turn scalar predictor
scores into probabilistic confidence estimates, then compute the F-measure at
the optima classification threshold, AUC, and expected calibration error for
confidence estimates.

Implement the following predictor [there may be more later]:

## Circuit overlap prediction

Implemented in `predictors/circuit_overlap/`. Adapted from
`~/code/arithmetic-inconsistencies/attribution/`.

### Running

```bash
# Single run (after models/run.py eval has completed)
python -m predictors.run fit-predict \
    --run-dir runs/arithmetic_parametric/arithmetic_by_format/gpt2_small/seed0 \
    --predictor circuit_overlap \
    --models-root /raid/lingo/models

# All completed runs
python -m predictors.run sweep --runs-dir runs --predictor circuit_overlap

# Collect results
python -m predictors.run aggregate --runs-dir runs
```

### Implementation decisions

#### Attribution method

The original code uses **activation patching** (three forward passes: clean,
corrupted, and clean-activations-on-corrupted-input) which requires
counterfactual prompts `x_prime`. The darpa3 datasets don't all have
counterfactuals, so we instead use **gradient × activation** attribution —
a single forward+backward pass on `−log P(y|x)` — which gives an attribution
tensor of the same shape `[n_layers, intermediate_size]` with no counterfactual
required. This is the standard "gradient × input" first-order Taylor attribution.

The hook target is the INPUT to each layer's final MLP linear
(`mlp.down_proj` for Llama/Qwen, `mlp.c_proj` for GPT-2), matching the
convention in the original patching code.

Attribution is extracted at the **last prompt-token position** (the `=` sign
or equivalent), the position whose logit determines the first output token.

Batch-level backward computes per-item gradients correctly because items don't
interact after their individual activation captures: `grad[i]` receives only
`(1/N) × d(loss_i)/d(act_i)`, and the shared `1/N` factor doesn't affect
ranking.

#### Reference circuit

The reference circuit is the weighted-mean attribution over val items, with
weight 1 for correct items and 0 for incorrect items. This concentrates the
reference on the neurons actually involved in the model's correct computations.

#### Overlap score

`sum_on_S` (sum of item attributions over the reference's top-1% neurons) is
the primary confidence signal, consistent with the original paper where this
metric gave the strongest point-biserial correlation with per-item accuracy
(rpb ≈ 0.38 for Llama-3.1-8B on English arithmetic).

#### Analysis

`predictors/base.py:analyze()` fits an IsotonicRegression calibrator on val
scores, then evaluates test with: AUC (raw scores), F1 at the val-optimal
threshold, and ECE on calibrated probabilities.
