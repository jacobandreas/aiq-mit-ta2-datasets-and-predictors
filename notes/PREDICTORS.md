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

### Attribution-patching variant (matches arithmetic-inconsistencies)

Added 2026-11-03 after the gradient × activation version fell to about
chance once its tokenization bug was fixed (the answer was missing its leading
space, so `"Answer:" + "A"` became one `:A` token). Code is in
`predictors/circuit_overlap/patching.py`, a port of
`arithmetic-inconsistencies/attribution/src/{generate,attribution}.py`, and
`predictors/circuit_overlap/patching_predictor.py`. In the MAGNET adapter
it's selected with `method="patching"`; the card is
`magnet_adapter/cards/circuit_overlap_patching.yaml`.

- **Counterfactuals.** These come from `magnet_adapter/make_counterfactuals.py`
  and are keyed by HELM instance id.
  - arithmetic_fixed uses the reference's sign-flipped `x_prime`. The ids are
    the same `prob_N`, and `symbolic` there is the reference's `numeric`.
  - EWOK uses the item's minimal-pair partner context (`_ctx1` ↔ `_ctx2`),
    which has the other answer.
  - arithmetic_parametric has no counterfactuals yet, because `*`/`/`-only
    items have no sign flip.
- **Method, as in the reference:**
  - Decode exactly K_i = max(len tok y, len tok y') greedy tokens for x and x'.
  - Keep items whose two prompts tokenize to the same length and whose two
    responses differ.
  - Compute the clean and corrupted baselines.
  - Use the three-pass attribution `grad ⊙ (clean − corrupted)` at the last
    prompt token, with float16 weights.
- **Reference circuit S.** S is the top 1% of the *unweighted* mean
  attribution over train items, without looking at labels. For cross_lingual
  only the symbolic train items are used, matching the reference's numeric
  circuit; the card selects them with `reference_id_regex: '^(symbolic_|ewok_)'`.
  For EWOK all train items are used.
- **Item score.** `sum_on_S`. Test items removed by the filters get the
  median kept score. The results report both `auc` (all items) and `auc_kept`
  (scored items only, which is what the reference reports).
- **Deviation.** The reference backpropagates the batch *mean*, which scales
  a short last batch differently. Here the batch sum is divided by the
  nominal batch size instead.
- **Smoke test** (Qwen3-4B, cross_lingual, 300 symbolic train and 300 test
  items): 84% of test items kept, kept-item AUC 0.70.
- **First results (2026-11-03, Qwen3-4B).** All-items AUC and AUC over the
  items circuit overlap could score:

  | split | all items | scored items | items scored |
  |---|---|---|---|
  | cross_lingual | 0.80 | 0.80 | 86% |
  | ewok_physical_to_social | 0.58 | 0.89 | 53% |
  | ewok_social_to_physical | 0.57 | 0.80 | 50% |

  On EWOK, about 30% of pairs fail the same-length filter, because the two
  contexts differ by a word.
- **Read position.** `position="first_diff"` (used for EWOK, at the user's
  suggestion) reads attribution at the first token where x and x' differ,
  which is usually the context word that flips the answer. It needs no
  length match, so the alignment filter is skipped. cross_lingual keeps the
  reference's last-token read.
- **Ensembling.** Items circuit overlap can't score now take the token prob
  rank percentile alone, rather than a median circuit overlap score averaged
  in. The node also writes per-item `scores.csv`, so ensembling can be
  changed without a GPU rerun.
- **arithmetic_parametric counterfactuals.**
  - **Construction:** `x'` swaps every operator, + ↔ − and × ↔ ÷. The
    operator sequence is recovered by re-rendering the item's operands with
    `datasets/arithmetic_parametric/generate.py`. Building them needs
    `num2words`.
  - **Gaps:** items whose swapped problem divides by zero or is non-integer
    get no counterfactual and stay unscored. That's 38% of all items. On
    the test sets, counterfactual coverage is 62% for by_format and 41% for
    by_skill, where items with only `*` mostly fail.
  - **Read position:** by_skill is symbolic, so it uses the last-token read.
    For by_format, only 26% of test items pass the same-length filter
    (Italian and Chinese operator words tokenize to different lengths), so
    it uses `first_diff`.
  - **Reference circuit:** for both splits it is built from all train
    items. Their ids don't encode format, so the symbolic items can't be
    picked out by id.
- **Read-position results (2026-11-03, Qwen3-4B).**
  - **`first_diff` doesn't work.** It brought circuit overlap to chance on
    the scored items in every split that used it: EWOK 0.44 and 0.47,
    by_format 0.48.
  - **The last-token read works.** On the scored items: cross_lingual 0.80,
    by_skill 0.98. Earlier last-token EWOK runs reached 0.80–0.89 on the
    aligned half of the items.
  - **Added `position="last_unaligned"`.** It reads each prompt at its own
    last token without the same-length filter, and is now used for
    by_format and both EWOK splits.
- **by_skill check.** Part of the 0.98 is separation between operator groups:
  `{−}` items are 99% correct and `{−,*}` items 31%. Within the 87 scored
  `{−,*}` items, circuit overlap AUC is still 0.90, while token prob is 0.22
  (inverted).
- **K from the model's completion (2026-11-04, at the user's request).**
  This is the default everywhere (`k_source: completion`). K is the token
  length of the model's own HELM completion of x, which HELM stops at the
  run's stop sequence, with a minimum of 1. The reference instead uses
  max(len y, len y') over the *gold* answers. That leaks label information,
  because a correct answer always fits K exactly, while a wrong one is
  padded with junk tokens or cut off. `k_source: gold` keeps the reference
  rule.
- **Fallback contrast (2026-11-04).** `fallback_contrast: true` applies to
  any item with no usable counterfactual: either none exists, or the
  model's answers to x and x' are identical. Such an item is contrasted
  with a random other prompt from the same call (the same split's train or
  test items) that has the same token length and different text, drawn
  with a fixed seed. If even that leaves the two answers identical, the
  item stays unscored. Results report `auc_own` and `auc_fallback`
  separately.
  - First result, by_skill: 979 of 1000 test items scored, up from 342.
    All-items AUC rose from 0.62 to 0.89; `auc_own` is 0.98 and
    `auc_fallback` 0.82. Combined beats token prob by +0.117.
  - Full run (card `circuit_overlap_patching.yaml`, 0.6 placeholder
    thresholds: VERIFIED). AUC = circuit overlap, all test items
    (unscored items median-filled); own/fallback = AUC on the items scored
    with each contrast; gain = combined minus token prob, all items.

    | split | scored | fallback | AUC | own | fallback | TP | combined | gain |
    |---|---|---|---|---|---|---|---|---|
    | cross_lingual | 1000/1016 | 135 | 0.819 | 0.795 | 0.858 | 0.865 | 0.882 | +0.017 |
    | by_format | 942/1000 | 473 | 0.803 | 0.902 | 0.728 | 0.945 | 0.925 | −0.020 |
    | by_skill | 979/1000 | 634 | 0.892 | 0.984 | 0.821 | 0.750 | 0.867 | +0.117 |
    | ewok_p2s | 3909/4530 | 475 | 0.663 | 0.904 | 0.620 | 0.851 | 0.841 | −0.011 |
    | ewok_s2p | 3642/4218 | 594 | 0.643 | 0.805 | 0.569 | 0.761 | 0.753 | −0.008 |

    Completion-length K leaves the own-contrast AUCs essentially unchanged
    from gold-length K. The fallback is strong on arithmetic and weak on
    EWOK, and there it pulls the combined score below token prob. EWOK
    still leaves ~13% unscored, because a random partner often gets the
    same letter answer.
- **Fallback redraw (2026-11-04, at the user's request).** The fallback now
  tries random length-matched prompts in a fixed random order until the
  model's answer differs from its answer to x, for at most
  `fallback_max_draws` (20) decodes per item. Responses are cached by
  (prompt, K), so a candidate already decoded at the same K costs nothing.
  Before this, a single draw left 13–14% of EWOK items unscored, because a
  random partner often got the same letter.
- **Cross-domain ensemble weight (2026-11-04, at the user's request).**
  `cards/circuit_overlap_patching_ensemble.yaml` (node
  `run_ensemble_card.py`, CPU only) reads the patching card's `scores.csv`
  files and scores combined = w·rank(CO) + (1−w)·rank(TP). Items without
  a CO score take the TP rank alone. w comes from the grid {0, 0.05, …, 1}
  and maximizes the mean combined AUC over the *other* domain's splits:
  arithmetic targets are tuned on the two EWOK splits, and EWOK targets on
  the three arithmetic splits. Results also report the equal-weight AUC
  and, for reference only, the target's oracle w. Run this card after the
  patching card. Its nodes are cached by algo_params, so delete them to
  recombine after re-running the patching card.
  - Results with the redraw (`fallback_max_draws` 20): 1 test item
    unscored in total. The patching card passes on every split at the 0.6
    placeholders. Cross-domain w: arithmetic splits get w = 0 (from EWOK),
    so the gain is exactly 0. EWOK splits get w = 0.5 (from arithmetic),
    with gains of −0.030 (p2s) and −0.027 (s2p). The ensemble card fails.
    The two domains want opposite weights; oracle w is 0.10–0.75 on
    arithmetic and 0 on both EWOK splits. EWOK fallback items score at or
    near chance (0.61, 0.53). Own-contrast AUCs are unchanged: 0.90 (p2s),
    0.81 (s2p), 0.98 (by_skill), 0.90 (by_format), 0.79 (cross_lingual).
- **Own-contrast-only ensemble (2026-11-04).** `combine_contrasts: own`
  (now the card default) uses CO only on own-counterfactual items;
  fallback-contrast items take the TP rank. Cross-domain w is 0.30 for
  arithmetic (tuned on EWOK) and 0.55 for EWOK (tuned on arithmetic). Card
  VERIFIED: every split gains over TP. combined vs TP: cross_lingual
  0.881 vs 0.865 (+0.016), by_format 0.949 vs 0.945 (+0.004), by_skill
  0.786 vs 0.750 (+0.035), ewok_p2s 0.852 vs 0.851 (+0.001), ewok_s2p
  0.764 vs 0.761 (+0.002). Oracle w: 0.35, 0.3, 1.0, 0.3, 0.3. The tuning
  curves are flat near their optima: 0.25–0.35 for arithmetic targets,
  0.45–0.60 for EWOK targets.
- **Decision (2026-11-04, user): fixed equal weight.** The ensemble card
  uses `weight: '0.5'` with `combine_contrasts: own`. Cross-domain tuning
  averaged 0.846 combined AUC against 0.850 for w = 0.5, so it gained
  nothing. `weight: 'tuned'` keeps it available. Card VERIFIED. combined
  vs TP: cross_lingual 0.877 vs 0.865 (+0.012), by_format 0.946 vs 0.945
  (+0.001), by_skill 0.808 vs 0.750 (+0.058), ewok_p2s 0.854 vs 0.851
  (+0.003), ewok_s2p 0.765 vs 0.761 (+0.004).
- **Thresholds (2026-11-04, user):** the patching card keeps a CO AUC threshold of 0.6 on every split.
