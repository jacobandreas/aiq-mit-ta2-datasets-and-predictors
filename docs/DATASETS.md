## Overview

The project will look at datasets in three domains: world modeling, arithmetic,
state tracking, and causal inference. 

Place the datasets in a data dierctory. The structure of this directory should
mirror `~/code/darpa`; in particular, it should be possible to run evals with the
helm-run calls described in `~/code/darpa/HELM_INTEGRATION.md`.

Each dataset consists of a collection of items; each item has a set of features.
Named splits are constructed by choosing items with subsets of specific
features.

### Data format

All data files are JSONL with one item per line. Each item has the fields:

```json
{"id": "...", "x": "...", "y": "...", "split": "train|val|test", "features": {...}}
```

- `x`: the prompt (input to the model, always ending with `=` or equivalent)
- `y`: the target (correct completion)
- `split`: partition membership for the enclosing split configuration
- `features`: domain-specific metadata used for split construction

### Directory layout

```
data/
  world_modeling/<domain>/train.jsonl, test.jsonl
  arithmetic_fixed/<language>/train.jsonl, test.jsonl
  arithmetic_parametric/arithmetic_by_format/train.jsonl, val.jsonl, test.jsonl
  arithmetic_parametric/arithmetic_by_skill/train.jsonl, val.jsonl, test.jsonl
  state_tracking/state_tracking_by_length/train.jsonl, val.jsonl, test.jsonl
  state_tracking/state_tracking_by_format/train.jsonl, val.jsonl, test.jsonl
  causal_inference/causal_default/train.jsonl, test.jsonl
  causal_inference/dags.json   (DAG metadata)
```

### Generator scripts

```bash
python -m datasets.arithmetic_parametric.generate --output-dir data/arithmetic_parametric
python -m datasets.state_tracking.generate        --output-dir data/state_tracking
python -m datasets.causal_inference.generate      --output-dir data/causal_inference
python -m datasets.arithmetic_fixed.convert \
    --source-dir ~/code/arithmetic-inconsistencies/arithmetic-dataset \
    --output-dir data/arithmetic_fixed
python -m datasets.world_modeling.generate        --output-dir data/world_modeling
```

Default sizes: 2000 train / 500 val / 1000 test items per split.
Causal inference generates ~25k train / ~22k test items across 100 DAGs.
World modeling mirrors the full EWOK-CORE HuggingFace dataset.

## World Modeling

Download from HuggingFace (`ewok-core/ewok-core-1.0`) and split by domain.
Each EWOK domain becomes a named split (e.g. `agent-properties`). Source
reference is `~/code/darpa` (resolves to `~/Library/CloudStorage/Dropbox/code/darpa`).

## Arithmetic (fixed)

Copy the dataset from `~/code/arithmetic-inconsistencies/arithmetic-dataset`.
Each language is a split

## Arithmetic (parametric)

Generate new code for this dataset.

### Features

- Operators that appear in the expression (subset of `{+, -, *, /}`).
  Items that share an operator set form a regime; the skill-composition
  split below carves regimes train vs. test.
- Format: symbolic (as in the example above), or natural language in one
  of `{English, Spanish, Italian, Chinese}`.

### Initial splits to generate

1. **Format generalization** (`arithmetic_by_format`). All splits use the
   full operator set `{+, -, *, /}` over single-digit operands; only the
   surface form varies.
   - train: symbolic + English
   - val: Spanish
   - test: Italian + Chinese

2. **Skill composition** (`arithmetic_by_skill`). All splits use symbolic
   format over single-digit operands; the operator combination is what
   varies.
   - train: items drawn uniformly from two operator pools — `{+, -}` and `{+, *}`
   - val: items using `{+, -, *}` together
   - test: items using `{-, *}` together

   This asks whether a model that has seen each of `-` and `*` paired
   with `+` (i.e. the train regimes `{+, -}` and `{+, *}`) can compose
   subtraction with multiplication directly. The val regime
   `{+, -, *}` lets us also look at three-operator composition as an
   intermediate level.

(Length-based splits were used in an earlier round of experiments; the
results were dominated by from-scratch capacity limits rather than
generalization, so they're not part of the current focus.)

### Implementation decisions

- Each expression has a uniformly random number of operands in `[2, 5]`.
- At least one operand uses the full digit count when length > 1; at
  length 1 every operand is in `[0, 9]`.
- Standard operator precedence (`*`, `/` before `+`, `-`; left-associative
  within each tier). Internally we evaluate via `fractions.Fraction` so
  division is exact.
- Only items whose final value is an integer are emitted; we
  reject-and-resample otherwise (including on division by zero).
- Natural-language items use `num2words` for English / Spanish / Italian.
  `num2words` has no Chinese converter, so we ship a hand-written one
  (`_to_chinese`) covering non-negative integers up to ~10^16 plus
  negatives via a `负` prefix.
- For natural-language items the target is rendered in the same language
  as the prompt (e.g. Spanish prompt → Spanish target). The doc's worked
  example `y = "10"` only covers the symbolic case.
- The dataset CLI accepts either `operators: [...]` (one fixed pool) or
  `operator_pools: [[...], [...]]` (per-item choice across multiple
  pools) in a split. When `operator_pools` has more than one pool, each
  item records the pool it was drawn from in
  `features.operator_pool` so downstream analysis can recover the
  regime.

## State tracking

Items in the state tracking domain essentially involve computing a product of
permutations, e.g. assuing an initial sequence ABC, x = "swap(1, 2), swap(2,
3)", y = "BCA"

### Features

- Item count: just 3 (ABC) or 5 (ABCDE).

- Problem length: number of swaps to perform (5 - 20).

- Format: Symbolic (in the example above, just "12 23"), code ("lst[1], lst[2] =
  lst[2], lst[1]; lst[2], lst[3] = lst[3], lst[2]"), natural langauge ("swap box
  1 and box 2; swap box 2 and box 3")

### Initial splits to generate

- Length 10 / Length 11-12 / Length 13-14 (`state_tracking_by_length`)

  An earlier round used length 5 / 6-7 / 8-20 but the very short training
  length encouraged per-permutation memorization rather than learning the
  swap mechanism. Length 10 gives the model enough sequence to learn an
  algorithm, with test length 13-14 close enough that smooth generalization
  is plausible.

- Symbolic / code / language (`state_tracking_by_format`)

### Implementation decisions

- The initial state is prepended to every prompt (e.g. `"ABC: 12 23 ="`,
  `"lst = ['A', 'B', 'C']; ...; lst ="`, `"Start with box A, box B, box C. ..."`).
  Otherwise the symbolic format cannot disambiguate 3-vs-5-item universes
  when the swap indices happen never to exceed 3.
- Indices are 1-based throughout, including the code format. This
  follows the doc's worked example verbatim (`lst[1], lst[2] = lst[2], lst[1]`
  for `swap(1, 2)`), even though Python's normal convention is 0-based.
- Target rendering matches the format: symbolic and language formats
  emit the final string `"BCA"`; the code format emits a Python list
  literal `"['B', 'C', 'A']"` to fit the surface form.

## Causal inference

This domain is slightly more complicated, because there are couplings between
items. We first construct a set of K random DAGs with varying topologies. Each
Each node in the DAG corresponds to a variable; it is either the AND or OR of
its parents. Then there are two kinds of items:

- *Obervations*: We choose some DAG i. We (uniformly at random) set its root
  nodes to either true or false. We then select a subset of nodes as
  to observe, and another node as the query; models must then determine the
  value of the query node. So items look like x = "DAG i: A=1, F=0, C=", y =
  "0".

- *Structural queries*": We choose some DAG i, choose a pair of nodes, and then
  prompt models to predict how they are related. Items look like x = "DAG i:
  relation between C -> F =", y="ancestor". (Can be "parent", "child",
  "sibling", "ancestor", "descendant", "other".)

When generating DAGs, we pre-determine which combinations of nodes can appear in
the training set and which are held out (ditto for structural queries).
Different DAGs can have different "training densities".

### Features

- Number of nodes in containing DAG [8, 16, 32]

- Training density of observations for containing DAG [0%, 25%, 50%, 75%, 100%]

- Training density of structural queries for containing DAG [0%, 25%, 50%, 75%, 100%]

### Initial splits to generate

First generate a fixed set of ~100 DAGs. Then generate splits with varying
distributions of # of observations / structrural queries per DAG, average
training densityfor test set items.

### Implementation decisions

- The DAG id is treated as an opaque label — the model never sees the
  graph structure in the prompt. Cross-DAG generalisation is not a
  target of this benchmark; the goal is to study within-DAG generalisation
  under varying training densities.
- DAGs are generated by ordering the nodes 0..n-1 and adding each
  forward edge `(i, j)` independently with probability
  `p_edge = min(0.5, 4 / (n - 1))` (configurable). Each non-root node is
  randomly assigned an `AND` or `OR` gate over its parents. If a sample
  happens to have no edges at all, we inject `(0, 1)` so AND/OR
  semantics are exercised.
- Node names: `A`..`Z` for the first 26 nodes, then `AA`..`AZ` for nodes
  beyond that (so `n=32` uses `A`..`AF`).
- The per-DAG train/test partition is materialised lazily via SHA-256
  hashing. Each DAG carries an `obs_partition_seed` and a
  `struct_partition_seed`; `is_train_obs(subset, query)` returns
  `hash(seed, subset, query) < density_obs`, and analogously for
  structural pairs. This keeps memory constant even for `n=32` (where
  the full enumeration of observation combos exceeds 100k).
- Observation items use a fixed observed-subset size `k=3` (configurable
  via `observed_subset_size`). Root values are sampled uniformly at item
  generation time, independent of which `(subset, query)` combo was
  drawn.
- Structural queries use ordered pairs `(a, b)`; `relation(a, b)` is
  computed with the precedence
  `parent > child > ancestor > descendant > sibling > other`
  (a node-pair that is both `parent` and `ancestor` reports `parent`;
  siblings that are also ancestors/descendants report the ancestral
  label).
- DAGs with `density_obs = 0%` have no train-eligible observation items,
  and DAGs with `density_obs = 100%` have no test-eligible ones; the CLI
  filters DAGs accordingly when sampling each split. Same for
  structural queries with `density_struct`.
