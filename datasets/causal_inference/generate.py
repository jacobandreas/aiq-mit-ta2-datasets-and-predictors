"""Generate causal_inference dataset.

Constructs ~100 random DAGs where each non-root node is an AND or OR gate.
Generates two item types:
  - Observations: given some observed node values, predict a query node
  - Structural queries: given a node pair, predict their relationship

Each DAG has obs_density and struct_density in {0, 0.25, 0.5, 0.75, 1.0};
train/test membership is determined by SHA-256 hash so the full item space
is never materialised.

Default split (causal_default):
  - train: items from DAGs with density >= 0.25 that hash into the train partition
  - test:  items from DAGs with density <= 0.75 that hash into the test partition

Usage:
    python -m datasets.causal_inference.generate \\
        --output-dir data/causal_inference \\
        --n-dags 100 --obs-per-dag 200 --struct-per-dag 100 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.utils import make_item, write_jsonl

# ── Node naming ──────────────────────────────────────────────────────────────

def node_name(i: int) -> str:
    if i < 26:
        return chr(ord("A") + i)
    return "A" + chr(ord("A") + (i - 26))


# ── DAG construction ─────────────────────────────────────────────────────────

class DAG:
    def __init__(self, n: int, dag_id: int, rng: random.Random, p_edge: Optional[float] = None):
        self.n = n
        self.dag_id = dag_id
        self.names = [node_name(i) for i in range(n)]

        if p_edge is None:
            p_edge = min(0.5, 4 / (n - 1))

        # Build adjacency: parents[j] = list of parent indices for node j
        self.parents: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p_edge:
                    self.parents[j].append(i)

        # Ensure at least one edge
        if all(len(p) == 0 for p in self.parents.values() if p is not None):
            self.parents[1].append(0)
        # Fix: nodes 1..n-1 that happened to get no parents via the loop above
        # are roots; that's intentional, not a bug.

        self.roots = [i for i in range(n) if not self.parents[i]]
        self.gate: dict[int, str] = {
            i: (rng.choice(["AND", "OR"]) if self.parents[i] else "ROOT")
            for i in range(n)
        }

        self.obs_partition_seed = rng.randint(0, 2**32)
        self.struct_partition_seed = rng.randint(0, 2**32)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, root_values: dict[int, bool]) -> dict[int, bool]:
        vals: dict[int, bool] = {}
        for i in range(self.n):
            if not self.parents[i]:
                vals[i] = root_values.get(i, False)
            elif self.gate[i] == "AND":
                vals[i] = all(vals[p] for p in self.parents[i])
            else:
                vals[i] = any(vals[p] for p in self.parents[i])
        return vals

    # ── Hash-based partition ─────────────────────────────────────────────────

    def _hash_float(self, seed: int, *parts: str) -> float:
        key = f"{seed}," + ",".join(parts)
        digest = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(digest, "big") / 2**256

    def is_train_obs(self, subset: frozenset[int], query: int, density: float) -> bool:
        subset_str = ",".join(self.names[i] for i in sorted(subset))
        query_str = self.names[query]
        return self._hash_float(self.obs_partition_seed, subset_str, query_str) < density

    def is_train_struct(self, a: int, b: int, density: float) -> bool:
        return self._hash_float(self.struct_partition_seed, self.names[a], self.names[b]) < density

    # ── Structural relation ───────────────────────────────────────────────────

    def ancestors(self, node: int) -> set[int]:
        result: set[int] = set()
        stack = list(self.parents[node])
        while stack:
            p = stack.pop()
            if p not in result:
                result.add(p)
                stack.extend(self.parents[p])
        return result

    def relation(self, a: int, b: int) -> str:
        anc_a = self.ancestors(a)
        anc_b = self.ancestors(b)
        if b in self.parents[a]:      # a is direct parent of b? no: parents[b] contains a
            pass
        # parents[b] = list of b's parents
        if a in self.parents[b]:
            return "parent"
        if b in self.parents[a]:
            return "child"
        if b in anc_a:
            return "ancestor"
        if a in anc_b:
            return "descendant"
        # siblings: share a common parent
        if set(self.parents[a]) & set(self.parents[b]):
            return "sibling"
        return "other"

    def to_dict(self) -> dict:
        return {
            "dag_id": self.dag_id,
            "n": self.n,
            "names": self.names,
            "parents": {str(k): v for k, v in self.parents.items()},
            "gate": self.gate,
            "obs_partition_seed": self.obs_partition_seed,
            "struct_partition_seed": self.struct_partition_seed,
        }


# ── Item rendering ────────────────────────────────────────────────────────────

def make_obs_item(
    dag: DAG,
    subset: frozenset[int],
    query: int,
    rng: random.Random,
    split: str,
    item_id: str,
) -> dict:
    root_vals = {r: rng.choice([True, False]) for r in dag.roots}
    all_vals = dag.evaluate(root_vals)

    obs_parts = ", ".join(
        f"{dag.names[i]}={int(all_vals[i])}" for i in sorted(subset)
    )
    x = f"dag_{dag.dag_id:03d}: {obs_parts}, {dag.names[query]}="
    y = str(int(all_vals[query]))

    return make_item(
        id=item_id,
        x=x,
        y=y,
        split=split,
        features={
            "type": "observation",
            "dag_id": dag.dag_id,
            "n_nodes": dag.n,
            "observed_nodes": sorted(dag.names[i] for i in subset),
            "query_node": dag.names[query],
        },
    )


def make_struct_item(
    dag: DAG,
    a: int,
    b: int,
    split: str,
    item_id: str,
) -> dict:
    rel = dag.relation(a, b)
    x = f"dag_{dag.dag_id:03d}: relation({dag.names[a]}, {dag.names[b]}) ="
    y = rel
    return make_item(
        id=item_id,
        x=x,
        y=y,
        split=split,
        features={
            "type": "structural",
            "dag_id": dag.dag_id,
            "n_nodes": dag.n,
            "node_a": dag.names[a],
            "node_b": dag.names[b],
        },
    )


# ── Split generation ──────────────────────────────────────────────────────────

DENSITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SIZES = [8, 16, 32]
OBS_SUBSET_SIZE = 3


def generate_dags(n_dags: int, rng: random.Random) -> list[DAG]:
    dags = []
    for dag_id in range(n_dags):
        n = rng.choice(N_SIZES)
        dags.append(DAG(n, dag_id, rng))
    return dags


def assign_densities(dags: list[DAG], rng: random.Random) -> dict[int, tuple[float, float]]:
    """Assign (obs_density, struct_density) to each DAG, balanced across DENSITIES."""
    assignments = {}
    for dag in dags:
        obs_d = rng.choice(DENSITIES)
        struct_d = rng.choice(DENSITIES)
        assignments[dag.dag_id] = (obs_d, struct_d)
    return assignments


def sample_obs_items(
    dag: DAG,
    density: float,
    n_items: int,
    split: str,
    id_prefix: str,
    rng: random.Random,
    id_counter: list[int],
) -> list[dict]:
    non_root = [i for i in range(dag.n) if dag.parents[i]]
    if len(non_root) < OBS_SUBSET_SIZE + 1:
        return []

    items = []
    attempts = 0
    while len(items) < n_items and attempts < n_items * 50:
        attempts += 1
        # Sample observed subset (non-root nodes, excluding query)
        candidates = rng.sample(range(dag.n), min(OBS_SUBSET_SIZE + 1, dag.n))
        subset = frozenset(candidates[:OBS_SUBSET_SIZE])
        query = candidates[OBS_SUBSET_SIZE]
        if query in subset:
            continue

        in_train = dag.is_train_obs(subset, query, density)
        if (split == "train") != in_train:
            continue

        item_id = f"{id_prefix}_{id_counter[0]:06d}"
        id_counter[0] += 1
        items.append(make_obs_item(dag, subset, query, rng, split, item_id))
    return items


def sample_struct_items(
    dag: DAG,
    density: float,
    n_items: int,
    split: str,
    id_prefix: str,
    rng: random.Random,
    id_counter: list[int],
) -> list[dict]:
    if dag.n < 2:
        return []
    items = []
    attempts = 0
    while len(items) < n_items and attempts < n_items * 50:
        attempts += 1
        a, b = rng.sample(range(dag.n), 2)
        in_train = dag.is_train_struct(a, b, density)
        if (split == "train") != in_train:
            continue
        item_id = f"{id_prefix}_{id_counter[0]:06d}"
        id_counter[0] += 1
        items.append(make_struct_item(dag, a, b, split, item_id))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/causal_inference")
    parser.add_argument("--n-dags", type=int, default=100)
    parser.add_argument("--obs-per-dag", type=int, default=200,
                        help="Target observation items per DAG per split")
    parser.add_argument("--struct-per-dag", type=int, default=100,
                        help="Target structural items per DAG per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    dags = generate_dags(args.n_dags, rng)
    densities = assign_densities(dags, rng)

    # Save DAG metadata
    dag_meta_path = out / "dags.json"
    dag_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dag_meta_path, "w") as f:
        json.dump(
            {
                "dags": [d.to_dict() for d in dags],
                "densities": {str(k): v for k, v in densities.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved {len(dags)} DAG metadata entries to {dag_meta_path}")

    # ── causal_default split ─────────────────────────────────────────────────
    split_dir = out / "causal_default"
    print("Generating causal_default …")

    train_items: list[dict] = []
    test_items: list[dict] = []
    obs_train_ctr = [0]
    obs_test_ctr = [0]
    struct_train_ctr = [0]
    struct_test_ctr = [0]

    for dag in dags:
        obs_d, struct_d = densities[dag.dag_id]

        # Train: density > 0 (some training items exist)
        if obs_d > 0:
            train_items += sample_obs_items(
                dag, obs_d, args.obs_per_dag, "train",
                "ci_obs_train", rng, obs_train_ctr,
            )
        if struct_d > 0:
            train_items += sample_struct_items(
                dag, struct_d, args.struct_per_dag, "train",
                "ci_struct_train", rng, struct_train_ctr,
            )

        # Test: density < 1 (some test items exist)
        if obs_d < 1.0:
            test_items += sample_obs_items(
                dag, obs_d, args.obs_per_dag, "test",
                "ci_obs_test", rng, obs_test_ctr,
            )
        if struct_d < 1.0:
            test_items += sample_struct_items(
                dag, struct_d, args.struct_per_dag, "test",
                "ci_struct_test", rng, struct_test_ctr,
            )

    write_jsonl(split_dir / "train.jsonl", train_items)
    write_jsonl(split_dir / "test.jsonl", test_items)
    print(f"  train={len(train_items)}, test={len(test_items)}")

    print("Done.")


if __name__ == "__main__":
    main()
