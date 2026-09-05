# Running the evaluation card in a container

## How Kitware runs this card

`python -m magnet.evaluation_new` reads the card, turns its `kwdagger:` block
into a DAG, and schedules the DAG through kwdagger. For
`magnet_adapter/cards/qwen_arithmetic_fixed_kwdagger.yaml` that is a single
node, `circuit_overlap`, whose command is `bash magnet_adapter/run_predictor_env.sh`.

The node runs as one `docker run` of the image built from the `Dockerfile` at
the repository root. The checkout, called `$REPO` below, is bind-mounted at its
own absolute path and the node's working directory is set there. `PYTHONPATH`
is forwarded into the container, so the node runs the mounted checkout, not the
copy baked into the image. Results land under `--output_path`.

The backend is tmux on a workstation and Slurm on a cluster. Per-node leasing
does not apply: the predictor loads Qwen3-4B itself for gradient attribution
and does no inference against a served endpoint.

## Build

```bash
cd $REPO
docker build -t mit-circuit-overlap-gpu .
```

`MAGNET_VERSION` in the Dockerfile is the aiq-magnet release the evaluator uses,
from PyPI (0.1.0, which also brings aiq-magnet-theory). To build against
another release:

```bash
docker build --build-arg MAGNET_VERSION=0.1.0 -t mit-circuit-overlap-gpu .
```

`TORCH_CUDA` selects the torch build (default `cu130`, the CUDA 13 wheels).
Pass `--build-arg TORCH_CUDA=cu128` for a driver that cannot run CUDA 13.

## Reproduce the June dry run

On the host you need the same aiq-magnet, docker with the NVIDIA container
toolkit, and tmux:

```bash
pip install "aiq-magnet[optional]==0.1.0"
export PYTHONPATH=$REPO
```

Two things are mounted from outside the checkout:

- `$DATA`: the HELM `benchmark_output` directory from the full generation.
  The June result used the run with 1016 request states in `cross_lingual`
  and 8984 in `cross_lingual_train`. A smaller generation silently changes
  the AUC (a 100-item copy gives 0.605), so check the counts before running.
- `$HF_HOME`: a HuggingFace cache that already holds `Qwen/Qwen3-4B`. The
  evaluator forwards `HF_HOME` into the container by value, but forwarding the
  name does not mount the directory. Without the mount the node downloads
  about 8 GB into a container whose home is `/tmp` and discards it on exit.

Then:

```bash
cd $REPO
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
python -m magnet.evaluation_new magnet_adapter/cards/qwen_arithmetic_fixed_kwdagger.yaml \
    --output_path runs/mit_circuit_overlap \
    --backend tmux \
    --container_image mit-circuit-overlap-gpu \
    --container_mounts "$REPO:$DATA:$HF_HOME" \
    --container_docker_args "--gpus device=0" \
    --params "matrix: {circuit_overlap.helm_runs_path: '$DATA', circuit_overlap.k_fraction: 0.01, circuit_overlap.batch_size: 4}"
```

A GPU is required. The June dry run, merged as `PhaseI_DryRun/MIT/CircuitOverlap`,
gave VERIFIED with a CircuitOverlap AUC of 0.721961 against the 0.65 threshold.
Our containerized reproduction lands on the same number to six digits.

The verdict is written to
`runs/mit_circuit_overlap/<hash>_<timestamp>/verdict.json`, with a `latest`
symlink beside it. The node's own `results.json` sits under
`runs/mit_circuit_overlap/_kwdagger/circuit_overlap/`.

kwdagger keys a node's output directory on its inputs, not on its command, so a
second run into the same `--output_path` with the same parameters reuses the
existing result. Use a fresh `--output_path` to force a re-run.

## Leasing

Does not apply. This card does no live inference against a served endpoint.

## What Kitware changes when evaluating

Our runner supplies the host-specific values: which GPU index, where the HELM
output and the HuggingFace cache are mounted from, tmux or Slurm as the
backend, and a provenance record written next to the verdict. The card, this
image and the command shape are exactly what is shown above.
