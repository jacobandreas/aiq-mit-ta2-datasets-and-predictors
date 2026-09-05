# syntax=docker/dockerfile:1.7
# MIT CircuitOverlap: the image the card's one node runs in.
#
# The card's kwdagger DAG has a single node, circuit_overlap, whose command is
# `bash magnet_adapter/run_predictor_env.sh`. That node runs as one
# `docker run` of this image, with this checkout bind-mounted at its own
# absolute path and the working directory set there. The copy baked below
# provides the environment; the mounted checkout provides the code, so editing
# the predictor or the card does not mean rebuilding.
#
# Build, from the repository root:
#   docker build -t mit-circuit-overlap-gpu .
#
# MAGNET_VERSION is the aiq-magnet release the evaluator runs against, from
# PyPI (0.1.0, released 2026-09-04; it also brings aiq-magnet-theory).
# `--build-arg MAGNET_VERSION=<version>` builds against another release.
#
# Never baked: the HELM benchmark_output the predictor reads, and the
# Qwen3-4B weights (HuggingFace cache). Both are mounted at run time. See
# docs/containerized_evaluation.md.
FROM python:3.11-slim

ARG MAGNET_VERSION=0.1.0

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# git: some dependencies install from source refs.
# libgomp1: torch's CPU kernels link against OpenMP, which slim does not carry.
#
# gcc/g++/libc6-dev are RUNTIME dependencies, not build ones. Triton, which
# torch uses for its GPU kernels, JIT-compiles at run time and shells out to a
# C compiler; without one the node dies on the first kernel launch with
# "Failed to find C compiler". That is invisible at build time, so do not
# move these into a purged layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git libgomp1 gcc g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# Triton searches PATH, but naming the compiler removes the ambiguity its own
# error message points at.
ENV CC=gcc

WORKDIR /opt/src

# Torch first, from the PyTorch index, in its own layer: the largest dependency
# and the one least likely to change. The build is pinned, not just the
# version, because a plain `pip install torch` resolves to the CPU wheel on
# some platforms and the run then fails for a reason that looks like a missing
# GPU. cu130 is the CUDA 13 build, which the newest GPUs (Blackwell) need;
# older cards can pass `--build-arg TORCH_CUDA=cu128`.
ARG TORCH_VERSION=2.13.0
ARG TORCH_CUDA=cu130
RUN pip install --no-cache-dir \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        "torch==${TORCH_VERSION}"

# The predictor's own stack. transformers/accelerate are commented out of
# requirements.txt ("only needed on server for model runs"); this node needs
# them. zstandard arrives via datasets and builds from source here, which is
# what the compiler above is also for.
COPY requirements.txt /tmp/mit-requirements.txt
RUN pip install --no-cache-dir -r /tmp/mit-requirements.txt \
 && pip install --no-cache-dir "transformers>=4.40" "accelerate>=0.28"

# magnet is a RUNTIME import of the predictor (magnet.instance_predictor,
# magnet.data_splits, magnet.backends.helm.helm_outputs), not only the thing
# that drives it. Pinned; kwdagger and cmd_queue arrive through magnet's own
# dependency list.
RUN pip install --no-cache-dir \
        "aiq-magnet[optional]==${MAGNET_VERSION}"

COPY . /opt/src/aiq-mit-ta2-datasets-and-predictors

# The predictor imports `predictors.*` and `magnet_adapter.*` as top-level
# packages from the repository root, and is not installed as a distribution.
# The bind-mounted checkout shadows this copy at run time: the evaluator
# forwards PYTHONPATH by value, and the node wrapper prepends its own root.
ENV PYTHONPATH=/opt/src/aiq-mit-ta2-datasets-and-predictors

WORKDIR /opt/src/aiq-mit-ta2-datasets-and-predictors

# Fail the BUILD rather than the run if the import surface is broken. This
# cannot check CUDA (the builder has no GPU), so it does not try.
RUN python -c "\
import torch, transformers, magnet, pandas, sklearn, scipy;\
from magnet.instance_predictor import InstancePredictor;\
from magnet.data_splits import TrainSplit;\
import predictors.circuit_overlap.attribution;\
print('mit-circuit-overlap image ok, torch', torch.__version__)"

CMD ["bash"]
