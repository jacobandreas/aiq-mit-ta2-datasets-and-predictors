#!/usr/bin/env bash
# Environment-carrying wrapper for magnet_adapter/run_predictor_card.py.
#
# New file. run_predictor_card.py and the legacy card are untouched.
#
# WHY THIS EXISTS. The kwdagger card's node used to run `python -u
# magnet_adapter/run_predictor_card.py`. Under the legacy route that is fine:
# MAGNET runs the node in a process that inherited the caller's environment, so
# the active venv is on PATH. Under kwdagger the node runs in a **cmd_queue
# tmux worker, which inherits nothing** -- a tmux session created against an
# already-running server gets that server's environment, not the orchestrator's.
# A bare `python` there is whatever the login shell happens to expose, which is
# either missing torch/transformers or, worse, a different interpreter that
# imports and quietly scores something else.
#
# magnet/containers.py names this trap for the containerized path
# (DEFAULT_CAPTURED_ENV). The host path has the same problem and no equivalent
# mechanism yet, so the node command carries its own. Sibling of
# ta1/VeriStressGT/aiq/run_mini_sweep_env.sh, same reason.
#
# Everything is derived from this script's own location: nothing about the
# caller survives the worker boundary.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB="$(cd "$HERE/.." && pwd)"

# Anchor the working directory. The node's `executable` is a relative path, so
# finding THIS script depends on the caller's cwd -- under Slurm that is
# sbatch's --chdir, which defaults to wherever the submitting process happened
# to be. Everything after this line is cwd-independent, which leaves exactly
# one such dependency instead of one per path the predictor touches.
cd "$SUB"

# `predictors.*` and `magnet_adapter.*` are imported as top-level packages from
# the submodule root. circuit_overlap_predictor.py inserts this on sys.path
# itself before it needs them, so this is belt-and-braces -- but it also makes
# the containerized case honest: MAGNET forwards PYTHONPATH into the container
# by value, which OVERRIDES the image's own ENV PYTHONPATH. Appending here
# means the node does not depend on which of the two won.
export PYTHONPATH="$SUB${PYTHONPATH:+:$PYTHONPATH}"

# Resolution order: an explicit override, a file the runner script drops beside
# this wrapper (the only channel that survives the worker boundary), the active
# venv if the worker did inherit one, then the superproject's guest venv, then
# PATH. run_developer_setup.sh installs into the ACTIVE venv, so the dropped
# file is what normally answers.
# The recorded path is per-machine. $HERE is inside the repo, which the
# aiq-gpu host and its guest VM SHARE over virtiofs, while their interpreters
# live under different (mutually invisible) homes. One unscoped file would have
# whichever side ran last handing the other a dead path. Scope it by host, and
# still check the result is executable here.
MIT_PY="${MIT_PYTHON:-}"
_rec="$HERE/.python_path.$(uname -n)"
if [[ -z "$MIT_PY" && -f "$_rec" ]]; then
    _cand="$(<"$_rec")"
    [[ -x "$_cand" ]] && MIT_PY="$_cand"
fi
if [[ -z "$MIT_PY" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    MIT_PY="$VIRTUAL_ENV/bin/python"
fi
if [[ -z "$MIT_PY" ]]; then
    for cand in "$SUB/../../.venv/bin/python" "$(command -v python || true)" \
                "$(command -v python3 || true)"; do
        [[ -n "$cand" && -x "$cand" ]] && { MIT_PY="$cand"; break; }
    done
fi

[[ -n "$MIT_PY" && -x "$MIT_PY" ]] || {
    echo "[wrapper] no usable interpreter (MIT_PYTHON, $HERE/.python_path," >&2
    echo "[wrapper] \$VIRTUAL_ENV, .venv, PATH all came up empty)" >&2
    exit 127
}

# Fail here rather than after the model loads, with a traceback from inside a
# tmux pane nobody is attached to.
"$MIT_PY" -c 'import torch, transformers' 2>/dev/null || {
    echo "[wrapper] $MIT_PY cannot import the predictor's dependencies" >&2
    echo "[wrapper] run_developer_setup.sh installs into the ACTIVE venv" >&2
    exit 127
}

exec "$MIT_PY" -u "$SUB/magnet_adapter/run_predictor_card.py" "$@"
