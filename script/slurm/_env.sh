#!/bin/bash
# Shared environment loader. Source this at the top of every sbatch script.
# Usage: source "$(dirname "$0")/_env.sh"

set -euo pipefail

# --- Strip any pre-activated uv .venv that would shadow conda --------------
if [ -n "${VIRTUAL_ENV:-}" ]; then
    # Drop $VIRTUAL_ENV/bin from PATH and unset venv vars.
    PATH=":${PATH}:"
    PATH=${PATH//:${VIRTUAL_ENV}\/bin:/:}
    PATH=${PATH#:}; PATH=${PATH%:}
    export PATH
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
fi

# --- Project paths -------------------------------------------------------
export PROJECT_PATH=/lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
export PROJECT_DATA=/lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer/datasets
export PROJECT_CKPT=/lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer/checkpoint
export PROJECT_PRETRAINED=/lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer/pretrained_models
mkdir -p "$PROJECT_DATA" "$PROJECT_CKPT" "$PROJECT_PRETRAINED" "$PROJECT_PATH/logs"

# --- Caches on the project filesystem ------------------------------------
export HF_HOME=/lus/lfs1aip2/projects/u6hx/huanan/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TORCH_HOME=/lus/lfs1aip2/projects/u6hx/huanan/.cache/torch
export PIP_CACHE_DIR=/lus/lfs1aip2/projects/u6hx/huanan/.cache/pip
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

# --- Cluster modules -----------------------------------------------------
# GH200 / aarch64 — use CUDA 12.6 toolchain.
module purge >/dev/null 2>&1 || true
module load PrgEnv-gnu cuda/12.6 gcc-native/14.2 >/dev/null 2>&1 || true

# --- Conda ---------------------------------------------------------------
# Pick whichever miniconda3 actually exists (HOME path is the conda init-ed one,
# the lus path is a project-side mirror used in some sessions).
if [ -d "$HOME/miniconda3" ]; then
    export CONDA_ROOT=$HOME/miniconda3
else
    export CONDA_ROOT=/lus/lfs1aip2/projects/u6hx/huanan/miniconda3
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

export U2T_ENV=${U2T_ENV:-u2t}
# Activate only if the env exists; setup_env.sbatch creates it.
if conda env list | awk '{print $1}' | grep -qx "$U2T_ENV"; then
    conda activate "$U2T_ENV"
fi

# --- Misc ----------------------------------------------------------------
# Avoid huge tokenizer warnings on each fork.
export TOKENIZERS_PARALLELISM=false
# Force torch + datasets caches into /lus
export TMPDIR=${TMPDIR:-/lus/lfs1aip2/projects/u6hx/huanan/.cache/tmp}
mkdir -p "$TMPDIR"

# Make python see project src/
export PYTHONPATH=$PROJECT_PATH:${PYTHONPATH:-}

echo "[env] PROJECT_PATH=$PROJECT_PATH"
echo "[env] CONDA_PREFIX=${CONDA_PREFIX:-<not active>}"
echo "[env] python=$(command -v python || echo none)"
