#!/bin/bash
#SBATCH --job-name=u2t-setup-env
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=logs/setup_env-%j.out
#SBATCH --error=logs/setup_env-%j.err

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

# --- 1. Create env if absent --------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$U2T_ENV"; then
    echo "[setup] creating conda env $U2T_ENV (python 3.11) ..."
    conda create -y -n "$U2T_ENV" python=3.11
fi
conda activate "$U2T_ENV"
echo "[setup] python -> $(command -v python)"

# --- 2. Install PyTorch aarch64 + CUDA 12.6 ------------------------------
# Use the official cu126 index — matches the cluster's loaded cuda/12.6.
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# --- 3. Install the rest of the deps -------------------------------------
python -m pip install -r script/slurm/requirements-aarch64.txt

# Verify the few versions that bit us during initial bring-up.
python - <<'PY'
import importlib, sys
need = {
    "transformers": "4.54.1",
    "tokenizers":   "0.21",     # >=0.21, <0.22
    "accelerate":   "1.13.0",
    "trl":          "0.19.1",
}
ok = True
for name, want in need.items():
    got = importlib.import_module(name).__version__
    print(f"  {name:>14s}: {got}  (need {want})")
    if not got.startswith(want.split('.')[0]):
        ok = False
if not ok:
    print("WARN: at least one library version is unexpected; see HOTFIXES.md",
          file=sys.stderr)
PY

# --- 4. Sanity check -----------------------------------------------------
python - <<'PY'
import torch, transformers, accelerate, monai, nibabel
print("torch        :", torch.__version__, "cuda", torch.version.cuda)
print("torch.cuda   :", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print("device       :", torch.cuda.get_device_name(0))
print("transformers :", transformers.__version__)
print("accelerate   :", accelerate.__version__)
print("monai        :", monai.__version__)
print("nibabel      :", nibabel.__version__)
PY

echo "[setup] DONE"
