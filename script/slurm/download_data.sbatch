#!/bin/bash
#SBATCH --job-name=u2t-dldata
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/dldata-%j.out
#SBATCH --error=logs/dldata-%j.err

# Downloads CT-RATE volumes + CT-RATE-Thinking JSONL bundle.
# CT-RATE is large (~TBs). For an initial smoke test pass --max-files via env.

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

python -m pip install -q "huggingface_hub[cli]" datasets

export THINK_DIR="$PROJECT_DATA/CT-RATE-Thinking"
export CTRATE_DIR="$PROJECT_DATA/CT-RATE"

# --- CT-RATE-Thinking (lightweight: JSONLs only) ------------------------
if [ ! -d "$THINK_DIR/.git" ] && [ -z "$(ls -A "$THINK_DIR" 2>/dev/null || true)" ]; then
    echo "[dl] CT-RATE-Thinking -> $THINK_DIR"
    huggingface-cli download AlpachinoNLP/CT-RATE-Thinking \
        --repo-type dataset --local-dir "$THINK_DIR" \
        --local-dir-use-symlinks False
else
    echo "[dl] $THINK_DIR already populated"
fi

# --- CT-RATE volumes -----------------------------------------------------
# CT-RATE layout (per huggingface listing):
#   dataset/{train,valid}/<pid>/<study>/<series>.nii.gz
#   dataset/metadata/*.csv
#   dataset/radiology_text_reports/*.csv
#   dataset/multi_abnormality_labels/*.csv
#
# Modes:
#   SMOKE=1 (default)  : download only the first ~20 validation volumes
#   SMOKE=0 FULL=0     : full validation split (~5k volumes, ~hundreds of GB)
#   SMOKE=0 FULL=1     : everything (train + valid; ~TBs)
SMOKE=${SMOKE:-1}
FULL=${FULL:-0}

mkdir -p "$CTRATE_DIR/dataset"

# Always pull labels + reports (small).
huggingface-cli download ibrahimhamamci/CT-RATE \
    --repo-type dataset --local-dir "$CTRATE_DIR" \
    --local-dir-use-symlinks False \
    --include "dataset/metadata/**" \
              "dataset/radiology_text_reports/**" \
              "dataset/multi_abnormality_labels/**" \
              "*.csv" "README.md"

if [ "$SMOKE" = "1" ] && [ "$FULL" = "0" ]; then
    echo "[dl] SMOKE mode — fetching first 20 validation volumes"
    python - <<'PY'
import os, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
files = api.list_repo_files("ibrahimhamamci/CT-RATE", repo_type="dataset")
valid = [f for f in files
         if f.startswith("dataset/valid/") and f.endswith(".nii.gz")
         and "ts_seg" not in f]
valid.sort()
take = valid[:20]
local = os.environ["CTRATE_DIR"]
for f in take:
    print("  ", f, flush=True)
    hf_hub_download(repo_id="ibrahimhamamci/CT-RATE", filename=f,
                    repo_type="dataset", local_dir=local,
                    local_dir_use_symlinks=False)
print(f"[dl] fetched {len(take)} validation volumes into {local}")
PY
elif [ "$FULL" = "1" ]; then
    echo "[dl] FULL mode — downloading entire dataset (TB-scale)"
    huggingface-cli download ibrahimhamamci/CT-RATE \
        --repo-type dataset --local-dir "$CTRATE_DIR" \
        --local-dir-use-symlinks False \
        --include "dataset/**"
else
    echo "[dl] VALID-FULL mode — downloading whole validation split"
    huggingface-cli download ibrahimhamamci/CT-RATE \
        --repo-type dataset --local-dir "$CTRATE_DIR" \
        --local-dir-use-symlinks False \
        --include "dataset/valid/**"
fi

du -sh "$THINK_DIR" "$CTRATE_DIR"
echo "[dl] DONE"
