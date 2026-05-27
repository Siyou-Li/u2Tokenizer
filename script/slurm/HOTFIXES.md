# Bring-up Hotfixes (Cray / GH200 / aarch64)

This file records the 15 fixes applied while getting the upstream code to run
end-to-end on an aarch64 + GH200 SLURM cluster. New clones should already
contain all of these; this is for posterity / regression reference.

## A. Shell / SLURM

| # | Symptom | Fix | Location |
|---|---|---|---|
| 1 | uv `.venv` shadows conda on PATH | strip `$VIRTUAL_ENV/bin` before sourcing conda | `script/slurm/_env.sh` |
| 2 | `--qos=workq_qos` rejected | only `normal` QOS available — line removed | `train_stage1/2.sbatch` |

## B. Python deps (aarch64 / Qwen3 / new trainer APIs)

| # | Symptom | Fix | Location |
|---|---|---|---|
| 3 | `torch==2.5.1` not on cu126 channel for aarch64 | install `torch 2.6.0+cu126` instead | `setup_env.sbatch` |
| 4 | `cannot import name Qwen3Config from transformers` | upgrade `transformers 4.46.1 → 4.54.1` + `tokenizers 0.20 → 0.21.4` | `requirements-aarch64.txt` |
| 5 | `Accelerator.unwrap_model() got unexpected keyword 'keep_torch_compile'` | upgrade `accelerate 1.0.1 → 1.13.0` | `requirements-aarch64.txt` |
| 6 | `cannot import name 'flush_left' from trl.trainer.utils` | upgrade `trl 0.9.6 → 0.19.1` | `requirements-aarch64.txt` |

## C. Config / dataset path

| # | Symptom | Fix | Location |
|---|---|---|---|
| 7 | `KeyError: 'project_path'` at import | add `project_path` to `config/project.json` | `config/project.json` |
| 8 | HF repo `Qwen/Qwen3-1.7B-Instruct` 404 | use `Qwen/Qwen3-1.7B` (Qwen3 default is already instruct) | `download_assets.sbatch`, `train_stage1.sbatch` |
| 9 | CT-RATE-Thinking val jsonl points at `CT-RATE/dataset/train/valid_X/...` (typo) | rewrite to `CT-RATE/dataset/valid_fixed/valid_X/...` when building smoke jsonl | smoke jsonl builder |
| 10 | huggingface-cli `--include "dataset/valid/*"` matched 0 files | use `dataset/**` glob; or iterate file list + `hf_hub_download` | `download_data.sbatch` |

## D. Trainer-side patches (Stage 1)

| # | Symptom | Fix | Location |
|---|---|---|---|
| 11 | `--enable_rpe True` not in ModelArguments | drop flag; control via `--attn_type rope` instead | `train_stage1.sbatch` |
| 12 | DDP "params unused" then "marked ready twice" | add `--ddp_find_unused_parameters True` + `--gradient_checkpointing_kwargs '{"use_reentrant": false}'` | `train_stage1.sbatch` |
| 13 | `trainer.tokenizer` returns `None` (deprecated in transformers ≥4.46) | fall back to `trainer.processing_class` in `safe_save_model_for_hf_trainer` | `src/train/train_stage1.py` |

## E. Trainer-side patches (Stage 2 — DPO)

| # | Symptom | Fix | Location |
|---|---|---|---|
| 14 | `ModuleNotFoundError: No module named 'train'` | absolute import `from src.train.dpo_u2trainer ...` | `src/train/train_stage2.py` |
| 15 | unrecognised CLI args `--lora_enable --vision_tower --pretrain_vision_model --enable_diffts --enable_dmtp` | Stage 2's `ModelArguments` doesn't define them — flags removed | `train_stage2.sbatch` |
| 16 | OOM on 95GB GH200 with policy + reference model | enable DeepSpeed ZeRO-2 (no CPU offload) via `--deepspeed config/ds_config_zero2_bf16.json`; bump GPUs to 4; cut `max_length 1024 → 512`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `train_stage2.sbatch` + `config/ds_config_zero2_bf16.json` |
| 17 | DeepSpeed wraps `nn.Module.__init__` and breaks einops `Rearrange(p1=...)` | avoid ZeRO-3 (which monkey-patches init); ZeRO-2 leaves init alone | `ds_config_zero2_bf16.json` |
| 18 | `cpu_adam` C++ extension fails to build (system `c++` < GCC 9) | drop `offload_optimizer.device=cpu` | `ds_config_zero2_bf16.json` |
| 19 | `int('90a')` parsing GH200 sm_90a when building `fused_adam` | remove the `"optimizer": {...}` block from ds_config (HF Trainer will use torch AdamW) | `ds_config_zero2_bf16.json` |
| 20 | `u2DPOTrainer.concatenated_forward() got unexpected kwarg 'is_ref_model'` (trl 0.19) | add `is_ref_model: bool = False` to the override | `src/train/dpo_u2trainer.py` |
| 21 | `AttributeError: 'TrainingArguments' has no 'lora_enable'` during save | `getattr(self.args, "lora_enable", False)` | `src/train/dpo_u2trainer.py` |

## How to verify after a fresh clone

```
sbatch script/slurm/setup_env.sbatch        # builds u2t conda env, ~5 min
sbatch script/slurm/download_assets.sbatch  # Qwen3-1.7B + M3D-CLIP, ~1 min
# (datasets already on shared /lus/lfs1aip2/projects/u6hx/datasets/)

# SFT smoke (10 steps, 16 samples)
CHECKPOINT_NAME=sft_smoke_qwen3 \
TRAIN_JSONL=$PWD/datasets/Fused_Dataset/train/ct_rate_smoke.jsonl \
VAL_JSONL=$PWD/datasets/Fused_Dataset/val/ct_rate_smoke.jsonl \
MAX_STEPS=10 SAVE_STEPS=10 EVAL_STEPS=10 \
sbatch --gres=gpu:2 --time=01:00:00 script/slurm/train_stage1.sbatch

# Eval smoke (HF public ckpt, 3 samples, BLEU+ROUGE)
MODEL_PATH=AlpachinoNLP/u2Qwen3-1.7B-Instruct \
DATA_DIR=$PWD/datasets/CT-RATE/dataset/valid_fixed \
MAX_SAMPLES=3 METRICS="bleu rouge" \
sbatch --gres=gpu:2 --time=01:00:00 script/slurm/eval_ct_rate.sbatch

# DPO data + DPO smoke
MAX_SAMPLES=4 N_CANDIDATES=3 sbatch --gres=gpu:2 script/slurm/build_dpo_pairs.sbatch
RUN_NAME=dpo_smoke_qwen3 \
SFT_CKPT=AlpachinoNLP/u2Qwen3-1.7B-Instruct \
DPO_JSONL=$PWD/datasets/Fused_Dataset/train/ct_rate_dpo_pairs.jsonl \
VAL_JSONL=$PWD/datasets/Fused_Dataset/val/ct_rate_smoke.jsonl \
NUM_EPOCHS=1 sbatch --gres=gpu:4 --time=01:00:00 script/slurm/train_stage2.sbatch
```

Expected timings (per smoke job, end-to-end including queue):
- setup_env: ~5 min
- download_assets: ~1 min
- train_stage1 smoke: ~2 min (loss 2.3-2.6)
- eval smoke (3 samples): ~1.5 min (BLEU ~46, R-L ~35)
- build_dpo_pairs (4×3): ~2.5 min
- train_stage2 smoke: ~2.5 min (loss ~0.34, +ve reward margin)
