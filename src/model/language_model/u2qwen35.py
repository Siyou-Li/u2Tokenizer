"""Qwen3.5 adapter (SKELETON — NOT YET FUNCTIONAL).

Why this file is a stub
-----------------------
Qwen3.5 (released 2026-02) is *not* a drop-in for Qwen3:

1. **transformers support:** Native classes (`Qwen3_5Config`,
   `Qwen3_5TextConfig`, `Qwen3_5ForCausalLM`) only appear in transformers
   5.x (verified missing in 4.57.6, present in 5.5.4). Upgrading the entire
   training pipeline to transformers 5.x is a separate, invasive change
   (trl, accelerate, peft, etc. all need re-pinning).

2. **Architecture:** Qwen3.5 uses a *hybrid* attention stack — every 4th
   layer is `full_attention`, the rest are `linear_attention` (DeltaNet /
   gated-delta style SSM-like blocks). The u2 stack assumes a uniform
   transformer decoder for things like RPE injection on past_key_values
   and the dmtp / diff-time-step heads — those abstractions need to be
   audited for linear-attention layers.

3. **Native multimodal:** The 9B release on HF is
   `Qwen3_5ForConditionalGeneration` (image + video preprocessors baked
   in) with `image_token_id=248056` and a 248k vocab. Our u2 architecture
   inserts its *own* `<im_patch>` token and feeds CT features through a
   separate projector. Either we (a) extract only the text branch
   (`text_config.model_type == "qwen3_5_text"`) and lose the multimodal
   nicely-integrated tokenizer, or (b) replace the upstream image path
   with ours. (a) is simpler but requires confirming gradient flow.

How to finish this port
-----------------------
1. Upgrade env: `pip install "transformers>=5.5,<6" "tokenizers>=0.22"`.
   Run the SFT smoke against the current Qwen3 path first to confirm
   nothing regressed (the SFT trainer is the most exposed surface).
2. Replace `Qwen3Config / Qwen3Model / Qwen3ForCausalLM` below with
   `Qwen3_5TextConfig / Qwen3_5TextModel / Qwen3_5ForCausalLM` (text-only
   subclasses landed in 5.x).
3. Patch `src/model/u2_arch.py::u2MetaModel.prepare_inputs_labels_for_*`
   to tolerate linear-attention layers: their `past_key_values` is a
   custom `DeltaNetCache`, not a tuple of (k, v) tensors.
4. In `src/train/train_stage1.py::_resolve_dataset_thinking` register
   "qwen35" as another thinking-on family.
5. Add a script/slurm/train_stage1_qwen35.sbatch that pins
   `--model_type qwen35` and the new MODEL path.

Until those steps land, importing this module raises NotImplementedError
so SLURM jobs fail fast with a useful message.
"""

import warnings


class _Qwen35NotReady:  # pragma: no cover - intentional fail-fast
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "u2Qwen35ForCausalLM is a skeleton — see "
            "src/model/language_model/u2qwen35.py module docstring for the "
            "outstanding port work. Use --model_type qwen3 (Qwen3-1.7B / 4B "
            "/ 8B) until the upgrade lands."
        )


# Public names kept stable so __init__.py and train_stage1.py imports don't
# crash on first import; they only blow up if someone actually tries to
# instantiate the class.
u2Qwen35Config = _Qwen35NotReady
u2Qwen35Model = _Qwen35NotReady
u2Qwen35ForCausalLM = _Qwen35NotReady


def _maybe_register():
    """Try to register the model with AutoConfig/AutoModelForCausalLM.

    Will silently no-op when transformers does not expose Qwen3_5 classes
    (i.e. < 5.x), which is the normal state today.
    """
    try:
        import transformers  # noqa: F401
        from transformers import Qwen3_5TextConfig  # type: ignore  # noqa: F401
    except Exception:
        return
    warnings.warn(
        "transformers exposes Qwen3_5TextConfig but u2qwen35.py adapter "
        "is still a stub. Skipping registration.",
        RuntimeWarning,
        stacklevel=2,
    )


_maybe_register()
