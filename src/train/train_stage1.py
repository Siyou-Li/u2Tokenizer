
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from dataclasses import dataclass, field
from src.dataset import FusedDataset
from src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset
from src.model.language_model import u2LlamaForCausalLM, u2Phi3ForCausalLM, u2Qwen3ForCausalLM
from src.train.sft_u2Trainer import u2Trainer
import os
import torch._dynamo
torch._dynamo.config.suppress_errors = False
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME


kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

from torch.distributed.elastic.multiprocessing.errors import record
@record
@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3, qwen3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})
    checkpoint_path: str = field(default=None, metadata={"help": "Path to Checkpoint"})
    
    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(256, 256, 32))
    input_image_size: tuple = field(default=(256, 256, 32))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # wandb
    wandb_project_name: Optional[str] = field(default="AMOS-MM", metadata={"help": "wandb project name"})
    wandb_run_name: Optional[str] = field(default="test", metadata={"help": "wandb run name"})

    # u2tokenizer config
    enable_u2tokenizer: bool = False
    u2t_num_heads: int = 8
    u2t_num_layers: int = 4
    u2t_top_k: int = 1024
    use_multi_scale: bool = True
    num_3d_query_token: int = 256
    attn_type: str = field(default="rma", metadata={"help": "Attention type for tokenizer [rma, rope]"})
    enable_diffts: bool = False
    enable_dmtp: bool = False

@dataclass
class DataArguments:
    train_jsonl_path: str = field(default="", metadata={"help": "Path to caption data."})
    train_base_path: str = field(default="", metadata={"help": "Path to image data."})
    val_jsonl_path: str = field(default="", metadata={"help": "Path to caption data."})
    val_base_path: str = field(default="", metadata={"help": "Path to image data."})
    
    # caption data
    data_root: str = field(default="/import/c4dm-04/siyoul/u2Tokenizer/datasets/M3D-Cap/", metadata={"help": "Root directory for all data."})
    cap_data_path: str = field(default="/import/c4dm-04/siyoul/u2Tokenizer/datasets/M3D-Cap/M3D_Cap_npy/M3D_Cap.json", metadata={"help": "Path to caption data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./u2/output/u2-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = False # fast
    dataloader_num_workers: int = 2
    report_to: str = "tensorboard"

def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def _is_resumable_trainer_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_trainer_state = os.path.isfile(os.path.join(path, "trainer_state.json"))
    has_weights = any(
        os.path.isfile(os.path.join(path, fname))
        for fname in (
            WEIGHTS_NAME,
            SAFE_WEIGHTS_NAME,
            "pytorch_model.bin",
            "model.safetensors",
        )
    )
    return has_trainer_state and has_weights


def _resolve_resume_checkpoint(resume_from_checkpoint: Optional[str], output_dir: str) -> Optional[str]:
    if not resume_from_checkpoint:
        return None

    resume_from_checkpoint = os.path.expanduser(resume_from_checkpoint)
    if not os.path.isdir(resume_from_checkpoint):
        raise ValueError(f"`resume_from_checkpoint` must be an existing directory, got: {resume_from_checkpoint}")

    if _is_resumable_trainer_checkpoint(resume_from_checkpoint):
        return resume_from_checkpoint

    last_checkpoint = get_last_checkpoint(resume_from_checkpoint)
    if last_checkpoint and _is_resumable_trainer_checkpoint(last_checkpoint):
        return last_checkpoint

    last_checkpoint = get_last_checkpoint(output_dir) if output_dir else None
    if last_checkpoint and _is_resumable_trainer_checkpoint(last_checkpoint):
        return last_checkpoint

    raise ValueError(
        "Could not find a resumable checkpoint. Expected either a Trainer checkpoint directory "
        "(containing `trainer_state.json` and model weights), or a directory containing `checkpoint-*` subfolders."
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
    
    # save tokenizer
    trainer.tokenizer.save_pretrained(output_dir)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module', 'u2tokenizer']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


@dataclass
class DataCollator:
    def __init__(self, seg_enable=False):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        images, input_ids, labels, attention_mask, question_ids = tuple(
            [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'question_ids'))

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
        question_ids = torch.cat([_.unsqueeze(0) for _ in question_ids], dim=0)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            question_ids=question_ids,
        )
        return return_dict


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    if local_rank == 0:
        wandb.init(project=model_args.wandb_project_name, name=model_args.wandb_run_name)

    resume_from_checkpoint = training_args.resume_from_checkpoint or model_args.checkpoint_path
    resume_from_checkpoint = _resolve_resume_checkpoint(resume_from_checkpoint, training_args.output_dir)
    if resume_from_checkpoint:
        rank0_print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)
    rank0_print("special tokens: ", tokenizer.special_tokens_map)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            rank0_print("Base model: ", model_args.model_name_or_path)
            model = u2LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        elif 'phi3' in model_args.model_type:
            rank0_print("Base model: ", model_args.model_name_or_path)
            model = u2Phi3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
                )
            
        elif 'qwen3' in model_args.model_type:
            rank0_print("Base model: ", model_args.model_name_or_path)
            model = u2Qwen3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    else:
        rank0_print("No vision tower is initialized.")

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm and not resume_from_checkpoint:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")
    elif model_args.pretrain_mllm and resume_from_checkpoint:
        rank0_print("Ignoring `pretrain_mllm` because training is resuming from a checkpoint.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module', 'u2tokenizer']]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()


    # ckpt = torch.load("PATH", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    max_length=data_args.max_length
    image_tokens_num=data_args.proj_out_num
    use_thinking = bool(model_args.model_type and "qwen3" in model_args.model_type.lower())
    # train_dataset = CapDataset(data_args, tokenizer, mode='train')
    # eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
    train_dataset = FusedDataset(
        data_args.train_base_path,
        data_args.train_jsonl_path,
        tokenizer,
        max_length=max_length,
        image_tokens_num=image_tokens_num,
        data_type="training",
        enable_u2tokenizer=model_args.enable_u2tokenizer,
        local_rank=local_rank,
        use_chat_template=False,
        use_thinking=use_thinking,
    )
    eval_dataset = FusedDataset(
        data_args.val_base_path,
        data_args.val_jsonl_path,
        tokenizer,
        max_length=max_length,
        image_tokens_num=image_tokens_num,
        data_type="valuation",
        enable_u2tokenizer=model_args.enable_u2tokenizer,
        local_rank=local_rank,
        use_chat_template=False,
        use_thinking=use_thinking,
    )
    data_collator = DataCollator()
    
    rank0_print("="*20 + " Training " + "="*20)
    trainer = u2Trainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics
                      )
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
