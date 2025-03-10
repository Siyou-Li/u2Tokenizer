import os
from typing import Any, Optional, Union
import torch
import numpy as np
from train.dpo_u2trainer import u2DPOTrainer
from trl.trainer.utils import pad
from trl.trainer.dpo_config import DPOConfig
from dataclasses import dataclass, field
from src.dataset.fused_dataset_dpo import FusedDataset
from transformers.data.data_collator import DataCollatorMixin
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from datasets import Dataset
from src.utils.u2Transform import u2Transform

image_transforms = u2Transform(mode='bilinear', data_type="training")
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
from torch.distributed.elastic.multiprocessing.errors import record
@record
@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    # wandb
    wandb_project_name: Optional[str] = field(default="AMOS-MM", metadata={"help": "wandb project name"})
    wandb_run_name: Optional[str] = field(default="test", metadata={"help": "wandb run name"})

    enable_u2tokenizer: Optional[bool] = field(default=False, metadata={"help": "Enable linear 3d tokenizer."})
    model_init_kwargs = None

@dataclass
class DataArguments:
    train_jsonl_path: str = field(default="", metadata={"help": "Path to caption data."})
    train_base_path: str = field(default="", metadata={"help": "Path to image data."})
    val_jsonl_path: str = field(default="", metadata={"help": "Path to caption data."})
    val_base_path: str = field(default="", metadata={"help": "Path to image data."})

@dataclass
class TrainingArguments(DPOConfig):

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
    warmup_num_steps: int = 100
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = False # fast
    dataloader_num_workers: int = 2
    report_to: str = "tensorboard"
    beta: float = 0.1
    model_init_kwargs = None
    ref_model_init_kwargs = None
    

@dataclass
class DataCollatorForPreference(DataCollatorMixin):

    pad_token_id: int = 128015
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        #print(examples)
        # Pad
        output = {}
        output["image"] = [example["image"] for example in examples]
        output["prompt_question_ids"] = [torch.tensor(example["prompt_question_ids"]) for example in examples]
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)

        return output
    


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    if local_rank == 0:
        wandb.init(project=model_args.wandb_project_name, name=model_args.wandb_run_name)

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)
    rank0_print("special tokens: ", tokenizer.special_tokens_map)

    rank0_print("="*20 + " Model preparation " + "="*20)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
        )
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
        )

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    max_length=data_args.max_length
    image_tokens_num=data_args.proj_out_num
    # train_dataset = CapDataset(data_args, tokenizer, mode='train')
    # eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
    train_dataset = FusedDataset(data_args.train_base_path, data_args.train_jsonl_path, tokenizer, max_length=max_length, image_tokens_num=image_tokens_num, data_type="training", enable_u2tokenizer=model_args.enable_u2tokenizer, local_rank=local_rank)
    eval_dataset = FusedDataset(data_args.val_base_path, data_args.val_jsonl_path, tokenizer, max_length=max_length, image_tokens_num=image_tokens_num, data_type="valuation", enable_u2tokenizer=model_args.enable_u2tokenizer, local_rank=local_rank)
    def gen_train():
        for idx in range(len(train_dataset)):
            yield train_dataset[idx]
    def gen_eval():
        for idx in range(20):
            yield eval_dataset[idx]
    train_dataset = Dataset.from_generator(gen_train)#, {"image": torch.float32, "question_ids": torch.int64, "prompt_input_ids": torch.int64, "chosen_input_ids": torch.int64, "rejected_input_ids": torch.int64})
    #eval_dataset = Dataset.from_generator(gen_eval)#.generator, {"image": torch.float32, "question_ids": torch.int64, "prompt_input_ids": torch.int64, "chosen_input_ids": torch.int64, "rejected_input_ids": torch.int64})
    data_collator = DataCollatorForPreference()
    
    rank0_print("="*20 + " Training " + "="*20)
    trainer = u2DPOTrainer(
                        model = model,
                        ref_model = model_ref,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=train_dataset,
                        processing_class=tokenizer,
                      )
    
    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
