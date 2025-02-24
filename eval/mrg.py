import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from green_score_accelerate import GREEN
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from torch.utils.data import DataLoader
from src.dataset.fused_dataset import FusedDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config
import textwrap


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

def find_all_linear_names(model):
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name)
    return list(lora_module_names)

class LlamedModel:
    def __init__(self, model_path: str, lora_weight_path: str = None):
        self.tokenizer =  AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,
            pad_token="<unk>",
            trust_remote_code=True
        )

        lamed_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if lora_weight_path:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=find_all_linear_names(lamed_model),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("Adding LoRA adapters only on LLM.")
            lamed_model = get_peft_model(lamed_model, lora_config)
            print("Load weights with LoRA")
            state_dict = torch.load(lora_weight_path, map_location=device)
            lamed_model.load_state_dict(state_dict, strict=True)
            print("Merge weights with LoRA")
            lamed_model = lamed_model.merge_and_unload()
        
        self.model = lamed_model.to(device).eval()

    def inference(self, image, question, temperature=1.0, top_p=0.9):

        image_tokens = "<im_patch>" * 256
        input_txt = image_tokens + question
        input_id = self.tokenizer(
            input_txt, add_special_tokens=False, max_length=1024, truncation=True, padding="max_length", return_tensors="pt", padding_side="right",
        )['input_ids'].to(device)
        question_ids = self.tokenizer(question_ids, add_special_tokens=False, max_length=1024, truncation=True, padding="max_length", return_tensors="pt", padding_side="right")['input_ids']
        generation = self.model.generate(image.to(device), input_id, question_ids=question_ids, max_new_tokens=768,
                                            do_sample=True, top_p=top_p, temperature=temperature)

        return self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

def check_character_and_length(answer):
    for ch in answer:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    if len(answer.replace(" ","")) < 20:
        return False
    return True

def mrg_annotation(dataloader, lamed_model):
    gt_report = []
    pred_report= []

    num = 0
    for batch in tqdm(dataloader):
        if batch is None:
            continue
        image = batch["image"]
        question = batch["question"][0]
        while True:
            pred = lamed_model.inference(image, question).strip()
            if check_character_and_length(pred):
                break

        gt_report.append(batch["answer"][0])
        pred_report.append(pred)
        num += 1
        if num == 10:
            break

    green_model_path = "/import/c4dm-04/siyoul/Med3DLLM/pretrained_models/GREEN-RadLlama2-7b"
    green_model = GREEN(green_model_path)
    mean, std, green_score_list, summary, result_df = green_model(refs=gt_report, hyps=pred_report)
    return mean
    
if __name__ == "__main__":
    lamed_model_path = config["project_path"] + "/checkpoint/amosmm_chatgpt_llama3.2_1b_l3dt_lora_0217@bs1_acc1_ep16_lr2e5_ws4_fused/checkpoint-18000"
    lora_weight_path = None
    llamed_model = LlamedModel(lamed_model_path, lora_weight_path)

    val_base_path = config["project_path"] + '/datasets'
    val_jsonl_path = config["project_path"] + '/datasets/Fused_Dataset/val/amos_mm_findings.jsonl'
    dataset = FusedDataset(
        val_base_path, 
        val_jsonl_path, 
        llamed_model.tokenizer, 
        max_length=1024, 
        image_tokens_num=256, 
        data_type="valuation",
        enable_linear_3d_tokenizer=False
    )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    mean = mrg_annotation(dataloader, llamed_model)
    print("Checkpoint: ", lamed_model_path.split("/")[-2])
    print("Mean: ", mean)