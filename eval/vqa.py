from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset.fused_dataset import FusedDataset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

def load_model(u2_model_path, lora_weight_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        u2_model_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
        pad_token="[PAD]",
        trust_remote_code=True
    )
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    u2_model = AutoModelForCausalLM.from_pretrained(
        u2_model_path,
        trust_remote_code=True,
    )
    if lora_weight_path:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=find_all_linear_names(u2_model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters only on LLM.")
        u2_model = get_peft_model(u2_model, lora_config)
        u2_model.print_trainable_parameters()
        print("Load weights with LoRA")
        state_dict = torch.load(lora_weight_path, map_location="cuda")
        u2_model.load_state_dict(state_dict, strict=True)
        print("Merge weights with LoRA")
        u2_model = u2_model.merge_and_unload()
        
    u2_model = u2_model.to("cuda")
    u2_model.eval()
    return tokenizer, u2_model

def inference(input_image, input_id, tokenizer, u2_model, temperature=1.0, top_p=0.9):

    input_id = tokenizer(input_id, return_tensors="pt")['input_ids'].to("cuda")
    image_pt = torch.from_numpy(input_image).to("cuda")

    generation = u2_model.generate(image_pt, input_id, max_new_tokens=1,
                                        do_sample=True, top_p=top_p, temperature=temperature)

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    return output_str, None
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def vqa_annotation(dataloader, tokenizer, u2_model):
    
    gt_report = []
    pred_report= []
    num = 0
    for batch in tqdm(dataloader):
        if batch is None:
            continue
        #print(batch)
        try:
            gt = batch["answer"][0]
            gt_report.append(gt)
            input_image = batch["image"].numpy()
            input_id = batch["question"][0]
            pred, _ = inference(input_image, input_id, tokenizer, u2_model)
            pred = pred.strip()[0]
            pred_report.append(pred)
            print("GT:", gt, "Pred:", pred, bool(gt == pred), len(gt), len(pred))
        except Exception as e:
            print(e)
        num += 1
    
    length = len(gt_report)
    num_correct = 0
    for gt, pred in zip(gt_report, pred_report):
        gt = gt.lower()
        pred = pred.lower()
        
        if gt == pred:
            num_correct += 1
    accuracy = num_correct / length
    
    return accuracy
    
def woker(tokenizer, u2_model):
    
    val_base_path = '/import/c4dm-04/siyoul/u2Tokenizer/datasets'
    val_jsonl_path = '/import/c4dm-04/siyoul/u2Tokenizer/datasets/Fused_Dataset/val/amos_mm_qa.jsonl'
    dataset = FusedDataset(
        val_base_path, 
        val_jsonl_path, 
        tokenizer, 
        max_length=2048, 
        image_tokens_num=256, 
        data_type="valuation"
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    accuracy = vqa_annotation(dataloader, tokenizer, u2_model)
    return accuracy

if __name__ == "__main__":
    #green_model_path="/import/c4dm-04/siyoul/u2Tokenizer/pretrained_models/GREEN-RadLlama2-7b"
    u2_model_path = "/import/c4dm-04/siyoul/u2Tokenizer/checkpoint/Med3dLLM_0113_mrg_llama3.2@1b_bs8_acc1_ep16_lr2e5_ws4_fused/checkpoint-218000"
    tokenizer, u2_model = load_model(u2_model_path)
    accuracy = woker(tokenizer, u2_model)
    print("Checkpoint: ", u2_model_path.split("/")[-2])
    print("Accuracy: ", accuracy)
