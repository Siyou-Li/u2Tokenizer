from green_score_accelerate import GREEN
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset.fused_dataset_m3d import FusedDataset
from src.dataset.amos_mm_monai_dataset import MRGDataset
from tqdm import tqdm
from src.utils.utils import normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
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

def load_model(green_model_path, u2_model_path, lora_weight_path=None):
    green_model = GREEN(
        green_model_path,
        output_dir="."
        )
    #green_model = green_model.to("cuda:{}".format(device))
    tokenizer = AutoTokenizer.from_pretrained(
        u2_model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        pad_token="<unk>",
        trust_remote_code=True
    )
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
        # u2_model.print_trainable_parameters()
        print("Load weights with LoRA")
        state_dict = torch.load(lora_weight_path, map_location="cuda")
        u2_model.load_state_dict(state_dict, strict=True)
        print("Merge weights with LoRA")
        u2_model = u2_model.merge_and_unload()
        
    
    with torch.no_grad():
        u2_model = u2_model.to("cuda")
    return green_model, tokenizer, u2_model

def green_score(pred_report, gt_report, categorize, green_model):

    # Initialize list to store scores for non-empty GT parts
    mean, std, green_score_list, summary, result_df = green_model(refs=gt_report, hyps=pred_report)
    print("GREEN Score Summary: ", summary)
    #print("GREEN Score Result: ", result_df)
    #result_df.to_csv("./eval/result_{}.csv".format(categorize[1]))
    # Print the average score
    print("{} - Average GREEN Score: {}".format(categorize[1],mean))
    return mean

def inference(image, input_ids, question_ids, model, temperature=1.0, top_p=0.9):

    generation = model.generate(image.to(device), input_ids.to(device), question_ids=question_ids.to(device), max_new_tokens=1024,
                                        do_sample=True, top_p=top_p, temperature=temperature)

    return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def mrg_annotation(dataloader, green_model, tokenizer, u2_model):
    
    gt_report = []
    pred_report= []
    num = 0
    for batch in tqdm(dataloader):
        if batch is None:
            continue
        #print(batch)
        #try:
        gt_report.append(batch["answer"][0])
        image = batch["image"]
        input_ids = batch["input_id"]
        question_ids = batch["question_ids"]
        
        pred = inference(image, input_ids, question_ids, u2_model)
        print(pred)
        pred_report.append(pred)
        if num == 10:
            break
        # except Exception as e:
        #     print(e)
        num += 1  
    mean_green_score = green_score(pred_report, gt_report, green_model)
    return mean_green_score
    
def woker(green_model, tokenizer, u2_model):
    
    val_base_path = '/import/c4dm-04/siyoul/u2Tokenizer/datasets'
    val_jsonl_path = '/import/c4dm-04/siyoul/u2Tokenizer/datasets/Fused_Dataset/val/amos_mm_findings.jsonl'
    dataset = FusedDataset(
        val_base_path, 
        val_jsonl_path, 
        tokenizer, 
        max_length=1024, 
        image_tokens_num=256, 
        data_type="valuation",
        enable_u2tokenizer=True
    )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    mean_green_score = mrg_annotation(dataloader, green_model, tokenizer, u2_model)
    # mean_green_scores.append(mean_green_score)
    # for categorize, mean_green_score in zip(["Chest","Abdomen","Pelvis"], mean_green_scores):
    return "Average GREEN Score: {}".format(mean_green_score)

if __name__ == "__main__":
    green_model_path="/import/c4dm-04/siyoul/u2Tokenizer/pretrained_models/GREEN-RadLlama2-7b"
    u2_model_path = "/import/c4dm-04/siyoul/u2Tokenizer/checkpoint/amosmm_chatgpt_llama3.2_1b_u2t_lora_0217@bs1_acc1_ep16_lr2e5_ws4_fused/checkpoint-18000"
    lora_weight_path = None
    green_model, tokenizer, u2_model = load_model(green_model_path, u2_model_path, lora_weight_path)
    print(u2_model_path.split("/")[-2])
    results = []
    results.append(woker(green_model, tokenizer, u2_model))
    for result in results:
        print(result)
    print(u2_model_path.split("/")[-2])