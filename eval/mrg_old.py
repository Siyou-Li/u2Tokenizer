from green_score_accelerate import GREEN
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset.amos_mm_monai_dataset import MRGDataset
from tqdm import tqdm
from src.utils.utils import normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def load_model(green_model_path, lamed_model_path, lora_weight_path=None):
    green_model = GREEN(
        green_model_path,
        output_dir="."
        )
    #green_model = green_model.to("cuda:{}".format(device))
    tokenizer = AutoTokenizer.from_pretrained(
        lamed_model_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
        pad_token="<unk>",
        trust_remote_code=True
    )
    lamed_model = AutoModelForCausalLM.from_pretrained(
        lamed_model_path,
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
        # lamed_model.print_trainable_parameters()
        print("Load weights with LoRA")
        state_dict = torch.load(lora_weight_path, map_location="cuda")
        lamed_model.load_state_dict(state_dict, strict=True)
        print("Merge weights with LoRA")
        lamed_model = lamed_model.merge_and_unload()
        
    
    with torch.no_grad():
        lamed_model = lamed_model.to("cuda")
    return green_model, tokenizer, lamed_model

def green_score(pred_report, gt_report, categorize, green_model, image_paths):

    # Initialize list to store scores for non-empty GT parts
    mean, std, green_score_list, summary, result_df = green_model(refs=gt_report, hyps=pred_report)
    print("GREEN Score Summary: ", summary)
    #print("GREEN Score Result: ", result_df)
    #result_df.to_csv("./eval/result_{}.csv".format(categorize[1]))
    # Print the average score
    print("{} - Average GREEN Score: {}".format(categorize[1],mean))
    return mean

def inference(input_image, input_id, tokenizer, lamed_model, temperature=1.0, top_p=0.9):

    # ## filter out special chars
    # input_str = bleach.clean(input_str)
    # # Model Inference
    # prompt = "<im_patch>" * 256 + input_str

    input_id = tokenizer(input_id, return_tensors="pt")['input_ids'].to("cuda")
    image_pt = torch.from_numpy(input_image).to("cuda")

    generation = lamed_model.generate(image_pt, input_id, max_new_tokens=512,
                                        do_sample=True, top_p=top_p, temperature=temperature)

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    return output_str, None
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def mrg_annotation(dataloader, categorize, green_model, tokenizer, lamed_model):
    
    gt_report = []
    pred_report= []
    image_paths = []
    num = 0
    for batch in tqdm(dataloader):
        if batch is None:
            continue
        #print(batch)
        #try:
        gt_report.append(batch["answer"][0])
        input_image = batch["image"].numpy()
        #print(input_image.shape)
        input_id = batch["question"][0]
        image_path = batch["image_path"][0]
        
        image_paths.append(image_path)
        pred, _ = inference(input_image, input_id, tokenizer, lamed_model)
        pred_report.append(pred)
        if num == 10:
            break
        # except Exception as e:
        #     print(e)
        num += 1
    lamed_model.to("cpu")    
    torch.cuda.empty_cache()
    mean_green_score = green_score(pred_report, gt_report, categorize, green_model, image_paths)
    lamed_model.to("cuda")
    return mean_green_score
    
def woker(green_model, tokenizer, lamed_model, categorize):
    
    image_dir = '/import/c4dm-04/siyoul/Med3DLLM/datasets/AMOS-MM/'
    json_path = '/import/c4dm-04/siyoul/Med3DLLM/datasets/AMOS-MM/dataset.json'
    dataset = MRGDataset(
        image_dir, json_path, tokenizer, 512, \
        categorize=categorize, data_type="validation"
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    mean_green_score = mrg_annotation(dataloader, categorize, green_model, tokenizer, lamed_model)
    # mean_green_scores.append(mean_green_score)
    # for categorize, mean_green_score in zip(["Chest","Abdomen","Pelvis"], mean_green_scores):
    return "{} - Average GREEN Score: {}".format(categorize[1], mean_green_score)

if __name__ == "__main__":
    categorizes = [["findings","chest"], ["findings","abdomen"], ["findings","pelvis"]]
    devices = [0, 1, 2]
    threads = []
    green_model_path="/import/c4dm-04/siyoul/Med3DLLM/pretrained_models/GREEN-RadLlama2-7b"
    lamed_model_path = "/import/c4dm-04/siyoul/Med3DLLM/checkpoint/Med3dLLM-1112-MRG-CoT/checkpoint-13179"
    lora_weight_path = None
    green_model, tokenizer, lamed_model = load_model(green_model_path, lamed_model_path, lora_weight_path)
    print(lamed_model_path.split("/")[-2])
    results = []
    for categorize in categorizes:
        results.append(woker(green_model, tokenizer, lamed_model, categorize))
    for result in results:
        print(result)
    print(lamed_model_path.split("/")[-2])