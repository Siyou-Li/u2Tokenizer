from green_score import GREEN
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset.fused_dataset import FusedDataset
from tqdm import tqdm
from src.utils.utils import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
check_model_name = "/import/c4dm-04/siyoul/Med3DLLM/pretrained_models/Qwen2.5-1.5B-Instruct"
check_model = AutoModelForCausalLM.from_pretrained(
    check_model_name,
    torch_dtype="auto",
    device_map="auto"
).eval()
check_tokenizer = AutoTokenizer.from_pretrained(check_model_name)

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
        model_max_length=768,
        padding_side="right",
        use_fast=False,
        pad_token="<|endoftext|>",
        trust_remote_code=True
    )
    # special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    # tokenizer.add_special_tokens(
    #     special_token
    # )
    #tokenizer.add_tokens("[SEG]")

    # if tokenizer.unk_token is not None and tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token
    lamed_model = AutoModelForCausalLM.from_pretrained(
        lamed_model_path,
        trust_remote_code=True,
    ).eval()
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
        state_dict = torch.load(lora_weight_path, map_location=device)
        lamed_model.load_state_dict(state_dict, strict=True)
        print("Merge weights with LoRA")
        lamed_model = lamed_model.merge_and_unload()
        
    
    with torch.no_grad():
        lamed_model = lamed_model.to(device)
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

def inference(image, question, tokenizer, lamed_model, temperature=1.0, top_p=0.9):

    # ## filter out special chars
    # input_str = bleach.clean(input_str)
    # # Model Inference
    # prompt = "<im_patch>" * 256 + input_str

    input_id = tokenizer(
        question, add_special_tokens=False, max_length=768, truncation=True, padding="max_length", return_tensors="pt", padding_side="right",
    )['input_ids'].to(device)
    # image_pt = torch.from_numpy(input_image).to(device)

    generation = lamed_model.generate(image.to(device), input_id, seg_enable=False, max_new_tokens=768,
                                        do_sample=True, top_p=top_p, temperature=temperature)

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    return output_str, None
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def check_chinese_and_quality(question, answer):
    for ch in answer:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    if len(answer) < 20:
        return False
    return True
    example = "Quention: Can you provide a diagnosis based on the fingings in chest in this image? Answer: Both sides of the chest are symmetrical. Scattered point-like translucence are seen in both lungs, and a few patchy high-density foci are seen in the low lobe of left lung. No other abnormal are seen in the lungs. The trachea and bronchi are unobstructed. The mediastinum and trachea are centered, and multiple slightly enlarged lymph nodes with higher density are seen in the mediastinum and bilateral pulmonary hila. The pleura is normal. The morphology and size of the heart and great vessels are normal, with a small amount of fluid in the pericardium. A high-density shadow is seen in the upper part of the esophagus. No obvious abnormal enhancement is seen in the chest."
    messages = [
        {"role": "system", "content": f"You are the Radiation LLM answer checker, please identify invalid answers (e.g. duplicate/meaningless/unrelated output) This is an example: {example}.if answers are checked by Yes otherwise No, do not output any other characters other than that"},
        {"role": "user", "content": f"Quention: {question} Answer: {answer}"}
    ]
    text = check_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = check_tokenizer([text], return_tensors="pt").to(check_model.device)

    generated_ids = check_model.generate(
        **model_inputs,
        max_new_tokens=10,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = check_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if "Yes" in response:
        return True
    else:
        return False

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
        image = batch["image"]
        #print(input_image.shape)
        question = batch["question"][0]
        image_path = batch["image_path"][0]
        image_paths.append(image_path)
        flag = False
        while flag == False:
            pred, _ = inference(image, question, tokenizer, lamed_model)
            flag = check_chinese_and_quality(question, pred)
            print(flag, pred)
        print(flag, pred)
        # pred, _ = inference(image, question, tokenizer, lamed_model)
        pred_report.append(pred)
            # if num == 10:
            #     break
        # except Exception as e:
        #     print(e)
        num += 1
        if num == 40:
            break
    lamed_model.to(device)    
    torch.cuda.empty_cache()
    try:
        mean_green_score = green_score(pred_report, gt_report, categorize, green_model, image_paths)
    except Exception as e:
        try:
            mean_green_score = green_score(pred_report, gt_report, categorize, green_model, image_paths)
        except Exception as e:
            mean_green_score = green_score(pred_report, gt_report, categorize, green_model, image_paths)
    lamed_model.to(device)
    return mean_green_score
    
def woker(green_model, tokenizer, lamed_model, categorize):
    
    val_base_path = '/import/c4dm-04/siyoul/Med3DLLM/datasets'
    val_jsonl_path = '/import/c4dm-04/siyoul/Med3DLLM/datasets/Fused_Dataset/val/amos_mm_findings.jsonl'
    dataset = FusedDataset(
        val_base_path, 
        val_jsonl_path, 
        tokenizer, 
        max_length=2048, 
        image_tokens_num=256, 
        data_type="valuation"
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    mean_green_score = mrg_annotation(dataloader, categorize, green_model, tokenizer, lamed_model)
    # mean_green_scores.append(mean_green_score)
    # for categorize, mean_green_score in zip(["Chest","Abdomen","Pelvis"], mean_green_scores):
    return "{} - Average GREEN Score: {}".format(categorize[1], mean_green_score)

if __name__ == "__main__":
    categorizes = [["findings","chest"], ["findings","abdomen"], ["findings","pelvis"]]
    green_model_path="/import/c4dm-04/siyoul/Med3DLLM/pretrained_models/GREEN-RadLlama2-7b"
    lamed_model_path = "/import/c4dm-04/siyoul/Med3DLLM/checkpoint/Med3dLLM_0205_mrg_phi2@bs2_acc1_ep16_lr2e5_ws2_fused/checkpoint-58000"
    lora_weight_path = None
    green_model, tokenizer, lamed_model = load_model(green_model_path, lamed_model_path, lora_weight_path)
    print(lamed_model_path.split("/")[-2])
    results = []
    for categorize in categorizes:
        results.append(woker(green_model, tokenizer, lamed_model, categorize))
    for result in results:
        print(result)
    print(lamed_model_path.split("/")[-2])
