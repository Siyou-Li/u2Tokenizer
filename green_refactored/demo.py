import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from green_refactored import GREEN, OpenAILLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from torch.utils.data import DataLoader
from src.dataset.fused_dataset import FusedDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config
import textwrap
import time


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=768,
            padding_side="right",
            use_fast=False,
            pad_token="<|endoftext|>",
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
        input_id = self.tokenizer(
            question, add_special_tokens=False, max_length=768, truncation=True, padding="max_length", return_tensors="pt", padding_side="right",
        )['input_ids'].to(device)

        generation = self.model.generate(image.to(device), input_id, seg_enable=False, max_new_tokens=768,
                                            do_sample=True, top_p=top_p, temperature=temperature)

        return self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

class AnswerValidator:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        EXAMPLE = textwrap.dedent("""\
            Quention: Can you provide a diagnosis based on the fingings in chest in this image?
            Answer: Both sides of the chest are symmetrical.
                Scattered point-like translucence are seen in both lungs, and a few patchy high-density foci are seen in the low lobe of left lung.
                No other abnormal are seen in the lungs. The trachea and bronchi are unobstructed.
                The mediastinum and trachea are centered, and multiple slightly enlarged lymph nodes with higher density are seen in the mediastinum and bilateral pulmonary hila.
                The pleura is normal. The morphology and size of the heart and great vessels are normal, with a small amount of fluid in the pericardium.
                A high-density shadow is seen in the upper part of the esophagus. No obvious abnormal enhancement is seen in the chest.
            """)
        self.SYSTEM_PROMPT = textwrap.dedent("""\
            You are the Radiation LLM answer checker, please identify invalid answers (e.g. duplicate/meaningless/unrelated output)

            This is an example:
            {example}.
            
            If answers are checked by Yes otherwise No, do not output any other characters other than that.
            """).format(example=EXAMPLE)

    def validate(self, question: str, answer: str) -> bool:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Quention: {question} Answer: {answer}"}
        ]
        text = self.tokenizer.apply_chat_template(         
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return "Yes" in response

def check_character_and_length(answer):
    for ch in answer:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    if len(answer.replace(" ","")) < 20:
        return False
    return True

def generate_predictions(dataloader, lamed_model):
    gt_report = []
    pred_report= []

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

    return gt_report, pred_report

def evaluate():
    lamed_model_path = config["project_path"] + "/checkpoint/checkpoint-18000"
    lora_weight_path = None
    llamed_model = LlamedModel(lamed_model_path, lora_weight_path)

    val_base_path = config["project_path"] + '/datasets'
    val_jsonl_path = config["project_path"] + '/datasets/Fused_Dataset/val/amos_mm_findings.jsonl'
    dataset = FusedDataset(
        val_base_path, 
        val_jsonl_path, 
        llamed_model.tokenizer, 
        max_length=2048, 
        image_tokens_num=256, 
        data_type="valuation"
    )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    gt_report, pred_report = generate_predictions(dataloader, llamed_model)


    # green_model_path = "/data/huanan/GREEN-RadLlama2-7b"
    # green_model_path = "/data/huanan/GREEN-RadPhi2"

    # 有多种类型的 LLM，这里选用远程 API 调用的类型
    llm_model = OpenAILLM("OpenAI LLM")
    green = GREEN(llm_model)

    mean, std, green_score_list, summary, result_df = green(refs=gt_report, hyps=pred_report)
    print(mean, std, green_score_list, summary)

def batch_task_evaluation():
    llm_model = OpenAILLM("OpenAI LLM")
    green = GREEN(llm_model)

    # 预测模型没有跑通，先用 example
    gt_report = [
        "Interstitial opacities without changes.",
        "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
        "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    ]
    pred_report = [
        "Interstitial opacities at bases without changes.",
        "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
        "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    ]

    # 用 GREEN 的方式，生成 prompt
    prompts = green.process_data(gt_report, pred_report)["prompt"]

    # 生成 batch 调用所输入的 jsonl 文件
    prompt_file = "test_me.jsonl"
    llm_model.generate_batch_file(prompts=prompts, file_name=prompt_file)

    # 上传 batch 文件到  openai
    batch_file_id = llm_model.upload_batch_file(prompt_file)

    # 通过 batch 文件，启动一个 batch 任务
    batch_id = llm_model.run_batch(batch_file_id).id

    # 轮询，直到任务成功
    while llm_model.probe_batch(batch_id) != "completed":
        time.sleep(10)

    # 将结果保存到文件
    llm_model.fetch_batch_result(batch_id, output_path="output.jsonl")

    # TODO：把结果塞到 green.completions 里，再调用 green.process_results() 就能计算分汁

if __name__ == "__main__":
    evaluate()