import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from config import config

base_path = config["project_path"]

tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_path, "pretrained_models/RadPhi-2"),
        cache_dir=None,
        model_max_length=768,
        padding_side="right",
        pad_token="<|endoftext|>",
        use_fast=False
        )
tokenizer.chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def cot_process(raw):
    question = raw["question"]
    options = raw["options"]
    answer = raw["answer"]
    reasoning = raw["reasoning"]
    input_messages = [
        {"from": "human", "value": "What condition is suggested by the findings in the right ascending colon? A. Diverticulitis, B. Crohn's Disease, C. Ulcerative Colitis, D. Ascending colon cancer"},
        {"from": "gpt", "value": "The findings of a soft tissue mass with obvious enhancement, and associated intestinal stenosis in the right ascending colon are consistent with ascending colon cancer. The size and characteristics of the lesion are more indicative of neoplasm than the inflammatory conditions listed in the other options. The Answer is D"},
        {"from": "human", "value": "{} A. {}, B. {}, C. {}, D. {}".format(question, options["A"], options["B"], options["C"], options["D"])}
    ]
    input_cot = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
        
    )
    answer_cot = f"{reasoning} The Answer is {answer}"
    return input_cot, answer_cot


json_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
with open(json_file_path, 'r') as f:
    raw_data = json.load(f)

cot_data = []

data_type = ["training"]

for data_t in data_type:
    for item in tqdm(raw_data[data_t]):
        image = item["image"]
        meta = item["meta"]
        for qa in item["labels"]["qa"]:
            input_cot, answer_cot = cot_process(qa)
            cot_data.append(json.dumps({
                "dataset": "AMOS-MM",
                "image": image,
                "task_type": "CoT",
                "synthesis": False,
                "question": input_cot,
                "answer": answer_cot
                # "labels": {"report": {"findings": {loc:findings[loc]}}}
            }, ensure_ascii=False))
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/amos_mm_cot.jsonl")
with open(output_file_path, 'w') as f:
    for item in cot_data:
        f.write(item + "\n")