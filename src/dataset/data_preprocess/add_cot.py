import json
from transformers import AutoTokenizer
from tqdm import tqdm

llama_tokenizer = AutoTokenizer.from_pretrained(
        "/home/lez/Siyou/Med3DLLM/pretrained_models/Llama-3.2-1B-Instruct",
        cache_dir=None,
        model_max_length=512,
        padding_side="left",
        pad_token="<unk>",
        use_fast=False
        )

def cot_process(raw):
    question = raw["question"]
    options = raw["options"]
    answer = raw["answer"]
    reasoning = raw["reasoning"]
    input_messages = [
        {"role": "user", "content": "What condition is suggested by the findings in the right ascending colon? A. Diverticulitis, B. Crohn's Disease, C. Ulcerative Colitis, D. Ascending colon cancer"},
        {"role": "assistant", "content": "The findings of a soft tissue mass with obvious enhancement, and associated intestinal stenosis in the right ascending colon are consistent with ascending colon cancer. The size and characteristics of the lesion are more indicative of neoplasm than the inflammatory conditions listed in the other options. The Answer is D"},
        {"role": "user", "content": "{} A. {}, B. {}, C. {}, D. {}".format(question, options["A"], options["B"], options["C"], options["D"])}
    ]
    input_cot = llama_tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_cot = f"{reasoning} The Answer is {answer}"
    return input_cot, answer_cot

json_file_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM/dataset.json"
chest_data_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_chest_extension_gemma2@27b.json"
abdomen_data_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_abdomen_extension_gemma2@27b.json"
pelvis_data_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_pelvis_extension_gemma2@27b.json"

with open(json_file_path, 'r') as f:
    data = json.load(f)
with open(chest_data_path, 'r') as f:
    chest_data = json.load(f)
with open(abdomen_data_path, 'r') as f:
    abdomen_data = json.load(f)
with open(pelvis_data_path, 'r') as f:
    pelvis_data = json.load(f)

data_type = ["training"]
mrg_type = {
    "chest": 
        {"training": chest_data["training"]},
    "abdomen":
        {"training":abdomen_data["training"]},
    "pelvis":
        {"training":pelvis_data["training"]}
    }

for loc in ["chest", "abdomen", "pelvis"]:
    for data_t in data_type:
        for item in tqdm(data[data_t]):
            image = item["image"]
            meta = item["meta"]
            for qa in item["labels"]["qa"]:
                input_cot, answer_cot = cot_process(qa)
                mrg_type[loc][data_t].append({
                    "image": image,
                    "is_extented": True,
                    "is_cot": True,
                    "meta": meta,
                    "category": loc,
                    "question": input_cot,
                    "answer": answer_cot
                    # "labels": {"report": {"findings": {loc:findings[loc]}}}
                })
    output_file_path = f"/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_{loc}_extension_gemma2@27b_cot.json"
    with open(output_file_path, 'w') as f:
        json.dump(mrg_type[loc], f, indent=4)