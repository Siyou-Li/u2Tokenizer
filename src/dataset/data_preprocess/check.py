import json
from transformers import AutoTokenizer
from tqdm import tqdm

llama_tokenizer = AutoTokenizer.from_pretrained(
        "/home/lez/Siyou/Med3DLLM/pretrained_models/Llama-3.2-1B-Instruct",
        cache_dir=None,
        model_max_length=1024,
        padding_side="left",
        pad_token="<unk>",
        use_fast=False
        )

chest_data_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_chest_extension_qwen32b.json"
abdomen_data_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_abdomen_extension_qwen32b_cot.json"
pelvis_data_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_pelvis_extension_qwen32b_cot.json"

with open(chest_data_path, 'r') as f:
    chest_data = json.load(f)
with open(abdomen_data_path, 'r') as f:
    abdomen_data = json.load(f)
with open(pelvis_data_path, 'r') as f:
    pelvis_data = json.load(f)

data = {"chest": chest_data["training"],"abdomen":abdomen_data["training"],"pelvis":pelvis_data["training"]}

for type in data.keys():
    mrg_data = data[type]
    max_length = 0
    max_token_length = 0
    max_length_answer = None
    print(f"[*] {type} - number of data: {len(mrg_data)}")
    for item in mrg_data:
        if len(item["answer"]) > max_length:
            max_length = len(item["answer"])
            max_length_answer = item
            max_token_length = len(llama_tokenizer.encode(item["answer"]))
    print(f"    max answer length: {max_length}, max answer token length: {max_token_length}")
    print(max_length_answer)
    break
