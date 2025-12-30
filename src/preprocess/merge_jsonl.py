# -*- encoding: utf-8 -*-
# @File        :   merge_jsonl.py
# @Time        :   2024/11/28 17:29:58
# @Author      :   Siyou
# @Description :

import json
import os
import random
from tqdm import tqdm

base_path = ""
dir_path = "datasets/CT-RATE-Thinking/train"

output_file_path = os.path.join(
    base_path,
    "datasets/u2/ct_rate_thinking_fused_train_dataset.jsonl"
)

if os.path.exists(output_file_path):
    os.remove(output_file_path)

file_names = os.listdir(os.path.join(base_path, dir_path))
fused_data = []

for file_name in file_names:
    with open(os.path.join(base_path, dir_path, file_name), "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            image_path = os.path.join(
                base_path, "datasets", item["image"]
            ).replace("/train/", "/train_fixed/")
            item["image"] = item["image"].replace("/train/", "/train_fixed/")
            if os.path.exists(image_path):
                fused_data.append(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )

random.shuffle(fused_data)

with open(output_file_path, "w", encoding="utf-8") as f:
    f.writelines(fused_data)
