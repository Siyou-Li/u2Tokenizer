# -*- encoding: utf-8 -*-
# @File        :   merge_jsonl.py
# @Time        :   2024/11/28 17:29:58
# @Author      :   Siyou
# @Description :

import json
import os
import pandas as pd
import random
from config import config
from tqdm import tqdm

base_path = config["project_path"]
dir_path = "datasets/AMOS-MM-Extension"

output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/dataset_chest_extension_qwen32b.jsonl")
if os.path.exists(output_file_path):
    os.remove(output_file_path)

file_names = os.listdir(os.path.join(base_path, dir_path))
fused_data = []
for file_name in file_names:
    with open(os.path.join(base_path, dir_path, file_name), 'r') as f:
        json_data = json.load(f)
        data = json_data["training"]
        for item in tqdm(data):
            if "is_cot" in item.keys():
                if item["is_cot"]:
                    continue
            else:
                item["image"] = "AMOS-MM" + item["image"][1:]
                fused_data.append(json.dumps(item) + "\n")

random.shuffle(fused_data)
with open(output_file_path, 'w') as f:
    for item in fused_data:
        f.write("%s" % item)

