# -*- encoding: utf-8 -*-
# @File        :   merge_jsonl.py
# @Time        :   2024/11/28 17:29:58
# @Author      :   Siyou
# @Description :

import json
import os
import pandas as pd
import random
base_path = "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Med3D_LLM"
dir_path = "datasets/Fused_Dataset/train"

output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/fused_train_dataset.jsonl")
if os.path.exists(output_file_path):
    os.remove(output_file_path)

file_names = os.listdir(os.path.join(base_path, dir_path))
fused_data = []
for file_name in file_names:
    with open(os.path.join(base_path, dir_path, file_name), 'r') as f:
        data = f.readlines()
        fused_data.extend(data)

random.shuffle(fused_data)
with open(output_file_path, 'w') as f:
    for item in fused_data:
        f.write("%s" % item)

