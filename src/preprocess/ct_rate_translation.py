# -*- encoding: utf-8 -*-
# @File        :   ct_rate_rewrite.py
# @Time        :   2025/01/12 09:47:02
# @Author      :   Siyou
# @Description :

import os
import json
import pandas as pd
from tqdm import tqdm
import random
from src.utils.vllm_func import translation
from src.utils.prompt_templates import Caption_templates
from config import config

base_path = config["project_path"]
csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/val/ct_rate_translation.jsonl")
with open(csv_file_path, 'r') as f:
    raw_data = pd.read_csv(csv_file_path, low_memory=False)

ct_rate_data = []
for idx, row in tqdm(raw_data.iterrows()):
    image_name = row["VolumeName"]
    image_name_split = image_name.split("_")
    image_path = image_name_split[0] + "_" + image_name_split[1] \
    + "/" + image_name_split[0] + "_" + image_name_split[1] + "_" + image_name_split[2]\
    + "/" + image_name
    findings = row["Findings_EN"]
    
    for i in range(1):
        prompt_question = random.choice(Caption_templates).format("findings")
        try:
            ans = translation(findings, "Chinese", "English")
            if os.path.exists(os.path.join(base_path, "datasets/CT-RATE/dataset/valid", image_path)):
                ct_rate_data.append(json.dumps({
                    "image": os.path.join("CT-RATE/dataset/valid", image_path),
                    "dataset": "CT-RATE",
                    "task_type": "VQA",
                    "synthesis": True,
                    "question": prompt_question,
                    "answer": ans,
                }, ensure_ascii=False))
        except Exception as e:
            print(e)
            continue

if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
with open(output_file_path, 'w') as f:
    for item in ct_rate_data:
        f.write("%s\n" % item)