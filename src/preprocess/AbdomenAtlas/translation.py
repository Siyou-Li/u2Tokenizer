# -*- encoding: utf-8 -*-
# @File        :   AbdomenAtlas.py
# @Time        :   2025/01/08 23:16:59
# @Author      :   Siyou
# @Description :

import os
from config import config
import json
import pandas as pd
from tqdm import tqdm
import random
from src.utils.vllm_func import translation
from src.utils.prompt_templates import Caption_templates_zh

base_path = config["project_path"]
csv_file_path = os.path.join(base_path, "datasets/AbdomenAtlas3.0Report/AbdomenAtlas3.0.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/abdomen_atlas3_translation.jsonl")
with open(csv_file_path, 'r') as f:
    raw_data = pd.read_csv(csv_file_path, low_memory=False)

if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
i = 0
with open(output_file_path, 'a') as f:
    for idx, row in tqdm(raw_data.iterrows()):
        if i < 1308:
            i += 1
            continue
        image = row["BDMAP ID"]
        structured_report = row["structured report"]
        narrative_report = row["narrative report"]
        fusion_structured_report = row["fusion structured report"]
        fusion_narrative_report = row["fusion narrative report"]
        radiologist_note = row["radiologist note"]
        
        try:
            ans = translation(structured_report, "Chinese", "English")
            prompt_question = random.choice(Caption_templates_zh).format("腹部")
            item = json.dumps({
                "image": os.path.join("AbdomenAtlasData", image, "ct.nii.gz"),
                "dataset": "AbdomenAtlasData3.0",
                "task_type": "VQA",
                "synthesis": True,
                "question": prompt_question,
                "answer": ans,
            }, ensure_ascii=False)
        except Exception as e:
            print(e)
            continue
        
        f.write("%s\n" % item)