# -*- encoding: utf-8 -*-
# @File        :   vqa_thinking_data_synthesis.py
# @Time        :   2025/05/23 21:50:33
# @Author      :   Siyou
# @Description :

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import pandas as pd
from tqdm import tqdm
from src.preprocess.qwen3_data_synthesis import vqa_thinking_batch, translation, vqa_thinking_translation_synthesis
from config import config
from src.preprocess.start_vllm_server import start_vllm_server
import itertools

base_path = config["project_path"]
test_mode = False
batch_size = 2

# AMO-MM
def amos_mm_vqa_thinking_synthesis(json_file_path, findings_file_path, data_type="training"):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    mrg_type = ["chest", "abdomen","pelvis"]
    raw_data = data[data_type]

    with open(findings_file_path, 'w') as f:  
        # process in 8 row batches
        for i in tqdm(range(0, len(raw_data), batch_size)):
            batch = raw_data[i:i+batch_size]
            image_paths = [os.path.join("AMOS-MM", item["image"][2:]) for item in batch]
            findings = [item["labels"]["report"]["findings"] for item in batch]

            image_paths_list = []
            findings_list = []

            for image_path, finding in zip(image_paths, findings):
                for loc in mrg_type:
                    if finding[loc] != "":
                        image_paths_list.append(image_path)
                        findings_list.append(finding[loc])
            try:
                outputs = vqa_thinking_batch(findings_list, image_paths_list)
                for item in outputs:
                    line = json.dumps({
                        "image": item["image"],
                        "dataset": "AMOS-MM",
                        "task_type": "VQA-Thinking",
                        "synthesis": True,
                        "report": item["report"],
                        "system_thinking": item["system_thinking"],
                        "question": item["question"],
                        "thinking": item["thinking"],
                        "answer": item["answer"],
                    }, ensure_ascii=False)
                    f.write(f"{line}\n")
                    f.flush()
            except Exception as e:
                print(e)
                continue
            global test_mode
            if test_mode:
                break
    print("Successfully synthesized the AMOS-MM dataset to {} using VQA thinking.".format(findings_file_path))

csv_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
findings_file_path = "datasets/Fused_Dataset/vqa_thinking/train/amos_mm_findings_vqa_thinking_.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="training")
findings_file_path = "datasets/Fused_Dataset/vqa_thinking/val/amos_mm_findings_vqa_thinking_.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="validation")
