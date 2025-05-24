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
from src.preprocess.qwen3_data_synthesis import vqa_thinking, vqa_thinking_batch
from config import config
from src.preprocess.start_vllm_server import start_vllm_server
import itertools

base_path = config["project_path"]
test_mode = True
batch_size = 8

# CT-RATE Training
def ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path):
    """
    Synthesize the CT-RATE dataset using VQA thinking.
    """
    
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'a') as f:
        # process in batch_size row batches
        for i in tqdm(range(0, len(raw_data), batch_size)):
            batch = raw_data.iloc[i:i+batch_size]
            findings = batch["Findings_EN"].tolist()
            image_paths = [os.path.join("CT-RATE/dataset/valid", image_name.split("_")[0] + "_" + image_name.split("_")[1] + "/" + image_name.split("_")[0] + "_" + image_name.split("_")[1] + "_" + image_name.split("_")[2] + "/" + image_name) for image_name in batch["VolumeName"].tolist()]
            try:
                outputs = vqa_thinking_batch(findings, image_paths)
                for item in outputs:
                    line = json.dumps({
                        "image": item["image"],
                        "dataset": "CT-RATE",
                        "task_type": "VQA-Thinking",
                        "synthesis": True,
                        "report": item["report"],
                        "system_thinking": item["system_thinking"],
                        "question": item["question"],
                        "thinking": item["thinking"],
                        "answer": item["answer"],
                    }, ensure_ascii=False)
                    f.write(f"{line}\n")
            except Exception as e:
                print(e)
                continue
            global test_mode
            if test_mode:
                break
    print("Successfully synthesized the CT-RATE dataset to {} using VQA thinking.".format(output_file_path))
    
# ## CT-RATE Validation
# csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/ct_rate_vqa_thinking.jsonl")
# ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path)
# ## CT-RATE Training
# csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/val/ct_rate_vqa_thinking.jsonl")
# ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path)


# AbdomenAtlas3.0
def abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path):
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'a') as f:
        # process in batch_size row batches
        for i in tqdm(range(0, len(raw_data), batch_size)):
            batch = raw_data.iloc[i:i+batch_size]
            image_paths = [os.path.join("AbdomenAtlas3.0Report", image_name, "ct.nii.gz") for image_name in batch["BDMAP ID"].tolist()]
            structured_report = batch["structured report"].tolist()
            try:
                outputs = vqa_thinking_batch(structured_report, image_paths)
                for item in outputs:
                    line = json.dumps({
                        "image": item["image"],
                        "dataset": "AbdomenAtlasData3.0",
                        "task_type": "VQA-Thinking",
                        "synthesis": True,
                        "report": item["report"],
                        "system_thinking": item["system_thinking"],
                        "question": item["question"],
                        "thinking": item["thinking"],
                        "answer": item["answer"],
                    }, ensure_ascii=False)
                    f.write(f"{line}\n")
            except Exception as e:
                print(e)
                continue
            global test_mode
            if test_mode:
                break
    print("Successfully synthesized the AbdomenAtlas3.0 dataset to {} using VQA thinking.".format(output_file_path))

## AbdomenAtlas3.0
csv_file_path = os.path.join(base_path, "datasets/AbdomenAtlas3.0Report/AbdomenAtlas3.0.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/abdomen_atlas3_vqa_thinking.jsonl")
abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path)


# AMO-MM
def amos_mm_vqa_thinking_synthesis(json_file_path, findings_file_path, data_type="training"):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    mrg_type = ["chest", "abdomen","pelvis"]
    raw_data = data[data_type]

    with open(findings_file_path, 'a') as f:  
        # process in 8 row batches
        for i in tqdm(range(0, len(raw_data), 8)):
            batch = raw_data[i:i+8]
            image_paths = [os.path.join("AMOS-MM", image[2:]) for image in batch["image"]]
            findings = [item["labels"]["report"]["findings"] for item in batch]

            image_paths_list = []
            findings_list = []

            for image_path, finding in zip(image_paths, findings):
                for loc in mrg_type:
                    if findings[loc] != "":
                        image_paths_list.append(image_path)
                        findings_list.append(finding[loc])

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
            global test_mode
            if test_mode:
                break
    print("Successfully synthesized the AMOS-MM dataset to {} using VQA thinking.".format(findings_file_path))

csv_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
findings_file_path = "datasets/Fused_Dataset/train/amos_mm_findings_vqa_thinking.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="training")
findings_file_path = "datasets/Fused_Dataset/val/amos_mm_findings_vqa_thinking.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="validation")
