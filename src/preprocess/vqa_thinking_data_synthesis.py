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
import random
from src.preprocess.qwen3_data_synthesis import vqa_thinking
from config import config
from src.preprocess.start_vllm_server import start_vllm_server
import time

base_path = config["project_path"]
test_mode = True
# CT-RATE Training
def ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path):
    """
    Synthesize the CT-RATE dataset using VQA thinking.
    """
    
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as f:
        for idx, row in tqdm(raw_data.iterrows()):
            image_name = row["VolumeName"]
            image_name_split = image_name.split("_")
            image_path = image_name_split[0] + "_" + image_name_split[1] \
            + "/" + image_name_split[0] + "_" + image_name_split[1] + "_" + image_name_split[2]\
            + "/" + image_name
            findings = row["Findings_EN"]
            try:
                outputs = vqa_thinking(findings)
                for item in outputs:
                    report = item["report"]
                    system_thinking = item["system_thinking"]
                    question = item["question"]
                    thinking = item["thinking"]
                    answer = item["answer"]
                    line = json.dumps({
                        "image": os.path.join("CT-RATE/dataset/valid", image_path),
                        "dataset": "CT-RATE",
                        "task_type": "VQA-Thinking",
                        "synthesis": True,
                        "report": report,
                        "system_thinking": system_thinking,
                        "question": question,
                        "thinking": thinking,
                        "answer": answer,
                    }, ensure_ascii=False)
                    print(line)
                    f.write(f"{line}\n")
            except Exception as e:
                print(e)
                continue
            if test_mode:
                raise Exception("test_mode is True, break the loop")
        f.close()
    print("Successfully synthesized the CT-RATE dataset to {} using VQA thinking.".format(output_file_path))
    
server_process = start_vllm_server()
if server_process:
    time.sleep(120)
## CT-RATE Validation
csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/ct_rate_vqa_thinking.jsonl")
ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path)
## CT-RATE Training
csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/val/ct_rate_vqa_thinking.jsonl")
ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path)
server_process.terminate()


# AbdomenAtlas3.0
def abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path):
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as f:
        for idx, row in tqdm(raw_data.iterrows()):
            image = row["BDMAP ID"]
            structured_report = row["structured report"]
            narrative_report = row["narrative report"]
            fusion_structured_report = row["fusion structured report"]
            fusion_narrative_report = row["fusion narrative report"]
            radiologist_note = row["radiologist note"]
            abdomen_atlas_data = []
            try:
                outputs = vqa_thinking(structured_report)
                for item in outputs:
                    report = item["report"]
                    system_thinking = item["system_thinking"]
                    question = item["question"]
                    thinking = item["thinking"]
                    answer = item["answer"]
                    abdomen_atlas_data.append(json.dumps({
                        "image": os.path.join("AbdomenAtlasData", image, "ct.nii.gz"),
                        "dataset": "AbdomenAtlasData3.0",
                        "task_type": "VQA-Thinking",
                        "synthesis": True,
                        "report": report,
                        "system_thinking": system_thinking,
                        "question": question,
                        "thinking": thinking,
                        "answer": answer,
                    }, ensure_ascii=False))
            except Exception as e:
                print(e)
                continue
            for item in abdomen_atlas_data:
                f.write("%s\n" % item)
            if test_mode:
                raise Exception("test_mode is True, break the loop")
        f.close()
    print("Successfully synthesized the AbdomenAtlas3.0 dataset to {} using VQA thinking.".format(output_file_path))

## AbdomenAtlas3.0
csv_file_path = os.path.join(base_path, "datasets/AbdomenAtlas3.0Report/AbdomenAtlas3.0.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/abdomen_atlas3_vqa_thinking.jsonl")
abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path)


# AMO-MM
def amos_mm_vqa_thinking_synthesis(json_file_path, findings_file_path, data_type="training"):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    data_type = [data_type]
    mrg_type = ["chest", "abdomen","pelvis"]

    with open(findings_file_path, 'w') as f:  
        for data_t in data_type:
            for item in tqdm(data[data_t]):
                image = "AMOS-MM/" + item["image"][2:]
                meta = item["meta"]
                findings = item["labels"]["report"]["findings"]
                findings_pairs = []
                for loc in mrg_type:
                    if findings[loc] != "":
                        try:
                            outputs = vqa_thinking(findings)
                            for item in outputs:
                                report = item["report"]
                                system_thinking = item["system_thinking"]
                                question = item["question"]
                                thinking = item["thinking"]
                                answer = item["answer"]
                                findings_pairs.append(json.dumps({
                                    "dataset": "AMOS-MM",
                                    "image": image,
                                    "task_type": "VQA-Thinking",
                                    "synthesis": True,
                                    "report": report,
                                    "system_thinking": system_thinking,
                                    "question": question,
                                    "thinking": thinking,
                                    "answer": answer,
                                }, ensure_ascii=False))
                        except Exception as e:
                            print(e)
                            continue
                        for item in findings_pairs:
                            f.write("%s\n" % item)
                        if test_mode:
                            raise Exception("test_mode is True, break the loop")
        f.close()
    print("Successfully synthesized the AMOS-MM dataset to {} using VQA thinking.".format(findings_file_path))

csv_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
findings_file_path = "datasets/Fused_Dataset/train/amos_mm_findings_vqa_thinking.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="training")
findings_file_path = "datasets/Fused_Dataset/val/amos_mm_findings_vqa_thinking.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="validation")
