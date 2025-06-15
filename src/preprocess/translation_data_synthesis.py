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
import random
from src.utils.prompt_templates import Caption_templates_zh

base_path = config["project_path"]
test_mode = False
batch_size = 2

# # CT-RATE Translation

# csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/val/ct_rate_translation_.jsonl")
# with open(csv_file_path, 'r') as f:
#     raw_data = pd.read_csv(csv_file_path, low_memory=False)

# if not os.path.exists(os.path.dirname(output_file_path)):
#     os.makedirs(os.path.dirname(output_file_path))
# with open(output_file_path, 'w') as f:
#     for idx, row in tqdm(raw_data.iterrows()):
#         image_name = row["VolumeName"]
#         image_name_split = image_name.split("_")
#         image_path = image_name_split[0] + "_" + image_name_split[1] \
#         + "/" + image_name_split[0] + "_" + image_name_split[1] + "_" + image_name_split[2]\
#         + "/" + image_name
#         findings = row["Findings_EN"]
        
#         prompt_question = random.choice(Caption_templates_zh).format("报告")
#         try:
#             ans = translation(findings, "Chinese", "English")
#             f.write("%s\n" % json.dumps({
#                 "image": os.path.join("CT-RATE/dataset/valid", image_path),
#                 "dataset": "CT-RATE",
#                 "task_type": "VQA",
#                 "synthesis": True,
#                 "question": prompt_question,
#                 "answer": ans,
#             }, ensure_ascii=False))
#             f.flush()

#         except Exception as e:
#             print(e)
#             continue

# csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/ct_rate_translation_.jsonl")
# with open(csv_file_path, 'r') as f:
#     raw_data = pd.read_csv(csv_file_path, low_memory=False)

# if not os.path.exists(os.path.dirname(output_file_path)):
#     os.makedirs(os.path.dirname(output_file_path))

# with open(output_file_path, 'w') as f:
#     for idx, row in tqdm(raw_data.iterrows()):
#         image_name = row["VolumeName"]
#         image_name_split = image_name.split("_")
#         image_path = image_name_split[0] + "_" + image_name_split[1] \
#         + "/" + image_name_split[0] + "_" + image_name_split[1] + "_" + image_name_split[2]\
#         + "/" + image_name
#         findings = row["Findings_EN"]
        
#         prompt_question = random.choice(Caption_templates_zh).format("报告")
#         try:
#             ans = translation(findings, "Chinese", "English")
#             f.write("%s\n" % json.dumps({
#                 "image": os.path.join("CT-RATE/dataset/valid", image_path),
#                 "dataset": "CT-RATE",
#                 "task_type": "VQA",
#                 "synthesis": True,
#                 "question": prompt_question,
#                 "answer": ans,
#             }, ensure_ascii=False))
#             f.flush()
#         except Exception as e:
#             print(e)
#             continue



# AbdomenAtlas3.0 Translation
csv_file_path = os.path.join(base_path, "datasets/AbdomenAtlas3.0Report/AbdomenAtlas3.0.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/abdomen_atlas3_translation_.jsonl")
with open(csv_file_path, 'r') as f:
    raw_data = pd.read_csv(csv_file_path, low_memory=False)

if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
i = 0
with open(output_file_path, 'a') as f:
    for idx, row in tqdm(raw_data.iterrows()):
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
        f.flush()

# AMOS-MM Translation
json_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
with open(json_file_path, 'r') as f:
    data = json.load(f)
data_type = ["training"]
mrg_type = ["chest", "abdomen","pelvis"]
findings_file_path = "datasets/Fused_Dataset/chinese/train/amos_mm_findings_translation_.jsonl"
vqa_file_path = "datasets/Fused_Dataset/chinese/train/amos_mm_vqa_translation_.jsonl"
with open(findings_file_path, 'w') as f, open(vqa_file_path, 'w') as f2:  
    for data_t in data_type:
        for item in tqdm(data[data_t]):
            image = "AMOS-MM/" + item["image"][2:]
            meta = item["meta"]
            findings = item["labels"]["report"]["findings"]
            findings_pairs = []
            vqa_pairs = []
            for qa in item["labels"]["qa"]:
                try:
                    question = qa["question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(qa["options"]["A"], qa["options"]["B"], qa["options"]["C"], qa["options"]["D"])
                    question = question + ' ' + choices
                    question = translation(question, "Chinese", "English")
                    answer = qa["answer"]
                    vqa_pairs.append(json.dumps({
                            "dataset": "AMOS-MM",
                            "image": image,
                            "is_extented": False,
                            "meta": meta,
                            "task_type": "VQA-Chioce",
                            "question": question,
                            "answer": answer
                            # "labels": {"report": {"findings": {loc:findings[loc]}}}
                        }, ensure_ascii=False))
                except Exception as e:
                        print(e)
                        continue
            for loc in mrg_type:
                if loc == "chest":
                    loc_zh = "胸部"
                elif loc == "abdomen":
                    loc_zh = "腹部"
                elif loc == "pelvis":
                    loc_zh = "盆腔"
                if findings[loc] != "":
                    try:
                        ans = translation(findings[loc], "Chinese", "English")
                        findings_pairs.append(json.dumps({
                            "dataset": "AMOS-MM",
                            "image": image,
                            "task_type": "VQA",
                            "synthesis": True,
                            "question": random.choice(Caption_templates_zh).format(loc_zh),
                            "answer": ans
                        }, ensure_ascii=False))
                    except Exception as e:
                        print(e)
                        continue
            for pair in findings_pairs:
                f.write(pair + "\n")
                f.flush()
            for pair in vqa_pairs:
                f2.write(pair + "\n")
                f2.flush()

json_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
with open(json_file_path, 'r') as f:
    data = json.load(f)
data_type = ["validation"]
mrg_type = ["chest", "abdomen","pelvis"]
findings_file_path = "datasets/Fused_Dataset/chinese/val/amos_mm_findings_translation_.jsonl"
vqa_file_path = "datasets/Fused_Dataset/chinese/val/amos_mm_vqa_translation_.jsonl"
with open(findings_file_path, 'w') as f, open(vqa_file_path, 'w') as f2:  
    for data_t in data_type:
        for item in tqdm(data[data_t]):
            image = "AMOS-MM/" + item["image"][2:]
            meta = item["meta"]
            findings = item["labels"]["report"]["findings"]
            findings_pairs = []
            vqa_pairs = []
            for qa in item["labels"]["qa"]:
                try:
                    question = qa["question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(qa["options"]["A"], qa["options"]["B"], qa["options"]["C"], qa["options"]["D"])
                    question = question + ' ' + choices
                    question = translation(question, "Chinese", "English")
                    answer = qa["answer"]
                    vqa_pairs.append(json.dumps({
                            "dataset": "AMOS-MM",
                            "image": image,
                            "is_extented": False,
                            "meta": meta,
                            "task_type": "VQA-Chioce",
                            "question": question,
                            "answer": answer
                            # "labels": {"report": {"findings": {loc:findings[loc]}}}
                        }, ensure_ascii=False))
                except Exception as e:
                        print(e)
                        continue
            for loc in mrg_type:
                if loc == "chest":
                    loc_zh = "胸部"
                elif loc == "abdomen":
                    loc_zh = "腹部"
                elif loc == "pelvis":
                    loc_zh = "盆腔"
                if findings[loc] != "":
                    try:
                        ans = translation(findings[loc], "Chinese", "English")
                        findings_pairs.append(json.dumps({
                            "dataset": "AMOS-MM",
                            "image": image,
                            "task_type": "VQA",
                            "synthesis": True,
                            "question": random.choice(Caption_templates_zh).format(loc_zh),
                            "answer": ans
                        }, ensure_ascii=False))
                    except Exception as e:
                        print(e)
                        continue
            for pair in findings_pairs:
                f.write(pair + "\n")
                f.flush()
            for pair in vqa_pairs:
                f2.write(pair + "\n")
                f2.flush()


# ## VQA Thinking Translation Synthesis
# # CT-RATE Translation
# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/ct_rate_vqa_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/ct_rate_vqa_thinking.jsonl")
# vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/val/ct_rate_vqa_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/val/ct_rate_vqa_thinking.jsonl")
# vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
# # AbdomenAtlas3.0 Translation
# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/abdomen_atlas3_vqa_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/abdomen_atlas3_vqa_thinking.jsonl")
# vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
# # AMOS-MM Translation
# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/amos_mm_findings_vqa_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/amos_mm_findings_vqa_thinking.jsonl")
# vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/val/amos_mm_findings_vqa_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/val/amos_mm_findings_vqa_thinking.jsonl")
# vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
