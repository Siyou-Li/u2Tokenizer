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
batch_size = 1
# CT-RATE Training
def ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path, data_type="train"):
    """
    Synthesize the CT-RATE dataset using VQA thinking.
    """
    
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as f:
        # process in batch_size row batches
        for i in tqdm(range(0, len(raw_data), batch_size)):
            batch = raw_data.iloc[i:i+batch_size]
            findings = batch["Findings_EN"].tolist()
            
            image_paths = [os.path.join("CT-RATE/dataset/{}".format(data_type), image_name.split("_")[0] + "_" + image_name.split("_")[1] + "/" + image_name.split("_")[0] + "_" + image_name.split("_")[1] + "_" + image_name.split("_")[2] + "/" + image_name) for image_name in batch["VolumeName"].tolist()]
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
csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/val/ct_rate_vqa_thinking.jsonl")
ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path, data_type="valid")
## CT-RATE Training
csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/ct_rate_vqa_thinking.jsonl")
ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path, data_type="train")


# AbdomenAtlas3.0
def abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path):
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as f:
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
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/abdomen_atlas3_vqa_thinking.jsonl")
abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path)


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
findings_file_path = "datasets/Fused_Dataset/vqa_thinking/train/amos_mm_findings_vqa_thinking.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="training")
findings_file_path = "datasets/Fused_Dataset/vqa_thinking/val/amos_mm_findings_vqa_thinking.jsonl"
amos_mm_vqa_thinking_synthesis(csv_file_path, findings_file_path, data_type="validation")

# CT-RATE Translation
import random
from src.utils.prompt_templates import Caption_templates_zh
csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/val/ct_rate_translation.jsonl")
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
        prompt_question = random.choice(Caption_templates_zh).format("报告")
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

csv_file_path = os.path.join(base_path, "datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/ct_rate_translation.jsonl")
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
        prompt_question = random.choice(Caption_templates_zh).format("报告")
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

# AbdomenAtlas3.0 Translation
csv_file_path = os.path.join(base_path, "datasets/AbdomenAtlas3.0Report/AbdomenAtlas3.0.csv")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/abdomen_atlas3_translation.jsonl")
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

# AMOS-MM Translation
json_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
with open(json_file_path, 'r') as f:
    data = json.load(f)
data_type = ["training"]
mrg_type = ["chest", "abdomen","pelvis"]
findings_file_path = "datasets/Fused_Dataset/chinese/train/amos_mm_findings_translation.jsonl"
vqa_file_path = "datasets/Fused_Dataset/chinese/train/amos_mm_vqa_translation.jsonl"
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
            for pair in vqa_pairs:
                f2.write(pair + "\n")

json_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")
with open(json_file_path, 'r') as f:
    data = json.load(f)
data_type = ["validation"]
mrg_type = ["chest", "abdomen","pelvis"]
findings_file_path = "datasets/Fused_Dataset/chinese/val/amos_mm_findings_translation.jsonl"
vqa_file_path = "datasets/Fused_Dataset/chinese/val/amos_mm_vqa_translation.jsonl"
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
            for pair in vqa_pairs:
                f2.write(pair + "\n")


## VQA Thinking Translation Synthesis
# CT-RATE Translation
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/ct_rate_vqa_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/ct_rate_vqa_thinking.jsonl")
vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/val/ct_rate_vqa_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/val/ct_rate_vqa_thinking.jsonl")
vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
# AbdomenAtlas3.0 Translation
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/abdomen_atlas3_vqa_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/abdomen_atlas3_vqa_thinking.jsonl")
vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
# AMOS-MM Translation
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/amos_mm_findings_vqa_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/amos_mm_findings_vqa_thinking.jsonl")
vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/val/amos_mm_findings_vqa_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/val/amos_mm_findings_vqa_thinking.jsonl")
vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
