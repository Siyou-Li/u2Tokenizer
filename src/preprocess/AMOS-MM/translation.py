import json
from tqdm import tqdm
import os 
from config import config
from eval import vqa
from src.utils.vllm_func import translation
from src.utils.prompt_templates import Caption_templates_zh
import random

base_path = config["project_path"]
test_mode = False
batch_size = 2

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
