import json
import ollama
from tqdm import tqdm
import os 
from config import config
from src.utils.vllm_func import rewrite, generate_qa
from src.utils.prompt_templates import Caption_templates
import random

base_path = config["project_path"]

json_file_path = os.path.join(base_path, "datasets/AMOS-MM/dataset.json")

with open(json_file_path, 'r') as f:
    data = json.load(f)

data_type = ["training"]
mrg_type = ["chest", "abdomen","pelvis"]


output_file_path = "datasets/Fused_Dataset/train/amos_mm_rewrite_qa.jsonl"
output_file_path = os.path.join(base_path, output_file_path)
with open(output_file_path, 'w') as f:    
    for data_t in data_type:
        for item in tqdm(data[data_t]):
            image = "AMOS-MM/" + item["image"][2:]
            meta = item["meta"]
            findings = item["labels"]["report"]["findings"]
            pairs = []
            for qa in item["labels"]["qa"]:
                question = qa["question"]
                choices = "Choices: A. {} B. {} C. {} D. {}".format(qa["options"]["A"], qa["options"]["B"], qa["options"]["C"], qa["options"]["D"])
                question = question + ' ' + choices
                answer = qa["answer"]
                pairs.append(json.dumps({
                        "dataset": "AMOS-MM",
                        "image": image,
                        "is_extented": False,
                        "meta": meta,
                        "task_type": "VQA-Chioce",
                        "question": question,
                        "answer": answer
                        # "labels": {"report": {"findings": {loc:findings[loc]}}}
                    }, ensure_ascii=False))
            for loc in mrg_type:
                if findings[loc] != "":
                    pairs.append(json.dumps({
                        "dataset": "AMOS-MM",
                        "image": image,
                        "is_extented": False,
                        "meta": meta,
                        "task_type": "VQA",
                        "category": loc,
                        "question": random.choice(Caption_templates).format(f"fingings in {loc}"),
                        "answer": findings[loc]
                        # "labels": {"report": {"findings": {loc:findings[loc]}}}
                    }, ensure_ascii=False))

                    # rewrite
                    for i in range(2):
                        try:
                            rewrite_findings = rewrite(findings[loc])
                            pairs.append(json.dumps({
                                "dataset": "AMOS-MM",
                                "image": image,
                                "is_extented": True,
                                "meta": meta,
                                "task_type": "VQA",
                                "category": loc,
                                "question": random.choice(Caption_templates).format(f"fingings in {loc}"),
                                "answer": rewrite_findings
                            }, ensure_ascii=False))
                        except Exception as e:
                            print(e)
                            continue
                    # qa
                    qa_pairs = generate_qa(findings[loc])
                    for qa_pair in qa_pairs:
                        if ("question" in qa_pair.keys()) and ("answer" in qa_pair.keys()):
                            pairs.append(json.dumps({
                                "dataset": "AMOS-MM",
                                "image": image,
                                "is_extented": True,
                                "meta": meta,
                                "task_type": "VQA",
                                "category": loc,
                                "question": qa_pair["question"],
                                "answer": qa_pair["answer"]
                            }, ensure_ascii=False))
            for pair in pairs:
                f.write(pair + "\n")