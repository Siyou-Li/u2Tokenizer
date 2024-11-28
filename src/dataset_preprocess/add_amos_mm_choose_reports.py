import json
import random
from tqdm import tqdm
import os 

base_path = "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Med3D_LLM/datasets"
json_file_path = os.path.join(base_path, "AMOS-MM/dataset.json")

with open(json_file_path, 'r') as f:
    data = json.load(f)

data_type = ["training"]
mrg_type = ["chest", "abdomen","pelvis"]


output_file_path = "Fused_Dataset/train/amos_mm_choose_reports.jsonl"
output_file_path = os.path.join(base_path, output_file_path)
pairs = []

with open(output_file_path, 'w') as f:    
    for data_t in data_type:
        for item in tqdm(data[data_t]):
            image = item["image"]
            meta = item["meta"]
            findings = item["labels"]["report"]["findings"]
            for loc in mrg_type:
                if findings[loc] != "":
                    pairs.append({
                        "dataset": "AMOS-MM",
                        "image": image,
                        "is_extented": False,
                        "meta": meta,
                        "category": loc,
                        "question": f"please provide a detailed caption outlining the fingings in {loc} of this image.",
                        "answer": findings[loc]
                        # "labels": {"report": {"findings": {loc:findings[loc]}}}
                    })

# generate single choice questions
choose_pairs_qa = []
max_len = 0
for pair in pairs:
    question = "Which of the following is the correct description of the findings in the " + pair["category"] + " of this image?"
    gt = pair["answer"]
    choices_item = random.sample(pairs, 9)
    choices = [choice["answer"] for choice in choices_item]
    if gt in choices:
        choices.remove(gt)
    correct_choice = random.randint(0, 6)
    choices.insert(correct_choice, gt)
    question += " A. " + choices[0] + " B. " + choices[1] + " C. " + choices[2] + " D. " + choices[3] + " E. " + choices[4] + " F. " + choices[5] + " G. " + choices[6]
    answer = "The correct answer is: " + chr(65 + correct_choice)
    if len(question) > max_len:
        max_len = len(question)
    choose_pairs_qa.append(json.dumps({
        "dataset": "AMOS-MM",
        "image": pair["image"],
        "task_type": "Select a report",
        "synthesis": True,
        "question": question,
        "answer": answer
    }, ensure_ascii=False))

with open(output_file_path, 'a') as f:
    for pair in choose_pairs_qa:
        f.write(pair + "\n")

print(max_len)
