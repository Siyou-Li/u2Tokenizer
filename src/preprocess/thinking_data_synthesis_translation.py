from src.preprocess.qwen3_data_synthesis import translation
import os
from config import config
import json
from tqdm import tqdm


base_path = config["project_path"]

def vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path):
    raw_data = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            try:
                    raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                print("Error loading json line: ", line)
    with open(output_file_path, 'w') as f:
        
        for item in tqdm(raw_data):
            translated_data = {}
            translated_data["image"] = item["image"]
            translated_data["dataset"] = item["dataset"]
            translated_data["task_type"] = item["task_type"]
            translated_data["synthesis"] = item["synthesis"]
            try:
                translated_data["question"] = translation(item["question"], "Chinese", "English")
                translated_data["answer"] = translation(item["answer"], "Chinese", "English")
                translated_data["report"] = translation(item["report"], "Chinese", "English")
                translated_data["system_thinking"] = translation(item["system_thinking"], "Chinese", "English")
                translated_data["thinking"] = translation(item["thinking"], "Chinese", "English")
                f.write(json.dumps(translated_data, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(e)
                continue
    f.close()
    print("Successfully translated the VQA thinking dataset to {}.".format(output_file_path))

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