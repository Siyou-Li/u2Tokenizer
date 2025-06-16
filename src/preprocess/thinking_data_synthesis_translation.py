from src.preprocess.qwen3_data_synthesis import translation
import os
from config import config
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

            translated_data["report"] = item["report"]
            translated_data["system_thinking"] = item["system_thinking"]
            translated_data["thinking"] = item["thinking"]
            try:
                translated_data["question"] = translation(item["question"], "Chinese", "English")
                translated_data["answer"] = translation(item["answer"], "Chinese", "English")
                # translated_data["report"] = translation(item["report"], "Chinese", "English")
                # translated_data["system_thinking"] = translation(item["system_thinking"], "Chinese", "English")
                # translated_data["thinking"] = translation(item["thinking"], "Chinese", "English")
                f.write(json.dumps(translated_data, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(e)
                continue
    f.close()
    print("Successfully translated the VQA thinking dataset to {}.".format(output_file_path))

# use arguments to run the script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Translate VQA thinking dataset to English.")
    parser.add_argument("--jsonl_file_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output JSONL file.")
    
    args = parser.parse_args()
    
    vqa_thinking_translation_synthesis(args.jsonl_file_path, args.output_file_path)