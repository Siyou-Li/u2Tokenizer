import json
from config import config
import os
from tqdm import tqdm

def convert_dpo_dataset(input_file, output_file):
    dpo_data = []
    with open(input_file, 'r') as f:
        data = f.readlines()
        for item in tqdm(data):
            item = json.loads(item)
            if item["prediction_scores"][0] != 0:
                dpo_example = {}
                dpo_example["image"] = item["image"]
                dpo_example["question"] = item["question"]
                dpo_example["answer"] = item["ground_truth"]
                dpo_example["chosen"] = item["predictions"][0]
                dpo_example["rejected"] = item["predictions"][-1]
                dpo_data.append(json.dumps(dpo_example))
            else:
                print(item["prediction_scores"])
    with open(output_file, 'w') as f:
        for item in dpo_data:
            f.write("%s\n" % item)

if __name__ == "__main__":
    base_path = config["project_path"]
    input_file = "datasets/DPO/amos_m3d_greened_merged.jsonl"
    output_file = "datasets/DPO/amos_m3d_greened_merged_stage2.jsonl"
    convert_dpo_dataset(os.path.join(base_path, input_file), os.path.join(base_path, output_file))
            