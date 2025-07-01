# -*- encoding: utf-8 -*-
# @File        :   vqa_thinking_data_synthesis.py
# @Time        :   2025/05/23 21:50:33
# @Author      :   Siyou
# @Description :

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
from tqdm import tqdm
import argparse
from src.preprocess.qwen3_data_synthesis import vqa_thinking_batch

def parse_args():
    parser = argparse.ArgumentParser(description='AMOS-MM VQA thinking synthesis')
    parser.add_argument('--input_file', type=str, default="datasets/AMOS-MM/dataset.json",
                        help='Input JSON file path')
    parser.add_argument('--output_file', type=str, default="output/amos_mm_findings_vqa_thinking_synthesis.jsonl",
                        help='Output JSONL file path')
    parser.add_argument('--data_type', type=str, default="training", choices=["training", "validation"],
                        help='Data type to process (default: training)')
    parser.add_argument('--start_line', type=int, default=0,
                        help='Start line for processing (default: 0)')
    parser.add_argument('--end_line', type=int, default=None,
                        help='End line for processing (default: None, process all)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for processing (default: 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode (process only first batch)')
    return parser.parse_args()

# AMO-MM
def amos_mm_vqa_thinking_synthesis(json_file_path, findings_file_path, data_type="training", start_line=0, end_line=None, batch_size=2, test_mode=False):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    mrg_type = ["chest", "abdomen","pelvis"]
    raw_data = data[data_type]

    # Apply start_line and end_line filtering
    if end_line is None:
        end_line = len(raw_data)
    raw_data = raw_data[start_line:end_line]

    if not os.path.exists(os.path.dirname(findings_file_path)):
        os.makedirs(os.path.dirname(findings_file_path))

    with open(findings_file_path, 'w') as f:  
        # process in batch_size row batches
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
            try:
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
                    f.flush()
            except Exception as e:
                print(e)
                continue
            if test_mode:
                break
    print("Successfully synthesized the AMOS-MM dataset to {} using VQA thinking.".format(findings_file_path))

def main():
    args = parse_args()
    
    # Use command line arguments
    json_file_path = args.input_file
    findings_file_path = args.output_file
    data_type = args.data_type
    start_line = args.start_line
    end_line = args.end_line
    batch_size = args.batch_size
    test_mode = args.test_mode
    
    amos_mm_vqa_thinking_synthesis(
        json_file_path, 
        findings_file_path, 
        data_type=data_type,
        start_line=start_line, 
        end_line=end_line, 
        batch_size=batch_size, 
        test_mode=test_mode
    )

if __name__ == "__main__":
    main()
