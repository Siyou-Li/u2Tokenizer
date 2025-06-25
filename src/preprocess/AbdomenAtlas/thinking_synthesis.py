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
import argparse
import logging
from src.preprocess.qwen3_data_synthesis import vqa_thinking_batch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='AbdomenAtlas VQA thinking synthesis')
    parser.add_argument('--input_file', type=str, default="datasets/AbdomenAtlas3.0Report/AbdomenAtlas3.0.csv",
                        help='Input CSV file path')
    parser.add_argument('--output_file', type=str, default="output/abdomen_atlas3_vqa_thinking_synthesis.jsonl",
                        help='Output JSONL file path')
    parser.add_argument('--start_line', type=int, default=0,
                        help='Start line for processing (default: 0)')
    parser.add_argument('--end_line', type=int, default=None,
                        help='End line for processing (default: None, process all)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for processing (default: 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode (process only first batch)')
    return parser.parse_args()

# AbdomenAtlas3.0
def abdomen_atlas_vqa_thinking_synthesis(csv_file_path, output_file_path, start_line=0, end_line=None, batch_size=2, test_mode=False):
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    # Apply start_line and end_line filtering
    if end_line is None:
        end_line = len(raw_data)
    raw_data = raw_data.iloc[start_line:end_line]

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
                    f.flush()
            except Exception as e:
                logger.error(e)
                continue
            if test_mode:
                break
    logger.info("Successfully synthesized the AbdomenAtlas3.0 dataset to {} using VQA thinking.".format(output_file_path))

def main():
    args = parse_args()
    
    # Use command line arguments
    csv_file_path = args.input_file
    output_file_path = args.output_file
    start_line = args.start_line
    end_line = args.end_line
    batch_size = args.batch_size
    test_mode = args.test_mode
    
    abdomen_atlas_vqa_thinking_synthesis(
        csv_file_path, 
        output_file_path, 
        start_line=start_line, 
        end_line=end_line, 
        batch_size=batch_size, 
        test_mode=test_mode
    )

if __name__ == "__main__":
    main()


