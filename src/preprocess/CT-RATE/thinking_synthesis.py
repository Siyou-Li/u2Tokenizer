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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='CT-RATE VQA thinking synthesis')
    parser.add_argument('--input_file', type=str, default="datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv",
                        help='Input CSV file path')
    parser.add_argument('--output_file', type=str, default="output/ct_rate_vqa_thinking_synthesis.jsonl",
                        help='Output JSONL file path')
    parser.add_argument('--data_type', type=str, default="train", choices=["train", "valid"],
                        help='Data type to process (default: train)')
    parser.add_argument('--start_line', type=int, default=0,
                        help='Start line for processing (default: 0)')
    parser.add_argument('--end_line', type=int, default=None,
                        help='End line for processing (default: None, process all)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for processing (default: 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode (process only first batch)')
    parser.add_argument('--skip_lines', type=int, default=0,
                        help='Number of lines to skip at the beginning (default: 0)')
    return parser.parse_args()

# CT-RATE Training
def ct_rate_vqa_thinking_synthesis(csv_file_path, output_file_path, data_type="train", start_line=0, end_line=None, batch_size=2, test_mode=False, skip_lines=0):
    """
    Synthesize the CT-RATE dataset using VQA thinking.
    """
    
    with open(csv_file_path, 'r') as f:
        raw_data = pd.read_csv(csv_file_path, low_memory=False)

    # Apply start_line and end_line filtering
    if end_line is None:
        end_line = len(raw_data)
    raw_data = raw_data.iloc[start_line:end_line]

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'a') as f:
        # process in batch_size row batches
        for i in tqdm(range(skip_lines * batch_size, len(raw_data), batch_size)):
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
                    f.flush()
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue
            if test_mode:
                break
    logger.info(f"Successfully synthesized the CT-RATE dataset to {output_file_path} using VQA thinking.")

def main():
    args = parse_args()
    
    # Use command line arguments
    csv_file_path = args.input_file
    output_file_path = args.output_file
    data_type = args.data_type
    start_line = args.start_line
    end_line = args.end_line
    batch_size = args.batch_size
    test_mode = args.test_mode
    skip_lines = args.skip_lines
    
    ct_rate_vqa_thinking_synthesis(
        csv_file_path, 
        output_file_path, 
        data_type=data_type,
        start_line=start_line, 
        end_line=end_line, 
        batch_size=batch_size, 
        test_mode=test_mode,
        skip_lines=skip_lines
    )

if __name__ == "__main__":
    main()
