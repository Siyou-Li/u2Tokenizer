import os
from config import config
from src.preprocess.qwen3_data_synthesis import report_thinking_translation_synthesis

base_path = config["project_path"]

# CT-RATE Translation
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis_10000_20000_refined_report_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis_10000_20000_refined_report_thinking_chinese.jsonl")
report_thinking_translation_synthesis(jsonl_file_path, output_file_path)

