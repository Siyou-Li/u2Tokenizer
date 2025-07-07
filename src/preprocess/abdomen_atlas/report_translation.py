import os
from config import config
from src.preprocess.qwen3_data_synthesis import report_thinking_translation_synthesis

base_path = config["project_path"]

# AbdomenAtlas3.0 Translation
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/abdomen_atlas_report_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/abdomen_atlas_report_thinking.jsonl")
report_thinking_translation_synthesis(jsonl_file_path, output_file_path)
