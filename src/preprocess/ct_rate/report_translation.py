import os
from config import config
from src.preprocess.qwen3_data_synthesis import report_thinking_translation_synthesis

base_path = config["project_path"]

# # CT-RATE Translation
# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis__30000_end_report_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis__30000_end_report_thinking_chinese.jsonl")
# report_thinking_translation_synthesis(jsonl_file_path, output_file_path)

# jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis_0_10000_report_thinking.jsonl")
# output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis_0_10000_report_thinking_chinese.jsonl")
# report_thinking_translation_synthesis(jsonl_file_path, output_file_path)

jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis_20000_30000_report_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/train/ct_rate_vqa_thinking_synthesis_20000_30000_report_thinking_chinese.jsonl")
report_thinking_translation_synthesis(jsonl_file_path, output_file_path)

jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/val/ct_rate_vqa_thinking_synthesis_report_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking_v2/val/ct_rate_vqa_thinking_synthesis_report_thinking_chinese.jsonl")
report_thinking_translation_synthesis(jsonl_file_path, output_file_path)