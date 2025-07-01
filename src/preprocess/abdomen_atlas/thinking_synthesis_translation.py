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
from src.preprocess.qwen3_data_synthesis import vqa_thinking_translation_synthesis
from config import config

base_path = config["project_path"]
test_mode = False
batch_size = 2

# AbdomenAtlas3.0 Translation
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/vqa_thinking/train/abdomen_atlas3_vqa_thinking.jsonl")
output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/chinese/train/abdomen_atlas3_vqa_thinking.jsonl")
vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path)
