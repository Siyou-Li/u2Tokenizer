# -*- encoding: utf-8 -*-
# @File        :   test.py
# @Time        :   2025/01/08 23:16:59
# @Author      :   Siyou
# @Description :

import os
from config import config
import json
import pandas as pd
from tqdm import tqdm
import random
from src.utils.vllm_func import rewrite, generate_qa
from src.utils.prompt_templates import Caption_templates

base_path = config["project_path"]
jsonl_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/amos_mm_rewrite.jsonl")

few_shot_prompt = \
    "Here are some examples: \
                1. The liver is normal in size and shape with regular surface. The liver parenchyma density is uniform, and the common bile duct and intrahepatic bile duct are not dilated. The spleen is normal in size and shape with no abnormal density in the parenchyma. The position, shape, and density of the pancreas are normal, with no dilation of the pancreatic duct. The gallbladder is not enlarged, and the wall is not thickened. Both kidneys are normal in size and shape, while multiple cystic low-density lesions are observed with no obvious enhancement. No obvious enlarged lymph nodes are seen in the posterior peritoneum. No obvious abnormal enhancing foci is seen in the upper and lower abdomen.\
                2. A nodular high-density focus is seen in the right lobe of the liver, and no abnormal density is found in the remaining liver parenchyma. The intrahepatic duct system is not obviously dilated. The gallbladder is not enlarged with uniform density enhancement inside, while a focal high-density shadow is seen near the neck of the gallbladder. The surface of the pancreas is rough, with still uniform density and clear adjacent fat intervals. The spleen is enlarged. Multiple enlarged lymph nodes are found in the abdominal cavity and retroperitoneum.\
                3. Bilateral chest are symmetrical. A few patchy high-density shadows with blurred edges are observed in the right lung upper lobe and both lung lower lobes, and a few consolidations are visible in the right lung lower lobe. No narrowing or obstruction of the trachea and bronchus can be seen. Esophageal dilation is observed with mixed low-density shadows inside. No obvious enlarged lymph nodes is seen in the mediastinum or bilateral pulmonary hila. No significant abnormalities are observed in the heart or major blood vessels. No abnormal enhanced lesions are observed in the enhanced scan.\
                4. The chest is symmetrical, the lung window shows clear lung texture with natural course. A few cord-like high-density foci are seen in the middle and lower lobe of right lung, while no abnormal solid foci is seen in the remaining lungs. The bilateral pulmonary hila are normal. The trachea and bronchi are unobstructed. The mediastinal window shows no deviation of the mediastinum, and the morphology of the heart and major blood vessels are normal. No mass or enlarged lymph nodes is seen in the mediastinum. No pleural effusion or pleural thickening is seen on either sides.\
                5. The bladder is well-filled, and the bladder wall is smooth. No abnormal density foci is observed in the bladder wall or inside the bladder. The position, contour, and size of the prostate are normal, with scattered high-density foci inside. The bilaterally symmetrical seminal vesicles show no abnormal density. The surrounding fat space is clear. No abnormal enhancement is observed in other regions.\
                6. The prostate volume is not large, with a smooth contour and high-density points in local areas. The angle between the bladder and seminal vesicle is clear, and no obvious focal thickening or nodular foci are seen on the bladder wall. High-density foci are seen in the pelvis."
abdomen_atlas_data = []
with open(jsonl_file_path, 'r') as f:
    for item in f.readlines():
        data = json.loads(item)
        if data["task_type"] == "VQA":
            data["question"] =  data["question"].replace(few_shot_prompt, "")

            # data["question"] = data["question"] + \
            
            abdomen_atlas_data.append(json.dumps(data, ensure_ascii=False))

output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/amos_mm_rewrite.jsonl")
if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
with open(output_file_path, 'w') as f:
    for item in abdomen_atlas_data:
        f.write("%s\n" % item)