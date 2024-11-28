import json
from src.utils.data_transforms import mask_transforms
from tqdm import tqdm
import os 
import pandas as pd
from scipy.sparse import csc_matrix
import torch

base_path = "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Med3D_LLM"
dataset_name = "datasets/M3D-RefSeg"
train_file_name = "M3D_RefSeg_train.csv"
test_file_name = "M3D_RefSeg_test.csv"

output_file_path = os.path.join(base_path, "datasets/Fused_Dataset/train/m3d_refseg.jsonl")
refseg_data = []
raw_data = pd.read_csv(os.path.join(base_path, dataset_name, train_file_name))
for idx, row in tqdm(raw_data.iterrows()):
    image = row["Image"]
    mask = row["Mask"]
    question = row["Question"]
    answer = row["Answer"]
    mask_path = os.path.join(base_path, dataset_name, "M3D_RefSeg", mask)
    try:
        mask_matrix = mask_transforms(mask_path)[0]
        mask_seq = torch.nonzero(mask_matrix).squeeze().tolist()
    except Exception as e:
        print(mask_path)
        print(f"Error: {e}")
        continue
    refseg_data.append(json.dumps({
        "image": os.path.join("M3D_RefSeg", image),
        "dataset": "M3D-RefSeg",
        "task_type": "SEG",
        "synthesis": False,
        "question": question,
        "answer": answer.replace("[SEG]", str(mask_seq)),
    }, ensure_ascii=False))

if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
with open(output_file_path, 'w') as f:
    for item in refseg_data:
        f.write("%s\n" % item)
