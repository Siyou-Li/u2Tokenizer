# -*- encoding: utf-8 -*-
# @File        :   fused_dataset.py
# @Time        :   2024/11/28 17:54:48
# @Author      :   Siyou
# @Description :

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from monai.utils import MAX_SEED, get_seed
from monai.transforms import Randomizable
from src.utils.data_transforms import train_transforms, val_transforms
from src.utils.linear_3d_transform import Linear3DTransform

class FusedDataset(Dataset, Randomizable):
    def __init__(
            self, base_path, jsonl_path, tokenizer, max_length,\
            image_tokens_num=1024, data_type="training", enable_linear_3d_tokenizer=True
            ):
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_tokens = "<im_patch>" * image_tokens_num #args.proj_out_num
        self.data_type = data_type
        self.annotations = self.load_annotations(os.path.join(base_path, jsonl_path))
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

        if self.data_type == "training" or self.data_type == "train":
            if enable_linear_3d_tokenizer:
                self.image_transforms = Linear3DTransform(mode='bilinear', data_type="training")
            else:
                self.image_transforms = train_transforms
        else:
            if enable_linear_3d_tokenizer:
                self.image_transforms = Linear3DTransform(mode='bilinear', data_type="validation")
            else:
                self.image_transforms = val_transforms     
        
    def randomize(self, data):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")
    
    def load_annotations(self, jsonl_path):
        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Error loading json line: ", line)
        return data

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_name = annotation['image']
        #dataset_name = annotation['dataset']
        prompt_question = annotation["question"]
        image_path = os.path.join(self.base_path, image_name)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        try:
            image = self.image_transforms(image_path)
        except Exception as e:
            if idx == self.__len__()-1:
                idx = 0
            return self.__getitem__(random.randint(0, self.__len__()-1))

        answer = annotation["answer"]
        # few_shot_prompt = \
        #     "Here are some answer examples: \
        #                 1. The liver is normal in size and shape with regular surface. The liver parenchyma density is uniform, and the common bile duct and intrahepatic bile duct are not dilated. The spleen is normal in size and shape with no abnormal density in the parenchyma. The position, shape, and density of the pancreas are normal, with no dilation of the pancreatic duct. The gallbladder is not enlarged, and the wall is not thickened. Both kidneys are normal in size and shape, while multiple cystic low-density lesions are observed with no obvious enhancement. No obvious enlarged lymph nodes are seen in the posterior peritoneum. No obvious abnormal enhancing foci is seen in the upper and lower abdomen.\
        #                 2. A nodular high-density focus is seen in the right lobe of the liver, and no abnormal density is found in the remaining liver parenchyma. The intrahepatic duct system is not obviously dilated. The gallbladder is not enlarged with uniform density enhancement inside, while a focal high-density shadow is seen near the neck of the gallbladder. The surface of the pancreas is rough, with still uniform density and clear adjacent fat intervals. The spleen is enlarged. Multiple enlarged lymph nodes are found in the abdominal cavity and retroperitoneum.\
        #                 3. Bilateral chest are symmetrical. A few patchy high-density shadows with blurred edges are observed in the right lung upper lobe and both lung lower lobes, and a few consolidations are visible in the right lung lower lobe. No narrowing or obstruction of the trachea and bronchus can be seen. Esophageal dilation is observed with mixed low-density shadows inside. No obvious enlarged lymph nodes is seen in the mediastinum or bilateral pulmonary hila. No significant abnormalities are observed in the heart or major blood vessels. No abnormal enhanced lesions are observed in the enhanced scan.\
        #                 "
        # question = "<|user|>\n" + self.image_tokens + prompt_question + "</s>\n<|assistant|>\n"

        question =  self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self.image_tokens + prompt_question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        # question = self.image_tokens + " " + str(self.categorize[1]) + ":"
        text_tensor = self.tokenizer(
            question + answer, add_special_tokens=False, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt", padding_side="right"
        )

        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        question_tensor = self.tokenizer(
            question, add_special_tokens=False, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt", padding_side="right"
        )

        question_len = torch.sum(question_tensor["attention_mask"][0])

        question_ids = self.tokenizer(
            prompt_question, add_special_tokens=False, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt", padding_side="right"
        )["input_ids"][0]
        label = input_id.clone()
        label[:question_len] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            label[label == self.tokenizer.pad_token_id] = -100
            if valid_len < len(label):
                label[valid_len] = self.tokenizer.eos_token_id
        else:
            label[label == self.tokenizer.pad_token_id] = -100

        ret = {
            'image': image,
            'image_path': image_path,
            'input_id': input_id,
            'label': label,
            'attention_mask': attention_mask,
            'question': question,
            'question_ids': question_ids,
            'prompt_question': prompt_question,
            'answer': answer,
            'question_type': "Caption",
        }
        return ret

if __name__ == '__main__':

    from transformers import AutoTokenizer

    base_path = "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Med3D_LLM/datasets"
    jsonl_path = "Fused_Dataset/fused_train_dataset.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(
        "/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Med3D_LLM/pretrained_models/Llama-3.2-1B-Instruct",
        cache_dir=None,
        model_max_length=2048,
    )
    dataset = FusedDataset(base_path, jsonl_path, tokenizer, 2048, data_type="training")
    #dataset = VQADataset(image_dir, json_path, output_size, patch_size, mode, tokenizer, max_length=2048, image_tokens_num=256, data_type="training")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        if batch is None:
            continue
        rets = batch
        print(rets["image_path"])
        print(rets["image"].shape)
        
    