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

class FusedDataset(Dataset, Randomizable):
    def __init__(
            self, base_path, jsonl_path, tokenizer, max_length,\
            image_tokens_num=1024, data_type="training"
            ):
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_tokens = "<im_patch>" * image_tokens_num #args.proj_out_num
        self.data_type = data_type
        self.annotations = self.load_annotations(os.path.join(base_path, jsonl_path))
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed
        #self.loader = LoadImage(NibabelReader, image_only=True, ensure_channel_first=False)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        if self.data_type == "training" or self.data_type == "train":
            self.image_transforms = self.train_transforms
        else:
            self.image_transforms = self.val_transforms     
        
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
        dataset_name = annotation['dataset']
        prompt_question = annotation["question"]
        image_path = os.path.join(self.base_path, dataset_name, image_name)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        image = self.image_transforms(image_path)

        answer = annotation["answer"]
        question = self.image_tokens + prompt_question

        # question = self.image_tokens + " " + str(self.categorize[1]) + ":"
        text_tensor = self.tokenizer(
            question + ' ' + answer, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        question_tensor = self.tokenizer(
            question, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        question_len = torch.sum(question_tensor["attention_mask"][0])

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
            'question_tensor': question_tensor,
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
        
    