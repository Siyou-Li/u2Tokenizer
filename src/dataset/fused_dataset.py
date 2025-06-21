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
from src.utils.u2Transform import u2Transform

class FusedDataset(Dataset, Randomizable):
    def __init__(
            self,
            base_path,
            jsonl_path,
            tokenizer,
            max_length,
            image_tokens_num=1024,
            data_type="training",
            enable_u2tokenizer=True,
            local_rank=-1,
            use_chat_template=True,
            dpo_mode=False,
            return_image=True,
    ):
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_tokens = "<im_patch>" * image_tokens_num  # args.proj_out_num
        self.data_type = data_type
        self.use_chat_template = use_chat_template
        self.return_image = return_image
        self.dpo_mode = dpo_mode

        self.annotations = self.load_annotations(os.path.join(base_path, jsonl_path))
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

        if local_rank != -1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = "cpu"

        if self.return_image:
            if self.data_type in ("training", "train"):
                self.image_transforms = (
                    u2Transform(mode="bilinear", data_type="training")
                    if enable_u2tokenizer
                    else train_transforms
                )
            else:
                self.image_transforms = (
                    u2Transform(mode="bilinear", data_type="validation")
                    if enable_u2tokenizer
                    else val_transforms
                )
        
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
        image_name = annotation["image"]
        prompt_question = annotation["question"]
        image_path = os.path.join(self.base_path, image_name)

        if self.return_image:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
            try:
                image = self.image_transforms(image_path)
            except Exception:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
        else:
            image = image_path

        if self.use_chat_template:
            question = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self.image_tokens + prompt_question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            question = self.image_tokens + prompt_question

        if self.dpo_mode:
            prompt_question_ids = self.tokenizer(
                prompt_question,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                padding_side="right",
            )["input_ids"][0]
            return {
                "image": image,
                "prompt_question_ids": prompt_question_ids,
                "prompt": question,
                "chosen": annotation.get("chosen"),
                "rejected": annotation.get("rejected"),
            }

        answer = annotation["answer"]
        text_tensor = self.tokenizer(
            question + answer,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            padding_side="right",
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

