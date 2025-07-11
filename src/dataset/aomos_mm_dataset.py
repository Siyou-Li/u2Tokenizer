# -*- encoding: utf-8 -*-
# @File        :   aomos_mm_dataset.py
# @Time        :   2024/08/07 01:37:35
# @Author      :   Siyou
# @Description :

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
from ..utils.prompt_templates import Caption_templates
from src.utils.NIfTI_processor import NIfTIProcessor
from src.dataset.ct_rate_dataset import CapDataset

class MRGDataset(Dataset):
    def __init__(
            self, image_dir, json_path, output_size, patch_size, mode, tokenizer, max_length,\
            image_tokens_num=256, seg_enable=False, categorize="chest", data_type="train"
            ):
        self.image_dir = image_dir
        self.output_size = output_size
        self.patch_size = patch_size
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_prompts = Caption_templates
        self.image_tokens = "<im_patch>" * image_tokens_num #args.proj_out_num
        self.seg_enable = seg_enable
        self.processor = NIfTIProcessor(self.output_size, self.patch_size, self.mode)
        if categorize[1] not in ["chest", "abdomen", "pelvis"]:
            raise ValueError("Invalid categorize value. Must be one of ['chest', 'abdomen', 'pelvis']")
        self.categorize = categorize
        if data_type not in ["training", "validation", "testing"]:
            raise ValueError("Invalid data_type value. Must be one of ['train', 'validation', 'test']")
        self.data_type = data_type
        self.annotations = self.load_annotations(json_path)
        
    
    def load_annotations(self, json_path):
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        return annotations[self.data_type]

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
        image_path = self.image_dir + image_name
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        try:
            preprocessed_img = self.processor.scale_image(image_path)
        except Exception as e:
            print(f"Error in scaling image: {image_path}")
            return self.__getitem__(random.randint(0, len(self.annotations) - 1))
        image = torch.unsqueeze(preprocessed_img, 0).to(torch.float32)
        answer = annotation['labels']['report'][self.categorize[0]][self.categorize[1]]
        if answer == "":
            return self.__getitem__(random.randint(0, len(self.annotations) - 1))
        # prompt_question = random.choice(self.caption_prompts).format("{} in {}".format(self.categorize[0],self.categorize[1]))
        prompt_question = "please provide a detailed caption outlining the {} in {} of this image." .format(self.categorize[0],self.categorize[1])
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
        if self.seg_enable:
            ret.update({'seg': torch.zeros_like(image)})
        return ret

class VQADataset(Dataset):
    def __init__(
            self, image_dir, json_path, output_size, patch_size, mode, tokenizer, max_length,\
            image_tokens_num=256, seg_enable=True, data_type="train"
            ):
        self.image_dir = image_dir
        self.output_size = output_size
        self.patch_size = patch_size
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_prompts = Caption_templates
        self.image_tokens = "<im_patch>" * image_tokens_num #args.proj_out_num
        self.seg_enable = seg_enable
        self.processor = NIfTIProcessor(self.output_size, self.patch_size, self.mode)
        if data_type not in ["training", "validation", "test"]:
            raise ValueError("Invalid data_type value. Must be one of ['training', 'validation', 'test']")
        self.data_type = data_type
        self.annotations = self.load_annotations(json_path)
    
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
    
    def load_annotations(self, json_path):
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        return annotations[self.data_type]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        image_name = data['image']
        image_path = self.image_dir + image_name

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        try:
            preprocessed_img = self.processor.scale_image(image_path)
        except Exception as e:
            print(f"Error in scaling image: {image_path}")
            return self.__getitem__(random.randint(0, len(self.annotations) - 1))
        image = torch.unsqueeze(preprocessed_img, 0).to(torch.float32)

        question = data["question"]
        choices = "Choices: A. {} B. {} C. {} D. {}".format(data["options"]["A"], data["options"]["B"], data["options"]["C"], data["options"]["D"])
        question = question + ' ' + choices
        #answer = "{}. {}".format(data["answer"], data["reasoning"])
        answer = data["answer"]

        question = self.image_tokens + ' ' + question
        text_tensor = self.tokenizer(
            question + ' ' + answer, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt",
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
            'answer': answer,
            'answer_choice': data["answer"],
            'question_type': data["type"],
        }

        if self.seg_enable:
            ret.update({'seg': torch.zeros_like(image)})

        return ret

class MRGMIXDatasets(Dataset):
    def __init__(self, image_dir, json_path, output_size, patch_size, mode, tokenizer, max_length,\
            image_tokens_num=256, seg_enable=False, data_type="train"):
        super(MRGMIXDatasets, self).__init__()
        image_dir2 = '/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Video-LLaVA/dataset/CT-RATE/dataset/train'
        json_path2 = '/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/datasets/CT-RATE/dataset/radiology_text_reports/train.json'
        self.ds_list = [
            MRGDataset(image_dir, json_path.format("chest"), output_size, patch_size, mode, tokenizer, max_length, categorize=["findings","chest"], data_type=data_type, image_tokens_num=image_tokens_num, seg_enable=seg_enable),
            MRGDataset(image_dir, json_path.format("abdomen"), output_size, patch_size, mode, tokenizer, max_length, categorize=["findings","abdomen"], data_type=data_type, image_tokens_num=image_tokens_num, seg_enable=seg_enable),
            MRGDataset(image_dir, json_path.format("pelvis"), output_size, patch_size, mode, tokenizer, max_length, categorize=["findings","pelvis"], data_type=data_type, image_tokens_num=image_tokens_num, seg_enable=seg_enable),
            ] 
        # self.ds_list.append(CapDataset(
        #     '/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Video-LLaVA/dataset/CT-RATE/dataset/train', 
        #     '/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Video-LLaVA/dataset/CT-RATE/dataset/radiology_text_reports/train.json',
        #       output_size, patch_size, mode, tokenizer, max_length, image_tokens_num))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
       
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

