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
from .prompt_templates import Caption_templates
from src.utils.NIfTI_processor import NIfTIProcessor
from src.dataset.ct_rate_dataset import CapDataset
from monai.utils import MAX_SEED, get_seed
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable
from monai.data.image_reader import NibabelReader
from monai.transforms import (
        LoadImage,
        Compose,
        CropForeground,
        Resize,
        ScaleIntensity,
        Lambda,
        Flip,
        Rotate90,
        RandScaleIntensity,
        ToTensor,
        EnsureType,
        RandShiftIntensity,
    )
from src.utils.utils import normalize

class MRGDataset(Dataset, Randomizable):
    def __init__(
            self, image_dir, json_path, tokenizer, max_length,\
            image_tokens_num=256, seg_enable=False, categorize=["findings","chest"], data_type="training"
            ):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_prompts = Caption_templates
        self.image_tokens = "<im_patch>" * image_tokens_num #args.proj_out_num
        self.seg_enable = seg_enable
        if categorize[1] not in ["chest", "abdomen", "pelvis"]:
            raise ValueError("Invalid categorize value. Must be one of ['chest', 'abdomen', 'pelvis']")
        self.categorize = categorize
        if data_type not in ["training", "validation", "testing"]:
            raise ValueError("Invalid data_type value. Must be one of ['train', 'validation', 'test']")
        self.data_type = data_type
        self.annotations = self.load_annotations(json_path)
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed
        #self.loader = LoadImage(NibabelReader, image_only=True, ensure_channel_first=False)
        self.image_transforms = Compose(
            [
                # preprocessing
                # EnsureChannelFirst(),
                LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                Lambda(func=normalize),
                Flip(spatial_axis=2),
                Rotate90(k=1, spatial_axes=(0, 1)),         
                EnsureType(track_meta=False),
                CropForeground(source_key="image"),
                Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear C, D, H, W,

                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),

                # common
                ToTensor(dtype=torch.float)
                ]
            )
        self.val_transforms = Compose(
                [
                    # preprocessing
                    # EnsureChannelFirst(),
                    LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                    Lambda(func=normalize),
                    Flip(spatial_axis=2),
                    Rotate90(k=1, spatial_axes=(0, 1)),         
                    EnsureType(track_meta=False),
                    CropForeground(source_key="image"),
                    Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear C, D, H, W,
                    # common
                    ToTensor(dtype=torch.float)
                    ]
                )
        if self.data_type == "training" or self.data_type == "train":
            self.image_transforms = self.image_transforms
        else:
            self.image_transforms = self.val_transforms     
        
    def randomize(self, data):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")
    
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
        prompt_question = annotation["question"]
        image_path = self.image_dir + image_name
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        image = self.image_transforms(image_path)
        # image = image.permute(0, 3, 1, 2)
        # answer = annotation['labels']['report'][self.categorize[0]][self.categorize[1]]
        answer = annotation["answer"]
        if answer == "":
            return self.__getitem__(random.randint(0, len(self.annotations) - 1))
        # prompt_question = random.choice(self.caption_prompts).format("{} in {}".format(self.categorize[0],self.categorize[1]))
        # prompt_question = "please provide a detailed caption outlining the {} in {} of this image." .format(self.categorize[0],self.categorize[1])
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
    def __init__(self, image_dir, json_path, tokenizer, max_length,\
            image_tokens_num=256, seg_enable=False, data_type="train", image_transforms=None):
        super(MRGMIXDatasets, self).__init__()
        self.ds_list = [
            MRGDataset(image_dir, json_path.format("chest"), tokenizer, max_length, categorize=["findings","chest"], data_type=data_type, image_tokens_num=image_tokens_num, seg_enable=seg_enable),
            MRGDataset(image_dir, json_path.format("abdomen"), tokenizer, max_length, categorize=["findings","abdomen"], data_type=data_type, image_tokens_num=image_tokens_num, seg_enable=seg_enable),
            MRGDataset(image_dir, json_path.format("pelvis"), tokenizer, max_length, categorize=["findings","pelvis"], data_type=data_type, image_tokens_num=image_tokens_num, seg_enable=seg_enable)
            ] 
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

if __name__ == '__main__':
    
    image_dir = '/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM/'
    json_path = '/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM/dataset_{}.json'
    #train_transforms = Compose([ScaleIntensity(minv=0.0, maxv=1.0), EnsureChannelFirst(minv=0.0, maxv=1.0), Resize((224, 224, 32), mode = "area")])
    mode = 'trilinear'

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/lez/Siyou/Med3DLLM/pretrained_models/Llama-3.2-1B-Instruct",
        cache_dir=None,
        model_max_length=2048,
    )
    #tokenizer.pad_token = tokenizer.unk_token
    #print(tokenizer.special_tokens_map)
    dataset = MRGMIXDatasets(image_dir, json_path, tokenizer, 2048, data_type="training")
    #dataset = VQADataset(image_dir, json_path, output_size, patch_size, mode, tokenizer, max_length=2048, image_tokens_num=256, data_type="training")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        if batch is None:
            continue
        rets = batch
        print(rets["image_path"])
        print(rets["image"].shape)
        # save the batch to a file
        # torch.save(rets, "./example_batch.pt")
        # rets = torch.load("./example_batch.pt")
        # print(rets["image_path"])
        break
    