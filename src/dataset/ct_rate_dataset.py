import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random
from ..utils.prompt_templates import Caption_templates
from src.utils.NIfTI_processor import NIfTIProcessor

class CapDataset(Dataset):
    def __init__(
            self, image_dir, json_path, output_size, patch_size, mode, tokenizer, max_length,\
            image_tokens_num=256, seg_enable=False
            ):
        """
        Initialize the MedicalDataset class.

        Args:
        - image_dir (str): Directory path where image files are located.
        - json_path (str): Path to the JSON file containing annotations.
        - output_size (tuple): Target size of the output image (D, H, W).
        - patch_size (tuple): Size of the patches to be extracted (d, h, w).
        - mode (str): Interpolation mode, e.g., 'trilinear'.
        """
        self.image_dir = image_dir
        self.output_size = output_size
        self.patch_size = patch_size
        self.mode = mode
        self.annotations = self.load_annotations(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_prompts = Caption_templates
        self.image_tokens = "<im_patch>" * image_tokens_num #args.proj_out_num
        self.seg_enable = seg_enable
        self.processor = NIfTIProcessor(self.output_size, self.patch_size, self.mode)
    
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
        """
        Load annotation data from the JSON file.
        
        Args:
        - json_path (str): Path to the JSON file.
        
        Returns:
        - annotations (list): List of annotation data.
        """
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        return annotations

    def __len__(self):
        """
        Return the size of the dataset.
        
        Returns:
        - (int): Size of the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Return a data item based on the index.
        
        Args:
        - idx (int): Index of the data item.
        
        Returns:
        - (tuple): Processed image patches and corresponding text annotations, or None if image file not found.
        """
        annotation = self.annotations[idx]
        image_name = annotation['VolumeName']
        image_name_split = image_name.split("_")
        image_path = self.image_dir + "/" + image_name_split[0] + "_" + image_name_split[1] \
        + "/" + image_name_split[0] + "_" + image_name_split[1] + "_" + image_name_split[2]\
        + "/" + image_name

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_name}")
            return None

        preprocessed_img = self.processor.scale_image(image_path)
        image = torch.unsqueeze(preprocessed_img, 0).to(torch.float32)
        answer = annotation['Findings_EN']
        prompt_question = random.choice(self.caption_prompts).format("findings")

        question = self.image_tokens + prompt_question

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
            'answer': answer,
            'question_type': "Caption",
        }
        if self.seg_enable:
            ret.update({'seg': torch.zeros_like(image)})
        return ret

def collate_fn(batch):
    """
    Custom collate function to filter out None values.
    
    Args:
    - batch (list): List of data items.
    
    Returns:
    - (tuple): Collated batch of data items.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

