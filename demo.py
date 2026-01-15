import argparse
import gzip
import inspect
import json
import os
import struct
import sys
import types

import torch.nn.functional as F
import torch
import nibabel as nib
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from monai.data.image_reader import NibabelReader
from monai.transforms import (
        LoadImage,
        Compose,
        CropForeground,
        ToTensor,
        SaveImage,
        ScaleIntensityRangePercentiles,
        RandRotate90,
        RandFlip,
        NormalizeIntensity,
        RandScaleIntensity,
        RandShiftIntensity
    )
from monai.transforms.spatial.functional import resize


_DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
}
class u2Transform:
    def __init__(self, mode='bilinear', device="cpu"):
        transforms = Compose(
                [
                #LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_max=1.0, b_min=0.0, clip=True),
                CropForeground(source_key="image"),
                #NormalizeIntensity(),   
                ToTensor(),
                ]
            )
        self.adaptive_transforms = transforms
        self.mode = mode
        self.save = SaveImage(separate_folder=False, output_postfix='')
        self.device = device

    def adaptive_resize(self, input_path, target_image_size=256, padding_size=32*8):
        """
        adaptive resize the NIfTI file to the target size.
        The minimum dimension is scaled to the target dimension, and other dimensions are scaled proportionally
        """
        data = nib.load(input_path).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]
        data = torch.tensor(data, device=self.device)
        data = self.adaptive_transforms(data)[0]
        data = torch.permute(data,(1, 2, 0))
        
        input_shape = data.shape
        # print(f"Input shape: {input_shape}")
        ratio = min([target_image_size / input_shape[i] for i in range(2)])
        scaling_shape = [int(input_shape[i] * ratio) for i in range(2)]
        # print(f"Scaling shape: {scaling_shape}")

        # padding the image to [padding_size, target_image_size, target_image_size]
        if padding_size >= input_shape[2]:
            scaling_shape.append(input_shape[2])
            data = resize(
                img=data.unsqueeze(0), 
                out_size=scaling_shape, 
                mode=self.mode,
                align_corners=True,
                dtype=None,
                input_ndim=3,
                anti_aliasing= True,
                anti_aliasing_sigma=None,
                lazy=False,
                transform_info=None,
                )
            pad_tuple = (0, padding_size - scaling_shape[2], 0, target_image_size - scaling_shape[1], 0, target_image_size - scaling_shape[0])
            data = F.pad(data, pad_tuple, mode='constant', value=0)
        else:
            scaling_shape.append(padding_size)
            data = resize(
                img=data.unsqueeze(0), 
                out_size=scaling_shape, 
                mode=self.mode,
                align_corners=True,
                dtype=None,
                input_ndim=3,
                anti_aliasing= True,
                anti_aliasing_sigma=None,
                lazy=False,
                transform_info=None,
                )
            # crop the image to [padding_size, target_image_size, target_image_size]
            pad_tuple = (0, 0, 0, target_image_size - scaling_shape[1], 0, target_image_size - scaling_shape[0])
            data = F.pad(data, pad_tuple, mode='constant', value=0)
            # data = data[:, :, :, :padding_size]
        # print("max:", data.max())
        # print("min:", data.min())
        # self.save(data, "/import/c4dm-04/siyoul/u2Tokenizer/amos_0001_resized.nii.gz")
        # print(f"Output shape: {data.shape}")
        data = torch.permute(data,(0, 3, 1, 2))
        # print(f"Output shape: {data.shape}")
        # split the date to multiple slices, every 32 slices is a batch
        data = data.view(-1, 32, target_image_size, target_image_size)
        # print(f"Output shape: {data.shape}")
        return data
        
    def __call__(self, *args, **kwds):
        return self.adaptive_resize(*args, **kwds)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoint/u2Qwen3-4B-Instruct")
    parser.add_argument("--image-path", default="example.nii.gz", help="NIfTI file (.nii or .nii.gz).")
    parser.add_argument("--question", default="你可以描述这张医学影像吗？", help="The question about the image.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    device = next(model.parameters()).device
    model.eval()

    target_dhw = tuple(int(x) for x in getattr(model.config, "image_size", (32, 256, 256)))
    image_transforms = u2Transform(mode="bilinear", device=device)
    image = image_transforms(args.image_path).unsqueeze(0).to(dtype)

    proj_out_num = getattr(getattr(model.get_model(), "mm_projector", None), "proj_out_num", 256)
    image_tokens = "<im_patch>" * int(proj_out_num)
    prompt = image_tokens + args.question

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    question_ids = tokenizer(args.question, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    # Transformers >=4.57 passes `cache_position` to forward() during generation; older custom model
    # implementations might not accept it yet.
    if "cache_position" not in inspect.signature(model.forward).parameters:
        original_forward = model.forward

        def _forward_compat(self, *f_args, **f_kwargs):
            f_kwargs.pop("cache_position", None)
            return original_forward(*f_args, **f_kwargs)

        model.forward = types.MethodType(_forward_compat, model)

    with torch.no_grad():
        output_ids = model.generate(
            images=image,
            inputs=input_ids,
            question_ids=question_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            # 强制指定停止符
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            # 防止复读机
            repetition_penalty=1.1
        )

    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
