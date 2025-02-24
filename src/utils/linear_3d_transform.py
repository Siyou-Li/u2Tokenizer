# -*- encoding: utf-8 -*-
# @File        :   ct_rate_resize.py
# @Time        :   2025/01/13 17:41:14
# @Author      :   Siyou
# @Description :

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
import torch.nn.functional as F
import torch
import nibabel as nib
import numpy as np
from config import config

base_path = config["project_path"]
class Linear3DTransform:
    def __init__(self, mode='bilinear', data_type="validation", device="cpu"):
        if data_type == "training":
            transforms = Compose(
                [
                #LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_max=1.0, b_min=0.0, clip=True),
                CropForeground(source_key="image"),
                #NormalizeIntensity(),   
                RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                RandFlip(prob=0.10, spatial_axis=0),
                RandFlip(prob=0.10, spatial_axis=1),
                RandFlip(prob=0.10, spatial_axis=2),
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
                ToTensor(),
                ]
            )
        else:
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
        # self.save(data, "/import/c4dm-04/siyoul/Med3DLLM/amos_0001_resized.nii.gz")
        # print(f"Output shape: {data.shape}")
        data = torch.permute(data,(0, 3, 1, 2))
        # print(f"Output shape: {data.shape}")
        # split the date to multiple slices, every 32 slices is a batch
        data = data.view(-1, 32, target_image_size, target_image_size)
        # print(f"Output shape: {data.shape}")
        return data
        
    def __call__(self, *args, **kwds):
        return self.adaptive_resize(*args, **kwds)

if __name__ == "__main__":
    l3d_t = Linear3DTransform()
    # shape: [512,512, 212]
    input_path = "/import/c4dm-04/siyoul/Med3DLLM/datasets/AMOS-MM/imagesTr/amos_7284.nii.gz"
    output_path = "/import/c4dm-04/siyoul/Med3DLLM/amos_0001_resized.nii.gz"
    data = l3d_t.adaptive_resize(input_path)
    print(data.shape)
    # import os
    # from multiprocessing import Process
    # from src.utils.array_split import array_split
    # image_dir = os.path.join(base_path, "datasets/CT-RATE/dataset/train")

    # num_workers = 32
    # def worker(work_id, sub_image_dir, image_dir=image_dir):
    #     nifti_utils = NlfTIUtils()
        
    #     for dir in sub_image_dir:
    #         nii_dirs_1 = os.listdir(os.path.join(image_dir, dir))
    #         for dir_1 in nii_dirs_1:
    #             nii_dirs_2 = os.listdir(os.path.join(image_dir, dir, dir_1))
    #             for file in nii_dirs_2:
    #                 path = os.path.join(image_dir, dir, dir_1, file)
    #                 print(f"worker-{work_id} Processing {path}")
    #                 try:
    #                     nifti_utils.NlfTI_adaptive_resize(path, path)
    #                 except Exception as e:
    #                     print(e)
    #                     print(path)
    #                     continue
    # # split the image dir
    # sub_image_dirs = array_split(os.listdir(image_dir), num_workers)
    # print("Number of image dirs:", len(os.listdir(image_dir)))
    # process_list = []
    # for i, sub_image_dir in enumerate(sub_image_dirs):
    #     p = Process(target=worker, args=(i, sub_image_dir))
    #     p.start()
    #     process_list.append(p)
    # for p in process_list:
    #     p.join()