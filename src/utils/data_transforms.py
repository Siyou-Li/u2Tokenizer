# -*- encoding: utf-8 -*-
# @File        :   data_transforms.py
# @Time        :   2024/11/28 13:13:12
# @Author      :   Siyou
# @Description :


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
        ToNumpy,
    )
from src.utils.utils import normalize
import torch

train_transforms = Compose(
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
val_transforms = Compose(
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

mask_transforms = Compose(
        [
            # preprocessing
            # EnsureChannelFirst(),
            LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
            Lambda(func=normalize),
            Flip(spatial_axis=2),
            Rotate90(k=1, spatial_axes=(0, 1)),         
            EnsureType(track_meta=False),
            CropForeground(source_key="image"),
            Resize(spatial_size=[8, 16, 16], mode='bilinear'),  # trilinear C, D, H, W,
            ToTensor()
            ]
        )