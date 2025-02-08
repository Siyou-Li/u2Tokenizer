"""
Parts copied from:
- https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
- https://github.com/scikit-image/scikit-image/blob/master/skimage/metrics/simple_metrics.py

Edited by: Robert Jan Schlimbach
"""
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import ssim
from einops import rearrange


def nmse(orig: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(pred - orig) ** 2 / torch.norm(orig) ** 2


def psnr(orig: torch.Tensor, pred: torch.Tensor, data_range: float) -> torch.Tensor:
    return 10 * torch.log10((data_range ** 2) / F.mse_loss(pred, orig))


class SSIM3DSlices(nn.Module):
    def __init__(self, *ssim_args, **ssim_kwargs):
        super().__init__()
        self.to_slices = partial(rearrange, pattern='b c h w d -> (b d) c h w')
        self.ssim_args = ssim_args
        self.ssim_kwargs = ssim_kwargs

    def forward(self, pred, target):
        pred, target = map(self.to_slices, (pred, target))
        return ssim(pred, target, *self.ssim_args, **self.ssim_kwargs)