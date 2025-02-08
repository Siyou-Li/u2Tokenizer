from typing import Optional, Tuple

import torch
from torch import nn
from einops import rearrange


def determine_num_groups(in_channels, preferred_channels_per_group=8):
    return max(in_channels // preferred_channels_per_group, 1)


def group_std(x: torch.Tensor, groups: Optional[int] = None, eps=1e-5): # type: ignore
    b, c, *dims = x.size()
    assert len(dims) >= 1

    if groups is None:
        groups = determine_num_groups(c, preferred_channels_per_group=8)

    x_g = torch.reshape(x, (b, groups, c // groups, *dims))

    var = torch.var(x_g, dim=tuple(torch.arange(2, x.dim()+1)), keepdim=True)
    std = torch.sqrt(var + eps)

    std = std.expand(-1, -1,  c // groups, *(-1 for _ in dims)).reshape(1, c, *(1 for _ in dims))

    return std


class SiLUVelocityFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor: # type: ignore
        ctx.save_for_backward(x, v)
        return x * torch.sigmoid(x*v)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # type: ignore
        x, v = ctx.saved_tensors

        xv = x*v
        sigmoid_xv = torch.sigmoid(xv)
        d_sigmoid = sigmoid_xv * (1 - sigmoid_xv)

        d_x = grad_output * (sigmoid_xv + xv * d_sigmoid)
        d_v = grad_output * (x**2 * d_sigmoid)

        return d_x, d_v


class SiLUVelocity(nn.Module):
    '''
    Performs `x * torch.sigmoid(v*x)`
    x, v need to be broadcastable w.r.t.
    '''
    def forward(self, x, v):
        return SiLUVelocityFunc.apply(x, v)


class EvoNorm3DS0(nn.Module):
    '''assumes non-linear, affine transformed EvoNorm S0'''
    def __init__(self, in_channels):
        super().__init__()

        self.silu_v = SiLUVelocity()

        self.v = nn.Parameter(torch.ones((in_channels, 1, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((in_channels, 1, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((in_channels, 1, 1, 1)))

    def forward(self, x):
        assert x.dim() == 5

        num = self.silu_v(x, self.v)
        std = group_std(x)

        return num * self.gamma / std + self.beta


def test_silu_velocity():
    module = SiLUVelocity()
    x1 = torch.randn((4, 2, 16, 16, 8), requires_grad=True).to(torch.double)
    x2 = x1.clone().detach().requires_grad_(True)
    v1 = nn.Parameter(torch.randn((x1.shape[1]))).to(torch.double)[:, None, None, None]
    v2 = v1.clone().detach().requires_grad_(True)

    x1.retain_grad()
    x2.retain_grad()
    v1.retain_grad()
    v2.retain_grad()

    torch.autograd.gradcheck(module, inputs=(x1, v1)) # backwards check

    res = module(x1, v1)
    true_res = x2 * torch.sigmoid(x2 * v2)

    assert (res == true_res).all() # forward check

    print("test passed succesfully")



if __name__ == '__main__':
    test_silu_velocity()