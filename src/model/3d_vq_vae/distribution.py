from typing import Optional, Union, Tuple, Type
from functools import cached_property
from math import pi

import torch
import torch.distributions as dist
from torch.distributions.mixture_same_family import MixtureSameFamily


class Logistic(dist.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc, self.scale = dist.utils.broadcast_all(loc, scale)

        zero, one = torch.Tensor([0, 1]).type_as(loc)

        base_distribution = dist.Uniform(zero, one).expand(self.loc.shape)
        transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=self.loc, scale=self.scale)]

        super(Logistic, self).__init__(base_distribution, transforms)


def mixture_nll_loss(
    x: torch.Tensor,
    base_dist: Type[dist.Distribution],
    n_mix: int,
    mixture_comp_logits: torch.Tensor,
    reduce_sum: bool = True,
    **base_dist_kwargs
) -> torch.Tensor:
    '''
    x has minimum dim of B x W
    expects every base_dist_kwarg to be a Sequence of length n_mix
    '''
    # mixture distribution needs to have channel last
    mixture_comp_logits, base_dist_kwargs = _fix_mixture_shapes(
        n_mix, mixture_comp_logits, **base_dist_kwargs
    )

    pi_k = dist.Categorical(logits=mixture_comp_logits)
    dists = base_dist(**base_dist_kwargs)

    nll_loss = generic_nll_loss(
        x,
        base_dist=MixtureSameFamily,
        mixture_distribution=pi_k,
        component_distribution=dists,
        reduce_sum=reduce_sum
    )

    return nll_loss


def sample_mixture(
    base_dist: Type[dist.Distribution],
    n_mix: int,
    mixture_comp_logits: torch.Tensor,
    greedy=True,
    **base_dist_kwargs
) -> torch.Tensor:


    mixture_comp_logits, base_dist_kwargs = _fix_mixture_shapes(
        n_mix, mixture_comp_logits, **base_dist_kwargs
    )

    if greedy:
        mixture_comp_logits = mixture_comp_logits.clone()

        pos_bool_idx = torch.nn.functional.one_hot(
            mixture_comp_logits.max(dim=-1)[1], n_mix
        ).to(dtype=torch.bool)

        mixture_comp_logits[pos_bool_idx] = 0
        mixture_comp_logits[~pos_bool_idx] = float('-inf')

    pi_k = dist.Categorical(logits=mixture_comp_logits)
    dists = base_dist(**base_dist_kwargs)

    mixture_model = MixtureSameFamily(
        mixture_distribution=pi_k,
        component_distribution=dists,
    )
    # max_loc_idx = mixture_model.log_prob(base_dist_kwargs['loc'].transpose(0,-1).squeeze()).max(dim=0)[1]
    # bool_mask = torch.nn.functional.one_hot(max_loc_idx).to(torch.bool)
    # locs = base_dist_kwargs['loc'].squeeze()[bool_mask].reshape(512,512,128)
    # return locs
    return mixture_model.sample()


def generic_nll_loss(
    x: torch.Tensor,
    base_dist: Type[dist.Distribution],
    reduce_sum: bool = True,
    **base_dist_kwargs
) -> torch.Tensor:

    nll_loss = -base_dist(**base_dist_kwargs).log_prob(x.squeeze())
    return (
        nll_loss
        if not reduce_sum
        else nll_loss.sum()
    )


def _fix_mixture_shapes(n_mix, mixture_comp_logits, **base_dist_kwargs):
    num_dims = len(mixture_comp_logits.shape)
    assert num_dims >= 2
    axes = (0, *range(2, num_dims), 1)

    _, channel, *_ = mixture_comp_logits.shape
    mixture_comp_logits = mixture_comp_logits.permute(axes)
    assert channel == n_mix
    for param, values in base_dist_kwargs.items():
        _, channel, *_ = values.shape
        assert channel == n_mix
        base_dist_kwargs[param] = values.permute(axes)

    return mixture_comp_logits, base_dist_kwargs


if __name__ == '__main__':
    n_mix = 10
    batch_size = 5

    dim = (28, 28)

    test_in = torch.randn((batch_size, 1, *dim))

    mix, locs, scales = torch.randn((3, batch_size, n_mix, *dim))
    scaled = scales.exp()

    loss = mixture_nll_loss(test_in, Logistic, n_mix, mix, loc=locs, scale=scales)
    print(loss)

    sample = sample_mixture(Logistic, n_mix, mix, loc=locs, scale=scales)
    print(sample.shape)

