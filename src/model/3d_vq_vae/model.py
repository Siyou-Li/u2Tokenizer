from functools import partial
from typing import Tuple
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from .layers import Encoder, Decoder, FixupResBlock, PreActFixupResBlock, EvonormResBlock, Encoder2
from utils import ExtractCenterCylinder
from .distribution import Logistic, mixture_nll_loss, generic_nll_loss
from .evaluate import nmse, psnr, SSIM3DSlices
from utils import sub_metric_log_dict



@torch.no_grad()
def _eval_metrics_log_dict(orig, pred):
    # FIXME: remove hardcoded data range
    metrics = (
        ('nmse', nmse),
        ('psnr', partial(psnr, data_range=4)),
    )
    return {
        func_name: func(orig, pred)
        for func_name, func in metrics
    }


class VQVAE(pl.LightningModule):

    # first in line is the default
    supported_metrics = ("huber",)

    def __init__(self, args: Namespace):

        super(VQVAE, self).__init__()

        self.save_hyperparameters()
        self._parse_input_args(args)

        self.encoder = Encoder2(
            in_channels=self.input_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_down_per_enc=self.n_blocks_per_bottleneck,
            n_pre_q_blocks=self.n_pre_quantization_blocks,
            n_post_downscale_blocks=self.n_post_downscale_blocks,
            n_post_upscale_blocks=self.n_post_upscale_blocks,
            num_embeddings=self.num_embeddings,
            resblock=self.resblock,

        )
        self.decoder = Decoder(
            out_channels=self.output_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_up_per_enc=self.n_blocks_per_bottleneck,
            n_post_q_blocks=self.n_post_quantization_blocks,
            n_post_upscale_blocks=self.n_post_upscale_blocks,
            resblock=self.resblock,
        )

        self.train_pre_loss_f_metrics = nn.ModuleDict({
            # 'ssim': SSIM3DSlices(),
        })
        self.val_pre_loss_f_metrics = nn.ModuleDict({
            'ssim': SSIM3DSlices(),
        })

        def init_fixupresblock(layer):
            if isinstance(layer, FixupResBlock) or isinstance(layer, PreActFixupResBlock):
                layer.initialize_weights(num_layers=self.num_layers)
        self.apply(init_fixupresblock)

    def forward(self, data):
        commitment_loss, quantizations, encoding_idx = zip(*self.encode(data))

        decoded = self.decode(quantizations)
        return decoded, (commitment_loss, quantizations, encoding_idx)

    def encode(self, data):
        return self.encoder(data)

    def decode(self, quantizations):
        return self.decoder(quantizations)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='val')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'val')

        pre_loss_f_metrics = (
            self.train_pre_loss_f_metrics
            if mode == 'train'
            else self.val_pre_loss_f_metrics
        )

        loss, log_dict = self.recon_loss_f(batch, batch_idx, **pre_loss_f_metrics)
        self.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()}, logger=True)

        return loss

    def loc_metric(self, batch, batch_idx, loss_f, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        x, num_valid_slices = batch

        loc, (commitment_loss, *_) = self(x)
        # loc = F.softplus(loc)
        loc = F.elu(loc)

        masks = torch.zeros_like(x, dtype=torch.bool, requires_grad=False)
        for idx, mask in zip(num_valid_slices, masks):
            mask[..., idx:] += True

        # Set padded values to 0
        loc = torch.masked_fill(loc, mask=masks, value=0)

        log_dict = {}

        # Pre loss metrics need full 3d
        log_dict.update({
            k: v for d in
            (sub_metric_log_dict(metric_name, metric(loc.half(), x.half())) for metric_name, metric in pre_loss_f_metrics.items())
            for k, v in d.items()
        })

        if self.pre_loss_f:
            loc, x = map(self.pre_loss_f, (loc, x))

        unreduced_recon_loss = loss_f(loc, x, reduction='none')

        for log_metric in (
            sub_metric_log_dict('recon_loss', unreduced_recon_loss),
            {f'commitment_loss_{i}': commitment_loss[i] for i in range(len(commitment_loss))},
            sub_metric_log_dict('loc', loc),
            _eval_metrics_log_dict(orig=x, pred=loc),
        ):
            log_dict.update(log_metric)


        recon_loss = unreduced_recon_loss.mean()
        commitment_loss = sum(commitment_loss)

        loss = recon_loss + commitment_loss

        self.log('recon loss', recon_loss, prog_bar=True, logger=False)
        self.log('commitment loss', commitment_loss, prog_bar=True, logger=False)

        return loss, log_dict

    def huber(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.smooth_l1_loss, **pre_loss_f_metrics)

    def _parse_input_args(self, args: Namespace):
        assert args.metric in self.supported_metrics

        if args.metric == 'huber':
            self.recon_loss_f = self.huber

        self.metric = args.metric
        self.lr = args.base_lr

        self.input_channels = args.input_channels
        self.output_channels = args.input_channels
        self.base_network_channels = args.base_network_channels
        self.n_bottleneck_blocks = args.n_bottleneck_blocks
        self.n_blocks_per_bottleneck = args.n_downscales_per_bottleneck
        self.n_pre_quantization_blocks = args.n_pre_quantization_blocks
        self.n_post_quantization_blocks = args.n_post_quantization_blocks
        self.n_post_upscale_blocks = args.n_post_upscale_blocks
        self.n_post_downscale_blocks = args.n_post_downscale_blocks

        assert len(args.num_embeddings) in (1, args.n_bottleneck_blocks)
        if len(args.num_embeddings) == 1:
            self.num_embeddings = [args.num_embeddings[0] for _ in range(args.n_bottleneck_blocks)]
        else:
            self.num_embeddings = args.num_embeddings

        resblocks = {'regular': FixupResBlock, 'pre-activation': PreActFixupResBlock, 'evonorm': EvonormResBlock}
        self.resblock = resblocks[args.block_type]

        # num_layers is defined as the longest path through the model
        n_down = args.n_bottleneck_blocks * args.n_downscales_per_bottleneck
        self.num_layers = (
            2 # input + output layer
            + 2 * n_down # down and up
            + args.n_pre_quantization_blocks
            + args.n_post_quantization_blocks
            + args.n_post_downscale_blocks * n_down
            + args.n_post_upscale_blocks * n_down
            + 1 # pre-activation block
        )

        #     (3 * args.n_bottleneck_blocks - 1)
        #     * args.n_blocks_per_bottleneck
        #     + (2 * args.n_bottleneck_blocks)
        # )

        self.pre_loss_f = ExtractCenterCylinder() if args.extract_center_cylinder else None


    @classmethod
    def add_model_specific_args(cls, parent_parser):


        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model specific arguments
        parser.add_argument('--input-channels', type=int, default=1)
        parser.add_argument('--base-network_channels', type=int, default=4)
        parser.add_argument('--n-bottleneck-blocks', type=int, default=3)
        parser.add_argument('--n-downscales-per-bottleneck', type=int, default=2)
        parser.add_argument('--n-pre-quantization-blocks', type=int, default=0)
        parser.add_argument('--n-post-quantization-blocks', type=int, default=0)
        parser.add_argument('--n-post-upscale-blocks', type=int, default=0)
        parser.add_argument('--n-post-downscale-blocks', type=int, default=0)
        parser.add_argument('--num-embeddings', type=int, default=256, nargs='+',
                            help=("Can be either a single int or multiple."
                                  " If multiple, number of args should be equal to n-bottleneck-blocks"))
        parser.add_argument('--block-type', type=str, default='pre-activation', choices=['regular', 'pre-activation', 'evonorm'])
        # parser.add_argument('--bottleneck-divisor', type=int, default=4,
                            # help='ignored if --block-type regular is used')

        # loss calculation specific
        parser.add_argument('--extract-center-cylinder', type=bool, default=True)
        parser.add_argument('--metric', choices=cls.supported_metrics, default=cls.supported_metrics[0])

        # Optimizer specific arguments
        parser.add_argument('--base_lr', default=1e-5, type=float)

        # Loss function specific arguments
        # FIXME: put this is a class or something, at least not here
        parser.add_argument('--n-mix', default=2)

        return parser