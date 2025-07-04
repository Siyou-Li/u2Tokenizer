import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryMultiheadAttention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.dense.weight)
        if self.wq.bias is not None:
            nn.init.zeros_(self.wq.bias)
        if self.wk.bias is not None:
            nn.init.zeros_(self.wk.bias)
        if self.wv.bias is not None:
            nn.init.zeros_(self.wv.bias)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, return_attn: bool = True, **kwargs
    ):
        if "need_weights" in kwargs:
            return_attn = kwargs.pop("need_weights")
        bsz, seq_len, _ = query.size()

        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self.split_heads(q, bsz)
        k = self.split_heads(k, bsz)
        v = self.split_heads(v, bsz)

        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(bsz, seq_len, self.d_model)
        output = self.dense(context)

        if return_attn:
            return output, attn
        return output
