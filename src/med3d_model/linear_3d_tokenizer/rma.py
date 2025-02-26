import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=512):
        super(RelativeMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        assert d_model % num_heads == 0

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.max_seq_len = max_seq_len
        # Relative bias table: one bias per relative position per head.
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_seq_len - 1, num_heads))
        self.init_weights()

    def init_weights(self):
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
        nn.init.zeros_(self.relative_bias)

    # Add this method to support _reset_parameters calls
    def _reset_parameters(self):
        self.init_weights()

    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.num_heads, self.depth)  # (B, seq_len, num_heads, depth)
        return x.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, depth)

    def forward(self, query, key, value, is_compress=False, return_attn=True, **kwargs):
        # Map 'need_weights' to 'return_attn' if provided.
        if 'need_weights' in kwargs:
            return_attn = kwargs.pop('need_weights')
        batch_size, seq_len, _ = query.size()

        query = self.wq(query)
        key = self.wk(key)
        if not is_compress:
            value = self.wv(value)
        query = self.split_heads(query, batch_size)  # (B, num_heads, seq_len, depth)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaling_factor = torch.sqrt(torch.tensor(self.depth, dtype=query.dtype, device=query.device))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor  # (B, num_heads, seq_len, seq_len)

        # Compute relative positional bias.
        positions = torch.arange(seq_len, device=query.device)
        rel_pos = positions[None, :] - positions[:, None]  # (seq_len, seq_len)
        rel_pos_index = rel_pos + self.max_seq_len - 1  # shift indices to be non-negative
        # Fetch bias for each relative position and rearrange to (1, num_heads, seq_len, seq_len)
        bias = self.relative_bias[rel_pos_index]  # (seq_len, seq_len, num_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        scores = scores + bias

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, value)  # (B, num_heads, seq_len, depth)
        context = context.permute(0, 2, 1, 3).contiguous()  # (B, seq_len, num_heads, depth)
        context = context.view(batch_size, seq_len, self.d_model)  # (B, seq_len, d_model)
        if not is_compress:
            output = self.dense(context)
        else:
            output = context

        if return_attn:
            return output, attention_weights
        return output