import torch
import torch.nn as nn
import torch.nn.functional as F
from .rma import RelativeMultiheadAttention

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        self.init_weights()

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

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

    def forward(self, query, value, is_compress=False, return_attn=False):
        batch_size = query.size(0)

        query = self.wq(query)
        key = self.wk(value)
        if not is_compress:
            value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Use a precomputed constant for scaling factor
        scaling_factor = torch.sqrt(torch.tensor(self.depth, dtype=query.dtype, device=query.device))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        if not is_compress:
            output = self.dense(context)
        else:
            output = context

        if return_attn:
            return output, attention_weights
        return output

class TextConditionTokenAttMap(nn.Module):
    def __init__(self, d_model, num_heads, enable_mu=False):
        super(TextConditionTokenAttMap, self).__init__()
        self.visual_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.text_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.dropout_cross = nn.Identity()
        self.norm_cross_v = nn.LayerNorm(d_model)
        self.norm_cross_t = nn.LayerNorm(d_model)
        if enable_mu:
            self.self_attention = RelativeMultiheadAttention(d_model, num_heads)
        else:
            self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.0)
        self.dropout_self = nn.Identity()
        self.norm_self = nn.LayerNorm(d_model)

        self.self_attention._reset_parameters()
        self.norm_cross_v.reset_parameters()
        self.norm_cross_t.reset_parameters()
        self.norm_self.reset_parameters()

    def forward(self, visual_query, visual_value, text_value, return_attn=False):
        self_out, self_attn = self.self_attention(visual_query, visual_query, visual_query, need_weights=True)
        # self_out = self_out + self.dropout2(self_out)
        self_out = self.norm_self(visual_query + self_out)
        cross_out, cross_attn_v = self.visual_cross_attention(self_out, visual_value, return_attn=True)
        cross_out = cross_out.squeeze(1)
        # cross_out = query + self.dropout1(cross_out)
        cross_out_visual = self.norm_cross_v(self_out + cross_out)
        cross_out_vt, cross_attn_t = self.text_cross_attention(cross_out_visual, text_value, return_attn=True)
        cross_out_vt = cross_out_vt.squeeze(1)
        cross_out_vt = self.norm_cross_t(cross_out_visual + cross_out_vt)

        if return_attn:
            return cross_out_vt, self_attn, cross_attn_v, cross_attn_t
        return cross_out_vt

class LinearAggregation(nn.Module):
    def __init__(self, d_model, num_heads):
        super(LinearAggregation, self).__init__()
        self.linear_aggregator = MultiHeadCrossAttention(d_model, num_heads)

    def forward(self, query_vt, visual_value):
        visual_compression = self.linear_aggregator(query_vt, visual_value, is_compress=True).squeeze(1)
        return visual_compression

class TextConditionTokenAggregatorModel(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, enable_mu=False):
        super(TextConditionTokenAggregatorModel, self).__init__()
        self.layers_vt = nn.ModuleList([TextConditionTokenAttMap(d_model, num_heads, enable_mu) for _ in range(num_layers)])
        self.layer_linagg = LinearAggregation(d_model, num_heads)

    def forward(self, query, visual_value, text_value, return_attn=False):
        attn_maps = {}
        for i, layer_vt in enumerate(self.layers_vt):
            # For demonstration, capture attention maps from the first layer.
            if return_attn and i == 0:
                query, self_attn, cross_attn_v, cross_attn_t = layer_vt(query, visual_value, text_value, return_attn=True)
                attn_maps['self_attn'] = self_attn
                attn_maps['cross_attn_visual'] = cross_attn_v
                attn_maps['cross_attn_text'] = cross_attn_t
            else:
                query = layer_vt(query, visual_value, text_value)
        visual_compression = self.layer_linagg(query, visual_value)
        if return_attn:
            return visual_compression, attn_maps
        return visual_compression

if __name__ == '__main__':
    # Example usage
    embed_size = 896
    num_heads = 8
    num_layers = 4
    model = TextConditionTokenAggregatorModel(embed_size, num_layers, num_heads)
    query = torch.randn(1, 64, embed_size)
    visual_data = torch.randn(1, 1792, embed_size)  # Example input: (batch_size, num_tokens, embed_size)
    text_data = torch.randn(1, 755, embed_size)
    output = model(query, visual_data, text_data)
    print(output.shape)