import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SpatioTemporalAttentionLayer, self).__init__()
        self.spatial_attention = nn.MultiheadAttention(embed_size, num_heads)
        self.temporal_attention = nn.MultiheadAttention(embed_size, num_heads)

        self.spatial_attention._reset_parameters()
        self.temporal_attention._reset_parameters()

    def forward(self, x):
        # x shape: (batch_size, num_frames, num_tokens, embed_size)
        b, t, n, e = x.size()

        # Spatial attention
        x = x.view(b * t, n, e)  # Merge batch and frames for spatial attention
        x, _ = self.spatial_attention(x, x, x)
        x = x.view(b, t, n, e)

        # Temporal attention
        x = x.permute(0, 2, 1, 3).contiguous()  # Prepare for temporal attention
        x = x.view(b * n, t, e)
        x, _ = self.temporal_attention(x, x, x)
        x = x.view(b, n, t, e).permute(0, 2, 1, 3).contiguous()  # Restore original shape

        return x


class SpatioTemporalSignificanceScoring(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers):
        super(SpatioTemporalSignificanceScoring, self).__init__()
        self.layers = nn.ModuleList([
            SpatioTemporalAttentionLayer(embed_size, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TokenSelection(nn.Module):
    def __init__(self, embed_size, top_k):
        super(TokenSelection, self).__init__()
        self.score_net = nn.Linear(embed_size, 1)
        self.top_k = top_k
        self.init_weights()

    def init_weights(self):
        if self.score_net.bias is not None:
            self.score_net.bias.data.zero_()

    def forward(self, x):
        # x shape: (batch_size, num_frames, num_tokens, embed_size)
        b, t, n, e = x.size()
        scores = self.score_net(x).squeeze(-1)  # Compute scores for each token
        scores = scores.view(b, -1)  # Flatten the frame and token dimensions

        # Select top-k tokens across all frames and tokens
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=1)

        # Convert flat indices to frame and token indices
        frame_indices = topk_indices // n
        token_indices = topk_indices % n

        # Gather top-k tokens
        topk_tokens = x[torch.arange(b).unsqueeze(1), frame_indices, token_indices]

        return topk_tokens


class SpatioTemporalVisualTokenRefinerModel(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale):
        super(SpatioTemporalVisualTokenRefinerModel, self).__init__()
        self.attention_network = SpatioTemporalSignificanceScoring(embed_size, num_heads, num_layers)
        self.token_selection = TokenSelection(embed_size, top_k)
        self.use_multi_scale = use_multi_scale

    def forward(self, x):
        x = self.attention_network(x)
        x = self.token_selection(x)

        if self.use_multi_scale:
            # Multi-scale pooling over frames
            scales = [1, 2, 4]
            pooled_outputs = []
            for scale in scales:
                if x.size(1) >= scale:  # Ensure the sequence is longer than the kernel size
                    pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=scale, stride=scale)
                    pooled_outputs.append(pooled.permute(0, 2, 1))
            # Concatenate pooled outputs along the token dimension
            x = torch.cat(pooled_outputs, dim=1)

        return x

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

    def forward(self, query, value, is_compress=False):
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

        return output

class TextConditionTokenAttMap(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TextConditionTokenAttMap, self).__init__()
        self.visual_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.text_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.dropout_cross = nn.Identity()
        self.norm_cross_v = nn.LayerNorm(d_model)
        self.norm_cross_t = nn.LayerNorm(d_model)

        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.0)
        self.dropout_self = nn.Identity()
        self.norm_self = nn.LayerNorm(d_model)

        self.self_attention._reset_parameters()
        self.norm_cross_v.reset_parameters()
        self.norm_cross_t.reset_parameters()
        self.norm_self.reset_parameters()

    def forward(self, visual_query, visual_value, text_value):
        self_out = self.self_attention(visual_query, visual_query, visual_query)[0]
        # self_out = self_out + self.dropout2(self_out)
        self_out = self.norm_self(visual_query + self_out)
        cross_out = self.visual_cross_attention(self_out, visual_value).squeeze(1)
        # cross_out = query + self.dropout1(cross_out)
        cross_out_visual = self.norm_cross_v(self_out + cross_out)
        cross_out_vt = self.text_cross_attention(cross_out_visual, text_value).squeeze(1)
        cross_out_vt = self.norm_cross_t(cross_out_visual + cross_out_vt)

        return cross_out_vt

class LinearAggregation(nn.Module):
    def __init__(self, d_model, num_heads):
        super(LinearAggregation, self).__init__()
        self.linear_aggregator = MultiHeadCrossAttention(d_model, num_heads)

    def forward(self, query_vt, visual_value):
        visual_compression = self.linear_aggregator(query_vt, visual_value, is_compress=True).squeeze(1)

        return visual_compression


class TextConditionTokenAggregatorModel(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(TextConditionTokenAggregatorModel, self).__init__()
        self.layers_vt = nn.ModuleList([TextConditionTokenAttMap(d_model, num_heads) for _ in range(num_layers)])
        self.layer_linagg = LinearAggregation(d_model, num_heads)

    def forward(self, query, visual_value, text_value):
        for layer_vt in self.layers_vt:
            query = layer_vt(query, visual_value, text_value)
        visual_compression = self.layer_linagg(query, visual_value)

        return visual_compression
    
class u2Tokenizer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale, num_3d_query_token, hidden_size):
        super(u2Tokenizer, self).__init__()
        self.svt_module = SpatioTemporalVisualTokenRefinerModel(embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, top_k=top_k, use_multi_scale=use_multi_scale)
        self.tta_module = TextConditionTokenAggregatorModel(d_model=embed_size, num_layers=num_layers, num_heads=num_heads)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_3d_query_token, hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)

    def forward(self, v_token, t_token):
        (B, T, N, E) = v_token.size()
        # make sure the query token is broadcastable
        query_tokens = self.query_tokens.expand(B, -1, -1)
        v_token = self.svt_module(v_token)
        align_token = self.tta_module(query_tokens, v_token, t_token)

        return align_token
    
def build_u2tokenizer_tower(config, **kwargs):
    return u2Tokenizer(
        embed_size=config.hidden_size,
        num_heads=config.u2t_num_heads,
        num_layers=config.u2t_num_layers,
        top_k=config.u2t_top_k,
        use_multi_scale=config.use_multi_scale,
        num_3d_query_token=config.num_3d_query_token,
        hidden_size=config.hidden_size
    )