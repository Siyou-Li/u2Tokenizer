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

class SpatioTemporalAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads, enable_rpe=False):
        super(SpatioTemporalAttentionLayer, self).__init__()
        if enable_rpe:
            self.spatial_attention = RelativeMultiheadAttention(embed_size, num_heads)
            self.temporal_attention = RelativeMultiheadAttention(embed_size, num_heads)
        else:
            self.spatial_attention = nn.MultiheadAttention(embed_size, num_heads)
            self.temporal_attention = nn.MultiheadAttention(embed_size, num_heads)

        self.spatial_attention._reset_parameters()
        self.temporal_attention._reset_parameters()

    def forward(self, x, return_attn=False):
        # x shape: (batch_size, num_frames, num_tokens, embed_size)
        b, t, n, e = x.size()

        # Spatial attention
        x = x.view(b * t, n, e)  # Merge batch and frames for spatial attention
        x, attn_spatial = self.spatial_attention(x, x, x)
        x = x.view(b, t, n, e)

        # Temporal attention
        x = x.permute(0, 2, 1, 3).contiguous()  # Prepare for temporal attention
        x = x.view(b * n, t, e)
        x, attn_temporal = self.temporal_attention(x, x, x)
        x = x.view(b, n, t, e).permute(0, 2, 1, 3).contiguous()  # Restore original shape

        if return_attn:
            return x, attn_spatial, attn_temporal
        return x


class SpatioTemporalSignificanceScoring(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, enable_rpe=False):
        super(SpatioTemporalSignificanceScoring, self).__init__()
        self.layers = nn.ModuleList([
            SpatioTemporalAttentionLayer(embed_size, num_heads, enable_rpe) for _ in range(num_layers)
        ])

    def forward(self, x, return_attn=False):
        attn_maps = {}
        for i, layer in enumerate(self.layers):
            # For illustration, capture attention maps only from the first layer.
            if return_attn and i == 0:
                x, attn_spatial, attn_temporal = layer(x, return_attn=True)
                attn_maps['spatial'] = attn_spatial  # shape: (b*t, n, n)
                attn_maps['temporal'] = attn_temporal  # shape: (b*n, t, t)
            else:
                x = layer(x)
        if return_attn:
            return x, attn_maps
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

class DifferentiableTokenSelection(nn.Module):
    def __init__(self, embed_size, top_k, tau=1.0):
        super(DifferentiableTokenSelection, self).__init__()
        # Extend the scoring layer to output top_k scores per token.
        self.score_net = nn.Linear(embed_size, top_k)
        self.top_k = top_k
        self.tau = tau

    def forward(self, x):
        # x shape: (batch_size, num_frames, num_tokens, embed_size)
        b, t, n, e = x.size()
        # Compute scores per token; output shape: (b, t, n, top_k)
        scores = self.score_net(x)
        scores = scores.view(b, t * n, self.top_k)  # flatten frames and tokens
        # Softmax with temperature over all tokens for each selection head
        weights = F.softmax(scores / self.tau, dim=1)  # (b, t*n, top_k)
        x_flat = x.view(b, t * n, e)  # (b, t*n, e)
        # For each selection head, compute a weighted sum of token embeddings.
        selected_tokens = []
        for r in range(self.top_k):
            w = weights[:, :, r].unsqueeze(-1)  # (b, t*n, 1)
            token_r = torch.sum(w * x_flat, dim=1)  # (b, e)
            selected_tokens.append(token_r.unsqueeze(1))
        selected_tokens = torch.cat(selected_tokens, dim=1)  # (b, top_k, e)
        return selected_tokens

class DynamicMultiScalePooling(nn.Module):
    def __init__(self, embed_size, scales=[1, 2, 4]):
        super(DynamicMultiScalePooling, self).__init__()
        self.scales = scales
        # A simple gating network to compute a scalar weight per pooled output.
        self.gate_fc = nn.Linear(embed_size, 1)

    def forward(self, x):
        # x shape: (B, num_tokens, embed_size)
        pooled_outputs = []
        gate_values = []
        for scale in self.scales:
            if x.size(1) >= scale:
                # Apply average pooling (pooling is done along the token dimension)
                pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=scale, stride=scale)
                pooled = pooled.permute(0, 2, 1)  # (B, new_len, embed_size)
                pooled_outputs.append(pooled)
                # Compute a gating value via global average pooling + linear layer.
                pooled_mean = pooled.mean(dim=1)  # (B, embed_size)
                gate = self.gate_fc(pooled_mean)   # (B, 1)
                gate_values.append(gate)
        if len(gate_values) == 0:
            return x
        # Compute dynamic weights over scales.
        gate_values = torch.cat(gate_values, dim=1)  # (B, num_scales)
        weights = F.softmax(gate_values, dim=1)        # (B, num_scales)
        weighted_pooled = []
        for i, pooled in enumerate(pooled_outputs):
            w = weights[:, i].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            weighted_pooled.append(pooled * w)
        # Concatenate the weighted pooled outputs along the token dimension.
        out = torch.cat(weighted_pooled, dim=1)
        return out
    
class SpatioTemporalVisualTokenRefinerModel(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale, enable_rpe=False, enable_diffts=False, enable_dmtp=False):
        super(SpatioTemporalVisualTokenRefinerModel, self).__init__()
        self.attention_network = SpatioTemporalSignificanceScoring(embed_size, num_heads, num_layers, enable_rpe)
        if enable_diffts:
            self.token_selection = DifferentiableTokenSelection(embed_size, top_k)
        else:
            self.token_selection = TokenSelection(embed_size, top_k)
        if enable_dmtp:
            self.dynamic_pool = DynamicMultiScalePooling(embed_size)
        self.enable_dmtp = enable_dmtp
        self.use_multi_scale = use_multi_scale

    def forward(self, x, return_attn=False):
        if return_attn:
            x, attn_maps = self.attention_network(x, return_attn=True)
        else:
            x = self.attention_network(x)
        x = self.token_selection(x)

        if self.use_multi_scale:
            if self.enable_dmtp:
                x = self.dynamic_pool(x)
            else:
                # Apply multi-scale pooling over the token dimension.
                scales = [1, 2, 4]
                pooled_outputs = []
                for scale in scales:
                    if x.size(1) >= scale:
                        pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=scale, stride=scale)
                        pooled_outputs.append(pooled.permute(0, 2, 1))
                x = torch.cat(pooled_outputs, dim=1)

        if return_attn:
            return x, attn_maps
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
    def __init__(self, d_model, num_heads, enable_rpe=False):
        super(TextConditionTokenAttMap, self).__init__()
        self.visual_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.text_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.dropout_cross = nn.Identity()
        self.norm_cross_v = nn.LayerNorm(d_model)
        self.norm_cross_t = nn.LayerNorm(d_model)
        if enable_rpe:
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
    def __init__(self, d_model, num_layers, num_heads, enable_rpe=False):
        super(TextConditionTokenAggregatorModel, self).__init__()
        self.layers_vt = nn.ModuleList([TextConditionTokenAttMap(d_model, num_heads, enable_rpe) for _ in range(num_layers)])
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
    
class u2Tokenizer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale, num_3d_query_token, hidden_size, enable_rpe=False, enable_diffts=False, enable_dmtp=False):
        super(u2Tokenizer, self).__init__()
        self.svt_module = SpatioTemporalVisualTokenRefinerModel(embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, top_k=top_k, use_multi_scale=use_multi_scale, enable_rpe=enable_rpe, enable_diffts=enable_diffts, enable_dmtp=enable_dmtp)
        self.tta_module = TextConditionTokenAggregatorModel(d_model=embed_size, num_layers=num_layers, num_heads=num_heads, enable_rpe=enable_rpe)
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
        hidden_size=config.hidden_size,
        enable_rpe=config.enable_rpe,
        enable_diffts=config.enable_diffts,
        enable_dmtp=config.enable_dmtp,
    )