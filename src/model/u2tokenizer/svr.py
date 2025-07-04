import torch
import torch.nn as nn
import torch.nn.functional as F
from .rma import RelativeMultiheadAttention
from .rope import RotaryMultiheadAttention

class SpatioTemporalAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads, attn_type="rma"):
        super(SpatioTemporalAttentionLayer, self).__init__()
        if attn_type == "rma":
            self.spatial_attention = RelativeMultiheadAttention(embed_size, num_heads)
            self.temporal_attention = RelativeMultiheadAttention(embed_size, num_heads)
        elif attn_type == "rope":
            self.spatial_attention = RotaryMultiheadAttention(embed_size, num_heads)
            self.temporal_attention = RotaryMultiheadAttention(embed_size, num_heads)
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
    def __init__(self, embed_size, num_heads, num_layers, attn_type="rma"):
        super(SpatioTemporalSignificanceScoring, self).__init__()
        self.layers = nn.ModuleList(
            [SpatioTemporalAttentionLayer(embed_size, num_heads, attn_type) for _ in range(num_layers)]
        )

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
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale, attn_type="rma", enable_diffts=False, enable_dmtp=False):
        super(SpatioTemporalVisualTokenRefinerModel, self).__init__()
        self.attention_network = SpatioTemporalSignificanceScoring(embed_size, num_heads, num_layers, attn_type)
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

if __name__ == '__main__':
    # Example usage
    embed_size = 512
    num_heads = 8
    num_layers = 4
    top_k = 1024
    model = SpatioTemporalVisualTokenRefinerModel(
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        top_k=top_k,
        use_multi_scale=True,
        attn_type="rope",
    )
    video_data = torch.randn(1, 64, 256, embed_size)
    output = model(video_data)
    print(output.shape)