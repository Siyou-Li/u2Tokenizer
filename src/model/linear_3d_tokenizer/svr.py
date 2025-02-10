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

if __name__ == '__main__':
    # Example usage
    embed_size = 512
    num_heads = 8
    num_layers = 4
    top_k = 1024
    model = SpatioTemporalVisualTokenRefinerModel(embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, top_k=top_k, use_multi_scale=True)
    video_data = torch.randn(1, 64, 256, embed_size)  # Example input: (batch_size, num_frames, num_tokens, embed_size)
    output = model(video_data)
    print(output.shape)