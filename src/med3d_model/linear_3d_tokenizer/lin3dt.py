import torch
import torch.nn as nn
from .svr import SpatioTemporalVisualTokenRefinerModel
from .tta import TextConditionTokenAggregatorModel

class Linear3DTokenizer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale, num_3d_query_token, hidden_size, enable_mu=False):
        super(Linear3DTokenizer, self).__init__()
        self.svt_module = SpatioTemporalVisualTokenRefinerModel(embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, top_k=top_k, use_multi_scale=use_multi_scale, enable_mu=enable_mu)
        self.tta_module = TextConditionTokenAggregatorModel(d_model=embed_size, num_layers=num_layers, num_heads=num_heads, enable_mu=enable_mu)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_3d_query_token, hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)

    def forward(self, v_token, t_token):
        (B, T, N, E) = v_token.size()
        # make sure the query token is broadcastable
        query_tokens = self.query_tokens.expand(B, -1, -1)
        v_token = self.svt_module(v_token)
        align_token = self.tta_module(query_tokens, v_token, t_token)

        return align_token
    