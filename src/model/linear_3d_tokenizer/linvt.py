import torch.nn as nn
from .svr import SpatioTemporalVisualTokenRefinerModel
from .tta import TextConditionTokenAggregatorModel

class Linear3DTokenizer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, top_k, use_multi_scale):
        super(Linear3DTokenizer, self).__init__()
        self.svt_module = SpatioTemporalVisualTokenRefinerModel(embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, top_k=top_k, use_multi_scale=use_multi_scale)
        self.tta_module = TextConditionTokenAggregatorModel(d_model=embed_size, num_layers=num_layers, num_heads=num_heads)

    def forward(self, v_quey, v_token, t_token):
        v_token = self.svt_module(v_token)
        align_token = self.tta_module(v_quey, v_token, t_token)

        return align_token