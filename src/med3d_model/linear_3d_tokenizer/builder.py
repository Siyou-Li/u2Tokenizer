from .lin3dt import Linear3DTokenizer

def build_linear3dtokenizer_tower(config, **kwargs):
    return Linear3DTokenizer(
        embed_size=config.hidden_size,
        num_heads=config.l3dt_num_heads,
        num_layers=config.l3dt_num_layers,
        top_k=config.l3dt_top_k,
        use_multi_scale=config.use_multi_scale,
        num_3d_query_token=config.num_3d_query_token,
        hidden_size=config.hidden_size,
        enable_rpe=config.enable_rpe,
        enable_diffts=config.enable_diffts,
        enable_dmtp=config.enable_dmtp,
    )