from .u2Tokenizer import u2Tokenizer

def build_u2tokenizer_tower(config, **kwargs):
    return u2Tokenizer(
        embed_size=config.hidden_size,
        num_heads=config.u2t_num_heads,
        num_layers=config.u2t_num_layers,
        top_k=config.u2t_top_k,
        use_multi_scale=config.use_multi_scale,
        num_3d_query_token=config.num_3d_query_token,
        hidden_size=config.hidden_size,
        attn_type=getattr(config, "attn_type", "rma"),
        enable_diffts=config.enable_diffts,
        enable_dmtp=config.enable_dmtp,
    )