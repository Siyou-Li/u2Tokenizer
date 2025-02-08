from transformers import LlamaConfig


class LamedConfig(LlamaConfig):
    model_type = "lamed_llama"