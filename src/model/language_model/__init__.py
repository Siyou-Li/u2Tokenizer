from .u2llama import u2LlamaForCausalLM
from .u2phi3 import u2Phi3ForCausalLM
from .u2qwen3 import u2Qwen3ForCausalLM

# Qwen3.5 adapter is a skeleton until transformers≥5.x is adopted; importing
# it here keeps the public symbol table stable but the class only fails on
# first use (see u2qwen35.py).
try:
    from .u2qwen35 import u2Qwen35ForCausalLM  # noqa: F401
except Exception:  # pragma: no cover
    pass