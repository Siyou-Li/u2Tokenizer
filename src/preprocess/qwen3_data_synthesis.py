# -*- encoding: utf-8 -*-
# @File        :   qwen3_data_synthesis.py
# @Time        :   2025/05/22 21:46:52
# @Author      :   Siyou
# @Description :

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import re
import os

os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

gpu_num = torch.cuda.device_count()
model_name = "pretrained_models/Qwen3-235B-A22B-GPTQ-Int4"
model_name = "pretrained_models/Qwen3-8B"

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096)

# Initialize the vLLM engine
llm = LLM(model=model_name, tensor_parallel_size=gpu_num)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def synthesize_data(prompt, enable_thinking=True):
    # Prepare the input to the model
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,  # Set to False to strictly disable thinking
    )

    # Generate outputs
    outputs = llm.generate([text], sampling_params)[0].outputs[0].text
    pattern = r'<think>(.*?)</think>\s*(.*)'

    # 使用 re.S 让 . 匹配换行
    m = re.search(pattern, text, re.S)
    if m:
        thinking = m.group(1).strip()
        output   = m.group(2).strip()
    else:
        thinking = ""
        output   = outputs
    return thinking, output

if __name__ == "__main__":
    # Example usage
    prompt = "Give me a short introduction to large language models."
    thinking, output = synthesize_data(prompt)
    print("Thinking:")
    print(thinking)
    print("Output:")
    print(output)
