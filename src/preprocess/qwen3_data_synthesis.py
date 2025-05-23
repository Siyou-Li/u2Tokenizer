# -*- encoding: utf-8 -*-
# @File        :   qwen3_data_synthesis.py
# @Time        :   2025/05/22 21:46:52
# @Author      :   Siyou
# @Description :

from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import re
import os

# Remove vLLM specific environment variables
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# # Initialize OpenAI client for local model
# client = OpenAI(
#     base_url="http://localhost:8088/v1",
#     api_key="not-needed"  # API key not needed for local model
# )

# # Remove vLLM specific configurations
# gpu_num = torch.cuda.device_count()
# # model_name = "pretrained_models/Qwen3-235B-A22B-GPTQ-Int4"
# model_name = "pretrained_models/Qwen3-8B"

# def synthesize_data(prompt, enable_thinking=True):
#     # Prepare the input to the model
#     messages = [
#         {"role": "user", "content": prompt}
#     ]

#     # Generate outputs using OpenAI client
#     response = client.chat.completions.create(
#         model=model_name,  # Model name doesn't matter for local deployment
#         messages=messages,
#         max_tokens=8192,
#         temperature=0.7,
#         top_p=0.8,
#         presence_penalty=1.5,
#         extra_body={
#             "top_k": 20,
#             "chat template kwargs": {"enable thinking": True},
#         })

#     return response.choices[0].message.reasoning_content, response.choices[0].message.content


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
    pattern = r'<think>\n(.*?)\n</think>\s*(.*)'

    m = re.search(pattern, outputs, flags=re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        output   = m.group(2).strip()
    else:
        thinking = ""
        output   = outputs
    return [(thinking, output)]

def synthesize_data_batch(prompts, enable_thinking=True):
    """
    prompts: List[str]
    enable_thinking: bool — whether to include <think> reasoning in the template
    Returns: (thinkings, outputs)
        thinkings: List[str]  the content captured between <think>…</think>
        outputs:   List[str]  the remaining generated content
    """
    # 1) Prepare each prompt with your chat template
    messages = [{"role": "user", "content": p} for p in prompts]
    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # 2) Send all to the LLM at once
    batch_results = llm.generate(texts, sampling_params)  # returns list of generations

    results = []
    pattern = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)

    # 3) Parse each result
    for result in batch_results:
        full_text = result.outputs[0].text
        m = pattern.search(full_text)
        if m:
            thinking = m.group(1).strip()
            output = m.group(2).strip()
        else:
            thinking = ""                 # no <think> block
            output = full_text.strip()
        results.append((thinking, output))

    return results

QUESTION_PROMPT_TPL = """
**original report:**{report}

**Question:** What information can we interprete from the report? Please list out as the form of questions, and questions only, in sequenced list.
""".strip()

THINKING_PROMPT_TPL = """
**original report:**{report}

**Question:** {question}

Please consider and answer the question in the following format:

Thinking: <thought process>

Answer: <answer to the question>
""".strip()

question_pattern = r".*?\d\. ?([^\n]*)"
thinking_pattern = r"Thinking: ?([^\n]*)"
answer_pattern = r"Answer: ?([^\n]*)"

def vqa_thinking(findings):
    # 1. generate questions
    prompt = QUESTION_PROMPT_TPL.format(report=findings)
    _, output = synthesize_data(prompt)[0]
    questions = re.findall(question_pattern, output, flags=re.M|re.S)

    # 2. generate thinking
    system_thinkings = []
    thinkings = []
    answers = []
    prompts = []
    for item in questions:
        prompt = THINKING_PROMPT_TPL.format(report=findings, question=item.strip())
        prompts.append(prompt)
    
    results = synthesize_data_batch(prompts, enable_thinking=True)
    for result in results:
        system_thinking, output = result
        system_thinkings.append(system_thinking)
        thinkings.append(re.findall(thinking_pattern, output, flags=re.M|re.S)[-1])
        answers.append(re.findall(answer_pattern, output, flags=re.M|re.S)[-1])

    # 3. format the results
    results = []
    for system_thinking, question, thinking, answer in zip(system_thinkings, questions, thinkings, answers):
        if len(system_thinking) > 20 and len(question) > 20 and len(thinking) > 20 and len(answer) > 20:
            results.append({
                "system_thinking": system_thinking,
                "question": question,
                "thinking": thinking,
                "answer": answer
            })

    return results


if __name__ == "__main__":
    # # Example usage
    # prompt = "Give me a short introduction to large language models."
    # thinking, output = synthesize_data(prompt)
    # print("Thinking:")
    # print(thinking)
    # print("Output:")
    # print(output)
    findings = """
    Multiple venous collaterals are noted in the anterior left chest wall, connecting to the anterior jugular vein at the right sternoclavicular junction, with a collapsed left subclavian vein suggestive of chronic occlusion; trachea and bronchi are patent; calcific plaques in the aortic arch are observed; mediastinal vascular structures, heart size, and thoracic aorta diameter are normal; no pericardial effusion or thickening; normal esophagus with no significant wall thickening; no enlarged lymph nodes in prevascular, pretracheal, subcarinal, or hilar regions; linear and subsegmental atelectasis, bronchial wall thickening, peribronchial tree-like reticulonodular densities, and minimal consolidation in the bilateral lower lobes suggestive of an infectious process; atrophic left kidney partially seen, right kidney not evaluable, normal other upper abdominal organs with no space-occupying lesions in the visible liver or bilateral adrenal glands; and thoracic vertebrae show anterior osteophyte extensions.
    """
    results = vqa_thinking(findings)
    for result in results:
        print("System Thinking:")
        print(result["system_thinking"])
        print("Question:")
        print(result["question"])
        print("Thinking:")
        print(result["thinking"])
        print("Answer:")
        print(result["answer"])
        print("-" * 50)