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
    Returns: List[Tuple[str, str]]
        thinkings: List[str]  the content captured between <think>…</think>
        outputs:   List[str]  the remaining generated content
    """
    # 1) Prepare each prompt with your chat template
    messages = [[{"role": "user", "content": p}] for p in prompts]
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
You are a radiology medicine expert.
Your task is to answer the following radiology medicine question, using the patient’s medical record report provided below.
When writing your thought process, imagine you are directly reviewing the patient’s radiology images (do not mention the report), and describe your logical reasoning step by step as an expert would.
Then, provide your final, correct answer to the question.
Your response will be used to guide and improve the training of a multimodal large language model for radiology medicine images.
And here is radiology report what you can see:
```
{report}
```

Now we have a question:
```
{question}
```

Please consider and answer the question in the following format:

Thinking: <thought process>

Answer: <answer to the question>
""".strip()

question_pattern = r".*?\d\. ?([^\n]*)"
thinking_pattern = r"Thinking: ?([^\n]*)"
answer_pattern = r"Answer: ?([^\n]*)"

def vqa_thinking(finding):
    # 1. generate questions
    prompt = QUESTION_PROMPT_TPL.format(report=finding)
    _, output = synthesize_data(prompt)[0]
    questions = re.findall(question_pattern, output, flags=re.M|re.S)
    
    # 2. generate thinking
    system_thinkings = []
    thinkings = []
    answers = []
    prompts = []
    for item in questions:
        prompt = THINKING_PROMPT_TPL.format(report=finding, question=item.strip())
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

class VQAThinking:
    def __init__(self, finding, image):
        self.finding = finding
        self.image = image
        self.questions = None
        self.thinkings = None
        self.answers = None
        self.system_thinkings = None

    @property
    def question_prompt(self):
        return QUESTION_PROMPT_TPL.format(report=self.finding)
    
    @property
    def thinking_prompts(self):
        return [THINKING_PROMPT_TPL.format(report=self.finding, question=q) for q in self.questions]

    def parse_questions_from_output(self, output):
        self.questions = re.findall(question_pattern, output, flags=re.M|re.S)

    def parse_thinkings_and_answers_from_outputs(self, outputs, system_thinkings):
        self.thinkings = [re.findall(thinking_pattern, output, flags=re.M|re.S)[-1] for output in outputs]
        self.answers = [re.findall(answer_pattern, output, flags=re.M|re.S)[-1] for output in outputs]
        self.system_thinkings = system_thinkings

    def format_results(self):
        results = []
        for question, thinking, answer, system_thinking in zip(self.questions, self.thinkings, self.answers, self.system_thinkings):
            if len(question) > 20 and len(thinking) > 20 and len(answer) > 20 and len(system_thinking) > 20:
                results.append({
                    "system_thinking": system_thinking,
                    "question": question,
                    "thinking": thinking,
                    "answer": answer,
                    "report": self.finding,
                    "image": self.image,
                })
        return results
    
def vqa_thinking_batch(findings, images):
    """
    findings: List[str] of N findings
    Returns: List[List[Dict]] of size N x M_i, where each inner list contains result dicts for each sub-result
    """

    vqa_thinkings = [VQAThinking(finding, image) for finding, image in zip(findings, images)]

    # 1. Generate questions
    question_outputs_batch = synthesize_data_batch([vqa_thinking.question_prompt for vqa_thinking in vqa_thinkings], enable_thinking=True)
    question_outputs = [out[1] for out in question_outputs_batch]
    for vqa_thinking, output in zip(vqa_thinkings, question_outputs):
        vqa_thinking.parse_questions_from_output(output)

    # 2. Generate thinkings
    thinking_prompts = [thinking_prompt for vqa_thinking in vqa_thinkings for thinking_prompt in vqa_thinking.thinking_prompts]
    thinking_outputs_batch = synthesize_data_batch(thinking_prompts, enable_thinking=True)
    system_thinkings = [out[0] for out in thinking_outputs_batch]
    thinking_outputs = [out[1] for out in thinking_outputs_batch]
    index = 0
    for vqa_thinking in vqa_thinkings:
        thinking_outputs_slice = thinking_outputs[index:index+len(vqa_thinking.questions)]
        system_thinkings_slice = system_thinkings[index:index+len(vqa_thinking.questions)]
        vqa_thinking.parse_thinkings_and_answers_from_outputs(thinking_outputs_slice, system_thinkings_slice)
        index += len(vqa_thinking.questions)
    
    # 3. Format the results
    results = []
    for vqa_thinking in vqa_thinkings:
        results.extend(vqa_thinking.format_results())
    return results

if __name__ == "__main__":
    # # Example usage
    # prompt = "Give me a short introduction to large language models."
    # thinking, output = synthesize_data(prompt)
    # print("Thinking:")
    # print(thinking)
    # print("Output:")
    # print(output)
    finding_1 = """
    Multiple venous collaterals are noted in the anterior left chest wall, connecting to the anterior jugular vein at the right sternoclavicular junction, with a collapsed left subclavian vein suggestive of chronic occlusion; trachea and bronchi are patent; calcific plaques in the aortic arch are observed; mediastinal vascular structures, heart size, and thoracic aorta diameter are normal; no pericardial effusion or thickening; normal esophagus with no significant wall thickening; no enlarged lymph nodes in prevascular, pretracheal, subcarinal, or hilar regions; linear and subsegmental atelectasis, bronchial wall thickening, peribronchial tree-like reticulonodular densities, and minimal consolidation in the bilateral lower lobes suggestive of an infectious process; atrophic left kidney partially seen, right kidney not evaluable, normal other upper abdominal organs with no space-occupying lesions in the visible liver or bilateral adrenal glands; and thoracic vertebrae show anterior osteophyte extensions.
    """
    finding_2 = """
    Give me a short introduction to large language models.
    """
    findings = [finding_1, finding_2]
    results = vqa_thinking_batch(findings, ["a.png", "b.png"])
    for result in results:
        print("[*]Report:")
        print(result["report"])
        print("[*]System Thinking:")
        print(result["system_thinking"])
        print("[*]Question:")
        print(result["question"])
        print("[*]Thinking:")
        print(result["thinking"])
        print("[*]Answer:")
        print(result["answer"])
        print("-" * 50)