# -*- encoding: utf-8 -*-
# @File        :   qwen3_data_synthesis.py
# @Time        :   2025/05/22 21:46:52
# @Author      :   Siyou
# @Description :

import random
from openai import OpenAI, AsyncOpenAI
import re
import logging
from config import config
import asyncio
import os
import json
from src.utils.prompt_templates import general_questions, general_questions_chinese

# Set up logger
logger = logging.getLogger(__name__)

model_name= config["openai_server"]["model_name"]
client = OpenAI(
    base_url=config["openai_server"]["base_url"],
    api_key=config["openai_server"]["api_key"],
)
async_client = AsyncOpenAI(
    base_url=config["openai_server"]["base_url"],
    api_key=config["openai_server"]["api_key"],
)

thinking_params = {
    "stream": False,
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "extra_body": {
        "top_k": 20, 
        "chat_template_kwargs": {"enable_thinking": True},
    },
}

non_thinking_params = {
    "stream": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "max_tokens": 8192,
    "extra_body": {
        "top_k": 20, 
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

def synthesize_data(prompt, enable_thinking=True):
    # Prepare the input to the model
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **(thinking_params if enable_thinking else non_thinking_params),
    )

    outputs = response.choices[0].message.content
    pattern = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)

    m = pattern.search(outputs)
    if m:
        thinking = m.group(1).strip()
        output   = m.group(2).strip()
    else:
        thinking = ""
        output   = outputs
    return thinking, output
    
async def synthesize_data_batch(prompts, enable_thinking=True):
    """
    prompts: List[str]
    enable_thinking: bool — whether to include <think> reasoning in the template
    Returns: List[Tuple[str, str]]
        thinkings: List[str]  the content captured between <think>…</think>
        outputs:   List[str]  the remaining generated content
    """
    message_batches = [[{"role": "user", "content": p}] for p in prompts]

    async def process_batch(messages):
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=messages,
            **(thinking_params if enable_thinking else non_thinking_params),
        )
        return response.choices

    batch_results = await asyncio.gather(*[process_batch(messages) for messages in message_batches])
    results = []
    pattern = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)

    for result in batch_results:
        full_text = result[0].message.content
        m = pattern.search(full_text)
        if m:
            thinking = m.group(1).strip()
            output = m.group(2).strip()
        else:
            thinking = ""
            output = full_text.strip()
        results.append((thinking, output))

    return results

QUESTION_PROMPT_TPL = """
Here is a medical radiology report for a CT image.
```
{report}
```

Imagine you are assessing a student who is looking at a CT image, you are going to ask a list of questions. Don't mention the report, just list out as the form of questions, and questions only, in sequenced list.
""".strip()

THINKING_PROMPT_TPL = """
You are a radiology medicine expert.
Your task is to answer the following radiology medicine question, using the patient’s medical record report provided below.
When writing your thought process, imagine you are directly reviewing the patient’s radiology images (do not mention the report), and describe your logical reasoning step by step as an expert would.
Then, provide your final, correct answer to the question.
Your response will be used to guide and improve the training of a multimodal large language model for radiology medicine images.
And here is the radiology report that you can see:
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
    _, output = synthesize_data(prompt, enable_thinking=False)
    questions = re.findall(question_pattern, output, flags=re.M|re.S)
    
    # 2. generate thinking
    system_thinkings = []
    thinkings = []
    answers = []
    for item in questions:
        prompt = THINKING_PROMPT_TPL.format(report=finding, question=item.strip())
        system_thinking, output= synthesize_data(prompt, enable_thinking=True)
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
                "answer": answer,
                "report": finding,
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
            if len(question) > 20 and len(thinking) > 20 and len(answer) > 20:
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
    question_outputs_batch = asyncio.run(synthesize_data_batch([vqa_thinking.question_prompt for vqa_thinking in vqa_thinkings], enable_thinking=False))
    question_outputs = [out[1] for out in question_outputs_batch]
    for vqa_thinking, output in zip(vqa_thinkings, question_outputs):
        vqa_thinking.parse_questions_from_output(output)

    # 2. Generate thinkings
    thinking_prompts = [thinking_prompt for vqa_thinking in vqa_thinkings for thinking_prompt in vqa_thinking.thinking_prompts]
    thinking_outputs_batch = asyncio.run(synthesize_data_batch(thinking_prompts, enable_thinking=False))
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


TRANSLATION = """
This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_input}
""".strip()

def translation(source_input, target_lang, source_lang, enable_thinking=False):
    messages=[
        {
            'role': 'user',
            'content': TRANSLATION.format(source_input=source_input, target_lang=target_lang, source_lang=source_lang),
        }]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **(thinking_params if enable_thinking else non_thinking_params),
    )

    outputs = response.choices[0].message.content
    return outputs

def vqa_thinking_translation_synthesis(jsonl_file_path, output_file_path, source_lang="English", target_lang="Chinese", enable_thinking=False):
    """
    Synthesize VQA thinking data with translation.
    jsonl_file_path: str, path to the input JSONL file
    output_file_path: str, path to the output JSONL file
    source_lang: str, language of the input text
    target_lang: str, language for the translation
    enable_thinking: bool, whether to enable thinking in the synthesis
    """

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()

    with open(output_file_path, 'w') as f:
        for line in lines:
            data = json.loads(line)
            question = data["question"]
            translated_question= translation(question, target_lang=target_lang, source_lang=source_lang, enable_thinking=enable_thinking)
            data["question"] = translated_question
            refined_thinking = data["refined_thinking"]
            translated_refined_thinking = translation(refined_thinking, target_lang=target_lang, source_lang=source_lang, enable_thinking=enable_thinking)
            data["refined_thinking"] = translated_refined_thinking
            answer = data["answer"]
            translated_answer = translation(answer, target_lang=target_lang, source_lang=source_lang, enable_thinking=enable_thinking)
            data["answer"] = translated_answer                           
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def report_thinking_translation_synthesis(jsonl_file_path, output_file_path, source_lang="English", target_lang="Chinese", enable_thinking=False):
    """
    Synthesize report thinking data with translation.
    jsonl_file_path: str, path to the input JSONL file
    output_file_path: str, path to the output JSONL file
    source_lang: str, language of the input text
    target_lang: str, language for the translation
    enable_thinking: bool, whether to enable thinking in the synthesis
    """
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()

    with open(output_file_path, 'w') as f:
        for line in lines:
            data = json.loads(line)

            # Translate the question
            question = data["question"]
            try:
                question_index = general_questions.index(question)
                translated_question = general_questions_chinese[question_index]
            except ValueError:
                question_index = random.randint(0, len(general_questions) - 1)
                question = data["question"] = general_questions[question_index]
                translated_question = translation(question, target_lang=target_lang, source_lang=source_lang, enable_thinking=enable_thinking)
            data["question"] = translated_question

            # Translate the refined thinking
            thinking_after = data["thinking_after"]
            translated_thinking_after = translation(thinking_after, target_lang=target_lang, source_lang=source_lang, enable_thinking=enable_thinking)
            data["thinking_after"] = translated_thinking_after
            # Remove the thinking_before
            data.pop("thinking_before")

            # Translate the report
            report = data["report"]
            translated_report = translation(report, target_lang=target_lang, source_lang=source_lang, enable_thinking=enable_thinking)
            data["report"] = translated_report

            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # # Example usage
    finding_1 = """
    Multiple venous collaterals are noted in the anterior left chest wall, connecting to the anterior jugular vein at the right sternoclavicular junction, with a collapsed left subclavian vein suggestive of chronic occlusion; trachea and bronchi are patent; calcific plaques in the aortic arch are observed; mediastinal vascular structures, heart size, and thoracic aorta diameter are normal; no pericardial effusion or thickening; normal esophagus with no significant wall thickening; no enlarged lymph nodes in prevascular, pretracheal, subcarinal, or hilar regions; linear and subsegmental atelectasis, bronchial wall thickening, peribronchial tree-like reticulonodular densities, and minimal consolidation in the bilateral lower lobes suggestive of an infectious process; atrophic left kidney partially seen, right kidney not evaluable, normal other upper abdominal organs with no space-occupying lesions in the visible liver or bilateral adrenal glands; and thoracic vertebrae show anterior osteophyte extensions.
    """
    finding_2 = """
    Give me a short introduction to large language models.
    """
    findings = [finding_1, finding_2]
    results = vqa_thinking_batch(findings, images=["image1.png", "image2.png"])
    logger.info(results)
    for result in results:
        logger.info(f"""[*]Report:
{result["report"]}

[*]System Thinking:
{result["system_thinking"]}

[*]Question:
{result["question"]}

[*]Thinking:
{result["thinking"]}

[*]Answer:
{result["answer"]}
{'-' * 50}""")