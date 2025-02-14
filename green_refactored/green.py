import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import time
import warnings
import json
from openai import OpenAI

from config import config

# Import necessary functions (ensure these are available in your environment)
from green_score.utils import (
    make_prompt,
    clean_responses,
    compute_largest_cluster,
    flatten_values_lists_of_list_dicts_to_dict,
)

from transformers.utils import logging

# Set the logging level for the transformers library to ERROR to suppress warnings that have been resolved
logging.get_logger("transformers").setLevel(logging.ERROR)

class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_response(self, batch):
        raise NotImplementedError()

class GREENLLM(LLM):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False if "Phi" in model_name else True,
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )

        chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        self.tokenizer.chat_template = chat_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.clean_up_tokenization_spaces = True
        assert self.tokenizer.padding_side == "left"

        self.is_distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1

    def tokenize_batch_as_chat(self, batch):
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        return batch

    @torch.inference_mode()
    def get_response(self, batch):
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        batch = [
            [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
            for prompt in batch["prompt"]
        ]

        batch = self.tokenize_batch_as_chat(batch)

        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=2048,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

class OpenAILLM(LLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        base_url = config["openai_server"]["base_url"]
        if not base_url:
            base_url = None
        self.client = OpenAI(
            base_url=base_url,
            api_key=config["openai_server"]["api_key"],
        )

    def get_response(self, batch):
        responses = []
        for prompt in batch["prompt"]:
            response = self.client.chat.completions.create(model=config["openai_server"]["model_name"], messages= [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]).choices[0].message.content
            responses.append(response)

        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)
        
        return response_list

    def generate_batch_file(self, prompts, file_name):
        for i, prompt in enumerate(tqdm(iterable=prompts, desc="Generating jsonl")):
            request = {
                "custom_id": f"green_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 6000,
                },
            }
            with open(file_name, "a") as f:
                json.dump(request, f)
                f.write("\n")
    
    def upload_batch_file(self, file_name) -> str:
        return self.client.files.create(file=open(file_name, "rb"), purpose="batch").id

    def run_batch(self, batch_file_id) -> str:
        return self.client.batches.create(
            input_file_id=batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

    def probe_batch(self, batch_id):
        batch_status = self.client.batches.retrieve(batch_id)
        print(batch_status)
        return batch_status.status

    def fetch_batch_result(self, batch_id, output_path="output.jsonl"):
        batch_status = self.client.batches.retrieve(batch_id)
        if batch_status.status == "completed":
            file_response = self.client.files.content(batch_status.output_file_id)
            with open(output_path, "w") as f:
                for line in file_response.content.splitlines():
                    line = json.loads(line)
                    f.write(line["response"]["body"]["choices"][0]["message"]["content"] + "\n")


class QuantizedLLM(LLM):
    def __init__(self, model_name):
        super().__init__(model_name, compute_summary_stats)
        # model_name is "/data/huanan/GREEN/quantized_model_path"
        self.model = GPTQModel.load(model_name, trust_remote_code=True)
        self.tokenizer = self.model.tokenizer

    @torch.inference_mode()
    def get_response(self, batch):
        self.model.generate()
        raise NotImplementedError()

class GREEN:
    def __init__(self, llm_model: LLM, compute_summary_stats=True):

        warnings.filterwarnings(
            "ignore", message="A decoder-only architecture is being used*"
        )

        self.model = llm_model

        self.compute_summary_stats = compute_summary_stats

        self.batch_size = 4
        self.max_length = 2048

        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]
        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

    def __call__(self, refs, hyps):
        dataset = self.process_data(refs, hyps)

        self.dataset = dataset

        t = time.time()

        mean, std, green_scores, summary, results_df = self.infer()

        t = time.time() - t
        print("Seconds per example: ", t / len(refs))

        return mean, std, green_scores, summary, results_df

    def process_data(self, refs, hyps):
        print("Processing data to generate prompts")
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})

        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        dataset = dataset.map(prompting, batched=True)
        return dataset

    @torch.inference_mode()
    def infer(self):
        local_completions = []
        local_references = []

        print("==== Beginning Inference ====")
        for batch in tqdm(
            iterable=self.dataset.iter(batch_size=self.batch_size),
            total=len(self.dataset) // self.batch_size,
        ):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.model.get_response(batch))

        self.completions = local_completions
        self.prompts = local_references

        print("==== End Inference ====")

        if len(self.completions) != len(self.prompts):
            print("Length of prompts and completions are not equal!")

        return self.process_results()

    def process_results(self):
        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "green_analysis": self.completions,
                "green_score": self.green_scores,
                **self.error_counts,
            }
        )
        mean, std, summary = None, None, None

        if self.compute_summary_stats:
            mean, std, summary = self.compute_summary()

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        for position, sub_category in enumerate(self.sub_categories):
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )

        return sub_category_dict_sentences

    def compute_sentences(self, response):
        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict

    def compute_accuracy(self, responses):
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc

    def compute_summary(self):
        print("Computing summary ...")
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f"\n-------------{self.model.model_name}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
        for idx, sub_category in enumerate(self.sub_categories):
            accuracy = accuracies[sub_category]
            sentences = representative_sentences[sub_category]
            summary += f"{sub_category}: {accuracy}. \n {sentences} \n\n"
        summary += "----------------------------------\n"

        return mean, std, summary
