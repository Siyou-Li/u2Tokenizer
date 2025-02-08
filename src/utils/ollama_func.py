# -*- encoding: utf-8 -*-
# @File        :   ollama_func.py
# @Time        :   2025/01/09 16:05:27
# @Author      :   Siyou
# @Description :


import ollama
from config import config
import json

model_name= config["ollama_model"]

def rewrite_ollama(raw_text):
    
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': f'Can you help me rewrite the following CT report? \
            The description is accurate and smooth when the rewriting means unchanged. \
            Please fix the findings and impression into one sentence.\
            Please directly output the rewritten sentence in English. \"{raw_text}\"',
        },
    ])

    return response['message']['content']

def generate_qa_ollama(raw_text):
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': f'Please break down the following CT reports and generate a corresponding number of questions and answers based on the number of lesions \
            The description should be accurate, professional and fluent. \
            Please answer the generated results in json format (for example [{{"question":"","answer":""}},{{"question":"","answer":""}},...]). \
            Do not add any other text, and output in English. \"{raw_text}\"',
        },
    ])
    try:
        qa_pairs = json.loads(response['message']['content'])
    except Exception as e:
        print(e)
        return []
    return qa_pairs