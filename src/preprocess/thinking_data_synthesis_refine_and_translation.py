from openai import OpenAI, AsyncOpenAI
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import re
import argparse
import random
import json

API_KEY = "EMPTY"
BASE_URL = "http://localhost:8000/v1"
MODEL = "pretrained_models/Qwen3-235B-A22B-GPTQ-Int4"
# MODEL = "pretrained_models/Qwen3-8B"

parser = argparse.ArgumentParser(description="Translate VQA thinking dataset to English.")
parser.add_argument("--JSONLFile", type=str, required=True, help="Path to the input JSONL file.")

args = parser.parse_args()

client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL,
)
async_client = AsyncOpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL,
)

FILTER_TEMPLATE = """
You are an expert in radiology. Now you are reviewing a some questions and answers made by another expert.
You need to determine if the question is proper for a radiology exam, and the answer is correct.

If the question is proper for a radiology exam, and the answer is correct, return "Yes".
If the question is not proper for a radiology exam, or the answer is incorrect, return "No".
Do not return anything else.

The Report:
```
{report}
```
Question: {question}
Answer: {answer}
""".strip()

questions = [
            "Can you provide a caption consists of {} for this medical image?",
            "Describe the {} of the medical image you see.",
            "Please caption this medical scan with {}.",
            "What is the {} of this image?",
            "Describe this medical scan with {}.",
            "Please write a caption consists of {} for this image.",
            "Can you summarize with {} the images presented?",
            "Please caption this scan with {}.",
            "Please provide a caption consists of {} for this medical image.",
            "Can you provide a summary consists of {} of this radiograph?",
            "What are the {} presented in this medical scan?",
            "Please write a caption consists of {} for this scan.",
            "Can you provide a description consists of {} of this medical scan?",
            "Please caption this medical scan with {}.",
            "Can you provide a caption consists of {} for this medical scan?",
            # "Please generate a medical report based on this image.",
            # "Can you generate a diagnose report from this image.",
            "Could you analyze and provide a caption for the {} in this medical image?",
            # "Please describe the observations depicted in this medical scan.",
            "Can you summarize the {} of this image in a caption?",
            "What are the significant {} in this medical image?",
            "Please provide a detailed caption outlining the {} of this image.",
            "Could you interpret and describe the {} shown in this medical scan?",
            # "What conclusions can you draw from the observations in this image?",
            "Please write a descriptive caption based on the {} in this scan.",
            "What key {} can you identify from examining this medical image?",
            # "Could you generate a detailed report based on the observations in this image?",
            "Can you provide a diagnosis based on the {} in this image?",
            "Please generate a comprehensive report summarizing the {} in this image.",
            "Caption the {} in this medical image?",
            "Describe the {} you see.",
            "Caption this medical scan's {}.",
            "What are the {} here?",
            "Describe these {}.",
            "Summarize the {} in these images.",
            "Caption this scan's {}.",
            "Provide a caption for this medical image's {}.",
            "Summarize the {} of this radiograph.",
            "What {} are presented in this scan?",
            "Describe this scan's {}.",
            # "Generate a medical report based on this image.",
            # "Can you provide a diagnosis based on this image?",
]

async def query(content, enable_thinking=False):
    if not enable_thinking:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.6,
            top_p=0.95,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
    return response.choices[0].message.content.strip(), response.choices[0].message.reasoning_content.strip() if enable_thinking else None




async def main(args):
    df = pd.read_json(args.JSONLFile, lines=True)

    pbar = tqdm(total=len(df))
    dropped_records = []

    for i, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        report = row["report"]
        content = FILTER_TEMPLATE.format(question=question, answer=answer, report=report)
        response, _ = await query(content, enable_thinking=False)
        if not response.startswith("Yes"):
            print(f"Index {i} 被认为是不合适的问题或答案，将剔除，响应为 {response}")
            df.drop(index=i, inplace=True)
            dropped_records.append(i)
        pbar.update(1)
    pbar.close()

    if dropped_records:
        with open(f"{args.JSONLFile.replace('.jsonl','')}_dropped.txt", "w") as f:
            f.write(repr(dropped_records))
            f.flush()

    EDIT_TEMPLATE = """
    Help me edit the narrative below:
    - If the narrative refers to a report, you change it as if you see it from the radiology image
    - Edit only the places mentioned above, leave all other text the same 
    - Do not add/remove/change any other information
    - Directly output the result text

    **The narrative:**
    ```
    {report}
    ```
    """.strip()

    results = []
    for index, row in df.iterrows():
        result, _ = await query(EDIT_TEMPLATE.format(report=row["thinking"]), enable_thinking=False)
        result = result.strip('`\n')
        df.loc[index, "refined_thinking"] = result
    
    # df.to_json(f"{args.JSONLFile}_refined.jsonl", orient='records', lines=True)
    with open(f"{args.JSONLFile}_refined.jsonl", "w") as f:
        for index, row in df.iterrows():
            f.write(json.dumps({
                "image": row["image"],
                "report": row["report"],
                "question": row["question"],
                "thinking": row["thinking"],
                "refined_thinking": row["refined_thinking"],
                "answer": row["answer"]
            }, ensure_ascii=False) + "\n")
            f.flush()
    THINKING_TEMPLATE = """
    You are a radiology medicine expert. Now you are looking at a radiology image.
    Here is your self talk when viewing the image:
    ```
    {thinking_before}
    ```

    Please paraphrase the self talk text and output it as **thinking progress**. Remember:
    - Do not add/remove/alter any information
    - Mind the coherence and fluence of output
    - Deductions are prefered
    - Directly output the result text

    Your output:
    """.strip()

    groups = df.groupby("image")
    pbar = tqdm(total=len(groups))
    full_report = []
    for image, group in groups:
        report = group["report"].iloc[0]
        thinking_before = ""
        for i, row in group.iterrows():
            thinking_before += row["question"] + row["refined_thinking"] + row["answer"]
        content = THINKING_TEMPLATE.format(report=report, thinking_before=thinking_before)
        thinking_after, _ = await query(content, enable_thinking=False)
        thinking_after = re.sub(r"^(?:\*+)?Thinking Progress:(?:\*+)?", "", thinking_after, flags=re.IGNORECASE).strip('`\n')
        full_report.append(json.dumps({
            "image": image,
            "report": report,
            "question": questions[random.randint(0, len(questions) - 1)].format("findings"),
            "thinking_before": thinking_before,
            "thinking_after": thinking_after,
        }, ensure_ascii=False))
        pbar.update(1)
    pbar.close()

    with open(f"{args.JSONLFile.replace('.jsonl','')}_report_thinking.jsonl", "w") as f:
        for item in full_report:
            f.write(item + "\n")
            f.flush()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main(args))