import json
import ollama
from tqdm import tqdm
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def rewrite_ollama(raw_text):
    
    response = ollama.chat(model='gemma2:27b', messages=[
        {
            'role': 'user',
            'content': f'Can you help me rewrite the following CT report? \
            The description is accurate and smooth when the rewriting means unchanged. \
            Please directly output the rewritten sentence in English. \"{raw_text}\"',
        },
    ])

    return response['message']['content']

def generate_qa_ollama(raw_text):
    response = ollama.chat(model='gemma2:27b', messages=[
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

json_file_path = "/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM/dataset.json"

with open(json_file_path, 'r') as f:
    data = json.load(f)

data_type = ["training"]
mrg_type = {"chest":{}, "abdomen":{},"pelvis":{}}

for loc in mrg_type:
    for data_t in data_type:
        mrg_type[loc][data_t] = []
        for item in tqdm(data[data_t]):
            image = item["image"]
            meta = item["meta"]
            findings = item["labels"]["report"]["findings"]
            if findings[loc] != "":
                mrg_type[loc][data_t].append({
                    "image": image,
                    "is_extented": False,
                    "meta": meta,
                    "category": loc,
                    "question": f"please provide a detailed caption outlining the fingings in {loc} of this image.",
                    "answer": findings[loc]
                    # "labels": {"report": {"findings": {loc:findings[loc]}}}
                })
                # rewrite
                for i in range(8):
                    try:
                        rewrite_findings = rewrite_ollama(findings[loc])
                        mrg_type[loc][data_t].append({
                            "image": image,
                            "is_extented": True,
                            "meta": meta,
                            "category": loc,
                            "question": f"please provide a detailed caption outlining the fingings in {loc} of this image.",
                            "answer": rewrite_findings
                        })
                    except Exception as e:
                        print(e)
                        continue
                # qa
                qa_pairs = generate_qa_ollama(findings[loc])
                for qa_pair in qa_pairs:
                    if ("question" in qa_pair.keys()) and ("answer" in qa_pair.keys()):
                        mrg_type[loc][data_t].append({
                            "image": image,
                            "is_extented": True,
                            "meta": meta,
                            "category": loc,
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"]
                        })
    output_file_path = f"/home/lez/Siyou/Med3DLLM/datasets/AMOS-MM-Extension/dataset_{loc}_extension_gemma2@27b.json"
    with open(output_file_path, 'w') as f:
        json.dump(mrg_type[loc], f, indent=4, ensure_ascii=False)