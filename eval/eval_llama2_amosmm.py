import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
from peft import LoraConfig, get_peft_model
from src.utils.u2Transform import u2Transform
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

u2t = u2Transform(data_type="validation")

nltk.download('wordnet', quiet=True)

# 缓存评估器
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

def load_dataset_info(base_path):
    """加载数据集信息"""
    with open(os.path.join(base_path, 'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

def combine_findings(findings):
    """组合所有findings文本"""
    combined = []
    if findings.get('chest'):
        combined.append(findings['chest'])
    if findings.get('abdomen'):
        combined.append(findings['abdomen'])
    if findings.get('pelvis'):
        combined.append(findings['pelvis'])
    return ' '.join(combined).strip()

def calculate_green_score(text):
    """Calculate GREEN score components for medical report evaluation."""
    coherence_labels = ["coherent", "incoherent"]
    completeness_labels = ["complete", "incomplete"]
    correctness_labels = ["accurate", "inaccurate"]
    
    coherence_result = classifier(text, coherence_labels)
    completeness_result = classifier(text, completeness_labels)
    correctness_result = classifier(text, correctness_labels)
    
    coherence_score = coherence_result['scores'][coherence_result['labels'].index("coherent")]
    completeness_score = completeness_result['scores'][completeness_result['labels'].index("complete")]
    correctness_score = correctness_result['scores'][correctness_result['labels'].index("accurate")]
    
    green_score = (coherence_score + completeness_score + correctness_score) / 3
    
    return {
        'green_overall': round(green_score * 100, 2),
        'green_coherence': round(coherence_score * 100, 2),
        'green_completeness': round(completeness_score * 100, 2),
        'green_correctness': round(correctness_score * 100, 2)
    }

def load_model():
    device = torch.device('cuda') # 'cpu', 'cuda'
    dtype = torch.bfloat16 # or bfloat16, float16, float32

    model_name_or_path = '/import/c4dm-04/siyoul/u2Tokenizer/checkpoint/amosmm_chatgpt_stage_1/checkpoint-100080'
    lora_model_path = '/import/c4dm-04/siyoul/u2Tokenizer/checkpoint/amosmm_chatgpt_stage_1/model_with_lora.bin'
    state_dict = torch.load(lora_model_path, map_location="cpu")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    if lora_model_path is not None:
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=find_all_linear_names(base_model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(base_model, lora_config)
        # u2_model.print_trainable_parameters()
        print("Load weights with LoRA")
        model.load_state_dict(state_dict, strict=True)
        print("Merge weights with LoRA")
        model = model.merge_and_unload()
    return model.to(device), tokenizer

def generate_caption(model, tokenizer, image_file_path):
    # try:
    device = next(model.parameters()).device
    image = u2t(image_file_path)
    
    question = "Can you provide a caption consists of findings and expressions for this medical image?"
    question_ids = tokenizer(
        question, add_special_tokens=False, max_length=768, truncation=True, padding="max_length", return_tensors="pt", padding_side="right"
    )["input_ids"][0].to(device)
    image_tokens = "<im_patch>" * 256
    input_txt = image_tokens + question
    input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device)
    
    with torch.cuda.amp.autocast(): 
        generation = model.generate(
            image.unsqueeze(0).to(device=device), 
            input_ids,
            question_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=1.0
        )
    return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    # except Exception as e:
    #     print(f"Error generating caption: {str(e)}")
    #     return None

def evaluate_captions(references, hypothesis):
    try:
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        references = [preprocess_text(ref) for ref in references]
        hypothesis = preprocess_text(hypothesis)
        
        from nltk.translate.bleu_score import SmoothingFunction
        smoother = SmoothingFunction().method3
        weights = (1, 0, 0, 0)
        references_tokenized = [ref.split() for ref in references]
        hypothesis_tokenized = hypothesis.split()
        bleu_score = corpus_bleu([references_tokenized], [hypothesis_tokenized], 
                                weights=weights,
                                smoothing_function=smoother) * 100
        
        rouge_scores = rouge_scorer_instance.score(references[0], hypothesis)
        rouge1 = rouge_scores['rouge1'].fmeasure * 100
        rouge2 = rouge_scores['rouge2'].fmeasure * 100
        rougeL = rouge_scores['rougeL'].fmeasure * 100
        
        P, R, F1 = score([hypothesis], references, lang='en', verbose=False)
        bert_f1 = F1.mean().item()
        
        meteor_score_value = meteor_score([references[0].split()], hypothesis_tokenized) * 100
        
        # Calculate GREEN scores
        green_scores = calculate_green_score(hypothesis)
        
        return {
            'bleu': round(bleu_score, 2),
            'rouge1': round(rouge1, 2),
            'rouge2': round(rouge2, 2),
            'rougeL': round(rougeL, 2),
            'bert_score': round(bert_f1, 4),
            'meteor': round(meteor_score_value, 2),
            **green_scores
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def main():
    base_path = '/import/c4dm-04/siyoul/u2Tokenizer/datasets/AMOS-MM/'
    
    # 加载数据集信息
    dataset_info = load_dataset_info(base_path)
    validation_cases = [case for case in dataset_info['validation']]
    
    model, tokenizer = load_model()
    all_scores = []
    num = 0
    for case in tqdm(validation_cases, desc="Processing validation cases"):
        # 获取图像路径和ground truth
        image_path = os.path.join(base_path, case['image'].lstrip('./'))
        findings = combine_findings(case['labels']['report']['findings'])
        
        if not findings:
            print(f"No findings for {case['image']}")
            continue
            
        # 生成描述
        generated_text = generate_caption(model, tokenizer, image_path)
        if not generated_text:
            continue
            
        # 评估
        scores = evaluate_captions([findings], generated_text)
        if scores:
            scores['image'] = case['image']
            all_scores.append(scores)
            
            print(f"\nImage: {case['image']}")
            print(f"Generated: {generated_text}")
            print(f"Ground truth: {findings}")
            print(f"Scores: {scores}")
        num += 1
        if num == 30:
            break
    
    if all_scores:
        # 计算平均分数
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor',
                  'green_overall', 'green_coherence', 'green_completeness', 'green_correctness']
        
        avg_scores = {
            metric: np.mean([s[metric] for s in all_scores])
            for metric in metrics
        }
        
        print("\nAverage scores across all cases:")
        for metric, value in avg_scores.items():
            print(f"{metric}: {value:.4f}")
        
        # 保存结果
        with open('amos_evaluation_results.txt', 'w') as f:
            f.write("Individual scores:\n")
            for score in all_scores:
                f.write(f"{score}\n")
            f.write("\nAverage scores:\n")
            for metric, value in avg_scores.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"\nTotal processed cases: {len(all_scores)}")
    else:
        print("No scores were generated!")

if __name__ == '__main__':
    main()