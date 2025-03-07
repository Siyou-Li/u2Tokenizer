import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence, List, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
import numpy as np
import os
import json
import nibabel as nib
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import traceback
from green_score import GREEN  # Import the GREEN score implementation
import nltk
import gc
nltk.download('wordnet', quiet=True)

# Initialize the rouge scorer
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) 

def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, padding_side='left')  # Add padding_side='left'
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = f"<image{i*image_num+j}>"
                image_padding_token += image_token
                special_token["additional_special_tokens"].append(image_token)
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(special_token)
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens 

def combine_and_preprocess(question, image_list, image_padding_tokens):
    images = []
    new_questions = [_ for _ in question]
    padding_index = 0
    
    for img in image_list:
        # 读取nii.gz文件
        nifti_img = nib.load(img['img_path'])
        image = nifti_img.get_fdata()
        
        # 转换为tensor并归一化
        image = torch.from_numpy(image).float()
        image = (image - image.min()) / (image.max() - image.min())
        
        image = image.unsqueeze(0)
        image = image.repeat(3,1,1,1)
        
        target_H = 256
        target_W = 256
        target_D = 64
        
        image = image.unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image,
            size=(target_H, target_W, target_D),
            mode='trilinear',
            align_corners=False
        )
        
        image = image.unsqueeze(1)
        images.append(image)
        new_questions[img['position']] = f"<image>{image_padding_tokens[padding_index]}</image>" + new_questions[img['position']]
        padding_index += 1
    
    vision_x = torch.cat(images, dim=1)
    text = ''.join(new_questions)
    
    return text, vision_x

def combine_findings(findings):
    """组合所有findings文本"""
    combined = []
    if isinstance(findings, dict):
        if findings.get('chest'):
            combined.append(findings['chest'])
        if findings.get('abdomen'):
            combined.append(findings['abdomen'])
        if findings.get('pelvis'):
            combined.append(findings['pelvis'])
        # For other structures in the findings dict
        for key, value in findings.items():
            if key not in ['chest', 'abdomen', 'pelvis'] and isinstance(value, str):
                combined.append(value)
    elif isinstance(findings, str):
        combined.append(findings)
    return ' '.join(combined).strip()

def clear_gpu_memory():
    """清理GPU内存"""
    torch.cuda.empty_cache()
    gc.collect()

def evaluate_captions(reference, hypothesis, green_scorer=None):
    try:
        # Text preprocessing
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        reference_processed = preprocess_text(reference)
        hypothesis_processed = preprocess_text(hypothesis)
        
        # BLEU calculation
        smoother = SmoothingFunction().method3
        weights = (1, 0, 0, 0)
        references_tokenized = [reference_processed.split()]
        hypothesis_tokenized = hypothesis_processed.split()
        bleu_score = corpus_bleu([references_tokenized], [hypothesis_tokenized], 
                                weights=weights,
                                smoothing_function=smoother) * 100
        
        # ROUGE calculation
        rouge_scores = rouge_scorer_instance.score(reference_processed, hypothesis_processed)
        rouge1 = rouge_scores['rouge1'].fmeasure * 100
        rouge2 = rouge_scores['rouge2'].fmeasure * 100
        rougeL = rouge_scores['rougeL'].fmeasure * 100
        
        # BERTScore
        P, R, F1 = score([hypothesis_processed], [reference_processed], lang='en', verbose=False)
        bert_f1 = F1.mean().item()
        
        # METEOR
        meteor_score_value = meteor_score([reference_processed.split()], hypothesis_tokenized) * 100
        
        # 必须使用GREEN score进行评估
        if green_scorer is None:
            print("警告: GREEN 评分器未初始化，无法进行完整评估")
            return {
                'bleu': round(bleu_score, 2),
                'rouge1': round(rouge1, 2),
                'rouge2': round(rouge2, 2),
                'rougeL': round(rougeL, 2),
                'bert_score': round(bert_f1, 4),
                'meteor': round(meteor_score_value, 2)
            }
            
        # GREEN evaluation - using original text
        mean, std, green_score_list, summary, result_df = green_scorer([reference], [hypothesis])
        
        # Extract detailed error analysis
        if not result_df.empty:
            green_analysis = result_df.iloc[0]['green_analysis']
            error_counts = {
                'false_reports': result_df.iloc[0]['(a) False report of a finding in the candidate'],
                'missing_findings': result_df.iloc[0]['(b) Missing a finding present in the reference'],
                'wrong_location': result_df.iloc[0]['(c) Misidentification of a finding\'s anatomic location/position'],
                'wrong_severity': result_df.iloc[0]['(d) Misassessment of the severity of a finding'],
                'extra_comparison': result_df.iloc[0]['(e) Mentioning a comparison that isn\'t in the reference'],
                'missing_comparison': result_df.iloc[0]['(f) Omitting a comparison detailing a change from a prior study'],
                'matched_findings': result_df.iloc[0]['Matched Findings']
            }
        else:
            green_analysis = ""
            error_counts = {
                'false_reports': 0, 'missing_findings': 0, 'wrong_location': 0,
                'wrong_severity': 0, 'extra_comparison': 0, 'missing_comparison': 0,
                'matched_findings': 0
            }
            
        return {
            'bleu': round(bleu_score, 2),
            'rouge1': round(rouge1, 2),
            'rouge2': round(rouge2, 2),
            'rougeL': round(rougeL, 2),
            'bert_score': round(bert_f1, 4),
            'meteor': round(meteor_score_value, 2),
            'green_mean': mean,
            'green_std': std,
            'green_summary': summary,
            'green_score_list': green_score_list,
            'green_analysis': green_analysis,
            'error_counts': error_counts
        }
            
    except Exception as e:
        print(f"评估指标计算错误: {str(e)}")
        print(f"详细错误信息:", traceback.format_exc())
        return None

def main():
    # Initialize paths
    base_dir = '/rds/projects/l/lez-medical-ai/PengyaoQin/AMOS/'
    
    # Load the dataset
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        dataset = json.load(f)
    
    validation_data = dataset['validation']
    generated_results = []
    
    print("第一阶段：生成caption...")
    # Initialize the tokenizer and model
    text_tokenizer, image_padding_tokens = get_tokenizer('./Language_files')
    model = MultiLLaMAForCausalLM(
        lang_model_path='./Language_files',
    ).half()
    
    print("Loading RadFM checkpoint...")
    ckpt = torch.load('./pytorch_model.bin', map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.to('cuda')
    model.eval()
    
    # 第一阶段：生成caption
    for case in tqdm.tqdm(validation_data, desc="Generating captions"):
        try:
            image_path = os.path.join(base_dir, case['image'].lstrip('./'))
            case_id = os.path.basename(image_path)
            
            # Get ground truth
            findings = case['labels']['report']['findings']
            ground_truth = combine_findings(findings)
            
            if not ground_truth.strip():
                print(f"Empty report for {case_id}, skipping...")
                continue
            
            # Generate report
            question = "Please generate a radiology report for this scan."
            image = [{
                'img_path': image_path,
                'position': 0,
            }]
            
            text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)
            
            with torch.cuda.amp.autocast():
                lang_x = text_tokenizer(
                    text, 
                    max_length=2048, 
                    truncation=True, 
                    return_tensors="pt"
                )['input_ids'].to('cuda')
                
                vision_x = vision_x.to('cuda').half()
                
                generation = model.generate(
                    lang_x, 
                    vision_x
                )
                
                generated_caption = text_tokenizer.batch_decode(
                    generation,
                    skip_special_tokens=True
                )[0]
            
            # 保存结果
            generated_results.append({
                'case_id': case_id,
                'ground_truth': ground_truth,
                'generated': generated_caption,
                'image': case['image']
            })
            
        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")
            continue
    
    # 清理GPU内存
    print("清理GPU内存...")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU内存已清理")
    
    # 第二阶段：评估
    print("\n第二阶段：评估caption...")
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    all_scores = []
    
    for result in tqdm.tqdm(generated_results, desc="Evaluating captions"):
        try:
            scores = evaluate_captions(result['ground_truth'], result['generated'], green_scorer)
            scores['case_id'] = result['case_id']
            scores['image'] = result['image']
            all_scores.append(scores)
            
            # 打印每个样本的评估结果
            print(f"\nImage: {result['image']}")
            print(f"Ground truth: {result['ground_truth']}")
            print(f"Generated: {result['generated']}")
            
        except Exception as e:
            print(f"Error evaluating {result['case_id']}: {str(e)}")
            continue
    
    # 计算最终平均值
    metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor', 
              'green_mean', 'green_std']
    error_types = ['false_reports', 'missing_findings', 'wrong_location', 
                  'wrong_severity', 'extra_comparison', 'missing_comparison',
                  'matched_findings']
    
    avg_scores = {
        metric: np.mean([s[metric] for s in all_scores if metric in s])
        for metric in metrics
    }
    
    avg_errors = {
        error_type: np.mean([s['error_counts'][error_type] for s in all_scores if 'error_counts' in s])
        for error_type in error_types
    }
    
    # 保存结果
    with open('radfm_evaluation_results.txt', 'w') as f:
        f.write("Evaluation Results:\n")
        for scores in all_scores:
            f.write(f"\nCase ID: {scores['case_id']}\n")
            f.write(f"Image: {scores['image']}\n")
            
            for metric in metrics:
                if metric in scores:
                    f.write(f"{metric}: {scores[metric]:.4f}\n")
            
            if 'green_analysis' in scores:
                f.write(f"\nGREEN Analysis:\n{scores['green_analysis']}\n")
                f.write("\nError Counts:\n")
                for error_type, count in scores['error_counts'].items():
                    f.write(f"{error_type}: {count}\n")
            
            f.write("-" * 80 + "\n")
        
        # 写入最终平均值
        f.write("\nFinal Average Scores:\n")
        f.write("-" * 30 + "\n")
        for metric, value in avg_scores.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nAverage Error Counts:\n")
        f.write("-" * 30 + "\n")
        for error_type, value in avg_errors.items():
            f.write(f"{error_type}: {value:.4f}\n")
    
    # 打印最终平均值
    print("\nFinal Average Scores:")
    print("-" * 30)
    for metric, value in avg_scores.items():
        print(f"{metric:15s}: {value:.4f}")
    
    print("\nAverage Error Counts:")
    print("-" * 30)
    for error_type, value in avg_errors.items():
        print(f"{error_type:20s}: {value:.4f}")
    
    print(f"\nProcessed {len(all_scores)} cases successfully")
    print("Results have been saved to radfm_evaluation_results.txt")
if __name__ == "__main__":
    main()