import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
from green_score import GREEN
nltk.download('wordnet', quiet=True)

# 缓存评估器
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

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

def load_and_preprocess_image(image_path):
    """加载和预处理nii.gz图像"""
    try:
        img = sitk.ReadImage(image_path)
        img_array = sitk.GetArrayFromImage(img)
        
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        current_depth, current_height, current_width = img_array.shape
        depth_factor = 32 / current_depth
        height_factor = 256 / current_height
        width_factor = 256 / current_width
        
        resized_img = zoom(img_array, (depth_factor, height_factor, width_factor))
        
        if resized_img.shape != (32, 256, 256):
            raise ValueError(f"Unexpected shape after resize: {resized_img.shape}")
        
        resized_img = np.expand_dims(resized_img, axis=0)
        return resized_img
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def load_model():
    device = torch.device('cuda')
    dtype = torch.float16
    model_name = 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    return model.to(device), tokenizer

def generate_caption(model, tokenizer, image_array):
    try:
        device = next(model.parameters()).device
        image_pt = torch.from_numpy(image_array).unsqueeze(0).to(device=device, dtype=torch.float16)
        
        question = "Can you provide a caption consists of findings and expressions for this medical image?"
        image_tokens = "<im_patch>" * 256
        input_txt = image_tokens + question
        input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device)
        
        with torch.no_grad():
            generation = model.generate(
                image_pt, 
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=1.0
            )
        return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        return None
def evaluate_captions(reference, hypothesis, green_scorer):
    try:
        # 其他评估指标的预处理
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        reference_processed = preprocess_text(reference)
        hypothesis_processed = preprocess_text(hypothesis)
        
        # BLEU
        from nltk.translate.bleu_score import SmoothingFunction
        smoother = SmoothingFunction().method3
        weights = (1, 0, 0, 0)
        references_tokenized = [reference_processed.split()]
        hypothesis_tokenized = hypothesis_processed.split()
        bleu_score = corpus_bleu(references_tokenized, [hypothesis_tokenized], 
                                weights=weights,
                                smoothing_function=smoother) * 100
        
        # ROUGE
        rouge_scores = rouge_scorer_instance.score(reference_processed, hypothesis_processed)
        rouge1 = rouge_scores['rouge1'].fmeasure * 100
        rouge2 = rouge_scores['rouge2'].fmeasure * 100
        rougeL = rouge_scores['rougeL'].fmeasure * 100
        
        # BERTScore
        P, R, F1 = score([hypothesis_processed], [reference_processed], lang='en', verbose=False)
        bert_f1 = F1.mean().item()
        
        # METEOR
        meteor_score_value = meteor_score([reference_processed.split()], hypothesis_tokenized) * 100
        
        # GREEN evaluation - 使用原始文本
        mean, std, green_score_list, summary, result_df = green_scorer([reference], [hypothesis])
        
        # 提取详细的错误分析
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
            error_counts = {}
        
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
        print(f"Error calculating metrics: {str(e)}")
        print(f"Full error details:", traceback.format_exc())
        return None

def main():
    base_path = '/rds/projects/l/lez-medical-ai/PengyaoQin/AMOS/'
    
    # 加载数据集信息
    dataset_info = load_dataset_info(base_path)
    validation_cases = [case for case in dataset_info['validation']]
    
    # 初始化模型和评估器
    model, tokenizer = load_model()
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    all_scores = []
    
    for case in tqdm(validation_cases, desc="Processing validation cases"):
        try:
            # 获取图像路径和ground truth
            image_path = os.path.join(base_path, case['image'].lstrip('./'))
            findings = combine_findings(case['labels']['report']['findings'])
            
            if not findings:
                print(f"No findings for {case['image']}")
                continue
                
            # 加载和预处理图像
            image_array = load_and_preprocess_image(image_path)
            if image_array is None:
                continue
                
            # 生成描述
            generated_text = generate_caption(model, tokenizer, image_array)
            if not generated_text:
                continue
                
            # 评估
            scores = evaluate_captions(findings, generated_text, green_scorer)
            if scores:
                scores['image'] = case['image']
                all_scores.append(scores)
                
                print(f"\nImage: {case['image']}")
                print(f"Generated: {generated_text}")
                print(f"Ground truth: {findings}")
                print(f"Scores: {scores}")
                print("\nDetailed GREEN Analysis:")
                print(scores['green_analysis'])
                print("\nError Counts:")
                for error_type, count in scores['error_counts'].items():
                    print(f"{error_type}: {count}")
                
        except Exception as e:
            print(f"Error processing case {case['image']}: {str(e)}")
            continue
    
    if all_scores:
        # 计算平均分数
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor',
                  'green_mean', 'green_std']
        
        avg_scores = {
            metric: np.mean([s[metric] for s in all_scores])
            for metric in metrics
        }
        
        # 计算错误类型的平均数
        error_types = ['false_reports', 'missing_findings', 'wrong_location', 
                      'wrong_severity', 'extra_comparison', 'missing_comparison',
                      'matched_findings']
        
        avg_errors = {
            error_type: np.mean([s['error_counts'][error_type] for s in all_scores])
            for error_type in error_types
        }
        
        print("\nAverage scores across all cases:")
        for metric, value in avg_scores.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nAverage error counts across all cases:")
        for error_type, value in avg_errors.items():
            print(f"{error_type}: {value:.4f}")
        
        # 保存结果
        with open('amos_evaluation_results.txt', 'w') as f:
            f.write("Individual scores:\n")
            for score in all_scores:
                f.write(f"Image: {score['image']}\n")
                f.write(f"Scores: {score}\n")
                f.write(f"GREEN Analysis:\n{score['green_analysis']}\n")
                f.write(f"Error Counts:\n")
                for error_type, count in score['error_counts'].items():
                    f.write(f"{error_type}: {count}\n")
                f.write("-" * 80 + "\n")
                
            f.write("\nAverage scores:\n")
            for metric, value in avg_scores.items():
                f.write(f"{metric}: {value:.4f}\n")
                
            f.write("\nAverage error counts:\n")
            for error_type, value in avg_errors.items():
                f.write(f"{error_type}: {value:.4f}\n")
        
        print(f"\nTotal processed cases: {len(all_scores)}")
    else:
        print("No scores were generated!")
if __name__ == '__main__':
    main()