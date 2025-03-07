import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
from green_score import GREEN
nltk.download('wordnet', quiet=True)
import monai.transforms as mtf

# 缓存评估器
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def load_model():
    device = torch.device('cuda')
    dtype = torch.bfloat16
    model_name = 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
    
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

def generate_caption(model, tokenizer, image_path):
    import monai.transforms as mtf
    
    device = next(model.parameters()).device
    image_np = np.load(image_path)
    print(f"Original shape: {image_np.shape}")
    print(f"Data type: {image_np.dtype}")
    print(f"Min/Max values: {image_np.min()}, {image_np.max()}")
    image_np = image_np - image_np.min()
    image_np = image_np / (image_np.max() + 1e-8)  # 归一化到[0,1]
    # 直接处理维度
    if image_np.ndim == 3:
        image_np = np.transpose(image_np, (2, 0, 1))  # D,H,W -> H,W,D
    image_np = image_np[np.newaxis, np.newaxis, ...]  # 添加batch和channel维度
    
    # 使用torch进行resize
    image_pt = torch.from_numpy(image_np).to(device=device, dtype=torch.bfloat16)
    image_pt = torch.nn.functional.interpolate(
        image_pt,
        size=(256, 256, 32),
        mode='trilinear',
        align_corners=True
    )
    
    print(f"Final shape: {image_pt.shape}")
    
    question = "Please generate a medical report based on this image."
    image_tokens = "<im_patch>" * 256
    input_txt = image_tokens + question
    input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device)
    
    generation = model.generate(
        image_pt, 
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=1.0
    )
    return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

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
        bleu_score = corpus_bleu([references_tokenized], [hypothesis_tokenized], 
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
        return None

def main():
    npy_dir = '/rds/projects/l/lez-medical-ml/PengyaoQin/download/CT-RATE/dataset/npy_files/'
    csv_path = './Compare/valid_labels.csv'
    df = pd.read_csv(csv_path)
    
    # 初始化模型和评估器
    model, tokenizer = load_model()
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    
    all_scores = []
    error_cases = []
    
    for npy_file in tqdm(os.listdir(npy_dir), desc="Processing files"):
        try:
            if not npy_file.endswith('.npy'):
                continue
                
            bdmap_id = npy_file.replace('.npy', '.nii.gz')
            print(f"\nProcessing {bdmap_id}...")
            
            # 查找ground truth
            try:
                ground_truth = df[df['VolumeName'] == bdmap_id]['Findings_EN'].iloc[0]
            except (IndexError, KeyError) as e:
                print(f"No report found for {bdmap_id}")
                error_cases.append((bdmap_id, f"No report found: {str(e)}"))
                continue
            
            if pd.isna(ground_truth):
                print(f"Empty report for {bdmap_id}")
                error_cases.append((bdmap_id, "Empty report"))
                continue
            
            # 生成caption
            try:
                npy_path = os.path.join(npy_dir, npy_file)
                generated_caption = generate_caption(model, tokenizer, npy_path)
            except Exception as e:
                print(f"Error generating caption for {bdmap_id}: {str(e)}")
                error_cases.append((bdmap_id, f"Generation error: {str(e)}"))
                continue
            
            # 评估
            try:
                scores = evaluate_captions(ground_truth, generated_caption, green_scorer)
                if scores:
                    scores['bdmap_id'] = bdmap_id
                    scores['reference'] = ground_truth
                    scores['hypothesis'] = generated_caption
                    all_scores.append(scores)
                    
                    print(f"Generated: {generated_caption}")
                    print(f"Ground truth: {ground_truth}")
                    print(f"Scores: {scores}")
                    print("\nDetailed GREEN Analysis:")
                    print(scores['green_analysis'])
                    print("\nError Counts:")
                    for error_type, count in scores['error_counts'].items():
                        print(f"{error_type}: {count}")
            except Exception as e:
                print(f"Error evaluating captions for {bdmap_id}: {str(e)}")
                error_cases.append((bdmap_id, f"Evaluation error: {str(e)}"))
                continue
                
        except Exception as e:
            print(f"Unexpected error processing {npy_file}: {str(e)}")
            error_cases.append((npy_file, f"Unexpected error: {str(e)}"))
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
        
        # 保存详细结果
        with open('ct_rate_evaluation_results.txt', 'w') as f:
            f.write("Individual scores:\n")
            for score in all_scores:
                f.write(f"Image: {score['bdmap_id']}\n")
                f.write(f"Generated: {score['hypothesis']}\n")
                f.write(f"Reference: {score['reference']}\n")
                f.write("Metrics:\n")
                for metric in metrics:
                    f.write(f"{metric}: {score[metric]}\n")
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
        
        # 保存错误案例
        with open('ct_rate_error_cases.txt', 'w') as f:
            f.write("Failed cases:\n")
            for case_id, error in error_cases:
                f.write(f"{case_id}: {error}\n")
            
        print(f"\nTotal processed: {len(all_scores) + len(error_cases)}")
        print(f"Successful cases: {len(all_scores)}")
        print(f"Failed cases: {len(error_cases)}")
        
        print("\nAverage scores across all cases:")
        for metric, value in avg_scores.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nAverage error counts across all cases:")
        for error_type, value in avg_errors.items():
            print(f"{error_type}: {value:.4f}")
    else:
        print("No scores were generated!")

if __name__ == '__main__':
    main()