import os
import json
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
from green_score import GREEN
from linear_3d_transform import Linear3DTransform

# Download required nltk data
nltk.download('wordnet', quiet=True)

# Cache evaluator
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_abdomen_atlas_dataset(test_csv_path, full_dataset_csv_path):
    """加载Abdomen Atlas数据集信息"""
    # 读取测试ID列表
    test_df = pd.read_csv(test_csv_path)
    
    # 读取完整数据集
    full_df = pd.read_csv(full_dataset_csv_path)
    
    # 确保列名正确
    bdmap_id_col = 'BDMAP ID'
    if bdmap_id_col not in test_df.columns:
        # 尝试查找实际列名
        for col in test_df.columns:
            if 'bdmap' in col.lower() or 'id' in col.lower():
                bdmap_id_col = col
                print(f"使用列名: {bdmap_id_col}")
                break
    
    # 创建测试集信息
    test_cases = []
    for idx, row in test_df.iterrows():
        bdmap_id = row[bdmap_id_col]
        
        # 在完整数据集中查找对应记录
        matching_row = full_df[full_df[bdmap_id_col] == bdmap_id]
        
        if matching_row.empty:
            print(f"警告: 在完整数据集中未找到ID {bdmap_id}")
            continue
        
        # 获取narrative report
        report = matching_row['narrative report'].iloc[0]
        if pd.isna(report) or report == '':
            print(f"警告: ID {bdmap_id}的报告为空")
            continue
        
        # 构建图像路径
        image_path = os.path.join(
            '/rds/projects/l/lez-medical-ml/PengyaoQin/download/AbdomenAtalsData',
            bdmap_id,
            'ct.nii.gz'
        )
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
        
        test_cases.append({
            'bdmap_id': bdmap_id,
            'image_path': image_path,
            'report': report
        })
    
    return test_cases

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    device = torch.device('cuda')
    dtype = torch.float16
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_caption(model, tokenizer, image_path, body_part, l3dt):
    try:
        device = next(model.parameters()).device
        
        # 使用Linear3DTransform处理图像
        image = l3dt(image_path)
        image_pt = image.unsqueeze(0).to(device=device)
        
        # 对于Atlas数据集，我们使用abdominal作为body_part
        if body_part is None:
            body_part = "abdominal"
            
        question = f"Can you provide a diagnosis based on the findings in {body_part} in this image?"
        question_ids = tokenizer(
            question, add_special_tokens=False, max_length=768, truncation=True, 
            padding="max_length", return_tensors="pt", padding_side="right"
        )["input_ids"][0]
        
        proj_out_num = 256
        image_tokens = "<im_patch>" * proj_out_num
        input_txt = image_tokens + question
        input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
        
        with torch.cuda.amp.autocast(): 
            generation = model.generate(
                image_pt, 
                input_id,
                question_ids=question_ids.to(device=device),
                max_new_tokens=768,
                do_sample=True,
                top_p=0.9,
                temperature=1.0
            )
        
        return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        import traceback
        print(f"Full error details:", traceback.format_exc())
        return None

def evaluate_captions(reference, hypothesis, green_scorer):
    try:
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        reference_processed = preprocess_text(reference)
        hypothesis_processed = preprocess_text(hypothesis)
        
        # BLEU
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
        
        # GREEN evaluation
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
        import traceback
        print(f"Full error details:", traceback.format_exc())
        return None

def main():
    # 设置路径
    test_csv_path = 'IID_test.csv'
    full_dataset_csv_path = 'AbdomenAtlas3.0.csv'
    model_path = '/rds/projects/l/lez-medical-ai/PengyaoQin/Ourmodel/sync_models/checkpoint-40000/' 
    output_filename = 'amos_on_abdomen_atlas_results.txt'
    
    # 加载数据集信息
    test_cases = load_abdomen_atlas_dataset(test_csv_path, full_dataset_csv_path)
    print(f"加载了 {len(test_cases)} 个测试案例")
    
    # 初始化模型和评估器
    model, tokenizer = load_model_and_tokenizer(model_path)
    l3dt = Linear3DTransform(data_type="validation")
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    
    # 存储所有评分
    all_scores = []
    
    # 处理每个测试案例
    for case in tqdm(test_cases, desc="Processing Abdomen Atlas test cases"):
        try:
            # 获取图像路径和ground truth
            image_path = case['image_path']
            reference_text = case['report']
            
            if not reference_text:
                print(f"No report for {case['bdmap_id']}")
                continue
            
            # 为腹部生成描述（使用AMOS的生成方法）
            generated_text = generate_caption(model, tokenizer, image_path, "abdominal", l3dt)
            if not generated_text:
                continue
                
            # 评估
            scores = evaluate_captions(reference_text, generated_text, green_scorer)
            if scores:
                scores['bdmap_id'] = case['bdmap_id']
                all_scores.append(scores)
                
                print(f"\nImage: {case['bdmap_id']}")
                print(f"Generated: {generated_text}")
                print(f"Ground truth: {reference_text}")
                print(f"Scores: {scores}")
                print("\nDetailed GREEN Analysis:")
                print(scores['green_analysis'])
                print("\nError Counts:")
                for error_type, count in scores['error_counts'].items():
                    print(f"{error_type}: {count}")
                
        except Exception as e:
            print(f"Error processing case {case['bdmap_id']}: {str(e)}")
            continue
    
    # 计算和保存结果
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
            error_type: np.mean([s['error_counts'].get(error_type, 0) for s in all_scores])
            for error_type in error_types
        }
        
        # 输出平均分数
        print("\nAverage scores across all cases:")
        for metric, value in avg_scores.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nAverage error counts across all cases:")
        for error_type, value in avg_errors.items():
            print(f"{error_type}: {value:.4f}")
        
        # 保存结果
        with open(output_filename, 'w') as f:
            f.write("Individual scores:\n")
            for score in all_scores:
                f.write(f"Image: {score['bdmap_id']}\n")
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
        print(f"Results saved to {output_filename}")
    else:
        print("No scores were generated!")

if __name__ == '__main__':
    main()