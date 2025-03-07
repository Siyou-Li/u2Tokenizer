import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
from dataclasses import dataclass, field
import pandas as pd
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk
from scipy.ndimage import zoom
import SimpleITK as sitk
from green_score import GREEN  # 导入GREEN类
import traceback

nltk.download('wordnet')

def calculate_metrics(generated_text, reference_text, green_scorer):
    try:
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        reference_text = preprocess_text(reference_text)
        generated_text = preprocess_text(generated_text)
        
        # BLEU score
        smoother = SmoothingFunction().method3
        weights = (1, 0, 0, 0)  # Equal weights for 1-4 grams
        references_tokenized = [reference_text.split()]
        hypothesis_tokenized = generated_text.split()
        bleu_score = corpus_bleu([references_tokenized], [hypothesis_tokenized], 
                                weights=weights,
                                smoothing_function=smoother) * 100
        
        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_text, generated_text)
        rouge1 = rouge_scores['rouge1'].fmeasure * 100
        rouge2 = rouge_scores['rouge2'].fmeasure * 100
        rougeL = rouge_scores['rougeL'].fmeasure * 100
        
        # BERTScore
        P, R, F1 = score([generated_text], [reference_text], lang='en', verbose=False)
        bert_f1 = F1.mean().item() * 100
        
        # METEOR score
        meteor_score_value = meteor_score([reference_text.split()], generated_text.split()) * 100
        
        # GREEN score
        mean, std, green_score_list, summary, result_df = green_scorer([reference_text], [generated_text])
        
        # 提取详细的错误分析
        error_counts = {}
        if not result_df.empty:
            error_counts = {
                'false_reports': result_df.iloc[0]['(a) False report of a finding in the candidate'],
                'missing_findings': result_df.iloc[0]['(b) Missing a finding present in the reference'],
                'wrong_location': result_df.iloc[0]['(c) Misidentification of a finding\'s anatomic location/position'],
                'wrong_severity': result_df.iloc[0]['(d) Misassessment of the severity of a finding'],
                'extra_comparison': result_df.iloc[0]['(e) Mentioning a comparison that isn\'t in the reference'],
                'missing_comparison': result_df.iloc[0]['(f) Omitting a comparison detailing a change from a prior study'],
                'matched_findings': result_df.iloc[0]['Matched Findings']
            }
        
        return {
            'bleu': round(bleu_score, 2),
            'rouge1': round(rouge1, 2),
            'rouge2': round(rouge2, 2),
            'rougeL': round(rougeL, 2),
            'bert_score': round(bert_f1, 2),
            'meteor': round(meteor_score_value, 2),
            'green_mean': mean,
            'green_std': std,
            'green_summary': summary,
            'error_counts': error_counts
        }
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        print(traceback.format_exc())
        return None

def format_text(text):
    text = ' '.join(text.split())
    text = text.replace('.', '. ')
    return ' '.join(text.strip().split())

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

@dataclass
class AllArguments:
    model_name_or_path: str = field(default="GoodBaiBai88/M3D-LaMed-Llama-2-7B")
    proj_out_num: int = field(default=256)
    base_path: str = field(default="/rds/projects/l/lez-medical-ml/PengyaoQin/download/AbdomenAtalsData/")

def load_and_preprocess_image(base_path, bdmap_id):
    """加载和预处理nii.gz格式的图像，确保尺寸符合模型期望"""
    try:
        # 构建图像路径
        image_path = os.path.join(base_path, bdmap_id, "ct.nii.gz")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
            
        # 读取nii.gz文件
        print(f"Loading image from: {image_path}")
        img = sitk.ReadImage(image_path)
        img_array = sitk.GetArrayFromImage(img)
        
        print(f"Original image shape: {img_array.shape}")
        
        # 标准化
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        # 使用scipy.ndimage.zoom进行重采样 - 回到原始代码的方式
        target_shape = (32, 256, 256)  # D, H, W
        print(f"Target shape: {target_shape}")
        
        # 计算缩放因子
        resize_factors = [target_shape[i] / img_array.shape[i] for i in range(3)]
        print(f"Resize factors: {resize_factors}")
        
        # 缩放图像
        img_array_resized = zoom(img_array, resize_factors, order=1)
        print(f"Resized 3D image shape: {img_array_resized.shape}")
        
        # 添加通道维度
        img_array_reshaped = np.expand_dims(img_array_resized, axis=0)  # [C, D, H, W]
        print(f"Reshaped image format [C,D,H,W]: {img_array_reshaped.shape}")
        
        return img_array_reshaped
        
    except Exception as e:
        print(f"Error processing image {bdmap_id}: {str(e)}")
        print(traceback.format_exc())
        return None

def main():
    seed_everything(42)
    device = torch.device('cuda')
    # 使用bfloat16，这可能是关键差异
    dtype = torch.bfloat16
    
    # 更新路径
    base_path = "/rds/projects/l/lez-medical-ml/PengyaoQin/download/AbdomenAtalsData/"
    csv_path = '/rds/homes/p/pxq307/Compare/AbdomenAtlas3.0.csv'
    df = pd.read_csv(csv_path)
    
    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 初始化模型和GREEN评分器
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=512, padding_side="right", 
                                            use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, 
                                                device_map='auto', trust_remote_code=True).to(device)
    
    print("Loading GREEN scorer...")
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    
    question = "Please caption this medical scan with detailed findings."
    image_tokens = "<im_patch>" * args.proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device)
    
    all_scores = []
    
    # 遍历CSV中的所有BDMAP ID
    for bdmap_id in df['BDMAP ID'].unique():
        try:
            print(f"\n--- Processing BDMAP ID: {bdmap_id} ---")
            ground_truth_df = df[df['BDMAP ID'] == bdmap_id]
            
            if ground_truth_df.empty or pd.isna(ground_truth_df['narrative report'].iloc[0]):
                print(f"No report found for {bdmap_id}")
                continue
                
            ground_truth = ground_truth_df['narrative report'].iloc[0]
            
            # 加载和预处理图像
            image_array = load_and_preprocess_image(base_path, bdmap_id)
            if image_array is None:
                continue
            
            # 创建5D张量 [B,C,D,H,W]
            # 首先转为PyTorch张量
            image_pt = torch.from_numpy(image_array).to(dtype=dtype, device=device)
            
            # 然后添加batch维度，创建5D张量: [B,C,D,H,W]
            image_pt_5d = image_pt.unsqueeze(0)
            print(f"5D tensor shape [B,C,D,H,W]: {image_pt_5d.shape}")
            
            # 生成描述 - 使用与正常工作的代码相同的方式
            print("Generating caption...")
            try:
                generation = model.generate(
                    image_pt_5d, 
                    input_id,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,  # 与正常工作的代码保持一致
                    temperature=1.0
                )
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                generated_text = format_text(generated_texts[0])
                
                print(f"Ground truth: {ground_truth}")
                print(f"Generated: {generated_text}")
                
                # 计算评分
                print("Calculating metrics...")
                scores = calculate_metrics(generated_text, ground_truth, green_scorer)
                if scores:
                    scores['bdmap_id'] = bdmap_id
                    all_scores.append(scores)
                    
                    print(f"Scores: {scores}")
                    print("\nDetailed error analysis:")
                    for error_type, count in scores['error_counts'].items():
                        print(f"{error_type}: {count}")
            except RuntimeError as e:
                print(f"Error generating caption: {str(e)}")
                print(f"Model expected dimensions: Check above error for details")
                print(traceback.format_exc())
                continue
                
        except Exception as e:
            print(f"Error processing case {bdmap_id}: {str(e)}")
            print(traceback.format_exc())
            continue
    
    if all_scores:
        avg_scores = {
            metric: np.mean([s[metric] for s in all_scores if metric in s])
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor', 
                          'green_mean', 'green_std']
        }
        
        # 计算错误类型的平均数
        error_types = ['false_reports', 'missing_findings', 'wrong_location', 
                      'wrong_severity', 'extra_comparison', 'missing_comparison',
                      'matched_findings']
        
        avg_errors = {
            error_type: np.mean([s['error_counts'][error_type] for s in all_scores if 'error_counts' in s])
            for error_type in error_types
        }
        
        print("\nAverage scores:")
        for metric, value in avg_scores.items():
            print(f"Average {metric}: {value:.4f}")
            
        print("\nAverage error counts:")
        for error_type, value in avg_errors.items():
            print(f"{error_type}: {value:.4f}")
            
        # 保存结果到文件
        with open('evaluation_results.txt', 'w') as f:
            f.write("Individual scores:\n")
            for score in all_scores:
                f.write(f"BDMAP ID: {score['bdmap_id']}\n")
                f.write(f"Scores: {score}\n")
                if 'error_counts' in score:
                    f.write("Error counts:\n")
                    for error_type, count in score['error_counts'].items():
                        f.write(f"{error_type}: {count}\n")
                f.write("-" * 80 + "\n")
                
            f.write("\nAverage scores:\n")
            for metric, value in avg_scores.items():
                f.write(f"{metric}: {value:.4f}\n")
                
            f.write("\nAverage error counts:\n")
            for error_type, value in avg_errors.items():
                f.write(f"{error_type}: {value:.4f}\n")
    else:
        print("No scores were generated!")

if __name__ == "__main__":
    main()