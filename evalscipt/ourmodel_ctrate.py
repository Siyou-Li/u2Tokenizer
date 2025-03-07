import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
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

nltk.download('wordnet', quiet=True)
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def load_model_with_lora(model_name_or_path, lora_model_path=None):
    device = torch.device('cuda') # 'cpu', 'cuda'
    dtype = torch.bfloat16 # or bfloat16, float16, float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    
    if lora_model_path is not None:
        state_dict = torch.load(lora_model_path, map_location="cpu")
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
        print("Load weights with LoRA")
        model.load_state_dict(state_dict, strict=True)
        print("Merge weights with LoRA")
        model = model.merge_and_unload()
    else:
        model = base_model
        
    return model, tokenizer

def load_model_from_huggingface():
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

def generate_caption(model, tokenizer, image_array, l3dt=None):
    try:
        device = next(model.parameters()).device
        
        # 判断是否使用Linear3DTransform或直接使用预处理好的数组
        if l3dt is not None:
            # 假设image_array是文件路径
            image = l3dt(image_array)
            image_pt = image.unsqueeze(0).to(device=device)
        else:
            # 假设image_array已经是预处理好的numpy数组
            image_pt = torch.from_numpy(image_array).to(device=device, dtype=torch.float16)
        
        question = "Can you provide a caption consists of findings and expressions for this medical image?"
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
        return None

def evaluate_captions(reference, hypothesis, green_scorer):
    try:
        # 其他评估指标的预处理
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        reference_processed = preprocess_text(reference)
        hypothesis_processed = preprocess_text(hypothesis)
        
        # BLEU
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
        import traceback
        print(f"Full error details:", traceback.format_exc())
        return None
def main():
    import pandas as pd
    
    # 模型和tokenizer初始化
    model_name_or_path = './checkpoint-100080'
    lora_model_path = './checkpoint-100080/model_with_lora.bin'
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_with_lora(model_name_or_path, lora_model_path)
    l3dt = Linear3DTransform(data_type="validation")
    
    # 初始化GREEN评分器
    print("Initializing GREEN scorer...")
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    
    # 存储所有评估结果
    all_scores = []
    
    def process_nii_file(nii_path):
        try:
            # 生成caption
            image = l3dt(nii_path)
            
            proj_out_num = 256
            question = "Can you provide a caption consists of findings and expressions for this medical image?"
            question_ids = tokenizer(
                question, add_special_tokens=False, max_length=768, truncation=True, 
                padding="max_length", return_tensors="pt", padding_side="right"
            )["input_ids"][0]
            
            image_tokens = "<im_patch>" * proj_out_num
            input_txt = image_tokens + question
            input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device='cuda')
            
            with torch.cuda.amp.autocast():
                generation = model.generate(
                    image.unsqueeze(0).to(device='cuda'),
                    input_id,
                    question_ids=question_ids.to(device='cuda'),
                    max_new_tokens=768,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0
                )
            
            generated_text = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
            return generated_text
            
        except Exception as e:
            print(f"Error processing file {nii_path}: {str(e)}")
            return None
    
    def process_directory(base_dir, reference_df):
        """处理目录下的所有nii.gz文件，使用DataFrame作为reference"""
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    nii_path = os.path.join(root, file)
                    print(f"\nProcessing: {nii_path}")
                    reference_row = reference_df[reference_df['VolumeName'] == file]
                    if reference_row.empty:
                        print(f"No reference found for {file}")
                        continue
                    
                    reference_text = reference_row['Findings_EN'].iloc[0]
                    
                    # 生成caption
                    generated_text = process_nii_file(nii_path)
                    if not generated_text:
                        continue
                    
                    # 评估生成的文本
                    scores = evaluate_captions(reference_text, generated_text, green_scorer)
                    if scores:
                        scores['file'] = file
                        all_scores.append(scores)
                        
                        print(f"\nFile: {file}")
                        print(f"Generated: {generated_text}")
                        print(f"Reference: {reference_text}")
                        print(f"Scores: {scores}")
                        print("\nDetailed GREEN Analysis:")
                        print(scores['green_analysis'])
                        print("\nError Counts:")
                        for error_type, count in scores['error_counts'].items():
                            print(f"{error_type}: {count}")
    
    base_directory = "/rds/projects/l/lez-medical-ml/PengyaoQin/download/CT-RATE/dataset/valid/"  # 替换为实际的nii.gz文件目录
    csv_path = "/rds/homes/p/pxq307/Compare/valid_labels.csv"    # 替换为实际的CSV文件路径
    print("Loading reference data...")
    reference_df = pd.read_csv(csv_path)
    process_directory(base_directory, reference_df)
    if all_scores:
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor',
                  'green_mean', 'green_std']
        
        avg_scores = {
            metric: np.mean([s[metric] for s in all_scores])
            for metric in metrics
        }
        error_types = ['false_reports', 'missing_findings', 'wrong_location', 
                      'wrong_severity', 'extra_comparison', 'missing_comparison',
                      'matched_findings']
        
        avg_errors = {
            error_type: np.mean([s['error_counts'].get(error_type, 0) for s in all_scores])
            for error_type in error_types
        }
        #以txt格式存一下结果
        result_filename = 'evaluation_results.txt'
        with open(result_filename, 'w') as f:
            f.write("Individual scores:\n")
            for score in all_scores:
                f.write(f"File: {score['file']}\n")
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
        print(f"Results saved to {result_filename}")
    else:
        print("No scores were generated!")

if __name__ == '__main__':
    main()