import argparse
import torch
import json
import os
import numpy as np
import pandas as pd
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
nltk.download('wordnet', quiet=True)
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_captions(reference, hypothesis, green_scorer=None):
    try:
        def preprocess_text(text):
            return ' '.join(text.lower().split())
        
        reference_processed = preprocess_text(reference)
        hypothesis_processed = preprocess_text(hypothesis)
        smoother = SmoothingFunction().method3
        weights = (1, 0, 0, 0)
        references_tokenized = [reference_processed.split()]
        hypothesis_tokenized = hypothesis_processed.split()
        bleu_score = corpus_bleu([references_tokenized], [hypothesis_tokenized], 
                                weights=weights,
                                smoothing_function=smoother) * 100
        rouge_scores = rouge_scorer_instance.score(reference_processed, hypothesis_processed)
        rouge1 = rouge_scores['rouge1'].fmeasure * 100
        rouge2 = rouge_scores['rouge2'].fmeasure * 100
        rougeL = rouge_scores['rougeL'].fmeasure * 100
        
        # BERTScore
        P, R, F1 = score([hypothesis_processed], [reference_processed], lang='en', verbose=False)
        bert_f1 = F1.mean().item()
        
        # METEOR
        meteor_score_value = meteor_score([reference_processed.split()], hypothesis_tokenized) * 100
        
        # GREEN
        if green_scorer:
            try:
                mean, std, green_score_list, summary, result_df = green_scorer([reference_processed], [hypothesis_processed])
                
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
                print(f"GREEN评分错误: {str(e)}")
        
        return {
            'bleu': round(bleu_score, 2),
            'rouge1': round(rouge1, 2),
            'rouge2': round(rouge2, 2),
            'rougeL': round(rougeL, 2),
            'bert_score': round(bert_f1, 4),
            'meteor': round(meteor_score_value, 2)
        }
    except Exception as e:
        print(f"评估指标计算错误: {str(e)}")
        return None

def main(args):
    print(f"正在加载模型: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        local_files_only=True
    )
    
    print("模型加载成功")
    green_scorer = None
    if args.evaluate:
        try:
            from green_score import GREEN
            print("正在初始化GREEN评分...")
            green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
            print("GREEN评分初始化成功")
        except ImportError:
            print("警告: 无法导入GREEN评分模块，将仅使用其他评估指标")
        except Exception as e:
            print(f"初始化GREEN评分时出错: {str(e)}")
    
    df = None
    if args.evaluate and args.csv_path:
        print(f"读取CSV文件: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        print(f"CSV文件加载成功，包含 {len(df)} 行")

    print(f"读取输入JSON文件: {args.input_json}")
    with open(args.input_json, 'r') as file:
        data_val = json.load(file)
    print(f"JSON文件加载成功，包含 {len(data_val)} 个样本")

    output_save = []
    all_scores = []
    error_cases = []

    for element in tqdm(data_val, desc="处理样本"):
        try:
            if "image_path" in element:
                image_rel_path = element["image_path"]
                image_path = os.path.join(args.image_folder, image_rel_path)
                print(f"样本图像路径: {image_path}")
            else:
                print(f"样本图像: {element['image']}")

            conversations_save = []
            
            for conversation in element["conversations"]:
                if conversation["from"] == "human":
                    inp = conversation["value"]
                    
                    try:
                        prompt = inp
                        
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                        
                        streamer = TextStreamer(tokenizer, skip_special_tokens=True)
                        
                        with torch.inference_mode():
                            output_ids = model.generate(
                                input_ids,
                                do_sample=True if args.temperature > 0 else False,
                                temperature=args.temperature,
                                max_new_tokens=args.max_new_tokens,
                                streamer=streamer,
                                use_cache=True)

                        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                        
                        conversations_save.append({"question": inp, "answer": outputs})
                        
                        if args.evaluate and df is not None:
                            bdmap_id = element["image"]
                            
                            try:
                                ref_row = df[df['VolumeName'] == bdmap_id]
                                if not ref_row.empty and 'Findings_EN' in ref_row.columns:
                                    ground_truth = ref_row['Findings_EN'].iloc[0]
                                    
                                    if pd.isna(ground_truth):
                                        print(f"警告: {bdmap_id} 的参考文本为空")
                                        error_cases.append((bdmap_id, "参考文本为空"))
                                    else:                                        
                                        scores = evaluate_captions(ground_truth, outputs, green_scorer)
                                        if scores:
                                            scores['bdmap_id'] = bdmap_id
                                            scores['reference'] = ground_truth
                                            scores['hypothesis'] = outputs
                                            all_scores.append(scores)
                                            
                                            if args.verbose:
                                                print(f"\n生成文本: {outputs}")
                                                print(f"参考文本: {ground_truth}")
                                                print(f"评分: {scores}")
                                                if 'green_analysis' in scores:
                                                    print("\n详细GREEN分析:")
                                                    print(scores['green_analysis'])
                                                if 'error_counts' in scores:
                                                    print("\n错误计数:")
                                                    for error_type, count in scores['error_counts'].items():
                                                        print(f"{error_type}: {count}")
                                else:
                                    print(f"警告: 在CSV中找不到 {bdmap_id} 的记录")
                                    error_cases.append((bdmap_id, "在CSV中找不到记录"))
                            except Exception as e:
                                print(f"评估时出错: {str(e)}")
                                error_cases.append((bdmap_id, f"评估错误: {str(e)}"))
                    except Exception as e:
                        print(f"生成文本时出错: {str(e)}")
                        error_cases.append((element["image"], f"生成错误: {str(e)}"))
                
                if args.debug:
                    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

            output_save.append({
                "image": element["image"],
                "conversations_out": conversations_save
            })
        
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            if "image" in element:
                error_cases.append((element["image"], f"处理错误: {str(e)}"))
            else:
                error_cases.append(("未知样本", f"处理错误: {str(e)}"))

    # 保存生成结果
    print(f"保存输出到JSON文件: {args.output_json}")
    with open(args.output_json, "w") as json_file:
        json.dump(output_save, json_file, indent=4)
    
    # 如果有评估结果
    if args.evaluate and all_scores:
        # 计算平均分数
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor']
        if 'green_mean' in all_scores[0]:
            metrics.extend(['green_mean', 'green_std'])
        
        avg_scores = {
            metric: np.mean([s[metric] for s in all_scores if metric in s])
            for metric in metrics
        }
        
        # 计算错误类型的平均数
        if 'error_counts' in all_scores[0]:
            error_types = ['false_reports', 'missing_findings', 'wrong_location', 
                          'wrong_severity', 'extra_comparison', 'missing_comparison',
                          'matched_findings']
            
            avg_errors = {
                error_type: np.mean([s['error_counts'][error_type] for s in all_scores if 'error_counts' in s])
                for error_type in error_types
            }
        else:
            avg_errors = {}
        
        # 保存评估结果
        with open(args.evaluation_output, 'w') as f:
            f.write("个体评分:\n")
            for score in all_scores:
                f.write(f"图像: {score['bdmap_id']}\n")
                f.write(f"生成文本: {score['hypothesis']}\n")
                f.write(f"参考文本: {score['reference']}\n")
                f.write("指标:\n")
                for metric in metrics:
                    if metric in score:
                        f.write(f"{metric}: {score[metric]}\n")
                if 'green_analysis' in score:
                    f.write(f"GREEN分析:\n{score['green_analysis']}\n")
                if 'error_counts' in score:
                    f.write(f"错误计数:\n")
                    for error_type, count in score['error_counts'].items():
                        f.write(f"{error_type}: {count}\n")
                f.write("-" * 80 + "\n")
                
            f.write("\n平均评分:\n")
            for metric, value in avg_scores.items():
                f.write(f"{metric}: {value:.4f}\n")
                
            if avg_errors:
                f.write("\n平均错误计数:\n")
                for error_type, value in avg_errors.items():
                    f.write(f"{error_type}: {value:.4f}\n")
        
        # 保存错误案例
        with open(args.evaluation_output.replace('.txt', '_error_cases.txt'), 'w') as f:
            f.write("失败案例:\n")
            for case_id, error in error_cases:
                f.write(f"{case_id}: {error}\n")
            
        print(f"\n处理总数: {len(all_scores) + len(error_cases)}")
        print(f"成功案例: {len(all_scores)}")
        print(f"失败案例: {len(error_cases)}")
        
        print("\n所有案例的平均评分:")
        for metric, value in avg_scores.items():
            print(f"{metric}: {value:.4f}")
            
        if avg_errors:
            print("\n所有案例的平均错误计数:")
            for error_type, value in avg_errors.items():
                print(f"{error_type}: {value:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试并评估语言模型")
    # 模型参数
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备（cuda或cpu）")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成令牌数")
    parser.add_argument("--debug", action="store_true", help="启用调试输出")
    
    # 评估参数
    parser.add_argument("--evaluate", action="store_true", help="启用评估")
    parser.add_argument("--csv-path", type=str, default=None, help="包含标签的CSV文件路径")
    parser.add_argument("--input-json", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output-json", type=str, required=True, help="输出JSON文件路径")
    parser.add_argument("--evaluation-output", type=str, default="evaluation_results.txt", help="评估结果输出路径")
    parser.add_argument("--image-folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--verbose", action="store_true", help="输出详细信息")
    
    args = parser.parse_args()
    
    # 运行主函数
    main(args)