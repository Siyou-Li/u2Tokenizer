import os
import json
import numpy as np
import torch
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

def load_dataset_info(base_path):
    """Load dataset information"""
    with open(os.path.join(base_path, 'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

def load_model_and_tokenizer(model_path):
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
        
        # Use Linear3DTransform to process image
        image = l3dt(image_path)
        image_pt = image.unsqueeze(0).to(device=device)
        
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
    # Set paths
    base_path = '/rds/projects/l/lez-medical-ai/PengyaoQin/AMOS/'
    model_path = '/rds/projects/l/lez-medical-ai/PengyaoQin/Ourmodel/sync_models/checkpoint-2000/'
    
    # Load dataset info
    dataset_info = load_dataset_info(base_path)
    validation_cases = dataset_info['validation']
    
    # Limit to only 100 images
    max_images = 100
    validation_cases = validation_cases[:max_images]
    
    # Initialize model and evaluators
    model, tokenizer = load_model_and_tokenizer(model_path)
    l3dt = Linear3DTransform(data_type="validation")
    green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    
    # Store scores for each body part
    scores_by_part = {
        'chest': [],
        'abdomen': [],
        'pelvis': []
    }
    
    print(f"Processing {len(validation_cases)} validation cases (limited to {max_images})")
    
    for case in tqdm(validation_cases, desc="Processing validation cases"):
        try:
            image_path = os.path.join(base_path, case['image'].lstrip('./'))
            
            # Process each body part separately
            for body_part in ['chest', 'abdomen', 'pelvis']:
                reference = case['labels']['report']['findings'].get(body_part, '')
                
                # Skip if no findings for this body part
                if not reference:
                    continue
                
                # Generate caption for this body part
                generated_text = generate_caption(model, tokenizer, image_path, body_part, l3dt)
                if not generated_text:
                    continue
                
                # Evaluate
                scores = evaluate_captions(reference, generated_text, green_scorer)
                if scores:
                    scores['image'] = case['image']
                    scores['body_part'] = body_part
                    scores_by_part[body_part].append(scores)
                    
                    print(f"\nImage: {case['image']} - {body_part}")
                    print(f"Generated: {generated_text}")
                    print(f"Ground truth: {reference}")
                    print(f"Scores: {scores}")
                    
        except Exception as e:
            print(f"Error processing case {case['image']}: {str(e)}")
            continue
    
    # Calculate and save results
    if any(scores_by_part.values()):
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'meteor',
                  'green_mean', 'green_std']
        error_types = ['false_reports', 'missing_findings', 'wrong_location', 
                      'wrong_severity', 'extra_comparison', 'missing_comparison',
                      'matched_findings']
        
        # Calculate averages for each body part
        avg_scores_by_part = {}
        for body_part, scores in scores_by_part.items():
            if scores:
                avg_scores = {
                    metric: np.mean([s[metric] for s in scores])
                    for metric in metrics
                }
                avg_errors = {
                    error_type: np.mean([s['error_counts'].get(error_type, 0) for s in scores])
                    for error_type in error_types
                }
                avg_scores_by_part[body_part] = {'metrics': avg_scores, 'errors': avg_errors}
        
        # Calculate overall averages
        all_scores = [score for scores in scores_by_part.values() for score in scores]
        overall_avg_scores = {
            metric: np.mean([s[metric] for s in all_scores])
            for metric in metrics
        }
        overall_avg_errors = {
            error_type: np.mean([s['error_counts'].get(error_type, 0) for s in all_scores])
            for error_type in error_types
        }
        
        # Save results
        results_filename = 'amos_evaluation_results_100images.txt'
        with open(results_filename, 'w') as f:
            f.write(f"EVALUATION RESULTS (LIMITED TO {max_images} IMAGES)\n")
            f.write("=" * 50 + "\n")
            
            # Write results for each body part
            for body_part in ['chest', 'abdomen', 'pelvis']:
                if body_part in avg_scores_by_part:
                    f.write(f"\n{body_part.upper()} Results:\n")
                    f.write(f"Number of cases: {len(scores_by_part[body_part])}\n")
                    f.write("Average scores:\n")
                    for metric, value in avg_scores_by_part[body_part]['metrics'].items():
                        f.write(f"{metric}: {value:.4f}\n")
                    f.write("\nAverage error counts:\n")
                    for error_type, value in avg_scores_by_part[body_part]['errors'].items():
                        f.write(f"{error_type}: {value:.4f}\n")
            
            # Write overall results
            f.write("\nOVERALL Results:\n")
            f.write(f"Total processed cases: {len(all_scores)}\n")
            f.write("Average scores:\n")
            for metric, value in overall_avg_scores.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\nAverage error counts:\n")
            for error_type, value in overall_avg_errors.items():
                f.write(f"{error_type}: {value:.4f}\n")
        
        print(f"\nResults saved to {results_filename}")
    else:
        print("No scores were generated!")

if __name__ == '__main__':
    main()