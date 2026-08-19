"""
Multi-GPU parallel evaluation script for CT-RATE dataset

This script distributes evaluation tasks across multiple GPUs using multiprocessing.
Each worker process is assigned to a specific GPU via CUDA_VISIBLE_DEVICES.
Tasks are pre-allocated to workers in a round-robin fashion for optimal load distribution.

Usage:
    python ct_rate.py [OPTIONS]

Options:
    --model_path PATH       Path to model checkpoint or HuggingFace model ID
                           (default: AlpachinoNLP/u2Qwen3-1.7B-Instruct)
    --data_dir PATH        Directory containing .nii.gz CT scan files
                           (default: /mnt/siyou/dataset/CT-RATE/dataset/valid/)
    --csv_path PATH        CSV file with reference reports
                           (default: ./valid_labels.csv)
    --max_samples N        Maximum number of samples to evaluate (default: all)
    --num_gpus N          Number of GPUs to use (default: all available)
    --log_file PATH    Output file name for results
                           (default: evaluation_results.txt)
    --metrics METRIC [METRIC ...]
                           Metrics to evaluate: bleu, rouge, bert, meteor, green, all
                           (default: all metrics)

Examples:
    # Run on first 10 samples using 2 GPUs
    python ct_rate.py --max_samples 10 --num_gpus 2

    # Run only BLEU and ROUGE (fast metrics)
    python ct_rate.py --metrics bleu rouge --max_samples 10

    # Skip expensive GREEN metric
    python ct_rate.py --metrics bleu rouge bert meteor

    # Run only GREEN metric
    python ct_rate.py --metrics green --max_samples 5

    # Run on custom dataset
    python ct_rate.py --data_dir /path/to/data --csv_path /path/to/labels.csv

    # Use custom model checkpoint
    python ct_rate.py --model_path /path/to/checkpoint --max_samples 5
"""

import inspect
import os
import sys
import types
from multiprocessing import Manager, Process, set_start_method
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

default_csv_file = os.path.dirname(__file__) + "/valid_labels.csv"

# =============================================================================
# Constants
# =============================================================================

TARGET_IMAGE_SIZE = 256
PADDING_SIZE = 256  # 32 * 8
# -Thinking checkpoints spend a large share of the budget on the <think> block
# before the report, so leave generous headroom.
MAX_NEW_TOKENS = 2048
DEFAULT_PROMPT = "Can you provide a caption consists of findings and expressions for this medical image?"

METRIC_CHOICES = ["bleu", "rouge", "bert", "meteor", "green", "all"]

GREEN_ERROR_TYPES = [
    "false_reports",
    "missing_findings",
    "wrong_location",
    "wrong_severity",
    "extra_comparison",
    "missing_comparison",
    "matched_findings",
]


# =============================================================================
# Image Transform
# =============================================================================


class u2Transform:
    """3D CT scan preprocessing: intensity normalization, foreground cropping, and adaptive resizing."""

    def __init__(self, mode: str = "bilinear", device: str = "cuda:0"):
        import nibabel as nib
        import torch
        import torch.nn.functional as F
        from monai.transforms import (
            Compose,
            CropForeground,
            ScaleIntensityRangePercentiles,
            ToTensor,
        )
        from monai.transforms.spatial.functional import resize

        self.torch = torch
        self.F = F
        self.nib = nib
        self.resize = resize

        transforms = Compose(
            [
                ScaleIntensityRangePercentiles(
                    lower=0.5, upper=99.5, b_max=1.0, b_min=0.0, clip=True
                ),
                CropForeground(source_key="image"),
                ToTensor(),
            ]
        )
        self.adaptive_transforms = transforms
        self.mode = mode
        self.device = device

    def adaptive_resize(
        self,
        input_path: str,
        target_image_size: int = TARGET_IMAGE_SIZE,
        padding_size: int = PADDING_SIZE,
    ):
        data = self.nib.load(input_path).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]
        data = self.torch.tensor(data, device=self.device)
        data = self.adaptive_transforms(data)[0]
        data = self.torch.permute(data, (1, 2, 0))

        input_shape = data.shape
        ratio = min([target_image_size / input_shape[i] for i in range(2)])
        scaling_shape = [int(input_shape[i] * ratio) for i in range(2)]

        if padding_size >= input_shape[2]:
            scaling_shape.append(input_shape[2])
            data = self.resize(
                img=data.unsqueeze(0),
                out_size=scaling_shape,
                mode=self.mode,
                align_corners=True,
                dtype=None,
                input_ndim=3,
                anti_aliasing=True,
                anti_aliasing_sigma=None,
                lazy=False,
                transform_info=None,
            )
            pad_tuple = (
                0,
                padding_size - scaling_shape[2],
                0,
                target_image_size - scaling_shape[1],
                0,
                target_image_size - scaling_shape[0],
            )
            data = self.F.pad(data, pad_tuple, mode="constant", value=0)
        else:
            scaling_shape.append(padding_size)
            data = self.resize(
                img=data.unsqueeze(0),
                out_size=scaling_shape,
                mode=self.mode,
                align_corners=True,
                dtype=None,
                input_ndim=3,
                anti_aliasing=True,
                anti_aliasing_sigma=None,
                lazy=False,
                transform_info=None,
            )
            pad_tuple = (
                0,
                0,
                0,
                target_image_size - scaling_shape[1],
                0,
                target_image_size - scaling_shape[0],
            )
            data = self.F.pad(data, pad_tuple, mode="constant", value=0)

        data = self.torch.permute(data, (0, 3, 1, 2))
        data = data.view(-1, 32, target_image_size, target_image_size)
        return data

    def __call__(self, *args, **kwds):
        return self.adaptive_resize(*args, **kwds)


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_name_or_path: str, gpu_id: int = 0):
    """Load model and tokenizer with cache_position compatibility fix."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, trust_remote_code=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    model.eval()

    # Add cache_position compatibility for older models
    if "cache_position" not in inspect.signature(model.forward).parameters:
        original_forward = model.forward

        def _forward_compat(self, *f_args, **f_kwargs):
            f_kwargs.pop("cache_position", None)
            return original_forward(*f_args, **f_kwargs)

        model.forward = types.MethodType(_forward_compat, model)

    return model, tokenizer, device


# =============================================================================
# Caption Generation
# =============================================================================


def generate_caption(
    model, tokenizer, image_path: str, image_transforms: u2Transform, device: str
):
    """Generate radiology report caption for a CT scan."""
    import torch

    try:
        dtype = next(model.parameters()).dtype

        image = image_transforms(image_path)
        image_pt = image.unsqueeze(0).to(device=device, dtype=dtype)

        proj_out_num = getattr(
            getattr(model.get_model(), "mm_projector", None), "proj_out_num", 256
        )
        image_tokens = "<im_patch>" * int(proj_out_num)
        prompt = image_tokens + DEFAULT_PROMPT

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        attention_mask = (
            attention_mask.to(device) if attention_mask is not None else None
        )
        question_ids = tokenizer(
            DEFAULT_PROMPT, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)

        # Decoding parameters are inherited from the checkpoint's
        # generation_config.json (the released checkpoints ship do_sample=True,
        # temperature=0.7, top_p=0.8, top_k=20), so scores vary slightly
        # between runs.
        with torch.no_grad():
            output_ids = model.generate(
                images=image_pt,
                inputs=input_ids,
                question_ids=question_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                # Force stop sign
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                # Penalize repetition
                repetition_penalty=1.1,
            )

        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # -Thinking checkpoints emit "<think>\n{reasoning}\n</think>\n\n{report}";
        # only the report section must be scored. Split on the LAST closing tag
        # (matching the reference parser in the README quickstart, which rindexes
        # the </think> token id) so a stray quoted tag inside the reasoning does
        # not leak reasoning text into the scored report. If the think block
        # never closed, the report was never generated - treat it as a failure.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        elif "<think>" in text:
            print(f"Unclosed <think> block for {image_path}, no report generated")
            return None
        return text
    except Exception as e:
        print(f"Error generating caption for {image_path}: {str(e)}")
        return None


# =============================================================================
# Metrics Evaluation
# =============================================================================


def preprocess_text(text: str) -> str:
    """Normalize text for metric computation."""
    return " ".join(text.lower().split())


def compute_bleu(reference: str, hypothesis: str) -> dict:
    """Compute BLEU-1 score."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smoother = SmoothingFunction().method3
    weights = (1, 0, 0, 0)
    references_tokenized = reference.split()
    hypothesis_tokenized = hypothesis.split()
    # :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    # :type references: list(list(list(str)))
    # :param hypotheses: a list of hypothesis sentences
    # :type hypotheses: list(list(str))
    bleu_score_val = (
        corpus_bleu(
            [[references_tokenized]],
            [hypothesis_tokenized],
            weights=weights,
            smoothing_function=smoother,
        )
        * 100
    )
    return {"bleu": round(bleu_score_val, 2)}


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1/2/L scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure * 100, 2),
        "rouge2": round(scores["rouge2"].fmeasure * 100, 2),
        "rougeL": round(scores["rougeL"].fmeasure * 100, 2),
    }


def compute_bertscore(reference: str, hypothesis: str) -> dict:
    """Compute BERTScore."""
    from bert_score import score as bert_score

    # This line will warn: Some weights of RobertaModel were not initialized ...
    # That is expected. See https://github.com/Tiiiger/bert_score/issues/201
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return {"bert_score": round(F1.mean().item(), 4)}


def compute_meteor(reference: str, hypothesis: str) -> dict:
    """Compute METEOR score."""
    from nltk.translate.meteor_score import meteor_score

    meteor_score_value = meteor_score([reference.split()], hypothesis.split()) * 100
    return {"meteor": round(meteor_score_value, 2)}


def compute_green(green_scorer, reference: str, hypothesis: str) -> dict:
    """Compute GREEN metric with detailed analysis."""
    mean, std, green_score_list, summary, result_df = green_scorer(
        [reference], [hypothesis]
    )

    results = {
        "green_mean": mean,
        "green_std": std,
        "green_summary": summary,
        "green_score_list": green_score_list,
    }

    if not result_df.empty:
        results["green_analysis"] = result_df.iloc[0]["green_analysis"]
        results["error_counts"] = {
            "false_reports": result_df.iloc[0][
                "(a) False report of a finding in the candidate"
            ],
            "missing_findings": result_df.iloc[0][
                "(b) Missing a finding present in the reference"
            ],
            "wrong_location": result_df.iloc[0][
                "(c) Misidentification of a finding's anatomic location/position"
            ],
            "wrong_severity": result_df.iloc[0][
                "(d) Misassessment of the severity of a finding"
            ],
            "extra_comparison": result_df.iloc[0][
                "(e) Mentioning a comparison that isn't in the reference"
            ],
            "missing_comparison": result_df.iloc[0][
                "(f) Omitting a comparison detailing a change from a prior study"
            ],
            "matched_findings": result_df.iloc[0]["Matched Findings"],
        }
    else:
        results["green_analysis"] = ""
        results["error_counts"] = {}

    return results


def evaluate_captions(
    reference: str, hypothesis: str, green_scorer, enabled_metrics: list
) -> dict:
    """Evaluate generated caption against reference using specified metrics."""
    try:
        reference_processed = preprocess_text(reference)
        hypothesis_processed = preprocess_text(hypothesis)

        results = {}

        metric_handlers = {
            "bleu": lambda: compute_bleu(reference_processed, hypothesis_processed),
            "rouge": lambda: compute_rouge(reference_processed, hypothesis_processed),
            "bert": lambda: compute_bertscore(
                reference_processed, hypothesis_processed
            ),
            "meteor": lambda: compute_meteor(reference_processed, hypothesis_processed),
            "green": lambda: compute_green(green_scorer, reference, hypothesis),
        }

        for metric in enabled_metrics:
            if metric in metric_handlers:
                results.update(metric_handlers[metric]())

        return results
    except Exception as e:
        print(f"Error evaluating: {str(e)}")
        return None


# =============================================================================
# File Collection
# =============================================================================


def collect_all_files(base_directory: str, reference_df: pd.DataFrame) -> list:
    """Collect all .nii.gz files that have corresponding reference reports."""
    from pathlib import Path

    # Create set for O(1) lookup instead of O(n*m) DataFrame filtering
    valid_filenames = set(reference_df["VolumeName"].tolist())

    tasks = [
        (str(nii_path), nii_path.name)
        for nii_path in Path(base_directory).rglob("*.nii.gz")
        if nii_path.name in valid_filenames
    ]
    return tasks


# =============================================================================
# Worker Process
# =============================================================================


def worker_process(
    gpu_id: int,
    task_list: list,
    result_list,
    model_path: str,
    reference_df_dict: dict,
    metrics: list,
    progress_counter,
):
    """
    Worker process that handles evaluation on a specific GPU.

    IMPORTANT: All CUDA-related imports happen AFTER setting CUDA_VISIBLE_DEVICES
    """
    # Set GPU assignment BEFORE any CUDA imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(
        f"[GPU {gpu_id}] Starting worker with {len(task_list)} tasks, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )

    # NOW import CUDA-related libraries
    import nltk
    import torch

    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception as e:
        print(f"[GPU {gpu_id}] Warning: Failed to download wordnet: {e}")

    # Verify GPU assignment
    print(f"[GPU {gpu_id}] torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"[GPU {gpu_id}] torch.cuda.current_device() = {torch.cuda.current_device()}")
    if torch.cuda.is_available():
        print(f"[GPU {gpu_id}] GPU Name: {torch.cuda.get_device_name(0)}")

    # Convert reference_df from dict back to DataFrame
    reference_df = pd.DataFrame(reference_df_dict)

    # Track failed samples
    failed_samples = []

    try:
        # Load model
        print(f"[GPU {gpu_id}] Loading model...")
        model, tokenizer, device = load_model(
            model_path, gpu_id=0
        )  # GPU 0 since CUDA_VISIBLE_DEVICES is set
        image_transforms = u2Transform(mode="bilinear", device="cuda:0")

        # Initialize GREEN scorer only if needed
        green_scorer = None
        if "green" in metrics:
            print(f"[GPU {gpu_id}] Initializing GREEN scorer...")
            from green_score import GREEN, GREENLLM, OpenAILLM

            # green_llm = GREENLLM("StanfordAIMI/GREEN-radllama2-7b", device="cuda:0")
            green_llm = OpenAILLM()
            green_scorer = GREEN(green_llm)
        else:
            print(f"[GPU {gpu_id}] Skipping GREEN scorer (not in metrics)")

        print(f"[GPU {gpu_id}] Ready to process tasks (metrics: {metrics})")

        processed_count = 0
        for nii_path, file_name in task_list:
            try:
                reference_row = reference_df[reference_df["VolumeName"] == file_name]
                if reference_row.empty:
                    print(f"[GPU {gpu_id}] No reference for {file_name}")
                    failed_samples.append({"file": file_name, "reason": "no_reference"})
                    continue

                reference_text = reference_row["Findings_EN"].iloc[0]

                generated_text = generate_caption(
                    model, tokenizer, nii_path, image_transforms, device
                )
                if not generated_text:
                    failed_samples.append(
                        {"file": file_name, "reason": "generation_failed"}
                    )
                    continue

                scores = evaluate_captions(
                    reference_text, generated_text, green_scorer, metrics
                )
                if scores:
                    scores["file"] = file_name
                    scores["generated_text"] = generated_text
                    scores["reference_text"] = reference_text
                    result_list.append(scores)
                    processed_count += 1

                    # Update progress counter (increment atomically)
                    progress_counter.value += 1

                    print(
                        f"[GPU {gpu_id}] Processed {file_name} ({processed_count}/{len(task_list)})"
                    )
                else:
                    failed_samples.append(
                        {"file": file_name, "reason": "evaluation_failed"}
                    )

            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {file_name}: {str(e)}")
                failed_samples.append({"file": file_name, "reason": f"error: {str(e)}"})
                continue

        print(
            f"[GPU {gpu_id}] Completed all tasks, processed {processed_count}/{len(task_list)} files"
        )

        # Report failed samples at the end
        if failed_samples:
            print(f"[GPU {gpu_id}] Failed samples: {len(failed_samples)}")
            result_list.append({"worker_id": gpu_id, "failed_samples": failed_samples})

    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()


# =============================================================================
# Results Processing
# =============================================================================


def compute_average_scores(all_scores: list, enabled_metrics: list) -> tuple:
    """Compute average scores and error counts from all results."""
    # Collect all available metrics from results
    available_metrics = set()
    for s in all_scores:
        available_metrics.update(s.keys())

    # Remove non-metric keys
    non_metric_keys = {
        "file",
        "generated_text",
        "reference_text",
        "green_analysis",
        "green_summary",
        "green_score_list",
        "error_counts",
    }
    available_metrics -= non_metric_keys

    # Calculate average for numeric metrics
    avg_scores = {}
    for metric in available_metrics:
        try:
            values = [s[metric] for s in all_scores if metric in s]
            if values:
                avg_scores[metric] = np.mean(values)
        except (TypeError, KeyError):
            pass

    # Calculate average errors if GREEN was used
    avg_errors = {}
    if "green" in enabled_metrics and any("error_counts" in s for s in all_scores):
        avg_errors = {
            error_type: np.mean(
                [
                    s["error_counts"].get(error_type, 0)
                    for s in all_scores
                    if "error_counts" in s
                ]
            )
            for error_type in GREEN_ERROR_TYPES
        }

    return avg_scores, avg_errors


def save_results(
    all_scores: list,
    avg_scores: dict,
    avg_errors: dict,
    num_gpus: int,
    enabled_metrics: list,
    log_file: str,
    model_name_or_path: str,
    base_directory: str,
    csv_path: str,
    max_samples: Optional[int] = None,
):
    """Save evaluation results to file."""
    # Metric format mapping for cleaner code
    METRIC_FORMATS = {
        "bleu": ("BLEU", lambda s: f"{s['bleu']}"),
        "rouge1": ("ROUGE-1", lambda s: f"{s['rouge1']}"),
        "rouge2": ("ROUGE-2", lambda s: f"{s['rouge2']}"),
        "rougeL": ("ROUGE-L", lambda s: f"{s['rougeL']}"),
        "bert_score": ("BERTScore", lambda s: f"{s['bert_score']}"),
        "meteor": ("METEOR", lambda s: f"{s['meteor']}"),
        "green_mean": (
            "GREEN",
            lambda s: f"{s['green_mean']:.4f}±{s.get('green_std', 0):.4f}",
        ),
    }

    with open(log_file, "w") as f:
        f.write(f"Parallel Evaluation Results (Used {num_gpus} GPUs)\n")
        f.write(f"Evaluated Metrics: {', '.join(enabled_metrics)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Individual scores:\n")
        for score in all_scores:
            # Skip metadata entries (failed_samples records)
            if "worker_id" in score:
                continue

            f.write(f"File: {score['file']}\n")
            f.write(f"Generated: {score.get('generated_text', 'N/A')}\n")
            f.write(f"Reference: {score.get('reference_text', 'N/A')}\n")

            # Build score string using mapping
            score_parts = [
                f"{name}={fmt(score)}"
                for metric, (name, fmt) in METRIC_FORMATS.items()
                if metric in score
            ]

            if score_parts:
                f.write(f"Scores: {', '.join(score_parts)}\n")

            if "green_analysis" in score and score["green_analysis"]:
                f.write(f"GREEN Analysis:\n{score['green_analysis']}\n")
            if "error_counts" in score:
                f.write("Error Counts:\n")
                for error_type, count in score["error_counts"].items():
                    f.write(f"  {error_type}: {count}\n")
            f.write("-" * 80 + "\n")

        evaluated_count = sum(1 for score in all_scores if "file" in score)
        failed_count = sum(
            len(score["failed_samples"])
            for score in all_scores
            if "worker_id" in score and "failed_samples" in score
        )

        f.write("\n" + "=" * 80 + "\n")
        f.write("Average Scores:\n")
        f.write(
            f"(averaged over {evaluated_count} evaluated samples; "
            f"{failed_count} failed samples excluded, see summary below)\n"
        )
        f.write("=" * 80 + "\n")
        for metric, value in sorted(avg_scores.items()):
            f.write(f"{metric}: {value:.4f}\n")

        if avg_errors:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Average Error Counts:\n")
            f.write("=" * 80 + "\n")
            for error_type, value in avg_errors.items():
                f.write(f"{error_type}: {value:.4f}\n")

        # Add failed samples summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("Failed Samples Summary:\n")
        f.write("=" * 80 + "\n")

        all_failed = []
        for score in all_scores:
            if "worker_id" in score and "failed_samples" in score:
                all_failed.extend(score["failed_samples"])

        if all_failed:
            # Group by reason
            from collections import defaultdict

            by_reason = defaultdict(list)
            for failed in all_failed:
                by_reason[failed["reason"]].append(failed["file"])

            for reason, files in sorted(by_reason.items()):
                f.write(f"\n{reason} ({len(files)} files):\n")
                for file in files[:10]:  # Show first 10
                    f.write(f"  - {file}\n")
                if len(files) > 10:
                    f.write(f"  ... and {len(files) - 10} more\n")
        else:
            f.write("No failed samples\n")


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse
    import time

    import torch

    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel evaluation for CT-RATE dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="AlpachinoNLP/u2Qwen3-1.7B-Instruct",
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/siyou/dataset/CT-RATE/dataset/valid/",
        help="Directory containing .nii.gz CT scan files",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=default_csv_file,
        help="CSV file with reference reports (VolumeName and Findings_EN columns)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="evaluation_results.txt",
        help="Output file name for results",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["bleu", "rouge", "bert", "meteor", "green"],
        choices=METRIC_CHOICES,
        help='Metrics to evaluate. Use "all" or specify individual metrics (default: all)',
    )

    args = parser.parse_args()

    # Process metrics argument
    if "all" in args.metrics:
        args.metrics = ["bleu", "rouge", "bert", "meteor", "green"]

    # Remove duplicates while preserving order
    args.metrics = list(dict.fromkeys(args.metrics))

    # Configuration
    model_name_or_path = args.model_path
    base_directory = args.data_dir
    csv_path = args.csv_path

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")

    if num_gpus == 0:
        print("Error: No GPU available!")
        return

    # Limit GPUs if specified
    if args.num_gpus is not None:
        num_gpus = min(num_gpus, args.num_gpus)
    print(f"Will use {num_gpus} GPUs for parallel evaluation")

    # Load reference data
    print("Loading reference data...")
    reference_df = pd.read_csv(csv_path)
    # Only keep necessary columns to reduce memory overhead
    reference_df_dict = reference_df[["VolumeName", "Findings_EN"]].to_dict("list")

    # Collect tasks
    print("Collecting files...")
    all_tasks = collect_all_files(base_directory, reference_df)
    print(f"Found {len(all_tasks)} files to process")

    # Limit samples if specified
    if args.max_samples is not None and args.max_samples > 0:
        all_tasks = all_tasks[: args.max_samples]
        print(f"Limited to {len(all_tasks)} samples for evaluation")

    if len(all_tasks) == 0:
        print("No files to process!")
        return

    # Pre-assign tasks to each GPU
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for idx, task in enumerate(all_tasks):
        gpu_id = idx % num_gpus
        tasks_per_gpu[gpu_id].append(task)

    # Print task distribution
    print("\nTask distribution:")
    for gpu_id, tasks in enumerate(tasks_per_gpu):
        print(f"  GPU {gpu_id}: {len(tasks)} tasks")

    # Create shared structures
    manager = Manager()
    result_list = manager.list()
    progress_counter = manager.Value("i", 0)  # Shared progress counter

    # Start workers
    print(f"\nStarting {num_gpus} worker processes...")
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_process,
            args=(
                gpu_id,
                tasks_per_gpu[gpu_id],
                result_list,
                model_name_or_path,
                reference_df_dict,
                args.metrics,
                progress_counter,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for completion with progress bar
    print("\nWaiting for workers...")
    with tqdm(total=len(all_tasks), desc="Evaluation Progress", unit="sample") as pbar:
        last_count = 0
        while any(p.is_alive() for p in processes):
            current_count = progress_counter.value
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            import time

            time.sleep(0.5)

        # Final update
        pbar.update(progress_counter.value - last_count)

    for i, p in enumerate(processes):
        p.join()
        print(f"Worker {i} finished")

    # Process results
    all_scores = list(result_list)

    print(f"\n{'=' * 80}")
    print("Evaluation Complete!")
    print(f"{'=' * 80}")
    print("Configuration:")
    print(f"  Model: {model_name_or_path}")
    print(f"  Data Directory: {base_directory}")
    print(f"  CSV Path: {csv_path}")
    print(f"  GPUs Used: {num_gpus}")
    if args.max_samples:
        print(f"  Max Samples: {args.max_samples}")
    print(f"  Metrics: {', '.join(args.metrics)}")
    print(f"{'=' * 80}")

    # Count successful and failed samples
    successful_count = sum(1 for s in all_scores if "worker_id" not in s)
    failed_count = sum(
        len(s["failed_samples"])
        for s in all_scores
        if "worker_id" in s and "failed_samples" in s
    )

    print(f"\nTotal processed: {successful_count} cases")
    if failed_count > 0:
        print(f"Failed samples: {failed_count} cases")

        # Show failure breakdown
        from collections import defaultdict

        by_reason = defaultdict(int)
        for score in all_scores:
            if "worker_id" in score and "failed_samples" in score:
                for failed in score["failed_samples"]:
                    by_reason[failed["reason"]] += 1

        print("Failure breakdown:")
        for reason, count in sorted(by_reason.items()):
            print(f"  - {reason}: {count}")

    if all_scores and successful_count > 0:
        # Filter out failed sample records for averaging
        score_data = [s for s in all_scores if "worker_id" not in s]

        avg_scores, avg_errors = compute_average_scores(score_data, args.metrics)

        save_results(
            all_scores,
            avg_scores,
            avg_errors,
            num_gpus,
            args.metrics,
            args.log_file,
            model_name_or_path,
            base_directory,
            csv_path,
            args.max_samples,
        )

        print(f"\nResults saved to {args.log_file}")
        print("\nAverage Scores:")
        for metric, value in avg_scores.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("No scores generated!")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
