# -*- encoding: utf-8 -*-
# @File        :   data_synthesis.py
# @Description :   Unified data synthesis preprocessing for CT-RATE / AMOS-MM / AbdomenAtlas.
#                  Merges qwen3_data_synthesis, thinking_refine_and_vqa_filter, and
#                  thinking_synthesis_preprocess into a single module with a unified CLI.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from collections import defaultdict
import json
import logging
import random
import re

import pandas as pd
from tqdm import tqdm

from src.preprocess.llm_utils import (
    async_translate_batch,
    batch_query,
    query_with_retry,
    synthesize_data_batch,
)
from src.utils.prompt_templates import general_questions, general_questions_chinese

logger = logging.getLogger(__name__)

# =============================================================================
# Prompt templates (loaded from src/preprocess/prompts/*.prompt)
# =============================================================================

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, f"{name}.prompt"), "r") as f:
        return f.read().strip()


QUESTION_PROMPT_TPL = _load_prompt("question")
THINKING_PROMPT_TPL = _load_prompt("thinking")
FILTER_TEMPLATE = _load_prompt("filter")
EDIT_TEMPLATE = _load_prompt("edit")
REPORT_THINKING_TEMPLATE = _load_prompt("report_thinking")

question_pattern = r".*?\d\. ?([^\n]*)"
thinking_pattern = r"Thinking:\s*(.*?)(?=\nAnswer:|\Z)"
answer_pattern = r"Answer:\s*(.*)"


# =============================================================================
# JSONL I/O helpers
# =============================================================================

def _read_jsonl_with_resume(jsonl_file_path, output_file_path, resume):
    """Read JSONL input, optionally skipping already-processed lines for resume.

    Returns (records, skip_count) or (None, 0) if nothing to process.
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_file_path, "r") as f:
        lines = f.readlines()

    skip_count = 0
    if resume and os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            skip_count = sum(1 for _ in f)
        logger.info("Resuming: skipping %d already-processed lines", skip_count)

    lines = lines[skip_count:]
    if not lines:
        logger.info("Nothing to translate (all lines already processed)")
        return None, 0

    logger.info("Translating %d lines from %s", len(lines), jsonl_file_path)
    records = [json.loads(line) for line in lines]
    return records, skip_count


def _load_processed_images(output_file):
    """Load set of already-processed image paths from an output JSONL file."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed.add(json.loads(line)["image"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return processed


def _write_jsonl(records, output_file_path, resume, skip_count):
    """Write records to a JSONL file, appending if resuming."""
    mode = "a" if resume and skip_count > 0 else "w"
    with open(output_file_path, mode) as f:
        for data in records:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logger.info("Translation complete: %s", output_file_path)


# =============================================================================
# VQA thinking core
# =============================================================================

class VQAThinking:
    def __init__(self, finding, image):
        self.finding = finding
        self.image = image
        self.questions = None
        self.thinkings = None
        self.answers = None

    @property
    def question_prompt(self):
        return QUESTION_PROMPT_TPL.format(report=self.finding)

    @property
    def thinking_prompts(self):
        return [
            THINKING_PROMPT_TPL.format(report=self.finding, question=q)
            for q in self.questions
        ]

    def parse_questions_from_output(self, output):
        self.questions = re.findall(question_pattern, output, flags=re.M)

    def parse_thinkings_and_answers_from_outputs(self, outputs):
        thinkings = []
        answers = []
        for output in outputs:
            t_matches = re.findall(thinking_pattern, output, flags=re.M | re.S)
            if t_matches:
                thinkings.append(t_matches[-1].strip())
            else:
                logger.warning(
                    "Failed to parse thinking from output: %s...", output[:100]
                )
                thinkings.append("")

            a_matches = re.findall(answer_pattern, output, flags=re.M | re.S)
            if a_matches:
                answers.append(a_matches[-1].strip())
            else:
                logger.warning(
                    "Failed to parse answer from output: %s...", output[:100]
                )
                answers.append("")

        self.thinkings = thinkings
        self.answers = answers

    def format_results(self):
        results = []
        for question, thinking, answer in zip(
            self.questions, self.thinkings, self.answers
        ):
            if len(question) > 20 and len(thinking) > 20 and len(answer) > 20:
                results.append(
                    {
                        "question": question,
                        "thinking": thinking,
                        "answer": answer,
                        "report": self.finding,
                        "image": self.image,
                    }
                )
        return results


# =============================================================================
# Per-report pipeline: synthesis -> filter -> refine
# =============================================================================

async def _process_single_report_async(finding, image, max_concurrent=5, retry_times=3, retry_delay=1.0):
    """Process one report through the full pipeline: synthesis -> filter -> refine."""
    vqa = VQAThinking(finding, image)

    # Step 1: Generate questions
    q_results = await synthesize_data_batch(
        [vqa.question_prompt], enable_thinking=False,
    )
    vqa.parse_questions_from_output(q_results[0][1])

    if not vqa.questions:
        logger.warning("No questions generated for report: %s...", finding[:80])
        return []

    # Step 2: Generate thinking + answers
    t_results = await synthesize_data_batch(
        vqa.thinking_prompts, enable_thinking=False,
    )
    vqa.parse_thinkings_and_answers_from_outputs([r[1] for r in t_results])
    qa_results = vqa.format_results()

    if not qa_results:
        return []

    # Step 3: Filter improper QA pairs
    filter_prompts = [
        FILTER_TEMPLATE.format(
            question=r["question"], answer=r["answer"], report=r["report"],
        )
        for r in qa_results
    ]
    filter_responses = await batch_query(
        filter_prompts,
        enable_thinking=False,
        max_concurrent=max_concurrent,
        batch_size=len(filter_prompts),
        retry_times=retry_times,
        retry_delay=retry_delay,
    )
    filtered = [
        r for r, (resp, _) in zip(qa_results, filter_responses)
        if resp and resp.startswith("Yes")
    ]

    if not filtered:
        return []

    logger.debug("Filtered: %d/%d QA pairs kept", len(filtered), len(qa_results))

    # Step 4: Refine thinking (remove report references)
    refine_prompts = [EDIT_TEMPLATE.format(report=r["thinking"]) for r in filtered]
    refine_responses = await batch_query(
        refine_prompts,
        enable_thinking=False,
        max_concurrent=max_concurrent,
        batch_size=len(refine_prompts),
        retry_times=retry_times,
        retry_delay=retry_delay,
    )
    for r, (refined, _) in zip(filtered, refine_responses):
        r["refined_thinking"] = refined.strip('`\n') if refined else r["thinking"]

    return filtered


async def _process_reports_async(findings, images, max_concurrent_reports=1, **kwargs):
    """Process a batch of reports through the full pipeline with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent_reports)

    async def _process_one(finding, image):
        async with semaphore:
            return await _process_single_report_async(finding, image, **kwargs)

    all_results = await asyncio.gather(
        *[_process_one(f, img) for f, img in zip(findings, images)]
    )
    return [item for sublist in all_results for item in sublist]


# =============================================================================
# Report thinking generation
# =============================================================================

async def generate_report_thinking(df, jsonl_file, *, output_file=None, report_lookup=None, enable_batch=False, batch_size=10, max_concurrent=5, retry_times=3, retry_delay=1.0, resume=True):
    logger.info("Starting report thinking generation...")
    report_file = output_file or f"{jsonl_file.replace('.jsonl','')}_report_thinking.jsonl"

    image_groups = df.groupby('image', sort=False)
    num_questions = len(general_questions)

    # Resume: skip already-processed images
    processed_images = set()
    if resume:
        processed_images = _load_processed_images(report_file)
        if processed_images:
            logger.info("Resuming: %d images already processed, will skip", len(processed_images))
            remaining = len(image_groups) - len(processed_images & set(image_groups.groups.keys()))
            if remaining <= 0:
                logger.info("Nothing to process (all images already processed)")
                return
            logger.info("Remaining: %d images to process", remaining)

    if enable_batch:
        logger.info(f"Using batch mode (batch_size: {batch_size}, max_concurrent: {max_concurrent})")

        contents = []
        image_names = []
        thinking_befores = []

        for current_image, group in image_groups:
            if current_image in processed_images:
                continue
            thinking_before = ""
            for _, row in group.iterrows():
                thinking_before += " " + row["question"] + row["thinking"] + row["answer"]
            thinking_before = thinking_before.strip()

            contents.append(REPORT_THINKING_TEMPLATE.format(thinking_before=thinking_before))
            image_names.append(current_image)
            thinking_befores.append(thinking_before)

        results = await batch_query(
            contents,
            enable_thinking=False,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
            retry_times=retry_times,
            retry_delay=retry_delay,
        )

        for i, (thinking_after, _) in enumerate(results):
            if thinking_after is None:
                logger.warning(f"Image {image_names[i]} query failed, skipping")
                continue
            question_index = random.randint(0, num_questions - 1)
            thinking_after = re.sub(r"^(?:\*+)?Thinking Progress:(?:\*+)?", "", thinking_after, flags=re.IGNORECASE).strip('`\n')
            result_line = json.dumps({
                "image": image_names[i],
                "report": report_lookup[image_names[i]] if report_lookup and image_names[i] in report_lookup else df[df['image'] == image_names[i]]['report'].iloc[0],
                "question": general_questions[question_index],
                "thinking_before": thinking_befores[i],
                "thinking_after": thinking_after,
            }, ensure_ascii=False)
            with open(report_file, "a") as f:
                f.write(result_line + "\n")
                f.flush()
    else:
        logger.info("Using sequential mode")
        current_image = ""
        pbar = tqdm(total=len(image_groups))
        for index, row in df.iterrows():
            if row["image"] != current_image:
                current_image = row["image"]
                if current_image in processed_images:
                    pbar.update(1)
                    continue
                logger.info(f"Processing image: {current_image}")
                logger.info(f"Total number of questions: {len(df[df['image'] == current_image])}")
                thinking_before = ""
                question_index = random.randint(0, num_questions - 1)
                subset = df.iloc[index:index+20]
                for _, inner_row in subset[subset["image"] == current_image].iterrows():
                    thinking_before += " " + inner_row["question"] + inner_row["thinking"] + inner_row["answer"]
                thinking_before = thinking_before.strip()
                content = REPORT_THINKING_TEMPLATE.format(thinking_before=thinking_before)
                thinking_after, _ = await query_with_retry(content, enable_thinking=False, max_retries=retry_times, delay=retry_delay)
                if thinking_after is None:
                    logger.warning(f"Image {current_image} query failed, skipping")
                    pbar.update(1)
                    continue
                thinking_after = re.sub(r"^(?:\*+)?Thinking Progress:(?:\*+)?", "", thinking_after, flags=re.IGNORECASE).strip('`\n')
                result_line = json.dumps({
                    "image": current_image,
                    "report": report_lookup[current_image] if report_lookup and current_image in report_lookup else row["report"],
                    "question": general_questions[question_index],
                    "thinking_before": thinking_before,
                    "thinking_after": thinking_after,
                }, ensure_ascii=False)
                with open(report_file, "a") as f:
                    f.write(result_line + "\n")
                    f.flush()
                pbar.update(1)
            else:
                continue
        pbar.close()

    logger.info(f"Report generation complete, saved to: {report_file}")


# =============================================================================
# Translation pipelines
# =============================================================================

def vqa_thinking_translation_synthesis(
    jsonl_file_path,
    output_file_path,
    source_lang="English",
    target_lang="Chinese",
    enable_thinking=False,
    resume=False,
    max_concurrent=1,
):
    """
    Synthesize VQA thinking data with translation.
    jsonl_file_path: str, path to the input JSONL file
    output_file_path: str, path to the output JSONL file
    source_lang: str, language of the input text
    target_lang: str, language for the translation
    enable_thinking: bool, whether to enable thinking in the synthesis
    resume: bool, if True, skip already-processed images and append to output
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_file_path, "r") as f:
        records = [json.loads(line) for line in f]

    # Load already-processed images for resume (always check output file)
    processed_images = _load_processed_images(output_file_path)
    if processed_images:
        logger.info("Resuming: %d images already processed, will skip", len(processed_images))

    records = [r for r in records if r["image"] not in processed_images]
    if not records:
        logger.info("Nothing to translate (all images already processed)")
        return

    logger.info("Translating %d records from %s", len(records), jsonl_file_path)

    # Collect all texts to translate
    all_texts = []
    for data in records:
        all_texts.append(data["question"])
        all_texts.append(data["thinking"])
        all_texts.append(data["answer"])

    # Batch translate
    translated = asyncio.run(
        async_translate_batch(all_texts, target_lang, source_lang, enable_thinking, concurrency=max_concurrent)
    )

    # Reassemble
    for i, data in enumerate(records):
        data["original_question"] = data["question"]
        data["question"] = translated[i * 3]
        data["thinking"] = translated[i * 3 + 1]
        data["answer"] = translated[i * 3 + 2]

    # Group by image and write each group atomically
    grouped = defaultdict(list)
    for data in records:
        grouped[data["image"]].append(data)

    mode = "a" if processed_images else "w"
    with open(output_file_path, mode) as f:
        for image, items in grouped.items():
            for data in items:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

    logger.info("Translation complete: %s", output_file_path)


def report_thinking_translation_synthesis(
    jsonl_file_path,
    output_file_path,
    source_lang="English",
    target_lang="Chinese",
    enable_thinking=False,
    resume=False,
    max_concurrent=1,
):
    """
    Synthesize report thinking data with translation.
    jsonl_file_path: str, path to the input JSONL file
    output_file_path: str, path to the output JSONL file
    source_lang: str, language of the input text
    target_lang: str, language for the translation
    enable_thinking: bool, whether to enable thinking in the synthesis
    resume: bool, if True, skip already-processed lines and append to output
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_file_path, "r") as f:
        records = [json.loads(line) for line in f]

    # Load already-processed images for resume
    processed_images = set()
    if resume:
        processed_images = _load_processed_images(output_file_path)
        if processed_images:
            logger.info("Resuming: %d images already processed, will skip", len(processed_images))

    records = [r for r in records if r["image"] not in processed_images]
    if not records:
        logger.info("Nothing to translate (all images already processed)")
        return

    logger.info("Translating %d records from %s", len(records), jsonl_file_path)

    # Build lookup for pre-translated questions
    question_lookup = {
        q: c for q, c in zip(general_questions, general_questions_chinese)
    }

    # Collect texts that actually need API translation
    # For each record we need: question (maybe), thinking_after, report
    texts_to_translate = []
    # Track which record/field each text corresponds to
    translation_map = []  # list of (record_index, field_name)

    for i, data in enumerate(records):
        question = data["question"]
        if question not in question_lookup:
            # Unknown question: reassign a known one
            data["question"] = random.choice(general_questions)

        texts_to_translate.append(data["thinking_after"])
        translation_map.append((i, "thinking_after"))

        texts_to_translate.append(data["report"])
        translation_map.append((i, "report"))

    # Batch translate all texts
    translated = asyncio.run(
        async_translate_batch(
            texts_to_translate, target_lang, source_lang, enable_thinking, concurrency=max_concurrent
        )
    )

    # Reassemble results
    for idx, (record_i, field_name) in enumerate(translation_map):
        records[record_i][field_name] = translated[idx]

    # Apply question translations from lookup
    for data in records:
        data["question"] = question_lookup[data["question"]]
        data.pop("thinking_before", None)

    # Group by image and write each group atomically
    grouped = defaultdict(list)
    for data in records:
        grouped[data["image"]].append(data)

    mode = "a" if processed_images else "w"
    with open(output_file_path, mode) as f:
        for image, items in grouped.items():
            for data in items:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

    logger.info("Translation complete: %s", output_file_path)


# =============================================================================
# Dataset-specific preprocessing
# =============================================================================

DATASET_NAMES = {
    "ct_rate": "CT-RATE",
    "amos_mm": "AMOS-MM",
    "abdomen_atlas": "AbdomenAtlasData3.0",
}

AMOS_MM_MRG_TYPES = ["chest", "abdomen", "pelvis"]


# -- image path builders ------------------------------------------------------

def _build_ctrate_image_path(volume_name: str, data_type: str = "train") -> str:
    parts = volume_name.split("_")
    prefix = f"{parts[0]}_{parts[1]}"
    mid = f"{prefix}_{parts[2]}"
    return f"CT-RATE/dataset/{data_type}/{prefix}/{mid}/{volume_name}"


def _build_amos_mm_image_path(image_path: str) -> str:
    if image_path.startswith("./"):
        return f"AMOS-MM/{image_path[2:]}"
    return image_path


def _build_abdomen_atlas_image_path(bdmap_id: str) -> str:
    return f"AbdomenAtlas3.0Report/{bdmap_id}/ct.nii.gz"


# -- dataset-specific load & extract ------------------------------------------

def _load_data(args):
    """Load and slice dataset. Returns (data, is_dataframe)."""
    if args.dataset == "ct_rate":
        data = pd.read_csv(args.input_file, low_memory=False)
    elif args.dataset == "amos_mm":
        with open(args.input_file, "r") as f:
            data = json.load(f)[args.data_type]
    elif args.dataset == "abdomen_atlas":
        data = pd.read_csv(args.input_file, low_memory=False)
        split_file = os.path.join(os.path.dirname(args.input_file), f"IID_{args.split}.csv")
        split_ids = pd.read_csv(split_file, low_memory=False)["BDMAP ID"]
        data = data[data["BDMAP ID"].isin(split_ids)]

    end = args.end_line or len(data)
    if args.dataset == "amos_mm":
        data = data[args.start_line:end]
    else:
        data = data.iloc[args.start_line:end]
    return data


def _extract_batch(args, batch):
    """Extract (findings_list, image_paths_list) from a batch."""
    if args.dataset == "ct_rate":
        findings = batch["Findings_EN"].tolist()
        image_paths = [_build_ctrate_image_path(n, args.data_type) for n in batch["VolumeName"].tolist()]
        return findings, image_paths

    if args.dataset == "amos_mm":
        findings_list, image_paths_list = [], []
        for item in batch:
            image_path = _build_amos_mm_image_path(item["image"])
            finding = item["labels"]["report"]["findings"]
            for loc in AMOS_MM_MRG_TYPES:
                if finding[loc] != "":
                    image_paths_list.append(image_path)
                    findings_list.append(finding[loc])
        return findings_list, image_paths_list

    # abdomen_atlas
    findings = batch["structured report"].tolist()
    image_paths = [_build_abdomen_atlas_image_path(n) for n in batch["BDMAP ID"].tolist()]
    return findings, image_paths


def _get_batch(data, i, batch_size, dataset):
    if dataset == "amos_mm":
        return data[i:i + batch_size]
    return data.iloc[i:i + batch_size]


def _run_vqa_pipeline(args):
    """Core VQA pipeline: question generation -> thinking synthesis -> filter -> refine."""
    dataset_name = DATASET_NAMES[args.dataset]
    data = _load_data(args)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    pipeline_kwargs = dict(
        max_concurrent=args.max_concurrent,
        max_concurrent_reports=args.max_concurrent_reports,
        retry_times=args.retry_times,
        retry_delay=args.retry_delay,
    )

    # Load already-processed images for resume
    processed_images = _load_processed_images(args.output_file)
    if processed_images:
        logger.info("Resuming: %d images already processed, will skip", len(processed_images))

    start_idx = args.skip_batches * args.batch_size
    with open(args.output_file, "a") as f:
        for i in tqdm(range(start_idx, len(data), args.batch_size)):
            batch = _get_batch(data, i, args.batch_size, args.dataset)
            findings, image_paths = _extract_batch(args, batch)

            # Filter out already-processed images
            pairs = [(fin, img) for fin, img in zip(findings, image_paths) if img not in processed_images]
            if not pairs:
                continue
            findings, image_paths = zip(*pairs)

            try:
                results = asyncio.run(
                    _process_reports_async(list(findings), list(image_paths), **pipeline_kwargs)
                )
                # Group results by image and write each group atomically
                grouped = defaultdict(list)
                for item in results:
                    grouped[item["image"]].append(item)

                for image, items in grouped.items():
                    for item in items:
                        line = json.dumps({
                            "image": item["image"],
                            "question": item["question"],
                            "thinking": item["refined_thinking"],
                            "answer": item["answer"],
                        }, ensure_ascii=False)
                        f.write(f"{line}\n")
                    f.flush()
                    processed_images.add(image)
            except Exception as e:
                logger.error(f"Error processing batch at index {i}: {e}")
                continue

            if args.test_mode:
                break

    logger.info(f"Done: {args.output_file}")


# =============================================================================
# Task wrappers (argparse -> business logic)
# =============================================================================

def _common_kwargs(args):
    """Extract common keyword arguments from an argparse namespace."""
    return dict(
        enable_batch=args.enable_batch,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        retry_times=args.retry_times,
        retry_delay=args.retry_delay,
    )


def _task_vqa(args):
    """Run VQA data synthesis for a dataset."""
    dataset_name = DATASET_NAMES[args.dataset]
    logger.info(f"[{dataset_name}] {args.input_file} -> {args.output_file}")
    _run_vqa_pipeline(args)


def _task_report_translation(args):
    """Translate report thinking data."""
    logger.info(f"Running report_translation: {args.input_file} -> {args.output_file}")
    report_thinking_translation_synthesis(
        args.input_file,
        args.output_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        resume=args.resume,
        max_concurrent=args.max_concurrent,
    )
    logger.info(f"Successfully translated to {args.output_file}")


def _task_vqa_translation(args):
    """Translate VQA thinking data."""
    logger.info(f"Running vqa_translation: {args.input_file} -> {args.output_file}")
    vqa_thinking_translation_synthesis(
        args.input_file,
        args.output_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_concurrent=args.max_concurrent,
    )
    logger.info(f"Successfully translated to {args.output_file}")


def _task_report(args):
    """Generate report thinking."""
    input_file = args.input_file

    logger.info(f"Running report: {input_file}")
    df = pd.read_json(input_file, lines=True)
    logger.info(f"Input contains {len(df)} records")

    report_lookup = None
    if args.original_input:
        from types import SimpleNamespace
        load_args = SimpleNamespace(
            dataset=args.dataset,
            input_file=args.original_input,
            data_type=args.data_type,
            split=args.split,
            start_line=0,
            end_line=None,
        )
        original_data = _load_data(load_args)
        findings, image_paths = _extract_batch(load_args, original_data)
        report_lookup = dict(zip(image_paths, findings))
        logger.info(f"Loaded {len(report_lookup)} report entries from original input")

    asyncio.run(generate_report_thinking(df, input_file, output_file=args.output_file, report_lookup=report_lookup, resume=args.resume, **_common_kwargs(args)))
    report_file = args.output_file or f"{input_file.replace('.jsonl','')}_report_thinking.jsonl"
    logger.info(f"Report generation complete, saved to {report_file}")


# =============================================================================
# Unified CLI
# =============================================================================

def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified CLI for data synthesis preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_synthesis.py vqa -d ct_rate -i input.csv -o output.jsonl
  python data_synthesis.py vqa_translation -i input.jsonl -o output_chinese.jsonl
  python data_synthesis.py report -i input.jsonl
  python data_synthesis.py report_translation -i input.jsonl -o output_chinese.jsonl
        """,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (DEBUG level, includes third-party libraries)",
    )
    subparsers = parser.add_subparsers(dest="task", required=True, help="Task to run")

    # -- helpers for shared argument groups --

    def add_vqa_args(subparser):
        subparser.add_argument(
            "-d", "--dataset", type=str, default="ct_rate",
            choices=list(DATASET_NAMES), help="Dataset (default: ct_rate)",
        )
        subparser.add_argument(
            "-i", "--input-file", type=str, required=True, help="Input file path",
        )
        subparser.add_argument(
            "-o", "--output-file", type=str, required=True, help="Output JSONL file path",
        )
        subparser.add_argument(
            "--data-type", type=str, default="train",
            help="Data split type (ct_rate: train/valid, amos_mm: training/validation)",
        )
        subparser.add_argument(
            "--split", type=str, default="train",
            help="Split file suffix for abdomen_atlas (default: train)",
        )
        subparser.add_argument(
            "--start-line", type=int, default=0, help="Start line (default: 0)",
        )
        subparser.add_argument(
            "--end-line", type=int, default=None, help="End line (default: all)",
        )
        subparser.add_argument(
            "--batch-size", type=int, default=2, help="Batch size (default: 2)",
        )
        subparser.add_argument(
            "--skip-batches", type=int, default=0,
            help="Number of batches to skip (default: 0)",
        )
        subparser.add_argument(
            "--max-concurrent", type=int, default=5,
            help="Maximum number of concurrent LLM requests (default: 5)",
        )
        subparser.add_argument(
            "--max-concurrent-reports", type=int, default=1,
            help="Maximum number of reports to process concurrently (default: 1)",
        )
        subparser.add_argument(
            "--retry-times", type=int, default=3,
            help="Number of retries for failed requests (default: 3)",
        )
        subparser.add_argument(
            "--retry-delay", type=float, default=1.0,
            help="Delay between retries in seconds (default: 1.0)",
        )
        subparser.add_argument(
            "--test-mode", action="store_true", help="Process only first batch",
        )

    def add_translation_args(subparser):
        subparser.add_argument(
            "-i", "--input-file", type=str, required=True, help="Input JSONL file path",
        )
        subparser.add_argument(
            "-o", "--output-file", type=str, required=True, help="Output JSONL file path",
        )
        subparser.add_argument(
            "--source-lang", type=str, default="English",
            help="Source language (default: English)",
        )
        subparser.add_argument(
            "--target-lang", type=str, default="Chinese",
            help="Target language (default: Chinese)",
        )
        subparser.add_argument(
            "--max-concurrent", type=int, default=1,
            help="Maximum number of concurrent translation requests (default: 1)",
        )

    def add_processing_args(subparser):
        subparser.add_argument(
            "-i", "--input-file", type=str, required=True, help="Input JSONL file path",
        )
        subparser.add_argument(
            "-o", "--output-file", type=str, default=None,
            help="Output JSONL file path (default: auto-generated from input file)",
        )
        subparser.add_argument(
            "--batch-size", type=int, default=10,
            help="Batch size for concurrent LLM calls (default: 10)",
        )
        subparser.add_argument(
            "--max-concurrent", type=int, default=5,
            help="Maximum number of concurrent requests (default: 5)",
        )
        subparser.add_argument(
            "--retry-times", type=int, default=3,
            help="Number of retries for failed requests (default: 3)",
        )
        subparser.add_argument(
            "--retry-delay", type=float, default=1.0,
            help="Delay between retries in seconds (default: 1.0)",
        )
        subparser.add_argument(
            "--enable-batch", action="store_true",
            help="Enable batch processing for better performance",
        )

    # -- VQA subcommands --

    p_vqa = subparsers.add_parser(
        "vqa", help="Run VQA data synthesis for a dataset",
    )
    add_vqa_args(p_vqa)
    p_vqa.set_defaults(func=_task_vqa)

    p_vqa_trans = subparsers.add_parser(
        "vqa_translation", help="Translate VQA thinking data",
    )
    add_translation_args(p_vqa_trans)
    p_vqa_trans.set_defaults(func=_task_vqa_translation)

    # -- report subcommands --

    p_report = subparsers.add_parser(
        "report", help="Generate report thinking",
    )
    add_processing_args(p_report)
    p_report.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from previous run, skipping already-processed images (default: True)",
    )
    p_report.add_argument(
        "--original-input", type=str, default=None,
        help="Original dataset file for report field lookup (uses _load_data to read)",
    )
    p_report.add_argument(
        "-d", "--dataset", type=str, default="ct_rate",
        choices=list(DATASET_NAMES),
        help="Dataset type for original input (default: ct_rate)",
    )
    p_report.add_argument(
        "--data-type", type=str, default="train",
        help="Data split type for original input (default: train)",
    )
    p_report.add_argument(
        "--split", type=str, default="train",
        help="Split file suffix for abdomen_atlas original input (default: train)",
    )
    p_report.set_defaults(func=_task_report)

    p_report_trans = subparsers.add_parser(
        "report_translation", help="Translate report thinking data",
    )
    add_translation_args(p_report_trans)
    p_report_trans.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from previous run, skipping already-processed images (default: True)",
    )
    p_report_trans.set_defaults(func=_task_report_translation)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not args.verbose:
        # Silence noisy third-party loggers
        for name in ("httpx", "openai", "backoff", "httpcore", "__main__"):
            logging.getLogger(name).setLevel(logging.WARNING)

    args.func(args)


if __name__ == "__main__":
    main()
