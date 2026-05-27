"""Build DPO preference pairs (chosen / rejected) from a Stage1 SFT model.

For each row in --src_jsonl (must have 'image' + 'question' + 'answer' fields):
  1. Sample N candidate answers from --model (temperature > 0).
  2. Score every candidate against the gold 'answer' with the chosen metric.
  3. Emit one DPO row with the best candidate as 'chosen' and worst as 'rejected'.

Output schema (one JSON per line):
    {
      "image":    "<same as src>",
      "question": "<same as src>",
      "chosen":   "<best sampled candidate>",
      "rejected": "<worst sampled candidate>",
      "scores":   [...]
    }

Metric options:
  - rougeL  (default; cheap, no external service)
  - bleu
  - green   (requires GREEN scorer — OpenAI API or local vLLM via config/project.json)

Multi-GPU friendly: pass --num_gpus N to shard rows round-robin across workers.
"""
import argparse
import inspect
import json
import os
import sys
import types
from multiprocessing import Manager, Process, set_start_method
from typing import List

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


TARGET_IMAGE_SIZE = 256
PADDING_SIZE = 256
DEFAULT_PROMPT_SUFFIX = ""  # use question verbatim


def make_image_transform(device: str):
    # Local import to avoid spawning the heavy CUDA modules in the parent.
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

    transforms = Compose(
        [
            ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_max=1.0, b_min=0.0, clip=True),
            CropForeground(source_key="image"),
            ToTensor(),
        ]
    )

    def adaptive(path, target=TARGET_IMAGE_SIZE, padding=PADDING_SIZE):
        data = nib.load(path).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]
        data = torch.tensor(data, device=device)
        data = transforms(data)[0]
        data = torch.permute(data, (1, 2, 0))
        shape = data.shape
        ratio = min(target / shape[i] for i in range(2))
        scaling = [int(shape[i] * ratio) for i in range(2)]
        if padding >= shape[2]:
            scaling.append(shape[2])
            data = resize(
                img=data.unsqueeze(0), out_size=scaling, mode="bilinear",
                align_corners=True, dtype=None, input_ndim=3,
                anti_aliasing=True, anti_aliasing_sigma=None,
                lazy=False, transform_info=None,
            )
            pad_t = (0, padding - scaling[2], 0, target - scaling[1], 0, target - scaling[0])
            data = F.pad(data, pad_t, mode="constant", value=0)
        else:
            scaling.append(padding)
            data = resize(
                img=data.unsqueeze(0), out_size=scaling, mode="bilinear",
                align_corners=True, dtype=None, input_ndim=3,
                anti_aliasing=True, anti_aliasing_sigma=None,
                lazy=False, transform_info=None,
            )
            pad_t = (0, 0, 0, target - scaling[1], 0, target - scaling[0])
            data = F.pad(data, pad_t, mode="constant", value=0)
        data = torch.permute(data, (0, 3, 1, 2))
        return data.view(-1, 32, target, target)
    return adaptive


def load_model(model_name_or_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, dtype=dtype, device_map="auto"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
        )
    model.eval()
    if "cache_position" not in inspect.signature(model.forward).parameters:
        original_forward = model.forward

        def _compat(self, *a, **kw):
            kw.pop("cache_position", None)
            return original_forward(*a, **kw)

        model.forward = types.MethodType(_compat, model)
    return model, tokenizer


def sample_candidates(model, tokenizer, image, question, n, max_new_tokens, temperature, top_p):
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    image_pt = image.unsqueeze(0).to(device=device, dtype=dtype)
    proj_out_num = getattr(getattr(model.get_model(), "mm_projector", None), "proj_out_num", 256)
    image_tokens = "<im_patch>" * int(proj_out_num)
    prompt = image_tokens + question

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    question_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    cand: List[str] = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                images=image_pt,
                inputs=input_ids,
                question_ids=question_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                repetition_penalty=1.1,
            )
        cand.append(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
    return cand


def score(metric, reference, candidates):
    ref = " ".join(reference.lower().split())
    if metric == "rougeL":
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return [scorer.score(ref, " ".join(c.lower().split()))["rougeL"].fmeasure for c in candidates]
    if metric == "bleu":
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        smoother = SmoothingFunction().method3
        return [
            corpus_bleu(
                [[ref.split()]],
                [" ".join(c.lower().split()).split()],
                weights=(1, 0, 0, 0),
                smoothing_function=smoother,
            )
            for c in candidates
        ]
    if metric == "green":
        from green_score import GREEN, OpenAILLM
        scorer = GREEN(OpenAILLM())
        mean, _, slist, _, _ = scorer([reference] * len(candidates), candidates)
        return slist
    raise ValueError(f"Unknown metric {metric}")


def worker(gpu_id, rows, out_path, model_path, base_path, n, metric, max_new_tokens, temperature, top_p, counter):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch  # noqa

    print(f"[GPU {gpu_id}] worker booting on {len(rows)} rows", flush=True)
    model, tokenizer = load_model(model_path)
    img_tf = make_image_transform("cuda:0")
    with open(out_path, "w") as f:
        for row in rows:
            img_path = os.path.join(base_path, row["image"])
            if not os.path.exists(img_path):
                print(f"[GPU {gpu_id}] missing {img_path}, skip", flush=True)
                continue
            try:
                image = img_tf(img_path)
                cands = sample_candidates(model, tokenizer, image, row["question"], n, max_new_tokens, temperature, top_p)
                scores = score(metric, row["answer"], cands)
                best, worst = int(np.argmax(scores)), int(np.argmin(scores))
                if best == worst:
                    print(f"[GPU {gpu_id}] tie on {row['image']}, skip", flush=True)
                    continue
                f.write(json.dumps({
                    "image": row["image"],
                    "question": row["question"],
                    "chosen": cands[best],
                    "rejected": cands[worst],
                    "scores": scores,
                }) + "\n")
                f.flush()
                counter.value += 1
            except Exception as e:
                print(f"[GPU {gpu_id}] err on {row['image']}: {e}", flush=True)
    print(f"[GPU {gpu_id}] DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--src_jsonl", required=True, help="Training JSONL with image/question/answer")
    p.add_argument("--base_path", required=True, help="Image base path (joined with row['image'])")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--metric", choices=["rougeL", "bleu", "green"], default="rougeL")
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    import torch
    n_gpus = torch.cuda.device_count() if args.num_gpus is None else args.num_gpus
    if n_gpus == 0:
        raise RuntimeError("No GPU available")

    with open(args.src_jsonl) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"loaded {len(rows)} src rows; sharding across {n_gpus} GPUs", flush=True)

    shards = [[] for _ in range(n_gpus)]
    for i, r in enumerate(rows):
        shards[i % n_gpus].append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    shard_outs = [f"{args.out_jsonl}.part{g}" for g in range(n_gpus)]

    mgr = Manager()
    counter = mgr.Value("i", 0)
    procs = []
    for g in range(n_gpus):
        p_ = Process(
            target=worker,
            args=(g, shards[g], shard_outs[g], args.model_path, args.base_path,
                  args.n_candidates, args.metric, args.max_new_tokens,
                  args.temperature, args.top_p, counter),
        )
        p_.start()
        procs.append(p_)
    for p_ in procs:
        p_.join()

    with open(args.out_jsonl, "w") as out:
        for s in shard_outs:
            if os.path.exists(s):
                with open(s) as f:
                    out.write(f.read())
                os.remove(s)
    print(f"wrote {counter.value} preference pairs -> {args.out_jsonl}")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
