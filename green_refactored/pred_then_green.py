# The script to generate predictions and evaluate them(case by case)using GREEN
# Type <script> --help to see the help message
# Example of generation which takes cases from 0 to 100 from the ct_rate dataset, and use the model checkpoint specified by -m, using GPU 0
# python <script> generate -d ct_rate -s 0 -e 100 -t 1 -m /data/huanan/final_checkpoint/ct_rate_mu2@bs1_acc1_ep4_lr4e6_ws4_fused/checkpoint-34912/ -g 0
# Example of evaluation which takes the generated file indexed 0, assesses the predictions using GREEN within GPU 0
# python <script> evaluate -d ct_rate -n 0 -g 0

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(dest="command")

generate_parser = sub_parsers.add_parser("generate", help="To generate predictions by the specified model")
generate_parser.add_argument("-d", "--data-type", choices=["ct_rate", "amos_mm"], default="ct_rate", help="Type of the model/dataset")
generate_parser.add_argument("-s", "--start", type=int, default=0, help="Start index of the dataset")
generate_parser.add_argument("-e", "--end", type=int, default=-1, help="End index of the dataset, -1 means the end of the dataset")
generate_parser.add_argument("-t", "--step", type=int, default=1, help="Step of the dataset, default is 1")
generate_parser.add_argument("-m", "--model", type=str, default="/data/huanan/final_checkpoint/ct_rate_mu2@bs1_acc1_ep4_lr4e6_ws4_fused/checkpoint-34912/", help="Path of the model")
generate_parser.add_argument("-g", "--gpus", type=str, default="0", help="Specify which GPUs to use, for example 0,1 means using GPU 0 and 1")

evaluate_parser = sub_parsers.add_parser("evaluate", help="To evaluate the predictions using GREEN")
evaluate_parser.add_argument("-d", "--data-type", choices=["ct_rate", "amos_mm"], default="ct_rate", help="Type of the model/dataset")
evaluate_parser.add_argument("-n", "--number", type=int, default=0, help="Number index to append to the prediction file")
evaluate_parser.add_argument("-g", "--gpus", type=str, default="0", help="Specify which GPUs to use, for example 0,1 means using GPU 0 and 1")
args = parser.parse_args()

import os
# must set CUDA_VISIBLE_DEVICES before import torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
from torch.utils.data import DataLoader, Subset
import json
from tqdm import tqdm
from green_refactored.lu2_model import Lu2Model
from green_refactored.green import GREEN, GREENLLM
from src.dataset.fused_dataset import FusedDataset


def generate_predictions():
    dataset_range = range(args.start, args.end, args.step)
    prediction_file = f"output/{args.data_type}_pred_{int(dataset_range[0])}.jsonl"
    lu2_model = Lu2Model(args.model)
    
    dataset = FusedDataset(
        base_path="datasets", 
        jsonl_path=f"Fused_Dataset/train/{args.data_type}_raw.jsonl", 
        tokenizer=lu2_model.tokenizer, 
        max_length=2048,
        data_type="training",
        enable_u2tokenizer=True)
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    
    dataloader = Subset(dataset, dataset_range)
    dataloader = DataLoader(dataloader, collate_fn=collate_fn, batch_size=1)

    results = []

    for batch in tqdm(dataloader):
        if batch is None:
            continue

        image = batch["image"]
        image_path = batch["image_path"][0][len(dataloader.dataset.dataset.base_path)+1:]
        question = batch["prompt_question"][0]
        ground_truth = batch["answer"][0]
        predictions = []
        with open(prediction_file, "a+") as f:
            for _ in range(8):
                while True:
                    pred = lu2_model.inference(image, question).strip()
                    if check_character_and_length(pred):
                        break
                predictions.append(pred)

            result = {
                "image": image_path,
                "question": question,
                "ground_truth": ground_truth,
                "predictions": predictions,
            }
            results.append(result)

            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    return results

def check_character_and_length(answer):
    for ch in answer:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    if len(answer.replace(" ","")) < 20:
        return False
    return True

def evaluate_with_green():
    # 从 generate 的结果中，取出一个 case，将其中 8 个 answer 分别与 ground truth 一起传到 green
    # 算出结果后取 mean 值，存起来
    # 同时，将 8 个 answer 根据 score 从高到低排序

    # 有多种类型的 LLM，这里选用远程 API 调用的类型
    # llm_model = OpenAILLM("OpenAI LLM")
    # green = GREEN(llm_model)

    # mean, std, green_score_list, summary, result_df = green(refs=gt_report, hyps=pred_report)
    # print(mean, std, green_score_list, summary)

    prediction_file = f"output/{args.data_type}_pred_{args.number}.jsonl"
    greened_file = f"output/{args.data_type}_greened_{args.number}.jsonl"

    llm_model = GREENLLM("/data/huanan/models/GREEN-RadPhi2")
    green = GREEN(llm_model, compute_summary_stats=False)

    with open(prediction_file, "r") as input_file:
        for line in input_file:
            record = json.loads(line)
            ground_truth = record["ground_truth"]
            predictions = record["predictions"]
            green_scores = {}
            for prediction in predictions:
                _, _, green_score_list, _, _ = green(refs=[ground_truth], hyps=[prediction])
                green_scores[prediction] = green_score_list[0]
            sorted_scored_predictions = sorted(green_scores.items(), key=lambda e: e[1], reverse=True)
            record["predictions"] = [item[0] for item in sorted_scored_predictions]
            record["prediction_scores"] = [item[1] for item in sorted_scored_predictions]
            with open(greened_file, "a+") as output_file:
                json.dump(record, output_file, ensure_ascii=False)
                output_file.write("\n")

if __name__ == "__main__":
    if args.command == "generate":
        generate_predictions()
    elif args.command == "evaluate":
        evaluate_with_green()
