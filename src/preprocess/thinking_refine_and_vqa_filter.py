from openai import OpenAI, AsyncOpenAI
import pandas as pd
from tqdm import tqdm
import os
import re
import argparse
import random
import json
import asyncio
import logging
from typing import List
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('thinking_refine_and_vqa_filter.log')
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process VQA thinking dataset with filtering, refinement, and report generation.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
parser.add_argument("--stage", type=str, default="all", 
                   choices=["filter", "refine", "report", "all"], 
                   help="Processing stage: filter (filter data), refine (refine thinking), report (generate report), all (run all stages)")
parser.add_argument("--output_dir", type=str, help="Output directory for processed files")
parser.add_argument("--skip-filter", action="store_true", help="Skip the filtering stage when running 'all'")
parser.add_argument("--skip-refine", action="store_true", help="Skip the refinement stage when running 'all'")
parser.add_argument("--skip-report", action="store_true", help="Skip the report generation stage when running 'all'")
parser.add_argument("--batch-size", type=int, default=10, help="Batch size for concurrent LLM calls")
parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum number of concurrent requests")
parser.add_argument("--retry-times", type=int, default=3, help="Number of retries for failed requests")
parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay between retries in seconds")
parser.add_argument("--enable-batch", action="store_true", help="Enable batch processing for better performance")

args = parser.parse_args()

async_client = AsyncOpenAI(
    api_key=config["openai_server"]["api_key"], 
    base_url=config["openai_server"]["base_url"],
)
model_name = config["openai_server"]["model_name"]

FILTER_TEMPLATE = """
You are an expert in radiology. Now you are reviewing a some questions and answers made by another expert.
You need to determine if the question is proper for a radiology exam, and the answer is correct.

If the question is proper for a radiology exam, and the answer is correct, return "Yes".
If the question is not proper for a radiology exam, or the answer is incorrect, return "No".
Do not return anything else.

The Report:
```
{report}
```
Question: {question}
Answer: {answer}
""".strip()

questions = [
            "Can you provide a caption consists of {} for this medical image?",
            "Describe the {} of the medical image you see.",
            "Please caption this medical scan with {}.",
            "What is the {} of this image?",
            "Describe this medical scan with {}.",
            "Please write a caption consists of {} for this image.",
            "Can you summarize with {} the images presented?",
            "Please caption this scan with {}.",
            "Please provide a caption consists of {} for this medical image.",
            "Can you provide a summary consists of {} of this radiograph?",
            "What are the {} presented in this medical scan?",
            "Please write a caption consists of {} for this scan.",
            "Can you provide a description consists of {} of this medical scan?",
            "Please caption this medical scan with {}.",
            "Can you provide a caption consists of {} for this medical scan?",
            # "Please generate a medical report based on this image.",
            # "Can you generate a diagnose report from this image.",
            "Could you analyze and provide a caption for the {} in this medical image?",
            # "Please describe the observations depicted in this medical scan.",
            "Can you summarize the {} of this image in a caption?",
            "What are the significant {} in this medical image?",
            "Please provide a detailed caption outlining the {} of this image.",
            "Could you interpret and describe the {} shown in this medical scan?",
            # "What conclusions can you draw from the observations in this image?",
            "Please write a descriptive caption based on the {} in this scan.",
            "What key {} can you identify from examining this medical image?",
            # "Could you generate a detailed report based on the observations in this image?",
            "Can you provide a diagnosis based on the {} in this image?",
            "Please generate a comprehensive report summarizing the {} in this image.",
            "Caption the {} in this medical image?",
            "Describe the {} you see.",
            "Caption this medical scan's {}.",
            "What are the {} here?",
            "Describe these {}.",
            "Summarize the {} in these images.",
            "Caption this scan's {}.",
            "Provide a caption for this medical image's {}.",
            "Summarize the {} of this radiograph.",
            "What {} are presented in this scan?",
            "Describe this scan's {}.",
            # "Generate a medical report based on this image.",
            # "Can you provide a diagnosis based on this image?",
]

async def query(content, enable_thinking=False):
    if not enable_thinking:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    else:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.6,
            top_p=0.95,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
    return response.choices[0].message.content.strip(), response.choices[0].message.reasoning_content.strip() if enable_thinking else None

async def query_with_retry(content, enable_thinking=False, max_retries=3, delay=1.0):
    """
    带重试机制的查询函数
    """
    for attempt in range(max_retries):
        try:
            return await query(content, enable_thinking)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"查询失败，已重试{max_retries}次: {e}")
                raise
            logger.warning(f"查询失败，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(delay)
    return None, None

async def batch_query(contents: List[str], enable_thinking=False, max_concurrent=5, batch_size=10):
    """
    批量查询函数，支持并发处理
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def query_with_semaphore(content, enable_thinking):
        async with semaphore:
            return await query_with_retry(content, enable_thinking, args.retry_times, args.retry_delay)
    
    results = []
    
    # 分批处理
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size} ({len(batch)} 个请求)")
        
        # 并发执行当前批次
        tasks = [query_with_semaphore(content, enable_thinking) for content in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"批次 {i//batch_size + 1} 中的请求 {j + 1} 失败: {result}")
                results.append((None, None))
            else:
                results.append(result)
        
        # 批次间延迟，避免过于频繁的请求
        if i + batch_size < len(contents):
            await asyncio.sleep(0.5)
    
    return results

def chunk_dataframe(df, chunk_size):
    """
    将DataFrame分块
    """
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

async def filter_data(df, jsonl_file):
    """
    第一步：过滤不合适的问题和答案
    """
    logger.info("开始过滤数据...")
    dropped_records = []

    if args.enable_batch:
        logger.info(f"使用批量处理模式 (批次大小: {args.batch_size}, 最大并发: {args.max_concurrent})")
        
        # 准备批量查询内容
        contents = []
        indices = []
        for i, row in df.iterrows():
            question = row["question"]
            answer = row["answer"]
            report = row["report"]
            content = FILTER_TEMPLATE.format(question=question, answer=answer, report=report)
            contents.append(content)
            indices.append(i)
        
        # 批量查询
        results = await batch_query(
            contents, 
            enable_thinking=False, 
            max_concurrent=args.max_concurrent, 
            batch_size=args.batch_size
        )
        
        # 处理结果
        for i, (response, _) in enumerate(results):
            if response is None:
                logger.warning(f"Index {indices[i]} 查询失败，跳过")
                continue
            if not response.startswith("Yes"):
                logger.warning(f"Index {indices[i]} 被认为是不合适的问题或答案，将剔除，响应为 {response}")
                df.drop(index=indices[i], inplace=True)
                dropped_records.append(indices[i])
    else:
        logger.info("使用单条处理模式")
        pbar = tqdm(total=len(df))
        for i, row in df.iterrows():
            question = row["question"]
            answer = row["answer"]
            report = row["report"]
            content = FILTER_TEMPLATE.format(question=question, answer=answer, report=report)
            response, _ = await query_with_retry(content, enable_thinking=False, max_retries=args.retry_times, delay=args.retry_delay)
            if response is None:
                logger.warning(f"Index {i} 查询失败，跳过")
                pbar.update(1)
                continue
            if not response.startswith("Yes"):
                logger.warning(f"Index {i} 被认为是不合适的问题或答案，将剔除，响应为 {response}")
                df.drop(index=i, inplace=True)
                dropped_records.append(i)
            pbar.update(1)
        pbar.close()

    if dropped_records:
        with open(f"{jsonl_file.replace('.jsonl','')}_dropped.txt", "w") as f:
            f.write(repr(dropped_records))
            f.flush()
    
    # 保存过滤后的数据
    filtered_file = f"{jsonl_file.replace('.jsonl','')}_filtered.jsonl"
    with open(filtered_file, "w") as f:
        for index, row in df.iterrows():
            f.write(json.dumps({
                "image": row["image"],
                "report": row["report"],
                "question": row["question"],
                "thinking": row["thinking"],
                "answer": row["answer"]
            }, ensure_ascii=False) + "\n")
            f.flush()
    
    logger.info(f"过滤完成，保存到: {filtered_file}")
    return df, filtered_file

async def refine_thinking(df, jsonl_file):
    """
    第二步：精炼thinking内容
    """
    logger.info("开始精炼thinking内容...")
    
    EDIT_TEMPLATE = """
    Help me edit the narrative below:
    - If the narrative refers to a report, you change it as if you see it from the radiology image
    - Edit only the places mentioned above, leave all other text the same 
    - Do not add/remove/change any other information
    - Directly output the result text

    **The narrative:**
    ```
    {report}
    ```
    """.strip()

    if args.enable_batch:
        logger.info(f"使用批量处理模式 (批次大小: {args.batch_size}, 最大并发: {args.max_concurrent})")
        
        # 准备批量查询内容
        contents = []
        indices = []
        for index, row in df.iterrows():
            content = EDIT_TEMPLATE.format(report=row["thinking"])
            contents.append(content)
            indices.append(index)
        
        # 批量查询
        results = await batch_query(
            contents, 
            enable_thinking=False, 
            max_concurrent=args.max_concurrent, 
            batch_size=args.batch_size
        )
        
        # 处理结果
        for i, (result, _) in enumerate(results):
            if result is None:
                logger.warning(f"Index {indices[i]} 查询失败，跳过")
                continue
            result = result.strip('`\n')
            df.loc[indices[i], "refined_thinking"] = result
    else:
        logger.info("使用单条处理模式")
        pbar = tqdm(total=len(df))
        for index, row in df.iterrows():
            result, _ = await query_with_retry(EDIT_TEMPLATE.format(report=row["thinking"]), enable_thinking=False, max_retries=args.retry_times, delay=args.retry_delay)
            if result is None:
                logger.warning(f"Index {index} 查询失败，跳过")
                pbar.update(1)
                continue
            result = result.strip('`\n')
            df.loc[index, "refined_thinking"] = result
            pbar.update(1)
        pbar.close()
    
    # 保存精炼后的数据
    refined_file = f"{jsonl_file}_refined.jsonl"
    with open(refined_file, "w") as f:
        for index, row in df.iterrows():
            f.write(json.dumps({
                "image": row["image"],
                "report": row["report"],
                "question": row["question"],
                "thinking": row["thinking"],
                "refined_thinking": row["refined_thinking"],
                "answer": row["answer"]
            }, ensure_ascii=False) + "\n")
            f.flush()
    
    logger.info(f"精炼完成，保存到: {refined_file}")
    return df, refined_file

async def generate_report_thinking(df, jsonl_file):
    """
    第三步：生成报告思考内容
    """
    logger.info("开始生成报告思考内容...")
    
    THINKING_TEMPLATE = """
    You are a radiology medicine expert. Now you are looking at a radiology image.
    Here is your self talk when viewing the image:
    ```
    {thinking_before}
    ```

    Please paraphrase the self talk text and output it as **thinking progress**. Remember:
    - Do not add/remove/alter any information
    - Mind the coherence and fluence of output
    - Deductions are prefered
    - Directly output the result text

    Your output:
    """.strip()

    # 按图像分组处理
    image_groups = df.groupby('image', sort=False)
    
    if args.enable_batch:
        logger.info(f"使用批量处理模式 (批次大小: {args.batch_size}, 最大并发: {args.max_concurrent})")
        
        # 准备批量查询内容
        contents = []
        image_names = []
        thinking_befores = []
        
        for current_image, group in image_groups:
            thinking_before = ""
            for _, row in group.iterrows():
                thinking_before += " " + row["question"] + row["refined_thinking"] + row["answer"]
            thinking_before = thinking_before.strip()
            
            content = THINKING_TEMPLATE.format(thinking_before=thinking_before)
            contents.append(content)
            image_names.append(current_image)
            thinking_befores.append(thinking_before)
        
        # 批量查询
        results = await batch_query(
            contents, 
            enable_thinking=False, 
            max_concurrent=args.max_concurrent, 
            batch_size=args.batch_size
        )
        
        # 处理结果并保存
        for i, (thinking_after, _) in enumerate(results):
            if thinking_after is None:
                logger.warning(f"图像 {image_names[i]} 查询失败，跳过")
                continue
                
            thinking_after = re.sub(r"^(?:\*+)?Thinking Progress:(?:\*+)?", "", thinking_after, flags=re.IGNORECASE).strip('`\n')
            result_line = json.dumps({
                "image": image_names[i],
                "report": df[df['image'] == image_names[i]]['report'].iloc[0],
                "question": questions[random.randint(0, len(questions) - 1)].format("findings"),
                "thinking_before": thinking_befores[i],
                "thinking_after": thinking_after,
            }, ensure_ascii=False)
            with open(f"{jsonl_file.replace('.jsonl','')}_report_thinking.jsonl", "a") as f:
                f.write(result_line + "\n")
                f.flush()
    else:
        logger.info("使用单条处理模式")
        current_image = ""
        pbar = tqdm(total=len(image_groups))
        for index, row in df.iterrows():
            if row["image"] != current_image:
                current_image = row["image"]
                logger.info(f"Processing image: {current_image}")
                logger.info(f"Total number of questions: {len(df[df['image'] == current_image])}")
                thinking_before = ""
                for _, inner_row in df.iloc[index:index+20][df["image"] == current_image].iterrows():
                    thinking_before += " " + inner_row["question"] + inner_row["refined_thinking"] + inner_row["answer"]
                thinking_before = thinking_before.strip()
                content = THINKING_TEMPLATE.format(thinking_before=thinking_before)
                thinking_after, _ = await query_with_retry(content, enable_thinking=False, max_retries=args.retry_times, delay=args.retry_delay)
                if thinking_after is None:
                    logger.warning(f"图像 {current_image} 查询失败，跳过")
                    pbar.update(1)
                    continue
                thinking_after = re.sub(r"^(?:\*+)?Thinking Progress:(?:\*+)?", "", thinking_after, flags=re.IGNORECASE).strip('`\n')
                result_line = json.dumps({
                    "image": current_image,
                    "report": row["report"],
                    "question": questions[random.randint(0, len(questions) - 1)].format("findings"),
                    "thinking_before": thinking_before,
                    "thinking_after": thinking_after,
                }, ensure_ascii=False)
                with open(f"{jsonl_file.replace('.jsonl','')}_report_thinking.jsonl", "a") as f:
                    f.write(result_line + "\n")
                    f.flush()
                pbar.update(1)
            else:
                continue
        pbar.close()
    
    report_file = f"{jsonl_file.replace('.jsonl','')}_report_thinking.jsonl"
    logger.info(f"报告生成完成，保存到: {report_file}")

async def main(args):
    # 确定输入文件
    input_file = args.input_file if args.input_file else args.input_file
    
    # 确定输出目录
    output_dir = args.output_dir if args.output_dir else os.path.dirname(input_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"处理模式: {args.stage}")
    logger.info(f"输入文件: {input_file}")
    if output_dir:
        logger.info(f"输出目录: {output_dir}")
    
    if args.stage == "filter":
        # 只运行过滤步骤
        logger.info("=== 运行过滤步骤 ===")
        df = pd.read_json(input_file, lines=True)
        logger.info(f"原始数据包含 {len(df)} 条记录")
        df_filtered, filtered_file = await filter_data(df, input_file)
        logger.info(f"过滤后剩余 {len(df_filtered)} 条记录")
        logger.info(f"过滤完成，结果保存到: {filtered_file}")
        
    elif args.stage == "refine":
        # 只运行精炼步骤
        logger.info("=== 运行精炼步骤 ===")
        df = pd.read_json(input_file, lines=True)
        logger.info(f"输入数据包含 {len(df)} 条记录")
        df_refined, refined_file = await refine_thinking(df, input_file)
        logger.info(f"精炼完成，结果保存到: {refined_file}")
        
    elif args.stage == "report":
        # 只运行报告生成步骤
        logger.info("=== 运行报告生成步骤 ===")
        df = pd.read_json(input_file, lines=True)
        logger.info(f"输入数据包含 {len(df)} 条记录")
        await generate_report_thinking(df, input_file)
        report_file = f"{input_file.replace('.jsonl','')}_report_thinking.jsonl"
        logger.info(f"报告生成完成，结果保存到: {report_file}")
        
    elif args.stage == "all":
        # 运行所有步骤
        logger.info("=== 运行所有处理步骤 ===")
        df = pd.read_json(input_file, lines=True)
        logger.info(f"原始数据包含 {len(df)} 条记录")
        
        # 第一步：过滤数据
        if not args.skip_filter:
            logger.info("\n--- 步骤1: 过滤数据 ---")
            df_filtered, filtered_file = await filter_data(df, input_file)
            logger.info(f"过滤后剩余 {len(df_filtered)} 条记录")
        else:
            logger.info("\n--- 跳过过滤步骤 ---")
            df_filtered = df
            filtered_file = input_file
        
        # 第二步：精炼thinking内容
        if not args.skip_refine:
            logger.info("\n--- 步骤2: 精炼thinking内容 ---")
            df_refined, refined_file = await refine_thinking(df_filtered, input_file)
        else:
            logger.info("\n--- 跳过精炼步骤 ---")
            df_refined = df_filtered
            refined_file = filtered_file
        
        # 第三步：生成报告思考内容
        if not args.skip_report:
            logger.info("\n--- 步骤3: 生成报告思考内容 ---")
            await generate_report_thinking(df_refined, input_file)
        else:
            logger.info("\n--- 跳过报告生成步骤 ---")
        
        logger.info("\n=== 所有处理步骤完成！ ===")
        
        # 输出文件总结
        logger.info("\n生成的文件:")
        if not args.skip_filter:
            logger.info(f"  - 过滤结果: {filtered_file}")
        if not args.skip_refine:
            logger.info(f"  - 精炼结果: {refined_file}")
        if not args.skip_report:
            report_file = f"{input_file.replace('.jsonl','')}_report_thinking.jsonl"
            logger.info(f"  - 报告结果: {report_file}")
    
    else:
        logger.error(f"错误: 未知的处理模式 '{args.stage}'")
        return

if __name__ == "__main__":
    asyncio.run(main(args))