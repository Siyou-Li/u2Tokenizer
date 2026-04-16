# -*- encoding: utf-8 -*-
# @File        :   llm_utils.py
# @Description :   LLM call infrastructure for data synthesis preprocessing

import asyncio
import logging
import re
from typing import List

import backoff
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

from config import config

logger = logging.getLogger(__name__)

_async_client = None
_model_name = None


def get_client():
    global _async_client, _model_name
    if _async_client is None:
        _model_name = config["openai_server"]["model_name"]
        _async_client = AsyncOpenAI(
            base_url=config["openai_server"]["base_url"],
            api_key=config["openai_server"]["api_key"],
        )
    return _async_client, _model_name


THINKING_PARAMS = {
    "stream": False,
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "extra_body": {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    },
}

NON_THINKING_PARAMS = {
    "stream": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "max_tokens": 8192,
    "extra_body": {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


async def synthesize_data_batch(prompts, enable_thinking=True, concurrency=4):
    """
    prompts: List[str]
    enable_thinking: bool -- whether to include <think> reasoning in the template
    Returns: List[Tuple[str, str]]
        thinkings: List[str]  the content captured between <think>...</think>
        outputs:   List[str]  the remaining generated content
    """
    message_batches = [[{"role": "user", "content": p}] for p in prompts]
    semaphore = asyncio.Semaphore(concurrency)

    client, model = get_client()

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APITimeoutError, APIConnectionError))
    async def process_one(messages):
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **(THINKING_PARAMS if enable_thinking else NON_THINKING_PARAMS),
            )
            return response.choices

    all_results = await asyncio.gather(
        *[process_one(messages) for messages in message_batches]
    )

    results = []
    pattern = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)

    for result in all_results:
        full_text = result[0].message.content
        m = pattern.search(full_text)
        if m:
            thinking = m.group(1).strip()
            output = m.group(2).strip()
        else:
            thinking = ""
            output = full_text.strip()
        results.append((thinking, output))

    return results


TRANSLATION_TPL = """
This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_input}
""".strip()


@backoff.on_exception(backoff.expo, (RateLimitError, APIError, APITimeoutError, APIConnectionError))
async def async_translation(
    source_input, target_lang, source_lang, enable_thinking=False
):
    client, model = get_client()
    messages = [
        {
            "role": "user",
            "content": TRANSLATION_TPL.format(
                source_input=source_input,
                target_lang=target_lang,
                source_lang=source_lang,
            ),
        }
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **(THINKING_PARAMS if enable_thinking else NON_THINKING_PARAMS),
    )

    return response.choices[0].message.content


async def async_translate_batch(
    texts, target_lang, source_lang, enable_thinking=False, concurrency=8
):
    """Translate a list of texts concurrently with semaphore-based rate control."""
    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(texts), desc="Translating")

    async def translate_one(text):
        async with semaphore:
            result = await async_translation(
                text, target_lang, source_lang, enable_thinking
            )
            pbar.update(1)
            return result

    try:
        results = await asyncio.gather(*[translate_one(t) for t in texts])
    finally:
        pbar.close()
    return results


async def query(content, enable_thinking=False):
    client, model = get_client()
    params = THINKING_PARAMS if enable_thinking else NON_THINKING_PARAMS
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        **params,
    )
    reasoning = (
        response.choices[0].message.reasoning_content.strip()
        if enable_thinking else None
    )
    return response.choices[0].message.content.strip(), reasoning


async def query_with_retry(content, enable_thinking=False, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            return await query(content, enable_thinking)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Query failed after {max_retries} retries: {e}")
                raise
            logger.warning(f"Query failed, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(delay)
    return None, None


async def batch_query(contents: List[str], enable_thinking=False, max_concurrent=5, batch_size=10, retry_times=3, retry_delay=1.0):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def query_with_semaphore(content, enable_thinking):
        async with semaphore:
            return await query_with_retry(content, enable_thinking, retry_times, retry_delay)

    results = []

    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size} ({len(batch)} requests)")

        tasks = [query_with_semaphore(content, enable_thinking) for content in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Request {j + 1} in batch {i//batch_size + 1} failed: {result}")
                results.append((None, None))
            else:
                results.append(result)

        if i + batch_size < len(contents):
            await asyncio.sleep(0.5)

    return results
