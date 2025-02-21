import asyncio
import json
import random
import backoff
import openai
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI
from config import config

from src.utils.prompt_templates import Caption_templates


async_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-njnlfiszqbrbdmffearogkutycgnhteqxjltsbzhglejljzo",
)
model_name="./Qwen2.5-32B-Instruct-AWQ/"

PROMPT = """
You are an expert radiologists. And your task is to paraphrase a given radiology report. 
You need to:
1. Take the following 3 examples for style of writing.
2. You MUST NOT change any meaning of the original report, nor add or remove any information, not event correction.
3. Give out the paraphrased report directly, without any other content.
4. In English only.

### example 1:
```
The spleen is of normal size, measuring 125.5 cc in volume, with a mean Hounsfield Unit (HU) value of 128.4 +/- 47.7. The liver is also of normal size, with a volume of 1295.6 cc and a mean HU value of 79.9 +/- 29.0. The pancreas is normal in size, with a volume of 60.7 cc and a mean HU value of 80.4 +/- 62.4. A small hypoattenuating lesion, consistent with a cyst, is present in the pancreas, measuring 0.6 x 0.4 cm in size and 0.8 cc in volume, with a HU value of 48.5 +/- 24.8. The kidneys are normal in size, with the right kidney measuring 158.5 cc and the left kidney measuring 196.3 cc, and a total kidney volume of 354.7 cc, with a mean HU value of 159.1 +/- 86.1.

IMPRESSION:
1. A small hypoattenuating lesion in the pancreas, consistent with a cyst, measuring 0.6 x 0.4 cm in size.
2. Normal size and appearance of the spleen, liver, pancreas, and kidneys.
```

### example 2:
```
The patient has an enlarged pancreas, measuring 184.6 cc in volume, with a mean Hounsfield Unit (HU) value of 58.0 +/- 43.0. Within the pancreas, a hypoattenuating lesion is identified in the head, measuring 3.3 x 2.7 cm in size, with a volume of 7.8 cc and a mean HU value of 36.2 +/- 26.1. This lesion is in contact with the bile duct and has a 25-degree contact with the portal vein and superior mesenteric vein (SMV), but does not encase these vessels. The tumor does not contact the superior mesenteric artery (SMA), aorta, inferior vena cava (IVC), celiac axis (CA), common hepatic artery (CHA), or splenic artery (SA).

The spleen is of normal size, with a volume of 146.9 cc and a mean HU value of 99.0 +/- 29.3. The liver is also normal in size, with a volume of 1552.4 cc and a mean HU value of 120.6 +/- 19.5. The kidneys are normal in size, with a total volume of 385.9 cc and a mean HU value of 128.5 +/- 49.1.

IMPRESSION:
1. Enlarged pancreas with a hypoattenuating mass in the head, measuring 3.3 x 2.7 cm, consistent with a pancreatic neoplasm.
2. The tumor is in contact with the portal vein and SMV, but does not encase these vessels.
3. The tumor does not contact the SMA, aorta, IVC, CA, CHA, or SA.
4. Normal-sized spleen, liver, and kidneys.
```

### example 3:
```
The spleen, liver, pancreas, and kidneys are all within normal limits in terms of size and appearance. The spleen measures 176.7 cc in volume, the liver measures 941.7 cc, and the kidneys measure 152.6 cc on the right and 122.6 cc on the left, with a total kidney volume of 275.2 cc. The mean Hounsfield Unit (HU) values for these organs are within normal ranges, with the spleen at 110.9 +/- 40.0, the liver at 122.4 +/- 27.7, the pancreas at 104.0 +/- 17.6, and the kidneys at 154.6 +/- 113.1.

IMPRESSION:
No evidence of tumor is observed in the liver, pancreas, or kidneys.
```

### The original report:
```
{}
```
""".strip()

base_path = config.project_path
raw_data_file_path = f"{base_path}/AbdomenAtlas3.0.csv"
output_file_path = f"{base_path}/output/abdomen_atlas3_rewrite7.jsonl"

# Load raw data
with open(raw_data_file_path, 'r') as f:
    raw_data = pd.read_csv(raw_data_file_path, low_memory=False)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def async_rewrite(prompt):
    response = (await async_client.chat.completions.create(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])).choices[0].message.content
    return response

async def gather_async_jobs(jobs):
    return await asyncio.gather(*jobs)

for idx, row in tqdm(iterable=raw_data.iterrows(), total=len(raw_data)):
    image = row["BDMAP ID"]
    narrative_report = row["narrative report"]

    if not narrative_report:
        continue

    with open(output_file_path, "a+") as f:
        paraphrased_reports = asyncio.run(gather_async_jobs(
            [async_rewrite(PROMPT.format(narrative_report)) for _ in range(8)]
        ))
        for paraphrased_report in paraphrased_reports:
            prompt_question = random.choice(Caption_templates).format("findings in abdomen")
            record = {
                "image": f"AbdomenAtlasData/{image}/ct.nii.gz",
                "dataset": "AbdomenAtlasData3.0",
                "task_type": "VQA",
                "synthesis": True,
                "question": prompt_question,
                "answer": paraphrased_report,
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
