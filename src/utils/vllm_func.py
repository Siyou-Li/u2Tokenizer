# -*- encoding: utf-8 -*-
# @File        :   ollama_func.py
# @Time        :   2025/01/09 16:05:27
# @Author      :   Siyou
# @Description :


from openai import OpenAI
from config import config
import json

model_name= config["vllm_model"]
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

def rewrite(raw_text):
    
    response = client.chat.completions.create(model=model_name, messages=[
        {
            'role': 'user',
            'content': f'You are a professional CT medical report expert, \
            Please help me rewrite the following CT report. \
            The description is accurate and smooth when the rewriting means unchanged. \
            Here are some examples: \
            1. The liver is normal in size and shape with regular surface. The liver parenchyma density is uniform, and the common bile duct and intrahepatic bile duct are not dilated. The spleen is normal in size and shape with no abnormal density in the parenchyma. The position, shape, and density of the pancreas are normal, with no dilation of the pancreatic duct. The gallbladder is not enlarged, and the wall is not thickened. Both kidneys are normal in size and shape, while multiple cystic low-density lesions are observed with no obvious enhancement. No obvious enlarged lymph nodes are seen in the posterior peritoneum. No obvious abnormal enhancing foci is seen in the upper and lower abdomen.\
            2. A nodular high-density focus is seen in the right lobe of the liver, and no abnormal density is found in the remaining liver parenchyma. The intrahepatic duct system is not obviously dilated. The gallbladder is not enlarged with uniform density enhancement inside, while a focal high-density shadow is seen near the neck of the gallbladder. The surface of the pancreas is rough, with still uniform density and clear adjacent fat intervals. The spleen is enlarged. Multiple enlarged lymph nodes are found in the abdominal cavity and retroperitoneum.\
            3. Bilateral chest are symmetrical. A few patchy high-density shadows with blurred edges are observed in the right lung upper lobe and both lung lower lobes, and a few consolidations are visible in the right lung lower lobe. No narrowing or obstruction of the trachea and bronchus can be seen. Esophageal dilation is observed with mixed low-density shadows inside. No obvious enlarged lymph nodes is seen in the mediastinum or bilateral pulmonary hila. No significant abnormalities are observed in the heart or major blood vessels. No abnormal enhanced lesions are observed in the enhanced scan.\
            4. The chest is symmetrical, the lung window shows clear lung texture with natural course. A few cord-like high-density foci are seen in the middle and lower lobe of right lung, while no abnormal solid foci is seen in the remaining lungs. The bilateral pulmonary hila are normal. The trachea and bronchi are unobstructed. The mediastinal window shows no deviation of the mediastinum, and the morphology of the heart and major blood vessels are normal. No mass or enlarged lymph nodes is seen in the mediastinum. No pleural effusion or pleural thickening is seen on either sides.\
            5. The bladder is well-filled, and the bladder wall is smooth. No abnormal density foci is observed in the bladder wall or inside the bladder. The position, contour, and size of the prostate are normal, with scattered high-density foci inside. The bilaterally symmetrical seminal vesicles show no abnormal density. The surrounding fat space is clear. No abnormal enhancement is observed in other regions.\
            6. The prostate volume is not large, with a smooth contour and high-density points in local areas. The angle between the bladder and seminal vesicle is clear, and no obvious focal thickening or nodular foci are seen on the bladder wall. High-density foci are seen in the pelvis.\
            \
            Please fix the findings and impression into one sentence.\
            Please directly output the rewritten sentence in English. \"{raw_text}\"',
        },
    ]).choices[0].message.content
    return response

def generate_qa(raw_text):
    response = client.chat.completions.create(model=model_name, messages=[
        {
            'role': 'user',
            'content': f'Please break down the following CT reports and generate a corresponding number of questions and answers based on the number of lesions \
            The description should be accurate, professional and fluent. \
            Please answer the generated results in json format (for example [{{"question":"","answer":""}},{{"question":"","answer":""}},...]). \
            Do not add any other text, and output in English. \"{raw_text}\"',
        },
    ]).choices[0].message.content
    try:
        qa_pairs = json.loads(response)
    except Exception as e:
        print(e)
        return []
    return qa_pairs