import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import textwrap

device = "cuda"
dtype = torch.bfloat16 # or bfloat16, float16, float32

class AnswerValidator:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        EXAMPLE = textwrap.dedent("""\
            Quention: Can you provide a diagnosis based on the fingings in chest in this image?
            Answer: Both sides of the chest are symmetrical.
                Scattered point-like translucence are seen in both lungs, and a few patchy high-density foci are seen in the low lobe of left lung.
                No other abnormal are seen in the lungs. The trachea and bronchi are unobstructed.
                The mediastinum and trachea are centered, and multiple slightly enlarged lymph nodes with higher density are seen in the mediastinum and bilateral pulmonary hila.
                The pleura is normal. The morphology and size of the heart and great vessels are normal, with a small amount of fluid in the pericardium.
                A high-density shadow is seen in the upper part of the esophagus. No obvious abnormal enhancement is seen in the chest.
            """)
        self.SYSTEM_PROMPT = textwrap.dedent("""\
            You are the Radiation LLM answer checker, please identify invalid answers (e.g. duplicate/meaningless/unrelated output)

            This is an example:
            {example}.
            
            If answers are checked by Yes otherwise No, do not output any other characters other than that.
            """).format(example=EXAMPLE)

    def validate(self, question: str, answer: str) -> bool:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Quention: {question} Answer: {answer}"}
        ]
        text = self.tokenizer.apply_chat_template(         
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return "Yes" in response
