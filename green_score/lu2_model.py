import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def find_all_linear_names(model):
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name)
    return list(lora_module_names)


class Lu2Model:
    def __init__(self, model_path: str, lora_weight_path: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=1024,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

        lu2_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        if lora_weight_path:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=find_all_linear_names(lu2_model),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("Adding LoRA adapters only on LLM.")
            lu2_model = get_peft_model(lu2_model, lora_config)
            print("Load weights with LoRA")
            state_dict = torch.load(lora_weight_path, map_location="cpu")
            lu2_model.load_state_dict(state_dict, strict=True)
            print("Merge weights with LoRA")
            lu2_model = lu2_model.merge_and_unload()
        self.model = lu2_model.eval()

    def inference(self, image, question, temperature=1.0, top_p=0.9):
        proj_out_num = 256
        # question = "Can you provide a diagnosis based on the findings in chest in this image?."
        question_ids = self.tokenizer(
                    question, add_special_tokens=False, max_length=768, truncation=True, padding="max_length", return_tensors="pt", padding_side="right"
                )["input_ids"][0]
        image_tokens = "<im_patch>" * proj_out_num
        input_txt = image_tokens + question
        input_id = self.tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device="cuda")

        with torch.amp.autocast("cuda"):
            generation = self.model.generate(image.to("cuda"), input_id, question_ids.to("cuda"), max_new_tokens=768,
                                                do_sample=True, top_p=top_p, temperature=temperature)

        return self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]