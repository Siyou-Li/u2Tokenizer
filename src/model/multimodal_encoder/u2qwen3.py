from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen3Config, Qwen3Model, Qwen3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..u2_arch import u2MetaModel, u2MetaForCausalLM


class u2Config(Qwen3Config):
    model_type = "u2Qwen3"


class u2Qwen3Model(u2MetaModel, Qwen3Model):
    config_class = u2Config
    def __init__(self, config: Qwen3Config):
        super(u2Qwen3Model, self).__init__(config)


class u2Qwen3ForCausalLM(u2MetaForCausalLM, Qwen3ForCausalLM):
    config_class = u2Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = u2Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            vision_input: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            raw_question_ids: Optional[torch.LongTensor] = None,
            vision_token_index: Optional[int] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        if inputs_embeds is None:
            (
                vision_input,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                emb_loss
            ) = self.prepare_inputs_for_multimodal(
                vision_input,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                vision_token_index,
                raw_question_ids
            )

        outputs =  super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        return (outputs, emb_loss)

    @torch.no_grad()
    def generate(
        self,
        vision_input: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        vision_token_index: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        raw_question_ids = kwargs.pop("raw_question_ids", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if vision_input is not None:
            (
                vision_input,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                vision_input,
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                vision_token_index,
                raw_question_ids
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        output_ids = super().generate(
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("u2Qwen3", u2Config)
AutoModelForCausalLM.register(u2Config, u2Qwen3ForCausalLM)