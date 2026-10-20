"""Unit tests for green_score: GREEN judge request construction and response parsing.

They use stub tokenizers / clients only, so no GPU, model weights or network access is needed:

    python -m unittest discover tests
"""
import sys
import types
import unittest
from unittest import mock

import jinja2

# green_score.green reads config/project.json at import time (via config.py); provide a stub.
sys.modules.setdefault("config", types.SimpleNamespace(config={"openai_server": {}}))

from green_score import green  # noqa: E402

GREEN_MODEL = "StanfordAIMI/GREEN-radllama2-7b"


def render(messages):
    return jinja2.Template(green.GREEN_CHAT_TEMPLATE).render(
        messages=messages, add_generation_prompt=True, eos_token="</s>"
    )


class FakeTokenizer:
    def __init__(self):
        self.chat_messages = None
        self.call_kwargs = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_messages = messages
        return "RENDERED"

    def __call__(self, text, **kwargs):
        self.call_kwargs = kwargs
        return {"input_ids": list(range(100))}


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.completion_calls = []
        self.chat_calls = []
        reply = types.SimpleNamespace
        self.completions = reply(create=lambda **kw: self.completion_calls.append(kw) or reply(choices=[reply(text="completion")]))
        self.chat = reply(completions=reply(
            create=lambda **kw: self.chat_calls.append(kw) or reply(choices=[reply(message=reply(content="chat"))])
        ))


def make_openai_llm(model_name):
    server = {"model_name": model_name, "base_url": "http://localhost:8003/v1", "api_key": "EMPTY"}
    tokenizer = FakeTokenizer()
    with mock.patch.object(green, "config", {"openai_server": server}), \
            mock.patch.object(green, "OpenAI", FakeClient), \
            mock.patch.object(green.AutoTokenizer, "from_pretrained", return_value=tokenizer) as from_pretrained:
        llm = green.OpenAILLM()
    return llm, tokenizer, from_pretrained


class GreenChatTemplateTest(unittest.TestCase):
    def test_from_value_messages_include_the_prompt(self):
        text = render([{"from": "human", "value": "PROMPT"}, {"from": "gpt", "value": ""}])
        self.assertIn("<|user|>\nPROMPT</s>", text)
        self.assertTrue(text.rstrip().endswith("<|assistant|>"))

    def test_role_content_messages_lose_the_prompt(self):
        # What a chat-completions request gives the server-side template: the prompt is dropped.
        text = render([{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": ""}])
        self.assertNotIn("PROMPT", text)


class OpenAILLMTest(unittest.TestCase):
    def test_green_judge_uses_completions_with_greenllm_inputs(self):
        llm, tokenizer, from_pretrained = make_openai_llm(GREEN_MODEL)
        self.assertEqual(from_pretrained.call_args.args[0], GREEN_MODEL)
        self.assertTrue(from_pretrained.call_args.kwargs["add_eos_token"])

        self.assertEqual(llm.get_response({"prompt": ["PROMPT"]}), ["completion"])

        self.assertEqual(tokenizer.chat_messages, [{"from": "human", "value": "PROMPT"}, {"from": "gpt", "value": ""}])
        self.assertEqual(tokenizer.call_kwargs, {"truncation": True, "max_length": green.GREEN_MAX_LENGTH})
        self.assertEqual(llm.client.chat_calls, [])
        (call,) = llm.client.completion_calls
        self.assertEqual(call["model"], GREEN_MODEL)
        self.assertEqual(call["prompt"], list(range(100)))
        self.assertEqual(call["max_tokens"], green.GREEN_MAX_LENGTH - 100)
        self.assertEqual(call["temperature"], 0)

    def test_tokenizer_config_key_overrides_model_name(self):
        server = {"model_name": GREEN_MODEL, "base_url": "", "api_key": "EMPTY", "tokenizer": "/models/green"}
        with mock.patch.object(green, "config", {"openai_server": server}), \
                mock.patch.object(green, "OpenAI", FakeClient), \
                mock.patch.object(green.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()) as from_pretrained:
            green.OpenAILLM()
        self.assertEqual(from_pretrained.call_args.args[0], "/models/green")

    def test_other_judges_use_greedy_chat_completions(self):
        llm, _, from_pretrained = make_openai_llm("gpt-4o")
        from_pretrained.assert_not_called()
        self.assertIsNone(llm.green_tokenizer)

        self.assertEqual(llm.get_response({"prompt": ["PROMPT"]}), ["chat"])

        self.assertEqual(llm.client.completion_calls, [])
        (call,) = llm.client.chat_calls
        self.assertEqual(call["messages"][0], {"role": "user", "content": "PROMPT"})
        self.assertEqual(call["temperature"], 0)


LETTERED = """[Explanation]:
Two findings match.

[Clinically Significant Errors]:
(a) False report of a finding in the candidate: 1. Nodule.
(b) Missing a finding present in the reference: 1. Effusion.
(c) Misidentification of a finding's anatomic location/position: 0.
(d) Misassessment of the severity of a finding: 0.
(e) Mentioning a comparison that isn't in the reference: 0.
(f) Omitting a comparison detailing a change from a prior study: 0.

[Clinically Insignificant Errors]:
(a) False report of a finding in the candidate: 0.

[Matched Findings]:
2. Cardiomegaly; Atelectasis."""

NUMBERED = LETTERED.replace("(a) False report of a finding in the candidate: 1.", "(1) False report: 1.") \
    .replace("(b) Missing", "(2) Missing").replace("(c) Mis", "(3) Mis").replace("(d) Mis", "(4) Mis") \
    .replace("(e) Mentioning", "(5) Mentioning").replace("(f) Omitting", "(6) Omitting")


class GreenParsingTest(unittest.TestCase):
    def setUp(self):
        self.scorer = green.GREEN(green.LLM("stub"), compute_summary_stats=False)
        self.categories = list(self.scorer.sub_categories)

    def test_compute_green(self):
        # 2 matched findings, 1 + 1 clinically significant errors -> 2 / (2 + 2)
        self.assertEqual(self.scorer.compute_green(LETTERED), 0.5)

    def test_numbered_sub_categories_do_not_change_later_parsing(self):
        numbered = self.scorer.compute_sentences(NUMBERED)
        self.assertEqual(self.scorer.sub_categories, self.categories)
        self.assertEqual(list(numbered), self.categories)
        self.assertEqual([s.strip() for s in numbered[self.categories[1]]], ["Effusion."])

        lettered = self.scorer.compute_sentences(LETTERED)
        self.assertEqual([s.strip() for s in lettered[self.categories[0]]], ["Nodule."])


if __name__ == "__main__":
    unittest.main()
