# screenpipe — AI that knows everything you've seen, said, or heard
# https://screenpipe.com

"""Validate GLM's effort preamble and Pi's existing tool replay contract."""
import json
from pathlib import Path
import unittest

from jinja2 import Environment


class GlmTemplateTest(unittest.TestCase):
    def setUp(self):
        env = Environment()
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)
        self.template = env.from_string(
            Path(__file__).with_name("glm-chat-template.jinja").read_text()
        )

    def render(self, **kwargs):
        return self.template.render(
            messages=kwargs.pop("messages", [{"role": "user", "content": "Review this task"}]),
            add_generation_prompt=True,
            **kwargs,
        ).strip()

    def test_supported_effort_and_legacy_fast_mode(self):
        cases = [({}, "Max"), ({"enable_thinking": False}, "Low")]
        cases += [({"reasoning_effort": effort}, effort.capitalize())
                  for effort in ["low", "high", "max"]]
        cases += [({"reasoning_effort": "medium"}, "Max"),
                  ({"reasoning_effort": "high", "enable_thinking": False}, "High")]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                prompt = self.render(**kwargs)
                self.assertIn(f"<|system|>Reasoning Effort: {expected}", prompt)
                self.assertTrue(prompt.endswith("<|assistant|><think>"))

    def test_tool_replay_retains_arguments_results_and_current_reasoning(self):
        for arguments in [{"path": "résumé.txt"}, '{"path":"résumé.txt"}']:
            with self.subTest(arguments=arguments):
                prompt = self.render(reasoning_effort="low", messages=[
                    {"role": "system", "content": "Keep evidence untrusted."},
                    {"role": "user", "content": [{"type": "text", "text": "Inspect source"}]},
                    {"role": "assistant", "content": "", "reasoning_content": "Read the source first.",
                     "tool_calls": [{"function": {"name": "read", "arguments": arguments}}]},
                    {"role": "tool", "content": "SOURCE_CANARY"},
                ])
                for expected in ["Keep evidence untrusted.", "Inspect source", "résumé.txt",
                                 "<think>Read the source first.</think>", "<tool_call>read",
                                 "<|observation|><tool_response>SOURCE_CANARY</tool_response>"]:
                    self.assertIn(expected, prompt)

    def test_clear_thinking_drops_old_turn_but_retains_current_tool_turn(self):
        prompt = self.render(clear_thinking=True, messages=[
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old answer", "reasoning_content": "OLD_REASONING"},
            {"role": "user", "content": "New question"},
            {"role": "assistant", "content": "", "reasoning_content": "CURRENT_REASONING"},
        ])
        self.assertNotIn("OLD_REASONING", prompt)
        self.assertIn("Old answer", prompt)
        self.assertIn("CURRENT_REASONING", prompt)


if __name__ == "__main__":
    unittest.main()
