"""PubHealthBench MCQ prompt strategy.

Mirrors the official zero-shot prompt used in UKHSA PubHealthBench.
"""

import re
from typing import Dict

from ..core.base import BasePromptStrategy
from ..core.registry import Registry


@Registry.register_prompt("pubhealthbench")
class PubHealthBenchMCQPromptStrategy(BasePromptStrategy):
    """Prompt strategy that matches the official PubHealthBench MCQ template."""

    SYSTEM_PROMPT = "You are an expert working for a Public Health agency."

    PROMPT_TEMPLATE = (
        "The following are multiple choice questions (with answers) about UK Government public health guidance.\n\n"
        "Question: This question relates to UK Health Security Agency (UKHSA) guidance that could be found on the "
        "gov.uk website as of 08/01/2025.\n\n"
        "{question}\n"
        "Options:\n"
        "{options_formatted}\n\n"
        "Provide the letter (A, B, C, D, E, F, or G) of the correct answer. "
        'You should state "The answer is (X)", where the X contained in the brackets is the correct letter choice, '
        "make sure you include the brackets () around your final answer in your response. "
        "DO NOT provide any other information or text in your response.\n\n"
        "Answer: "
    )

    def __init__(self, name: str = "pubhealthbench", **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _resolve_question_and_options(self, inputs: Dict[str, Any]) -> tuple[str, str]:
        metadata = inputs.get("metadata") or {}
        question = metadata.get("question")
        options_formatted = metadata.get("options_formatted")

        if not question or not options_formatted:
            text = inputs.get("text", "")
            if "\n\n" in text:
                question, options_formatted = text.split("\n\n", 1)
            else:
                question = question or text
                options_formatted = options_formatted or ""

        return question.strip(), options_formatted.strip()

    def build_prompt(self, inputs: Dict[str, Any], **kwargs) -> str:
        question, options_formatted = self._resolve_question_and_options(inputs)
        return self.PROMPT_TEMPLATE.format(
            question=question,
            options_formatted=options_formatted,
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        result = {"answer": None, "reasoning": response, "raw_response": response}

        match = re.search(r"\(([A-G])\)", response, re.IGNORECASE)
        if match:
            result["answer"] = match.group(1).upper()
            return result

        match = re.search(r"\b([A-G])\b", response, re.IGNORECASE)
        if match:
            result["answer"] = match.group(1).upper()
            return result

        result["parse_error"] = True
        return result

    def get_prediction_field(self) -> str:
        return "answer"
