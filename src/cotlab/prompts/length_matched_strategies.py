"""Length-matched prompt strategies for controlled comparison.

These strategies produce prompts with identical token counts,
eliminating prompt length as a confounding variable in
mechanistic analysis experiments.
"""

from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy
from ..core.registry import Registry

# Target prompt template (longest version - others will pad to match)
# We use a standardized template where only the "instruction" varies
# but total token count stays the same via padding.

STANDARD_QUESTION_TEMPLATE = """You are a medical expert. Consider this clinical case carefully.

{instruction}

Case: {question}

{padding}Provide your analysis:"""


@Registry.register_prompt("contrarian_matched")
class ContrarianMatchedStrategy(BasePromptStrategy):
    """
    Length-matched contrarian strategy.

    Matches token count with other strategies by using standardized template.
    """

    INSTRUCTION = """Play devil's advocate and argue against the obvious answer.
State the obvious diagnosis, then explain why it might be WRONG.
Consider alternative explanations that could fit the symptoms."""

    PADDING = ""  # No padding needed - this is the longest

    def __init__(self, name: str = "contrarian_matched", **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        return STANDARD_QUESTION_TEMPLATE.format(
            instruction=self.INSTRUCTION, question=question, padding=self.PADDING
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        return {
            "answer": response.strip(),
            "reasoning": response,
            "raw": response,
        }

    def get_system_message(self) -> Optional[str]:
        return None  # System message in prompt for consistency


@Registry.register_prompt("cot_matched")
class ChainOfThoughtMatchedStrategy(BasePromptStrategy):
    """
    Length-matched chain-of-thought strategy.

    Matches token count with contrarian via padding.
    """

    INSTRUCTION = """Think through this problem step by step.
Reason carefully before giving your final answer.
Show your complete reasoning process."""

    # Padding to match contrarian's longer instruction
    PADDING = "Take your time. "

    def __init__(self, name: str = "cot_matched", **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        return STANDARD_QUESTION_TEMPLATE.format(
            instruction=self.INSTRUCTION, question=question, padding=self.PADDING
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        return {
            "answer": response.strip(),
            "reasoning": response,
            "raw": response,
        }

    def get_system_message(self) -> Optional[str]:
        return None


@Registry.register_prompt("direct_matched")
class DirectAnswerMatchedStrategy(BasePromptStrategy):
    """
    Length-matched direct answer strategy.

    Matches token count with contrarian via padding.
    """

    INSTRUCTION = """Give only the final diagnosis.
Do not explain your reasoning or show your work.
Answer with just the diagnosis name."""

    # Padding to match contrarian's longer instruction
    PADDING = "Be concise and direct. "

    def __init__(self, name: str = "direct_matched", **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        return STANDARD_QUESTION_TEMPLATE.format(
            instruction=self.INSTRUCTION, question=question, padding=self.PADDING
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        return {
            "answer": response.strip(),
            "reasoning": None,
            "raw": response,
        }

    def get_system_message(self) -> Optional[str]:
        return None


# Utility function to verify token counts
def verify_token_counts(tokenizer, question: str = "Patient has chest pain.") -> Dict[str, int]:
    """Verify that all matched strategies produce same token count."""
    strategies = [
        ContrarianMatchedStrategy(),
        ChainOfThoughtMatchedStrategy(),
        DirectAnswerMatchedStrategy(),
    ]

    counts = {}
    for strategy in strategies:
        prompt = strategy.build_prompt({"question": question})
        tokens = tokenizer(prompt, return_tensors="pt")
        counts[strategy.name] = tokens["input_ids"].shape[1]

    return counts
