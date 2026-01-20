"""PubMedQA prompt strategy.

This strategy is designed for the PubMedQA dataset which requires answering
Yes/No/Maybe to research questions based on abstract contexts.
"""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry


@Registry.register_prompt("pubmedqa")
class PubMedQAPromptStrategy(BasePromptStrategy, StructuredOutputMixin):
    """Prompt strategy for PubMedQA (Yes/No/Maybe)."""

    SYSTEM_ROLE = """You are a medical researcher evaluating scientific studies.
You must answer the research question based ONLY on the provided context.
The valid answers are 'yes', 'no', or 'maybe'."""

    PROMPT_TEMPLATE = """## Research Context
{text}

## Instructions

1. Read the research question and abstract context carefully.
2. Determine if the context provides enough evidence to answer the question.
3. If the answer is clearly supported by the findings, answer 'yes' or 'no'.
4. If the answer is ambiguous, not fully supported, or conditional, answer 'maybe'.

{format_instructions}"""

    def __init__(
        self,
        name: str = "pubmedqa",
        output_format: str = "json",
        **kwargs,
    ):
        self._name = name
        self.output_format = output_format

    @property
    def name(self) -> str:
        return self._name

    def get_system_prompt(self) -> str:
        return self.SYSTEM_ROLE

    def build_prompt(
        self,
        inputs: Dict[str, Any],
        few_shot: Optional[bool] = None,
        **kwargs,
    ) -> str:
        # Loaders provide text as "Question: ...\n\nContext: ..."
        text = inputs.get("text", "")

        format_instructions = self._get_format_instructions()

        prompt = self.PROMPT_TEMPLATE.format(
            text=text,
            format_instructions=format_instructions,
        )
        return prompt

    def _get_format_instructions(self) -> str:
        if self.output_format == "json":
            return """Respond with a JSON object in this exact format:
```json
{"reasoning": "Explain your finding based on the context", "answer": "yes/no/maybe"}
```
The "answer" field MUST be one of: "yes", "no", "maybe"."""
        else:
            return (
                "Provide your reasoning, then state your final answer as 'yes', 'no', or 'maybe'."
            )

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract answer(yes/no/maybe)."""
        result = {"answer": None, "reasoning": None, "raw_response": response}

        # Clean response
        clean_response = response.strip()

        # Try JSON parsing
        try:
            json_match = re.search(r"\{[^{}]*\}", clean_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                ans = parsed.get("answer", "").lower().strip()
                if ans in ["yes", "no", "maybe"]:
                    result["answer"] = ans
                result["reasoning"] = parsed.get("reasoning", "")
                if result["answer"]:
                    return result
        except json.JSONDecodeError:
            pass

        # Fallback regex
        patterns = [
            r"answer[:\s]*[\"']?(yes|no|maybe)[\"']?",
            r"\b(yes|no|maybe)\b",
        ]

        for p in patterns:
            matches = re.findall(p, clean_response, re.IGNORECASE)
            if matches:
                # Take the last match as it's likely the conclusion
                result["answer"] = matches[-1].lower()
                break

        result["reasoning"] = response
        return result

    def get_prediction_field(self) -> str:
        return "answer"
