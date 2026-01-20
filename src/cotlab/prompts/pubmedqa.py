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

    SYSTEM_ROLE_CONTRARIAN = """You are a skeptical peer reviewer evaluating a study's claims.
Be highly critical. Do not accept the abstract's conclusions unless the evidence in the context is irrefutable.
Look for limitations, confounding factors, or ambiguity that would make the answer 'maybe' instead of a definite 'yes' or 'no'."""

    PROMPT_TEMPLATE = """## Research Context
{text}

## Instructions

1. Read the research question and abstract context carefully.
2. Determine if the context provides enough evidence to answer the question.
3. If the answer is clearly supported by the findings, answer 'yes' or 'no'.
4. If the answer is ambiguous, not fully supported, or conditional, answer 'maybe'.

{format_instructions}"""

    FEW_SHOT_EXAMPLES = [
        {
            "text": """Question: Is laparoscopic appendectomy safe in pregnant women?

Context: A retrospective review of 300 pregnant women who underwent appendectomy (150 laparoscopic, 150 open). There was no significant difference in fetal loss or preterm delivery between groups. Laparoscopic group had shorter hospital stay and lower wound infection rates.""",
            "reasoning": "The context provides a direct comparison between laparoscopic and open appendectomy in a substantial cohort. The key safety outcomes (fetal loss, preterm delivery) showed no significant difference, while secondary outcomes favored laparoscopy. This supports an affirmative answer regarding safety.",
            "answer": "yes",
        },
        {
            "text": """Question: Does Vitamin D supplementation prevent cancer?

Context: A systematic review of 10 randomized trials found inconsistent results. Three trials showed benefit, four showed no effect, and three were inconclusive. Heterogeneity in dosage and population was high.""",
            "reasoning": "The context describes 'inconsistent results' from a systematic review, with some trials showing benefit and others not. This ambiguity and lack of definitive consensus means a simple 'yes' or 'no' is not fully supported by the provided evidence.",
            "answer": "maybe",
        },
    ]

    def __init__(
        self,
        name: str = "pubmedqa",
        output_format: str = "json",
        few_shot: bool = True,
        contrarian: bool = False,
        answer_first: bool = False,
        **kwargs,
    ):
        self._name = name
        self.output_format = output_format
        self.few_shot = few_shot
        self.contrarian = contrarian
        self.answer_first = answer_first

    @property
    def name(self) -> str:
        return self._name

    def get_system_prompt(self) -> str:
        return self.SYSTEM_ROLE_CONTRARIAN if self.contrarian else self.SYSTEM_ROLE

    def build_prompt(
        self,
        inputs: Dict[str, Any],
        few_shot: Optional[bool] = None,
        **kwargs,
    ) -> str:
        # Loaders provide text as "Question: ...\n\nContext: ..."
        text = inputs.get("text", "")

        use_few_shot = few_shot if few_shot is not None else self.few_shot
        format_instructions = self._get_format_instructions()

        # Build few-shot examples
        examples_str = ""
        if use_few_shot:
            examples_str = self._build_few_shot_examples()

        prompt = self.PROMPT_TEMPLATE.format(
            text=text,
            format_instructions=format_instructions,
        )

        if examples_str:
            prompt = f"## Examples\n\n{examples_str}\n\n{prompt}"

        return prompt

    def _build_few_shot_examples(self) -> str:
        examples = []
        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            if self.answer_first:
                example = f"### Example {i}\n\n{ex['text']}\n\n**Answer:** {ex['answer']}\n\n**Reasoning:** {ex['reasoning']}"
            else:
                example = f"### Example {i}\n\n{ex['text']}\n\n**Reasoning:** {ex['reasoning']}\n\n**Answer:** {ex['answer']}"
            examples.append(example)
        return "\n\n".join(examples)

    def _get_format_instructions(self) -> str:
        if self.output_format == "json":
            if self.answer_first:
                return """Respond with a JSON object in this exact format:
```json
{"answer": "yes/no/maybe", "reasoning": "Explain your finding based on the context"}
```
The "answer" field MUST be one of: "yes", "no", "maybe"."""
            else:
                return """Respond with a JSON object in this exact format:
```json
{"reasoning": "Explain your finding based on the context", "answer": "yes/no/maybe"}
```
The "answer" field MUST be one of: "yes", "no", "maybe"."""
        else:
            if self.answer_first:
                return "State your final answer as 'yes', 'no', or 'maybe', followed by your reasoning."
            else:
                return "Provide your reasoning, then state your final answer as 'yes', 'no', or 'maybe'."

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
