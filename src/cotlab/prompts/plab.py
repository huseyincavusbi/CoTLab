"""PLAB MCQ prompt strategy.

Prompting strategy tailored for PLAB-style single-best-answer clinical questions.
"""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry


@Registry.register_prompt("plab")
class PLABPromptStrategy(BasePromptStrategy, StructuredOutputMixin):
    """Prompt strategy for PLAB-style medical licensing MCQs."""

    SYSTEM_ROLE = """You are an expert clinician taking the UK PLAB medical licensing exam.
Select the single best answer from the options using safe, guideline-consistent clinical reasoning."""

    SYSTEM_ROLE_CONTRARIAN = """You are a skeptical clinician reviewing a PLAB-style MCQ.
Critically evaluate distractors and common exam traps before choosing the single best answer."""

    PROMPT_TEMPLATE = """## PLAB Clinical Question

{question_text}

## Instructions

1. Focus on the most likely diagnosis/management based on the stem.
2. Consider exam distractors and select the single best answer.
3. Keep reasoning concise and clinically grounded.

{format_instructions}"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": """A 24-year-old woman at 10 weeks gestation presents with persistent vomiting, ketonuria, and weight loss. What is the most likely diagnosis?

A) Acute gastroenteritis
B) Hyperemesis gravidarum
C) Urinary tract infection
D) Peptic ulcer disease""",
            "reasoning": "Early pregnancy with severe vomiting, ketonuria, and weight loss is classic for hyperemesis gravidarum rather than simple nausea/vomiting of pregnancy or non-obstetric causes.",
            "answer": "B",
        },
        {
            "question": """A 68-year-old man has sudden, painless complete vision loss in one eye. Fundoscopy shows a pale retina with a cherry-red spot. What is the diagnosis?

A) Central retinal artery occlusion
B) Retinal detachment
C) Optic neuritis
D) Acute angle-closure glaucoma""",
            "reasoning": "Painless sudden monocular vision loss with a cherry-red spot indicates central retinal artery occlusion.",
            "answer": "A",
        },
    ]

    def __init__(
        self,
        name: str = "plab",
        few_shot: bool = False,
        output_format: str = "json",
        answer_first: bool = False,
        contrarian: bool = False,
        **kwargs,
    ):
        self._name = name
        self.few_shot = few_shot
        self.output_format = output_format
        self.answer_first = answer_first
        self.contrarian = contrarian

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
        use_few_shot = few_shot if few_shot is not None else self.few_shot
        question_text = inputs.get("text", "")
        format_instructions = self._get_format_instructions()
        examples_str = self._build_few_shot_examples() if use_few_shot else ""

        prompt = self.PROMPT_TEMPLATE.format(
            question_text=question_text,
            format_instructions=format_instructions,
        )
        if examples_str:
            prompt = f"## Examples\n\n{examples_str}\n\n{prompt}"
        return prompt

    def _build_few_shot_examples(self) -> str:
        examples = []
        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            if self.answer_first:
                example = f"### Example {i}\n\n{ex['question']}\n\n**Answer:** {ex['answer']}\n\n**Reasoning:** {ex['reasoning']}"
            else:
                example = f"### Example {i}\n\n{ex['question']}\n\n**Reasoning:** {ex['reasoning']}\n\n**Answer:** {ex['answer']}"
            examples.append(example)
        return "\n\n".join(examples)

    def _get_format_instructions(self) -> str:
        if self.output_format == "json":
            if self.answer_first:
                return """Respond with JSON only:
```json
{"answer": "X", "reasoning": "Brief clinical justification"}
```
Where X is a single option letter (A-G)."""
            return """Respond with JSON only:
```json
{"reasoning": "Brief clinical justification", "answer": "X"}
```
Where X is a single option letter (A-G)."""

        if self.answer_first:
            return "Give your final choice first as `The answer is (X)`, then a brief reason."
        return "Give a brief reason, then final choice as `The answer is (X)`."

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response and extract answer letter."""
        result = {"answer": None, "reasoning": None, "raw_response": response}

        # JSON-first parsing
        try:
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                answer = str(parsed.get("answer", "")).strip().upper()
                if re.fullmatch(r"[A-G]", answer):
                    result["answer"] = answer
                result["reasoning"] = parsed.get("reasoning", "")
                if result["answer"]:
                    return result
        except json.JSONDecodeError:
            pass

        # Plain-text fallbacks
        patterns = [
            r"\(([A-G])\)",
            r"(?:final\s+answer|answer|selection|choice)\s*[:\-]?\s*([A-G])\b",
            r"\b([A-G])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["answer"] = match.group(1).upper()
                break

        result["reasoning"] = response
        return result

    def get_prediction_field(self) -> str:
        return "answer"
