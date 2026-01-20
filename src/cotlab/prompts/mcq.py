"""MCQ (Multiple Choice Question) prompt strategy for medical benchmarks.

This is a generic prompt strategy for MCQ-style datasets like MedQA, MMLU, MedMCQA.
"""

from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry


@Registry.register_prompt("mcq")
class MCQPromptStrategy(BasePromptStrategy, StructuredOutputMixin):
    """Generic MCQ prompt strategy for medical question answering."""

    SYSTEM_ROLE = """You are an expert medical professional taking a medical licensing examination.
You must carefully analyze each question and select the single best answer from the options provided.
Think through your reasoning step by step before selecting your final answer."""

    PROMPT_TEMPLATE = """## Medical Question

{question_text}

## Instructions

Analyze this medical question carefully. Consider:
1. Key clinical findings and patient presentation
2. Relevant pathophysiology and mechanisms
3. Standard diagnostic and treatment guidelines
4. Why each option is correct or incorrect

After your analysis, select the single best answer from the provided options.

{format_instructions}"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": """A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. Laboratory studies show elevated TSH and low free T4. Which of the following is the most likely diagnosis?

A) Graves' disease
B) Hashimoto's thyroiditis
C) Thyroid adenoma
D) Subacute thyroiditis""",
            "reasoning": "The patient presents with classic symptoms of hypothyroidism: fatigue, weight gain, and cold intolerance. The lab findings confirm primary hypothyroidism with elevated TSH (the pituitary is trying to stimulate the thyroid) and low free T4 (the thyroid is not producing enough hormone). Hashimoto's thyroiditis is the most common cause of primary hypothyroidism in iodine-sufficient areas. Graves' disease causes hyperthyroidism. Thyroid adenoma typically presents as a nodule and doesn't usually cause hypothyroidism. Subacute thyroiditis can cause transient hypothyroidism but typically presents with pain and follows a viral illness.",
            "answer": "B",
        },
        {
            "question": """A 60-year-old man with a history of smoking presents with hemoptysis and a 3 cm lung mass on chest CT. Biopsy shows small round blue cells. Which of the following is the most appropriate next step?

A) Surgical resection
B) Radiation therapy alone
C) Chemotherapy with cisplatin and etoposide
D) Observation with repeat imaging in 3 months""",
            "reasoning": "The biopsy showing small round blue cells in a smoker with a lung mass is diagnostic of small cell lung cancer (SCLC). SCLC is highly aggressive but initially chemosensitive. Unlike non-small cell lung cancer, surgical resection is not typically indicated for SCLC because it has usually metastasized by the time of diagnosis. The standard first-line treatment is combination chemotherapy with cisplatin (or carboplatin) and etoposide, often combined with radiation for limited-stage disease. Observation would be inappropriate given the aggressive nature of this malignancy.",
            "answer": "C",
        },
    ]

    def __init__(
        self,
        name: str = "mcq",
        few_shot: bool = True,
        output_format: str = "json",
        answer_first: bool = False,
        **kwargs,
    ):
        self._name = name
        self.few_shot = few_shot
        self.output_format = output_format
        self.answer_first = answer_first

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
        use_few_shot = few_shot if few_shot is not None else self.few_shot

        # Get the question text (already formatted with options by the dataset loader)
        question_text = inputs.get("text", "")

        # Build format instructions based on output format
        format_instructions = self._get_format_instructions()

        # Build few-shot examples if enabled
        examples_str = ""
        if use_few_shot:
            examples_str = self._build_few_shot_examples()

        # Construct the prompt
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
                return """Respond with a JSON object in this exact format:
```json
{"answer": "X", "reasoning": "Your step-by-step explanation"}
```
Where X is the letter (e.g., A, B, C, D, ...) of the correct answer."""
            else:
                return """Respond with a JSON object in this exact format:
```json
{"reasoning": "Your step-by-step explanation", "answer": "X"}
```
Where X is the letter (e.g., A, B, C, D, ...) of the correct answer."""
        else:
            return "Provide your reasoning, then state your final answer as a single letter."

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract answer and reasoning."""
        import json
        import re

        result = {"answer": None, "reasoning": None, "raw_response": response}

        # Try JSON parsing first
        try:
            # Find JSON block
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                result["answer"] = parsed.get("answer", "").upper().strip()
                result["reasoning"] = parsed.get("reasoning", "")
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: Look for answer pattern
        answer_patterns = [
            r"(?:answer|selection|choice)[:\s]*([A-Z])",
            r"\b([A-Z])\s*(?:is|would be)\s+(?:the\s+)?(?:correct|best|right)",
            r"(?:^|\n)\s*([A-Z])\s*(?:\)|\.|\:)",
            r"\*\*([A-Z])\*\*",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["answer"] = match.group(1).upper()
                break

        # If still no answer, look for lone letter
        if not result["answer"]:
            letters = re.findall(r"\b([A-Z])\b", response)
            if letters:
                result["answer"] = letters[-1].upper()  # Take last mentioned

        result["reasoning"] = response
        return result

    def get_prediction_field(self) -> str:
        return "answer"
