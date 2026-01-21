"""Oncology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry

SYSTEM_ROLE = """You are a paediatric oncology expert.
Your goal is to identify malignancies in the given oncology report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""

SYSTEM_ROLE_CONTRARIAN = """You are a skeptical paediatric oncology expert.
Your goal is to identify malignancies in the given oncology report.
However, you must question obvious conclusions and consider alternative explanations.
Think rationally, play devil's advocate, and explain your reasoning."""


PROMPT_TEMPLATE = """Follow this structured reasoning on the attached oncology report:

1. **Abnormal Findings**: Determine whether abnormal findings are explicitly described in the report.
2. **Malignancy Assessment**: If abnormal findings exist, assess whether they represent malignancy.

The instruction is to answer questions strictly based on the content of the provided oncology report without making any assumptions.
Only make judgements on strong evidence.
Ignore benign findings (e.g., reactive lymphadenopathy, resolved infections) as they do not indicate malignancy.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Malignancy present
```json
{{
    "abnormal_findings": true,
    "malignancy": true,
    "evidence": {{
        "report_findings": ["lymphoblasts 85% of marrow", "B-cell ALL immunophenotype", "bone marrow infiltration"],
        "rationale": "The report explicitly identifies bone marrow infiltration with lymphoblasts and immunophenotyping consistent with acute lymphoblastic leukemia, confirming malignancy."
    }}
}}
```

Example 2: No malignancy
```json
{{
    "abnormal_findings": false,
    "malignancy": false,
    "evidence": {{
        "report_findings": ["normal CBC", "age-appropriate values", "no blast cells"],
        "rationale": "The report shows normal blood counts with no atypical or malignant cells. No evidence of malignancy."
    }}
}}
```

Oncology report:
\"\"\"
{report}
\"\"\"

Response:
"""

PROMPT_TEMPLATE_CONTRARIAN = """As a skeptical oncologist, follow this structured reasoning on the attached oncology report.
Question obvious patterns and consider alternative explanations before reaching your conclusion.

1. **Abnormal Findings**: Determine whether abnormal findings are explicitly described. Consider if what appears abnormal might be reactive or benign.
2. **Malignancy Assessment**: If abnormal findings exist, critically assess whether they represent malignancy. Question the obvious - could there be benign explanations?

Apply skeptical reasoning - if the report suggests malignancy, argue why it might NOT be malignant. If it seems benign, consider why it MIGHT be malignant.
Only make final judgements when evidence is overwhelming and alternative explanations are ruled out.
Ignore benign findings (e.g., reactive lymphadenopathy, resolved infections) as they do not indicate malignancy.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Malignancy present (after skeptical review)
```json
{{
    "abnormal_findings": true,
    "malignancy": true,
    "evidence": {{
        "report_findings": ["lymphoblasts 85% of marrow", "B-cell ALL immunophenotype", "bone marrow infiltration"],
        "rationale": "Initial skepticism: Could high lymphocyte count be reactive? However, the overwhelming proportion of lymphoblasts (85%), specific immunophenotyping confirming B-cell ALL, and marrow infiltration pattern provide definitive evidence. Alternative benign explanations (viral infection, stress response) are ruled out by the phenotype. Conclusion: malignancy confirmed despite initial skepticism."
    }}
}}
```

Example 2: No malignancy (skeptical analysis)
```json
{{
    "abnormal_findings": false,
    "malignancy": false,
    "evidence": {{
        "report_findings": ["normal CBC", "age-appropriate values", "no blast cells"],
        "rationale": "Applying skeptical reasoning: Could early malignancy be masked by normal counts? Playing devil's advocate against a benign conclusion: the report shows completely normal blood counts with explicit absence of blast cells. No features suggesting occult malignancy. Cannot conclude malignancy without evidence of abnormal cells or pathological findings."
    }}
}}
```

Oncology report:
\"\"\"
{report}
\"\"\"

Response:
"""
PROMPT_TEMPLATE_ANSWER_FIRST = """Review this oncology report and provide immediate assessment.

**Step 1 - Initial Diagnosis**: Based on your immediate review, state your initial conclusion about the presence of malignancy.

Initial Assessment: Is there malignancy? (State YES or NO immediately)

**Step 2 - Evidence Collection**: Now systematically gather and evaluate evidence:
- What findings SUPPORT your initial diagnosis?
- What findings CONTRADICT your initial diagnosis?
- Are there alternative explanations?

**Step 3 - Final Diagnosis**: Based on the evidence collected, confirm or revise your initial assessment.

Provide your response in JSON format. Follow the format of these two examples and give the output strictly in the json format.

Example 1: Initial YES, Confirmed malignancy
```json
{{
    "abnormal_findings": true,
    "malignancy": true,
    "evidence": {{
        "report_findings": ["lymphoblasts 85%", "B-cell ALL immunophenotype"],
        "rationale": "Initial diagnosis: YES - malignancy suspected. Supporting evidence: High blast count with specific immunophenotyping confirms acute lymphoblastic leukemia. Contradicting evidence: None identified. Final diagnosis: Confirmed malignancy."
    }}
}}
```

Example 2: Initial NO, Confirmed no malignancy
```json
{{
    "abnormal_findings": false,
    "malignancy": false,
    "evidence": {{
        "report_findings": ["normal CBC", "no blast cells"],
        "rationale": "Initial diagnosis: NO - appears normal. Supporting evidence: Normal blood counts with no atypical cells. Contradicting evidence: None suggesting malignancy. Final diagnosis: Confirmed no malignancy."
    }}
}}
```

Oncology report:
\"\"\"
{report}
\"\"\"

Response:
"""


@Registry.register_prompt("oncology")
class OncologyPromptStrategy(StructuredOutputMixin, BasePromptStrategy):
    """
    Structured JSON output for paediatric oncology malignancy detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with abnormal_findings, malignancy, evidence
    - Few-shot examples for format guidance
    """

    def __init__(
        self,
        name: str = "oncology",
        system_role: Optional[str] = None,
        contrarian: bool = False,
        few_shot: bool = True,
        answer_first: bool = False,
        direct_answer: bool = False,
        output_format: str = "json",
        **kwargs,
    ):
        self._name = name
        self.contrarian = contrarian
        self.few_shot = few_shot
        self.answer_first = answer_first
        self.direct_answer = direct_answer
        self.output_format = output_format
        if system_role:
            self.system_role = system_role
        else:
            self.system_role = SYSTEM_ROLE_CONTRARIAN if contrarian else SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        report = input_data.get("text", input_data.get("report", input_data.get("question", "")))

        # Select base template based on reasoning mode
        if self.answer_first:
            template = PROMPT_TEMPLATE_ANSWER_FIRST
        elif self.contrarian:
            template = PROMPT_TEMPLATE_CONTRARIAN
        else:
            template = PROMPT_TEMPLATE

        # Remove examples if few_shot=False
        if not self.few_shot:
            template = self._remove_few_shot_examples(template)
        elif self.output_format != "json":
            # Convert JSON examples to target format
            template = self._convert_examples_to_format(template)

        prompt = template.format(report=report)

        if self.direct_answer:
            prompt += "\n\nProvide ONLY the final answer. Do not include reasoning."

        return prompt

    def _convert_examples_to_format(self, template: str) -> str:
        """Convert JSON examples in template to target output format."""
        examples = [
            {
                "title": "Example 1: Malignancy present",
                "data": {
                    "abnormal_findings": True,
                    "malignancy": True,
                    "evidence": {
                        "report_findings": [
                            "lymphoblasts 85% of marrow",
                            "B-cell ALL immunophenotype",
                            "bone marrow infiltration",
                        ],
                        "rationale": "The report explicitly identifies bone marrow infiltration with lymphoblasts and immunophenotyping consistent with acute lymphoblastic leukemia, confirming malignancy.",
                    },
                    "_plain_answer": "MALIGNANCY DETECTED",
                },
            },
            {
                "title": "Example 2: No malignancy",
                "data": {
                    "abnormal_findings": False,
                    "malignancy": False,
                    "evidence": {
                        "report_findings": [
                            "normal CBC",
                            "age-appropriate values",
                            "no blast cells",
                        ],
                        "rationale": "The report shows normal blood counts with no atypical or malignant cells. No evidence of malignancy.",
                    },
                    "_plain_answer": "NORMAL",
                },
            },
        ]

        # Build examples in target format
        examples_str = ""
        for ex in examples:
            examples_str += f"\n{ex['title']}\n"
            examples_str += self._format_example(ex["data"]) + "\n"

        # Replace JSON examples section
        import re

        pattern = r"Example 1:.*?```\s*\n\nOncology report:"
        replacement = examples_str.strip() + "\n\nOncology report:"

        new_template = re.sub(pattern, replacement, template, flags=re.DOTALL)

        # Update instruction in header
        if self.output_format == "plain":
            new_template = new_template.replace(
                "give the output strictly in the json format",
                "provide your answer in plain text with FINAL ANSWER: at the end",
            )
        else:
            new_template = new_template.replace(
                "give the output strictly in the json format",
                f"give the output in {self.output_format.upper()} format",
            )

        return new_template

    def _remove_few_shot_examples(self, template: str) -> str:
        """Remove few-shot examples from template for ablation studies."""
        import re

        pattern = r"Example \d+:.*?(?=(?:Radiology report:|Cardiac imaging report:|Neuroimaging report:|Oncology report:))"
        cleaned = re.sub(pattern, "", template, flags=re.DOTALL)

        cleaned = cleaned.replace("Follow the format of these two examples and give", "Give")
        cleaned = cleaned.replace("follow the format of these two examples and give", "give")

        return cleaned

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse response from model (supports multiple formats)."""
        # Use mixin's multi-format parser if not JSON
        if self.output_format != "json":
            try:
                parsed = self._parse_formatted_response(response)
                is_positive = False
                if self.output_format in ["plain", "markdown"]:
                    ans_str = str(parsed.get("answer", "")).upper()
                    is_positive = "MALIGNANCY DETECTED" in ans_str
                else:
                    is_positive = parsed.get("malignancy", False)

                return {
                    "answer": "malignancy present" if is_positive else "benign",
                    "abnormal_findings": parsed.get("abnormal_findings", is_positive),
                    "malignancy": is_positive,
                    "reasoning": parsed.get("evidence", {}).get(
                        "rationale", parsed.get("reasoning", "")
                    ),
                    "findings": parsed.get("evidence", {}).get("report_findings", []),
                    "raw": response,
                    "parsed_json": parsed,
                }
            except Exception:
                pass  # Fall back

        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[^{}]*"malignancy"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                # Handle case where model outputs a list of reasoning steps/options
                # Take the last item as the final conclusion
                if len(parsed) > 0:
                    parsed = parsed[-1]
                else:
                    parsed = {}

            evidence = parsed.get("evidence", {})
            if isinstance(evidence, list):
                if len(evidence) > 0 and isinstance(evidence[0], dict):
                    evidence = evidence[0]
                else:
                    evidence = {}

            return {
                "answer": "malignancy present" if parsed.get("malignancy") else "benign",
                "abnormal_findings": parsed.get("abnormal_findings", False),
                "malignancy": parsed.get("malignancy", False),
                "reasoning": evidence.get("rationale", ""),
                "findings": evidence.get("report_findings", []),
                "raw": response,
                "parsed_json": parsed,
            }
        except json.JSONDecodeError:
            return {
                "answer": response.strip(),
                "reasoning": response,
                "raw": response,
                "parse_error": True,
            }

    def get_system_message(self) -> Optional[str]:
        return self.system_role

    def get_compatible_datasets(self) -> list[str]:
        """Oncology prompt is only compatible with oncology dataset."""
        return ["oncology"]

    def get_prediction_field(self) -> str:
        """Return the JSON field name used for binary classification."""
        return "malignancy"
