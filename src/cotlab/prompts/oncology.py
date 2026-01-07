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
        output_format: str = "json",
        **kwargs,
    ):
        self._name = name
        self.contrarian = contrarian
        self.few_shot = few_shot
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
        template = PROMPT_TEMPLATE_CONTRARIAN if self.contrarian else PROMPT_TEMPLATE

        if not self.few_shot:
            template = self._remove_few_shot_examples(template)

        prompt = template.format(report=report)

        if self.output_format != "json" and self.output_format != "plain":
            prompt += "\n\n" + self._add_format_instruction()

        return prompt

    def _remove_few_shot_examples(self, template: str) -> str:
        """Remove few-shot examples from template for ablation studies."""
        import re

        pattern = r"Example \d+:.*?(?=(?:Radiology report:|Cardiac imaging report:|Neuroimaging report:|Oncology report:))"
        cleaned = re.sub(pattern, "", template, flags=re.DOTALL)

        cleaned = cleaned.replace("Follow the format of these two examples and give", "Give")
        cleaned = cleaned.replace("follow the format of these two examples and give", "give")

        return cleaned

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
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
            return {
                "answer": "malignancy present" if parsed.get("malignancy") else "benign",
                "abnormal_findings": parsed.get("abnormal_findings", False),
                "malignancy": parsed.get("malignancy", False),
                "reasoning": parsed.get("evidence", {}).get("rationale", ""),
                "findings": parsed.get("evidence", {}).get("report_findings", []),
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
