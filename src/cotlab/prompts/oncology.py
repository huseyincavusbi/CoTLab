"""Oncology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy
from ..core.registry import Registry

SYSTEM_ROLE = """You are a paediatric oncology expert.
Your goal is to identify malignancies in the given oncology report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""


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


@Registry.register_prompt("oncology")
class OncologyPromptStrategy(BasePromptStrategy):
    """
    Structured JSON output for paediatric oncology malignancy detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with abnormal_findings, malignancy, evidence
    - Few-shot examples for format guidance
    """

    def __init__(self, name: str = "oncology", system_role: Optional[str] = None, **kwargs):
        self._name = name
        self.system_role = system_role or SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with oncology report."""
        report = input_data.get("text", input_data.get("report", input_data.get("question", "")))
        return PROMPT_TEMPLATE.format(report=report)

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
