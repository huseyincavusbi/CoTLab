"""Cardiology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy
from ..core.registry import Registry

SYSTEM_ROLE = """You are a paediatric cardiology expert.
Your goal is to identify congenital heart defects in the given cardiac imaging report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""


PROMPT_TEMPLATE = """Follow this structured reasoning on the attached cardiac imaging report:

1. **Cardiac Abnormality**: Determine whether a structural cardiac abnormality is explicitly described.
2. **Congenital Heart Defect**: If an abnormality is found, assess whether it constitutes a congenital heart defect.

The instruction is to answer questions strictly based on the content of the provided cardiac report without making any assumptions.
Only make judgements on strong evidence.
Ignore physiological variants (e.g., patent foramen ovale, trivial regurgitation) as they do not indicate CHD.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Congenital heart defect present
```json
{{
    "cardiac_abnormality": true,
    "congenital_heart_defect": true,
    "evidence": {{
        "report_findings": ["ventricular septal defect", "left-to-right shunt", "pulmonary artery pressure elevated"],
        "rationale": "The report explicitly identifies a large ventricular septal defect with hemodynamically significant shunt causing elevated pulmonary pressures, consistent with congenital heart disease requiring intervention."
    }}
}}
```

Example 2: Normal cardiac findings
```json
{{
    "cardiac_abnormality": false,
    "congenital_heart_defect": false,
    "evidence": {{
        "report_findings": ["normal cardiac structure", "physiological tricuspid regurgitation"],
        "rationale": "The report describes a structurally normal heart with only physiological findings. No evidence of congenital heart defect."
    }}
}}
```

Cardiac imaging report:
\"\"\"
{report}
\"\"\"
"""


@Registry.register_prompt("cardiology")
class CardiologyPromptStrategy(BasePromptStrategy):
    """
    Structured JSON output for paediatric cardiology CHD detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with cardiac_abnormality, congenital_heart_defect, evidence
    - Few-shot examples for format guidance
    """

    def __init__(self, name: str = "cardiology", system_role: Optional[str] = None, **kwargs):
        self._name = name
        self.system_role = system_role or SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with cardiac imaging report."""
        report = input_data.get("text", input_data.get("report", input_data.get("question", "")))
        return PROMPT_TEMPLATE.format(report=report)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from model.

        Expected format:
        {
            "cardiac_abnormality": bool,
            "congenital_heart_defect": bool,
            "evidence": {
                "report_findings": [...],
                "rationale": "..."
            }
        }
        """
        # Try to extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(
                r'\{[^{}]*"congenital_heart_defect"[^{}]*\}', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        # Parse JSON
        try:
            parsed = json.loads(json_str)
            return {
                "answer": "CHD present" if parsed.get("congenital_heart_defect") else "no CHD",
                "cardiac_abnormality": parsed.get("cardiac_abnormality", False),
                "congenital_heart_defect": parsed.get("congenital_heart_defect", False),
                "reasoning": parsed.get("evidence", {}).get("rationale", ""),
                "findings": parsed.get("evidence", {}).get("report_findings", []),
                "raw": response,
                "parsed_json": parsed,
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "answer": response.strip(),
                "reasoning": response,
                "raw": response,
                "parse_error": True,
            }

    def get_system_message(self) -> Optional[str]:
        return self.system_role

    def get_compatible_datasets(self) -> list[str]:
        """
        Cardiology prompt is only compatible with cardiology dataset.

        This prompt is specifically designed for congenital heart defect detection
        in cardiac imaging reports and should NOT be used for general medical QA.
        """
        return ["cardiology"]
