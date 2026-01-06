"""Neurology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy
from ..core.registry import Registry

SYSTEM_ROLE = """You are a paediatric neuroradiology expert.
Your goal is to identify neurological abnormalities in the given neuroimaging report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""


PROMPT_TEMPLATE = """Follow this structured reasoning on the attached neuroimaging report:

1. **Imaging Abnormality**: Determine whether a structural or functional brain abnormality is explicitly described.
2. **Neurological Significance**: If an abnormality is found, assess whether it represents a clinically significant neurological condition.

The instruction is to answer questions strictly based on the content of the provided neuroimaging report without making any assumptions.
Only make judgements on strong evidence.
Ignore normal variants (e.g., cavum septum pellucidum, benign enlarged subarachnoid spaces) as they do not indicate pathology.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Neurological abnormality present
```json
{{
    "imaging_abnormality": true,
    "neurological_abnormality": true,
    "evidence": {{
        "report_findings": ["dilated ventricles", "periventricular edema", "hydrocephalus"],
        "rationale": "The report explicitly identifies dilated lateral ventricles with transependymal CSF flow, consistent with hydrocephalus requiring intervention."
    }}
}}
```

Example 2: Normal neuroimaging findings
```json
{{
    "imaging_abnormality": false,
    "neurological_abnormality": false,
    "evidence": {{
        "report_findings": ["normal brain MRI", "age-appropriate myelination"],
        "rationale": "The report describes a structurally normal brain with appropriate myelination for age. No neurological abnormality identified."
    }}
}}
```

Neuroimaging report:
\"\"\"
{report}
\"\"\"
"""


@Registry.register_prompt("neurology")
class NeurologyPromptStrategy(BasePromptStrategy):
    """
    Structured JSON output for paediatric neurology abnormality detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with imaging_abnormality, neurological_abnormality, evidence
    - Few-shot examples for format guidance
    """

    def __init__(self, name: str = "neurology", system_role: Optional[str] = None, **kwargs):
        self._name = name
        self.system_role = system_role or SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with neuroimaging report."""
        report = input_data.get("text", input_data.get("report", input_data.get("question", "")))
        return PROMPT_TEMPLATE.format(report=report)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(
                r'\{[^{}]*"neurological_abnormality"[^{}]*\}', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        try:
            parsed = json.loads(json_str)
            return {
                "answer": "abnormality present"
                if parsed.get("neurological_abnormality")
                else "normal",
                "imaging_abnormality": parsed.get("imaging_abnormality", False),
                "neurological_abnormality": parsed.get("neurological_abnormality", False),
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
        """Neurology prompt is only compatible with neurology dataset."""
        return ["neurology"]
