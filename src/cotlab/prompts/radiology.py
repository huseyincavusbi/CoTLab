"""Radiology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy
from ..core.registry import Registry

SYSTEM_ROLE = """You are a radiology expert specialised in paedeatric radiology.
Your goal is to identify incidence of pathological fractures in the given radiology report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""


PROMPT_TEMPLATE = """Follow this structured reasoning on the attached radiology report:

1. **Fracture Mention**: Determine whether a bone fracture is explicitly described in the report.
2. **Pathological Nature**: If a bone fracture is mentioned, assess the nature of the fracture, check if there is a strong, direct and explicit indication that this fracture is of pathological nature.

The instruction is to answer questions strictly based on the content of the provided radiology report without making any assumptions or future projections/possibilities.
Only make judgements on strong reason.
Ignore non-bone fractures (e.g., device or lead fractures) as they do not indicate bone pathology.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Fracture present and signs of pathological fracture
```json
{{
    "fracture_mentioned": true,
    "pathological_fracture": true,
    "evidence": {{
        "report_findings": ["bilateral clavicle fractures", "right clavicle fracture shows some periosteal reaction and callus formation"],
        "rationale": "The report explicitly mentions bilateral clavicle fractures. The description of the right clavicle fracture includes periosteal reaction and callus formation, which are indicative of a pathological fracture."
    }}
}}
```

Example 2: Fracture present and signs of non-pathological fracture
```json
{{
    "fracture_mentioned": true,
    "pathological_fracture": false,
    "evidence": {{
        "report_findings": ["displaced fracture of the right sixth posterolateral rib"],
        "rationale": "The report mentions a displaced rib fracture, but does not provide any information that suggests it is a pathological fracture. Therefore, I cannot conclude that it is a pathological fracture."
    }}
}}
```

Radiology report:
\"\"\"
{report}
\"\"\"
"""


@Registry.register_prompt("radiology")
class RadiologyPromptStrategy(BasePromptStrategy):
    """
    Structured JSON output for radiology pathological fracture detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with fracture_mentioned, pathological_fracture, evidence
    - Few-shot examples for format guidance
    """

    def __init__(self, name: str = "radiology", system_role: Optional[str] = None, **kwargs):
        self._name = name
        self.system_role = system_role or SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with radiology report."""
        report = input_data.get("text", input_data.get("report", input_data.get("question", "")))
        return PROMPT_TEMPLATE.format(report=report)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from model.

        Expected format:
        {
            "fracture_mentioned": bool,
            "pathological_fracture": bool,
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
            json_match = re.search(r'\{[^{}]*"fracture_mentioned"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        # Parse JSON
        try:
            parsed = json.loads(json_str)
            return {
                "answer": "pathological"
                if parsed.get("pathological_fracture")
                else "non-pathological",
                "fracture_mentioned": parsed.get("fracture_mentioned", False),
                "pathological_fracture": parsed.get("pathological_fracture", False),
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
        Radiology prompt is only compatible with radiology dataset.

        This prompt is specifically designed for pathological fracture detection
        in radiology reports and should NOT be used for general medical QA.
        """
        return ["radiology"]
