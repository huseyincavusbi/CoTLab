"""Neurology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry

SYSTEM_ROLE = """You are a paediatric neuroradiology expert.
Your goal is to identify neurological abnormalities in the given neuroimaging report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""

SYSTEM_ROLE_CONTRARIAN = """You are a skeptical paediatric neuroradiology expert.
Your goal is to identify neurological abnormalities in the given neuroimaging report.
However, you must question obvious conclusions and consider alternative explanations.
Think rationally, play devil's advocate, and explain your reasoning."""


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

PROMPT_TEMPLATE_CONTRARIAN = """As a skeptical neuroradiologist, follow this structured reasoning on the attached neuroimaging report.
Question obvious patterns and consider alternative explanations before reaching your conclusion.

1. **Imaging Abnormality**: Determine whether a structural or functional brain abnormality is explicitly described. Consider if what appears abnormal might be a normal variant or artifact.
2. **Neurological Significance**: If an abnormality is found, critically assess whether it represents clinically significant pathology. Question the obvious - could there be alternative explanations?

Apply skeptical reasoning - if the report suggests abnormality, argue why it might NOT be pathological. If it seems normal, consider why it MIGHT indicate pathology.
Only make final judgements when evidence is overwhelming and alternative explanations are ruled out.
Ignore normal variants (e.g., cavum septum pellucidum, benign enlarged spaces) as they do not indicate pathology.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Neurological abnormality present (after skeptical review)
```json
{{
    "imaging_abnormality": true,
    "neurological_abnormality": true,
    "evidence": {{
        "report_findings": ["dilated ventricles", "periventricular edema", "hydrocephalus"],
        "rationale": "Initial skepticism: Could enlarged ventricles be a normal variant for age? However, the explicit mention of transependymal CSF flow and periventricular edema provides overwhelming evidence of active hydrocephalus. Alternative explanations (benign external hydrocephalus, measurement artifact) are ruled out by the edema. Conclusion: pathological hydrocephalus confirmed despite initial skepticism."
    }}
}}
```

Example 2: Normal neuroimaging (skeptical analysis)
```json
{{
    "imaging_abnormality": false,
    "neurological_abnormality": false,
    "evidence": {{
        "report_findings": ["normal brain MRI", "age-appropriate myelination"],
        "rationale": "Applying skeptical reasoning: Could subtle abnormalities be missed? Playing devil's advocate against a normal conclusion: the report explicitly states normal structure and appropriate myelination. No features suggesting pathology. Cannot conclude abnormality without evidence."
    }}
}}
```

Neuroimaging report:
\"\"\"
{report}
\"\"\"
"""
PROMPT_TEMPLATE_ANSWER_FIRST = """Review this neuroimaging report and provide immediate assessment.

**Step 1 - Initial Diagnosis**: Based on your immediate review, state your initial conclusion about the presence of significant neurological abnormality.

Initial Assessment: Is there a significant neurological abnormality? (State YES or NO immediately)

**Step 2 - Evidence Collection**: Now systematically gather and evaluate evidence:
- What findings SUPPORT your initial diagnosis?
- What findings CONTRADICT your initial diagnosis?
- Are there alternative explanations?

**Step 3 - Final Diagnosis**: Based on the evidence collected, confirm or revise your initial assessment.

Provide your response in JSON format. Follow the format of these two examples and give the output strictly in the json format.

Example 1: Initial YES, Confirmed abnormality
```json
{{
    "imaging_abnormality": true,
    "neurological_abnormality": true,
    "evidence": {{
        "report_findings": ["large subdural hematoma", "midline shift"],
        "rationale": "Initial diagnosis: YES - significant abnormality suspected. Supporting evidence: Large subdural collection with mass effect strongly indicates significant pathology. Contradicting evidence: None identified. Final diagnosis: Confirmed significant neurological abnormality."
    }}
}}
```

Example 2: Initial NO, Confirmed normal
```json
{{
    "imaging_abnormality": false,
    "neurological_abnormality": false,
    "evidence": {{
        "report_findings": ["normal brain parenchyma"],
        "rationale": "Initial diagnosis: NO - appears normal. Supporting evidence: No abnormal signal, mass, or structural abnormality. Contradicting evidence: None suggesting pathology. Final diagnosis: Confirmed no significant abnormality."
    }}
}}
```

Neuroimaging report:
\"\"\"
{report}
\"\"\"
"""


@Registry.register_prompt("neurology")
class NeurologyPromptStrategy(StructuredOutputMixin, BasePromptStrategy):
    """
    Structured JSON output for paediatric neurology abnormality detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with imaging_abnormality, neurological_abnormality, evidence
    - Few-shot examples for format guidance
    """

    def __init__(
        self,
        name: str = "neurology",
        system_role: Optional[str] = None,
        contrarian: bool = False,
        few_shot: bool = True,
        answer_first: bool = False,
        output_format: str = "json",
        **kwargs,
    ):
        self._name = name
        self.contrarian = contrarian
        self.few_shot = few_shot
        self.answer_first = answer_first
        self.output_format = output_format
        if system_role:
            self.system_role = system_role
        else:
            self.system_role = SYSTEM_ROLE_CONTRARIAN if contrarian else SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with neuroimaging report."""
        report = input_data.get("text", input_data.get("report", input_data.get("question", "")))

        # Select template based on reasoning mode (priority: answer_first > contrarian > standard)
        if self.answer_first:
            template = PROMPT_TEMPLATE_ANSWER_FIRST
        elif self.contrarian:
            template = PROMPT_TEMPLATE_CONTRARIAN
        else:
            template = PROMPT_TEMPLATE

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
