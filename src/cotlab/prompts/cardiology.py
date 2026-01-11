"""Cardiology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry

SYSTEM_ROLE = """You are a paediatric cardiology expert.
Your goal is to identify congenital heart defects in the given cardiac imaging report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""

SYSTEM_ROLE_CONTRARIAN = """You are a skeptical paediatric cardiology expert.
Your goal is to identify congenital heart defects in the given cardiac imaging report.
However, you must question obvious conclusions and consider alternative explanations.
Think rationally, play devil's advocate, and explain your reasoning."""


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

Response:
"""

PROMPT_TEMPLATE_CONTRARIAN = """As a skeptical cardiologist, follow this structured reasoning on the attached cardiac imaging report.
Question obvious patterns and consider alternative explanations before reaching your conclusion.

1. **Cardiac Abnormality**: Determine whether a structural cardiac abnormality is explicitly described. Consider if what appears abnormal might be a normal variant for age.
2. **Congenital Heart Defect**: If an abnormality is found, critically assess whether it constitutes a congenital heart defect. Question the obvious diagnosis - could there be alternative explanations?

Apply skeptical reasoning - if the report suggests CHD, argue why it might NOT be CHD. If it seems normal, consider why it MIGHT indicate CHD.
Only make final judgements when evidence is overwhelming and alternative explanations are ruled out.
Ignore physiological variants (e.g., patent foramen ovale, trivial regurgitation) as they do not indicate CHD.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Congenital heart defect present (after skeptical review)
```json
{{
    "cardiac_abnormality": true,
    "congenital_heart_defect": true,
    "evidence": {{
        "report_findings": ["ventricular septal defect", "left-to-right shunt", "elevated pulmonary pressures"],
        "rationale": "Initial skepticism: Could the shunt be physiological? However, the combination of large VSD with hemodynamically significant shunt causing elevated pulmonary pressures provides overwhelming evidence. Alternative explanations (innocent murmur, transient finding) are ruled out by the severity and persistence. Conclusion: CHD confirmed despite initial skepticism."
    }}
}}
```

Example 2: Normal cardiac findings (skeptical analysis)
```json
{{
    "cardiac_abnormality": false,
    "congenital_heart_defect": false,
    "evidence": {{
        "report_findings": ["normal cardiac structure", "physiological tricuspid regurgitation"],
        "rationale": "Applying skeptical reasoning: While trivial TR could suggest valve pathology, the report explicitly states it is physiological. Playing devil's advocate against a CHD diagnosis: structurally normal heart with only age-appropriate physiological findings. No features suggesting congenital abnormality. Cannot conclude CHD without stronger evidence."
    }}
}}
```

Cardiac imaging report:
\"\"\"
{report}
\"\"\"

Response:
"""

PROMPT_TEMPLATE_ANSWER_FIRST = """Review this cardiac imaging report and provide immediate assessment.

**Step 1 - Initial Diagnosis**: Based on your immediate review, state your initial conclusion about the presence of congenital heart defect (CHD).

Initial Assessment: Is there a congenital heart defect? (State YES or NO immediately)

**Step 2 - Evidence Collection**: Now systematically gather and evaluate evidence:
- What findings SUPPORT your initial diagnosis?
- What findings CONTRADICT your initial diagnosis?
- Are there alternative explanations?

**Step 3 - Final Diagnosis**: Based on the evidence collected, confirm or revise your initial assessment.

Provide your response in JSON format with the following structure:
{{
    "cardiac_abnormality": true or false,
    "congenital_heart_defect": true or false,
    "evidence": {{
        "report_findings": [...list of relevant findings...],
        "rationale": "Your reasoning including initial assessment, supporting/contradicting evidence, and final conclusion"
    }}
}}

Follow the format of these two examples and give the output strictly in the json format.

Example 1: Initial YES, Confirmed CHD
```json
{{
    "cardiac_abnormality": true,
    "congenital_heart_defect": true,
    "evidence": {{
        "report_findings": ["ventricular septal defect", "left-to-right shunt"],
        "rationale": "Initial diagnosis: YES - CHD suspected. Supporting evidence: Large VSD with hemodynamically significant shunt strongly indicates congenital defect. Contradicting evidence: None identified. Final diagnosis: Confirmed congenital heart defect."
    }}
}}
```

Example 2: Initial NO, Confirmed no CHD
```json
{{
    "cardiac_abnormality": false,
    "congenital_heart_defect": false,
    "evidence": {{
        "report_findings": ["normal cardiac structure", "physiological TR"],
        "rationale": "Initial diagnosis: NO - appears normal. Supporting evidence: Structurally normal heart with only trace physiological regurgitation. Contradicting evidence: None suggesting structural defect. Final diagnosis: Confirmed no congenital heart defect."
    }}
}}
```

Cardiac imaging report:
\"\"\"
{report}
\"\"\"

Response:
"""


@Registry.register_prompt("cardiology")
class CardiologyPromptStrategy(StructuredOutputMixin, BasePromptStrategy):
    """
    Structured JSON output for paediatric cardiology CHD detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with cardiac_abnormality, congenital_heart_defect, evidence
    - Few-shot examples for format guidance
    """

    def __init__(
        self,
        name: str = "cardiology",
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
        """Build prompt with cardiac imaging report."""
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

        return prompt

    def _convert_examples_to_format(self, template: str) -> str:
        """Convert JSON examples in template to target output format."""
        examples = [
            {
                "title": "Example 1: Congenital heart defect present",
                "data": {
                    "cardiac_abnormality": True,
                    "congenital_heart_defect": True,
                    "evidence": {
                        "report_findings": [
                            "ventricular septal defect",
                            "left-to-right shunt",
                            "pulmonary artery pressure elevated",
                        ],
                        "rationale": "The report explicitly identifies a large ventricular septal defect with hemodynamically significant shunt causing elevated pulmonary pressures, consistent with congenital heart disease requiring intervention.",
                    },
                    "_plain_answer": "CHD DETECTED",
                },
            },
            {
                "title": "Example 2: Normal cardiac findings",
                "data": {
                    "cardiac_abnormality": False,
                    "congenital_heart_defect": False,
                    "evidence": {
                        "report_findings": [
                            "normal cardiac structure",
                            "physiological tricuspid regurgitation",
                        ],
                        "rationale": "The report describes a structurally normal heart with only physiological findings. No evidence of congenital heart defect.",
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

        pattern = r"Example 1:.*?```\s*\n\nCardiac imaging report:"
        replacement = examples_str.strip() + "\n\nCardiac imaging report:"

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
        """
        Parse response from model (supports multiple formats).

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
        # Use mixin's multi-format parser if not JSON
        if self.output_format != "json":
            try:
                parsed = self._parse_formatted_response(response)
                # Map extracted values to standardized keys
                is_positive = False
                if self.output_format in ["plain", "markdown"]:
                    # Parse final answer string
                    ans_str = str(parsed.get("answer", "")).upper()
                    is_positive = "CHD DETECTED" in ans_str
                else:
                    is_positive = parsed.get("congenital_heart_defect", False)

                return {
                    "answer": "CHD present" if is_positive else "no CHD",
                    "cardiac_abnormality": parsed.get("cardiac_abnormality", is_positive),
                    "congenital_heart_defect": is_positive,
                    "reasoning": parsed.get("evidence", {}).get(
                        "rationale", parsed.get("reasoning", "")
                    ),
                    "findings": parsed.get("evidence", {}).get("report_findings", []),
                    "raw": response,
                    "parsed_json": parsed,
                }
            except Exception:
                pass  # Fall back to JSON parsing

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

    def get_prediction_field(self) -> str:
        """Return the JSON field name used for binary classification."""
        return "congenital_heart_defect"
