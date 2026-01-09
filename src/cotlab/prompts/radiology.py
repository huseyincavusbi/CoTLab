"""Radiology-specific prompt strategy with JSON structured output."""

import json
import re
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry

SYSTEM_ROLE = """You are a radiology expert specialised in paedeatric radiology.
Your goal is to identify incidence of pathological fractures in the given radiology report.
Do not make assumptions or diagnoses from the text.
Think rationally and explain your reasoning."""

SYSTEM_ROLE_CONTRARIAN = """You are a skeptical radiology expert specialised in paedeatric radiology.
Your goal is to identify incidence of pathological fractures in the given radiology report.
However, you must question obvious conclusions and consider alternative explanations.
Think rationally, play devil's advocate, and explain your reasoning."""


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

PROMPT_TEMPLATE_CONTRARIAN = """As a skeptical diagnostician, follow this structured reasoning on the attached radiology report.
Question obvious patterns and consider alternative explanations before reaching your conclusion.

1. **Fracture Mention**: Determine whether a bone fracture is explicitly described in the report. Consider if what appears to be a fracture might be a normal variant or imaging artifact.
2. **Pathological Nature**: If a bone fracture is mentioned, critically assess whether it is truly pathological. Question the obvious diagnosis - could there be alternative explanations for the findings?

The instruction is to answer questions strictly based on the content of the provided radiology report.
Apply skeptical reasoning - if the report suggests a pathological fracture, argue why it might NOT be pathological. If it seems non-pathological, consider why it MIGHT be pathological.
Only make final judgements when the evidence is overwhelming and alternative explanations are ruled out.
Ignore non-bone fractures (e.g., device or lead fractures) as they do not indicate bone pathology.
Follow the format of these two examples and give the output strictly in the json format.

Example 1: Fracture present and signs of pathological fracture (after skeptical review)
```json
{{
    "fracture_mentioned": true,
    "pathological_fracture": true,
    "evidence": {{
        "report_findings": ["bilateral clavicle fractures", "right clavicle fracture shows some periosteal reaction and callus formation"],
        "rationale": "Initial skepticism: Could these findings represent normal bone remodeling? However, the bilateral nature and explicit mention of periosteal reaction and callus formation in the context of reported pathology provide overwhelming evidence. Alternative explanations (trauma, normal variant) are ruled out by the report's clinical context. Conclusion: pathological fracture confirmed despite initial skepticism."
    }}
}}
```

Example 2: Fracture present but questioning pathological nature (skeptical analysis)
```json
{{
    "fracture_mentioned": true,
    "pathological_fracture": false,
    "evidence": {{
        "report_findings": ["displaced fracture of the right sixth posterolateral rib"],
        "rationale": "The report mentions a displaced rib fracture. Applying skeptical reasoning: While displacement might suggest pathology, it could equally result from mechanical trauma. The report provides no explicit indicators of metabolic bone disease, no mention of osteopenia, no description of abnormal bone architecture. Playing devil's advocate against a pathological diagnosis: the lack of corroborating features means this is more likely traumatic. Cannot conclude pathological fracture without stronger evidence."
    }}
}}
```

Radiology report:
\"\"\"
{report}
\"\"\"
"""

PROMPT_TEMPLATE_ANSWER_FIRST = """Review this radiology report and provide immediate assessment.

**Step 1 - Initial Diagnosis**: Based on your immediate review, state your initial conclusion about the presence of a pathological fracture.

Initial Assessment: Is this a pathological fracture? (State YES or NO immediately)

**Step 2 - Evidence Collection**: Now systematically gather and evaluate evidence:
- What findings SUPPORT your initial diagnosis?
- What findings CONTRADICT your initial diagnosis?
- Are there alternative explanations?

**Step 3 - Final Diagnosis**: Based on the evidence collected, confirm or revise your initial assessment.

Provide your response in JSON format with the following structure:
{{
    "fracture_mentioned": true or false,
    "pathological_fracture": true or false,
    "evidence": {{
        "report_findings": [...list of relevant findings...],
        "rationale": "Your reasoning including initial assessment, supporting/contradicting evidence, and final conclusion"
    }}
}}

Follow the format of these two examples and give the output strictly in the json format.

Example 1: Initial YES, Confirmed pathological
```json
{{
    "fracture_mentioned": true,
    "pathological_fracture": true,
    "evidence": {{
        "report_findings": ["bilateral clavicle fractures", "periosteal reaction", "callus formation"],
        "rationale": "Initial diagnosis: YES - pathological fracture suspected. Supporting evidence: bilateral nature of fractures and periosteal reaction strongly indicate pathological process. Contradicting evidence: None identified. Final diagnosis: Confirmed pathological fracture."
    }}
}}
```

Example 2: Initial NO, Confirmed non-pathological
```json
{{
    "fracture_mentioned": true,
    "pathological_fracture": false,
    "evidence": {{
        "report_findings": ["displaced rib fracture"],
        "rationale": "Initial diagnosis: NO - likely traumatic. Supporting evidence: single displaced rib fracture with no additional pathological features. Contradicting evidence: None suggesting metabolic disease. Final diagnosis: Confirmed non-pathological fracture."
    }}
}}
```

Radiology report:
\"\"\"
{report}
\"\"\"
"""


@Registry.register_prompt("radiology")
class RadiologyPromptStrategy(StructuredOutputMixin, BasePromptStrategy):
    """
    Structured JSON output for radiology pathological fracture detection.

    Uses structured JSON output format with:
    - Clear step-by-step reasoning instructions
    - JSON output with fracture_mentioned, pathological_fracture, evidence
    - Few-shot examples for format guidance
    """

    def __init__(
        self,
        name: str = "radiology",
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
        # Choose system role based on contrarian mode
        if system_role:
            self.system_role = system_role
        else:
            self.system_role = SYSTEM_ROLE_CONTRARIAN if contrarian else SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with radiology report."""
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
        # Example data for radiology
        examples = [
            {
                "title": "Example 1: Fracture present and signs of pathological fracture",
                "data": {
                    "fracture_mentioned": True,
                    "pathological_fracture": True,
                    "evidence": {
                        "report_findings": [
                            "bilateral clavicle fractures",
                            "periosteal reaction and callus formation",
                        ],
                        "rationale": "The report explicitly mentions bilateral clavicle fractures with periosteal reaction and callus formation, indicative of a pathological fracture.",
                    },
                    "_plain_answer": "PATHOLOGICAL",
                },
            },
            {
                "title": "Example 2: Fracture present but non-pathological",
                "data": {
                    "fracture_mentioned": True,
                    "pathological_fracture": False,
                    "evidence": {
                        "report_findings": [
                            "displaced fracture of the right sixth posterolateral rib"
                        ],
                        "rationale": "The report mentions a displaced rib fracture, but provides no information suggesting it is pathological.",
                    },
                    "_plain_answer": "NON-PATHOLOGICAL",
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

        pattern = r"Example 1:.*?```\s*\n\nRadiology report:"
        replacement = examples_str.strip() + "\n\nRadiology report:"

        new_template = re.sub(pattern, replacement, template, flags=re.DOTALL)

        # Also update the format instruction in the header
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

        # Remove everything from "Example 1:" to just before "Radiology report:" or similar
        # Match: Example 1: ... Example 2: ... up until the report section
        pattern = r"Example \d+:.*?(?=(?:Radiology report:|Cardiac imaging report:|Neuroimaging report:|Oncology report:))"
        cleaned = re.sub(pattern, "", template, flags=re.DOTALL)

        # Clean up instruction text that references examples
        cleaned = cleaned.replace("Follow the format of these two examples and give", "Give")
        cleaned = cleaned.replace("follow the format of these two examples and give", "give")

        return cleaned

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse response from model (supports multiple formats).

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
        # Plain text parsing - extract FINAL ANSWER
        if self.output_format == "plain":
            # Look for FINAL ANSWER pattern
            final_match = re.search(
                r"FINAL ANSWER:\s*(PATHOLOGICAL|NON-PATHOLOGICAL|NO FRACTURE)",
                response,
                re.IGNORECASE,
            )
            if final_match:
                answer_text = final_match.group(1).upper()
                is_pathological = answer_text == "PATHOLOGICAL"
                has_fracture = answer_text != "NO FRACTURE"
                return {
                    "answer": "pathological" if is_pathological else "non-pathological",
                    "fracture_mentioned": has_fracture,
                    "pathological_fracture": is_pathological,
                    "reasoning": response,
                    "findings": [],
                    "raw": response,
                }
            # Fallback: look for pathological/non-pathological keywords
            if (
                "pathological fracture" in response.lower()
                and "non-pathological" not in response.lower()
            ):
                return {
                    "answer": "pathological",
                    "fracture_mentioned": True,
                    "pathological_fracture": True,
                    "reasoning": response,
                    "raw": response,
                }

        # Use mixin's multi-format parser if not JSON/plain
        if self.output_format not in ("json", "plain"):
            try:
                parsed = self._parse_formatted_response(response)
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
            except Exception:
                pass  # Fall back to JSON parsing

        # Original JSON parsing logic
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[^{}]*"fracture_mentioned"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

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

    def get_prediction_field(self) -> str:
        """Return the JSON field name used for binary classification."""
        return "pathological_fracture"

