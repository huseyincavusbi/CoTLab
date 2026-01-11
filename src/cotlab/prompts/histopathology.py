"""Histopathology report quality rating prompt strategy."""

import json
import re
from typing import Any, Dict

from ..core.base import BasePromptStrategy
from ..core.registry import Registry

SYSTEM_ROLE = """You are a histopathology expert specializing in evaluating report quality.
Your goal is to assess the clinical accuracy, completeness, and clarity of histopathology reports.
Think rationally and explain your reasoning for quality ratings."""

SYSTEM_ROLE_CONTRARIAN = """You are a skeptical histopathology expert specializing in evaluating report quality.
Your goal is to critically assess histopathology reports and identify potential errors or omissions.
Question obvious conclusions and look for missing information. Think rationally and explain your reasoning."""


# Standard prompt with few-shot examples - INCLUDES SOURCE CASE FOR COMPARISON
PROMPT_TEMPLATE = """You are an expert histopathologist evaluating the quality of a GENERATED pathology report.

IMPORTANT: You must compare the generated report against the ORIGINAL CASE FINDINGS.
A report that looks well-written but does NOT match the original case should be rated 0 (Poor).

Rate on a scale of 0-2:
- 0 = Poor: Report does NOT match the original case, has major errors, or is misleading
- 1 = Partially correct: Matches some aspects of original case but incomplete or has minor errors
- 2 = Good: Accurately reflects the original case findings with proper terminology

Example 1 (Score: 0 - Poor - Well-written but WRONG diagnosis):
Original Case: "Invasive ductal carcinoma with lymph node metastasis"
Generated Report: "Benign fibrous lesion with no evidence of malignancy"
Rating: 0 - Report is well-structured but completely misses the malignancy. WRONG!

Example 2 (Score: 2 - Good - Matches original case):
Original Case: "Well-differentiated adenocarcinoma, pT2N0, clear margins"
Generated Report: "Sections show well-differentiated adenocarcinoma with glandular architecture. Margins clear. pT2N0."
Rating: 2 - Report accurately reflects the original case findings.

Example 3 (Score: 1 - Partial - Missing key details):
Original Case: "Hepatocellular carcinoma, grade 4/4, with vascular invasion"
Generated Report: "Hepatocellular carcinoma identified. Features consistent with malignancy."
Rating: 1 - Correct diagnosis but missing grade and vascular invasion status.

---

ORIGINAL CASE FINDINGS:
\"\"\"
{source_case}
\"\"\"

GENERATED REPORT TO EVALUATE:
\"\"\"
{report}
\"\"\"

Compare the generated report against the original case. Does it accurately reflect the findings?

Provide your response in JSON format:
```json
{{
    "quality_score": 0, 1, or 2,
    "reasoning": "Brief explanation comparing report to original case"
}}
```

Response:
"""

# Contrarian prompt - skeptical evaluation WITH source case comparison
PROMPT_TEMPLATE_CONTRARIAN = """You are a skeptical histopathologist reviewing a generated pathology report. Be critical and compare against the original case.

IMPORTANT: Compare the generated report against the ORIGINAL CASE FINDINGS.
Question any discrepancies. A well-written report that doesn't match the case is WRONG.

Look for:
- Does the diagnosis match the original case?
- Missing critical information from the original
- Incorrect or fabricated details
- Inappropriate terminology

Rate on a scale of 0-2:
- 0 = Poor: Does NOT match original case or has major errors
- 1 = Partially correct: Matches some but not all key findings
- 2 = Good: Accurately reflects original case

**Step 1**: Does the generated report match the ORIGINAL CASE?
**Step 2**: What key findings are missing or wrong?
**Step 3**: Final rating based on comparison.

ORIGINAL CASE FINDINGS:
\"\"\"
{source_case}
\"\"\"

GENERATED REPORT TO EVALUATE:
\"\"\"
{report}
\"\"\"

Provide your response in JSON format:
```json
{{
    "quality_score": 0, 1, or 2,
    "reasoning": "Brief explanation comparing report to original case"
}}
```

Response:
"""

# Answer-first prompt - conclude then justify WITH source case
PROMPT_TEMPLATE_ANSWER_FIRST = """You are an expert histopathologist evaluating a generated pathology report.

Compare the GENERATED REPORT against the ORIGINAL CASE FINDINGS.
First state your rating (0, 1, or 2), then explain why.

Rating scale:
- 0 = Poor: Does NOT match original case
- 1 = Partial: Matches some findings but incomplete
- 2 = Good: Accurately reflects original case

ORIGINAL CASE FINDINGS:
\"\"\"
{source_case}
\"\"\"

GENERATED REPORT:
\"\"\"
{report}
\"\"\"

Provide your response in JSON format:
```json
{{
    "quality_score": <0, 1, or 2>,
    "reasoning": "Justification comparing report to original case"
}}
```

Response:
"""


@Registry.register_prompt("histopathology")
class HistopathologyPromptStrategy(BasePromptStrategy):
    """Prompt strategy for histopathology report quality rating.

    3-class classification:
    - 0: Poor/incorrect
    - 1: Partially correct
    - 2: Good/accurate

    Supports:
    - few_shot: Include examples in prompt
    - answer_first: Rate first, then explain
    - contrarian: Skeptical evaluation mode
    """

    def __init__(
        self,
        name: str = "histopathology",
        output_format: str = "json",
        few_shot: bool = True,
        answer_first: bool = False,
        contrarian: bool = False,
        **kwargs,
    ):
        self._name = name
        self.output_format = output_format.lower()
        self.few_shot = few_shot
        self.answer_first = answer_first
        self.contrarian = contrarian

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with histopathology report and source case."""
        report = input_data.get("text", input_data.get("report", ""))
        # Get source case from metadata for comparison
        metadata = input_data.get("metadata", {})
        source_case = metadata.get("ground_truth", "Not provided")

        # Select template (priority: answer_first > contrarian > standard)
        if self.answer_first:
            template = PROMPT_TEMPLATE_ANSWER_FIRST
        elif self.contrarian:
            template = PROMPT_TEMPLATE_CONTRARIAN
        else:
            template = PROMPT_TEMPLATE

        # Format with both source case and report (all templates now use source_case)
        prompt = template.format(report=report, source_case=source_case)

        # Remove examples if few_shot=False
        if not self.few_shot and not self.answer_first and not self.contrarian:
            prompt = self._remove_few_shot_examples(prompt)

        return prompt

    def _remove_few_shot_examples(self, template: str) -> str:
        """Remove few-shot examples from template."""
        # Find and remove example sections
        lines = template.split("\n")
        filtered = []
        skip = False
        for line in lines:
            if line.startswith("Example 1") or line.startswith("Example 2"):
                skip = True
            elif skip and line.startswith("Histopathology Report"):
                skip = False
            if not skip:
                filtered.append(line)
        return "\n".join(filtered)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse response to extract quality score."""
        result = {
            "raw_response": response,
            "parse_success": False,
            "quality_score": None,
        }

        # Try to extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result.update(parsed)
                result["parse_success"] = True
            except json.JSONDecodeError:
                pass

        # Fallback: try direct JSON parse
        if not result["parse_success"]:
            try:
                parsed = json.loads(response)
                result.update(parsed)
                result["parse_success"] = True
            except json.JSONDecodeError:
                pass

        # Fallback: find quality_score in text
        if not result["parse_success"] or result.get("quality_score") is None:
            score_match = re.search(r'"quality_score"\s*:\s*(\d)', response)
            if score_match:
                result["quality_score"] = int(score_match.group(1))
                result["parse_success"] = True
            else:
                # Last resort: find any 0, 1, or 2 alone
                simple_match = re.search(r"\b([012])\b", response)
                if simple_match:
                    result["quality_score"] = int(simple_match.group(1))
                    result["parse_success"] = True

        return result

    def get_system_message(self) -> str:
        """Return mode-appropriate system message."""
        if self.contrarian:
            return SYSTEM_ROLE_CONTRARIAN
        return SYSTEM_ROLE

    def get_compatible_datasets(self) -> list[str]:
        """Histopathology prompt works with histopathology dataset."""
        return ["histopathology"]

    def get_prediction_field(self) -> str:
        """Return the JSON field name used for classification."""
        return "quality_score"
