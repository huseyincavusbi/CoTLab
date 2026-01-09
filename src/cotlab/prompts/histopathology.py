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


# Standard prompt with few-shot examples
PROMPT_TEMPLATE = """You are an expert histopathologist evaluating the quality of a generated pathology report.

Rate the following histopathology report on a scale of 0-2:
- 0 = Poor/incorrect: Major errors, missing critical information, or misleading content
- 1 = Partially correct: Some accurate information but incomplete or minor errors
- 2 = Good/accurate: Clinically accurate, complete, and well-structured

Example 1 (Score: 2 - Good):
```
**Histological Findings:** Sections show a well-differentiated adenocarcinoma with glandular architecture. The tumor infiltrates the muscularis propria. Margins are clear. No lymphovascular invasion identified.
**Diagnosis:** Moderately differentiated adenocarcinoma, pT2N0.
```
Rating: 2 - Complete findings, proper terminology, clear diagnosis.

Example 2 (Score: 0 - Poor):
```
The tissue looks abnormal with some cells that appear different. There might be something wrong but hard to tell.
```
Rating: 0 - Vague findings, no medical terminology, no diagnosis.

Histopathology Report to Rate:
\"\"\"
{report}
\"\"\"

Analyze the report for clinical accuracy, completeness, terminology, and clarity.

Provide your response in JSON format:
```json
{{
    "quality_score": 0, 1, or 2,
    "reasoning": "Brief explanation of your rating"
}}
```
"""

# Contrarian prompt - more skeptical evaluation
PROMPT_TEMPLATE_CONTRARIAN = """You are a skeptical histopathologist reviewing a generated pathology report. Be critical and look for errors.

Question the accuracy of this report. Look for:
- Missing critical information
- Potential diagnostic errors
- Incomplete descriptions
- Inappropriate terminology

Rate on a scale of 0-2:
- 0 = Poor/incorrect
- 1 = Partially correct
- 2 = Good/accurate

**Step 1 - Initial Impression**: What is your gut reaction to this report's quality?
**Step 2 - Critical Analysis**: What problems or omissions do you notice?
**Step 3 - Final Rating**: Based on critical review, what score does it deserve?

Histopathology Report:
\"\"\"
{report}
\"\"\"

Provide your response in JSON format:
```json
{{
    "quality_score": 0, 1, or 2,
    "reasoning": "Brief explanation of your rating"
}}
```
"""

# Answer-first prompt - conclude then justify
PROMPT_TEMPLATE_ANSWER_FIRST = """You are an expert histopathologist evaluating the quality of a generated pathology report.

First, state your quality rating (0, 1, or 2), then explain why.

Rating scale:
- 0 = Poor/incorrect
- 1 = Partially correct
- 2 = Good/accurate

Histopathology Report:
\"\"\"
{report}
\"\"\"

Provide your response in JSON format:
```json
{{
    "quality_score": <your rating 0, 1, or 2>,
    "reasoning": "Justification for your rating"
}}
```
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
        """Build prompt with histopathology report."""
        report = input_data.get("text", input_data.get("report", ""))

        # Select template (priority: answer_first > contrarian > standard)
        if self.answer_first:
            template = PROMPT_TEMPLATE_ANSWER_FIRST
        elif self.contrarian:
            template = PROMPT_TEMPLATE_CONTRARIAN
        else:
            template = PROMPT_TEMPLATE

        prompt = template.format(report=report)

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
