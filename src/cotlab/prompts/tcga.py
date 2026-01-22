"""TCGA Cancer Type Classification Prompt Strategy."""

import json
from typing import Any, Dict, Optional

from ..core.base import BasePromptStrategy, StructuredOutputMixin
from ..core.registry import Registry

SYSTEM_ROLE = """You are an expert pathologist specializing in cancer classification.
Your goal is to identify the specific TCGA Cancer Type Code (e.g., BRCA, LUAD) from the pathology report.
Think rationally and explain your reasoning step-by-step."""

SYSTEM_ROLE_CONTRARIAN = """You are a skeptical pathologist reviewing a cancer classification.
Question obvious conclusions and consider alternate cancer type codes before deciding."""

PROMPT_TEMPLATE = """Follow this structured reasoning on the attached pathology report:

1. **Anatomic Site**: Identify the organ and specific site of the specimen.
2. **Histological Findings**: Identify the specific tumor type, morphology, and grade described.
3. **TCGA Code Selection**: Match the findings to one of the 32 valid TCGA codes.

Valid Codes:
ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM

Follow the format of these examples and give the output strictly in the json format.

Example 1: Kidney Renal Clear Cell Carcinoma
```json
{{
    "cancer_type": "KIRC",
    "reasoning": "1. Site: Kidney (Left Upper Pole). 2. Findings: Renal cell carcinoma, conventional (clear cell) type, Fuhrman Grade II. 3. Code: Kidney Renal Clear Cell Carcinoma maps to KIRC."
}}
```

Example 2: Breast Invasive Carcinoma
```json
{{
    "cancer_type": "BRCA",
    "reasoning": "1. Site: Breast. 2. Findings: Invasive ductal carcinoma. 3. Code: Breast Invasive Carcinoma maps to BRCA."
}}
```

Pathology report:
\"\"\"
{report}
\"\"\"

Response:
"""

PROMPT_TEMPLATE_ANSWER_FIRST = """Review the pathology report and provide an immediate TCGA code.

**Step 1 - Initial Code**: Based on your first read, state the single best TCGA code.
**Step 2 - Evidence**: Justify the code with site and histology evidence.

Valid Codes:
ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM

Provide your response in JSON format. Follow the format of these two examples and give the output strictly in the json format.

Example 1: Initial BRCA, then justification
```json
{
    "cancer_type": "BRCA",
    "reasoning": "Initial code: BRCA. Site: Breast. Histology: Invasive ductal carcinoma. This maps to TCGA BRCA."
}
```

Example 2: Initial LUAD, then justification
```json
{
    "cancer_type": "LUAD",
    "reasoning": "Initial code: LUAD. Site: Lung. Histology: Adenocarcinoma with glandular features. This maps to TCGA LUAD."
}
```

Pathology report:
\"\"\"
{report}
\"\"\"

Response:
"""


@Registry.register_prompt("tcga")
class TCGAPromptStrategy(StructuredOutputMixin, BasePromptStrategy):
    """
    Structured JSON output for TCGA cancer type classification.
    """

    def __init__(
        self,
        name: str = "tcga",
        system_role: Optional[str] = None,
        few_shot: bool = True,
        contrarian: bool = False,
        answer_first: bool = False,
        **kwargs,
    ):
        self._name = name
        self.few_shot = few_shot
        self.contrarian = contrarian
        self.answer_first = answer_first
        if system_role:
            self.system_role = system_role
        else:
            self.system_role = SYSTEM_ROLE_CONTRARIAN if contrarian else SYSTEM_ROLE

    @property
    def name(self) -> str:
        return self._name

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt with pathology report."""
        report = input_data.get("text", input_data.get("report", ""))

        template = PROMPT_TEMPLATE_ANSWER_FIRST if self.answer_first else PROMPT_TEMPLATE

        if not self.few_shot:
            template = self._remove_few_shot_examples(template)

        prompt = template.format(report=report)
        return prompt

    def _remove_few_shot_examples(self, template: str) -> str:
        import re

        pattern = r"Example \d+:.*?```json.*?```"
        cleaned = re.sub(pattern, "", template, flags=re.DOTALL)
        return cleaned

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response."""
        try:
            # Try to find JSON block
            import re

            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try raw brace match
                json_match = re.search(r'\{[^{}]*"cancer_type"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            parsed = json.loads(json_str)
            return {
                "answer": parsed.get("cancer_type", "UNKNOWN"),
                "cancer_type": parsed.get("cancer_type"),
                "reasoning": parsed.get("reasoning", ""),
                "raw": response,
                "parsed_json": parsed,
            }
        except Exception:
            return {"answer": "ERROR", "raw": response, "parse_error": True}

    def get_system_message(self) -> Optional[str]:
        return self.system_role

    def get_compatible_datasets(self) -> list[str]:
        return ["tcga"]

    def get_prediction_field(self) -> str:
        return "cancer_type"
