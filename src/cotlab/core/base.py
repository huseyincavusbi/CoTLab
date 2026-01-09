"""Base classes and data structures for the CoT research framework."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GenerationOutput:
    """Standard output format for model generation."""

    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None

    def __repr__(self) -> str:
        return f"GenerationOutput(text={self.text[:50]}..., tokens={len(self.tokens)})"


@dataclass
class ExperimentResult:
    """JSON-serializable experiment result."""

    experiment_name: str
    model_name: str
    prompt_strategy: str
    metrics: Dict[str, Any]
    raw_outputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.__dict__, indent=2, default=str)

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class BasePromptStrategy(ABC):
    """Abstract base class for prompt construction strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        ...

    @abstractmethod
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Build a prompt from input data.

        Args:
            input_data: Dictionary with at least 'question' key

        Returns:
            Formatted prompt string
        """
        ...

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response to extract answer and reasoning.

        Args:
            response: Raw model output

        Returns:
            Dictionary with 'answer', 'reasoning', and any other extracted fields
        """
        ...

    def get_system_message(self) -> Optional[str]:
        """Return system message if applicable."""
        return None

    def get_compatible_datasets(self) -> Optional[List[str]]:
        """
        Return list of compatible dataset names, or None if compatible with all.

        Override this in specialized prompts to restrict usage.
        """
        return None


# Output format options
class OutputFormat:
    """Supported output formats for structured responses."""

    PLAIN = "plain"
    JSON = "json"
    TOON = "toon"
    TOML = "toml"
    XML = "xml"
    YAML = "yaml"
    MARKDOWN = "markdown"

    @classmethod
    def all(cls) -> list:
        return [cls.PLAIN, cls.JSON, cls.TOON, cls.TOML, cls.XML, cls.YAML, cls.MARKDOWN]


# Plain text output schema for medical diagnosis tasks
PLAIN_OUTPUT_SCHEMA = """\

Provide your answer in plain text. At the end of your response, clearly state:

FINAL ANSWER: [your diagnosis]
CONFIDENCE: [0-100]

Example:
Based on the clinical findings, the patient shows signs consistent with the condition.

FINAL ANSWER: positive
CONFIDENCE: 85"""

PLAIN_COT_SCHEMA = """\

Provide your reasoning step by step in plain text. At the end, clearly state:

FINAL ANSWER: [your diagnosis]
CONFIDENCE: [0-100]

Example:
Step 1: The report mentions...
Step 2: This suggests...
Step 3: Therefore...

FINAL ANSWER: positive
CONFIDENCE: 85"""

# JSON output schema for medical diagnosis tasks
JSON_OUTPUT_SCHEMA = """\

Output your answer ONLY in this JSON format:
```json
{
    "diagnosis": "[your diagnosis]",
    "confidence": [0-100],
    "reasoning": "[brief explanation]"
}
```"""

JSON_COT_SCHEMA = """\

Output your answer ONLY in this JSON format:
```json
{
    "step_by_step": ["Step 1: ...", "Step 2: ..."],
    "diagnosis": "[your diagnosis]",
    "confidence": [0-100]
}
```"""

# TOON (Token-Oriented Object Notation) - 40% fewer tokens than JSON
TOON_OUTPUT_SCHEMA = """\

Output your answer ONLY in this TOON format (indentation-based, no braces):
```
diagnosis: [your diagnosis]
confidence: [0-100]
reasoning: [brief explanation]
```"""

TOON_COT_SCHEMA = """\

Output your answer ONLY in this TOON format:
```
step_by_step: [3]
  Step 1: ...
  Step 2: ...
  Step 3: ...
diagnosis: [your diagnosis]
confidence: [0-100]
```"""

# TOML format
TOML_OUTPUT_SCHEMA = """\

Output your answer ONLY in this TOML format:
```toml
[response]
diagnosis = "[your diagnosis]"
confidence = 0-100
reasoning = "[brief explanation]"
```"""

TOML_COT_SCHEMA = """\

Output your answer ONLY in this TOML format:
```toml
[response]
step_by_step = ["Step 1: ...", "Step 2: ..."]
diagnosis = "[your diagnosis]"
confidence = 0-100
```"""

# XML format
XML_OUTPUT_SCHEMA = """\

Output your answer ONLY in this XML format:
```xml
<response>
  <diagnosis>[your diagnosis]</diagnosis>
  <confidence>[0-100]</confidence>
  <reasoning>[brief explanation]</reasoning>
</response>
```"""

XML_COT_SCHEMA = """\

Output your answer ONLY in this XML format:
```xml
<response>
  <step_by_step>
    <step>Step 1: ...</step>
    <step>Step 2: ...</step>
  </step_by_step>
  <diagnosis>[your diagnosis]</diagnosis>
  <confidence>[0-100]</confidence>
</response>
```"""

# YAML format
YAML_OUTPUT_SCHEMA = """\

Output your answer ONLY in this YAML format:
```yaml
diagnosis: "[your diagnosis]"
confidence: 0-100
reasoning: "[brief explanation]"
```"""

YAML_COT_SCHEMA = """\

Output your answer ONLY in this YAML format:
```yaml
step_by_step:
  - "Step 1: ..."
  - "Step 2: ..."
diagnosis: "[your diagnosis]"
confidence: 0-100
```"""

# Markdown structured format
MARKDOWN_OUTPUT_SCHEMA = """\

Output your answer ONLY in this structured Markdown format:
## Diagnosis
[your diagnosis]

## Confidence
[0-100]

## Reasoning
[brief explanation]"""

MARKDOWN_COT_SCHEMA = """\

Output your answer ONLY in this structured Markdown format:
## Step-by-Step Reasoning
1. Step 1: ...
2. Step 2: ...

## Diagnosis
[your diagnosis]

## Confidence
[0-100]"""


class StructuredOutputMixin:
    """Mixin to add multi-format structured output capability to any prompt strategy.

    Supports: PLAIN, JSON, TOON, TOML, XML, YAML, MARKDOWN

    Usage:
        class MyStrategy(StructuredOutputMixin, BasePromptStrategy):
            def __init__(self, output_format="plain", ...):
                self.output_format = output_format
                self.cot_format = False  # Set True for CoT inside structure
    """

    output_format: str = "plain"
    cot_format: bool = False

    # Backward compatibility with json_output
    @property
    def json_output(self) -> bool:
        return self.output_format == OutputFormat.JSON

    @json_output.setter
    def json_output(self, value: bool):
        if value:
            self.output_format = OutputFormat.JSON

    def _get_format_schema(self) -> tuple:
        """Return (output_schema, cot_schema) for current format."""
        schemas = {
            OutputFormat.PLAIN: (PLAIN_OUTPUT_SCHEMA, PLAIN_COT_SCHEMA),
            OutputFormat.JSON: (JSON_OUTPUT_SCHEMA, JSON_COT_SCHEMA),
            OutputFormat.TOON: (TOON_OUTPUT_SCHEMA, TOON_COT_SCHEMA),
            OutputFormat.TOML: (TOML_OUTPUT_SCHEMA, TOML_COT_SCHEMA),
            OutputFormat.XML: (XML_OUTPUT_SCHEMA, XML_COT_SCHEMA),
            OutputFormat.YAML: (YAML_OUTPUT_SCHEMA, YAML_COT_SCHEMA),
            OutputFormat.MARKDOWN: (MARKDOWN_OUTPUT_SCHEMA, MARKDOWN_COT_SCHEMA),
        }
        return schemas.get(self.output_format, ("", ""))

    def _add_format_instruction(self) -> str:
        """Return format instruction to append to prompt."""
        output_schema, cot_schema = self._get_format_schema()
        return cot_schema if self.cot_format else output_schema

    def _format_example(self, example_data: Dict[str, Any], format_type: str = None) -> str:
        """Format a single example in the specified format.

        Args:
            example_data: Dictionary with example data
            format_type: Output format (json, yaml, toml, xml, plain). Uses self.output_format if None.

        Returns:
            Formatted example string with code block markers
        """
        import json

        fmt = format_type or self.output_format

        if fmt == OutputFormat.JSON or fmt == "json":
            return f"```json\n{json.dumps(example_data, indent=4)}\n```"

        elif fmt == OutputFormat.YAML or fmt == "yaml":
            # Simple YAML conversion
            def to_yaml(obj, indent=0):
                lines = []
                prefix = "  " * indent
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, (dict, list)):
                            lines.append(f"{prefix}{k}:")
                            lines.append(to_yaml(v, indent + 1))
                        else:
                            val = (
                                str(v).lower()
                                if isinstance(v, bool)
                                else f'"{v}"'
                                if isinstance(v, str)
                                else v
                            )
                            lines.append(f"{prefix}{k}: {val}")
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, str):
                            lines.append(f'{prefix}- "{item}"')
                        else:
                            lines.append(f"{prefix}- {to_yaml(item, indent + 1)}")
                return "\n".join(lines)

            return f"```yaml\n{to_yaml(example_data)}\n```"

        elif fmt == OutputFormat.TOML or fmt == "toml":
            # Simple TOML conversion
            def to_toml(obj, section=""):
                lines = []
                if section:
                    lines.append(f"[{section}]")
                for k, v in obj.items():
                    if isinstance(v, dict):
                        lines.append(to_toml(v, f"{section}.{k}" if section else k))
                    elif isinstance(v, list):
                        items = ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in v)
                        lines.append(f"{k} = [{items}]")
                    elif isinstance(v, bool):
                        lines.append(f"{k} = {str(v).lower()}")
                    elif isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    else:
                        lines.append(f"{k} = {v}")
                return "\n".join(lines)

            return f"```toml\n{to_toml(example_data)}\n```"

        elif fmt == OutputFormat.PLAIN or fmt == "plain":
            # Plain text - use plain_answer if available
            plain_answer = example_data.get("_plain_answer", "")
            reasoning = example_data.get("evidence", {}).get("rationale", "")
            return f"{reasoning}\n\nFINAL ANSWER: {plain_answer}"

        elif fmt == OutputFormat.XML or fmt == "xml":

            def to_xml(obj, root="response"):
                lines = [f"<{root}>"]
                for k, v in obj.items():
                    if isinstance(v, dict):
                        lines.append(to_xml(v, k))
                    elif isinstance(v, list):
                        lines.append(f"  <{k}>")
                        for item in v:
                            lines.append(f"    <item>{item}</item>")
                        lines.append(f"  </{k}>")
                    else:
                        val = str(v).lower() if isinstance(v, bool) else v
                        lines.append(f"  <{k}>{val}</{k}>")
                lines.append(f"</{root}>")
                return "\n".join(lines)

            return f"```xml\n{to_xml(example_data)}\n```"

        elif fmt == OutputFormat.TOON or fmt == "toon":
            # TOON: Simple Key-Value pairs
            lines = []
            # Extract main fields from example data
            plain_answer = example_data.get("_plain_answer", "")
            reasoning = example_data.get("evidence", {}).get("rationale", "")

            lines.append(f"diagnosis: {plain_answer}")
            lines.append(f"reasoning: {reasoning}")
            # Add other flat keys if they exist and aren't dicts/lists
            for k, v in example_data.items():
                if k not in ["evidence", "_plain_answer"] and not isinstance(v, (dict, list)):
                    lines.append(f"{k}: {str(v).lower() if isinstance(v, bool) else v}")
            return f"```toon\n{chr(10).join(lines)}\n```"

        elif fmt == OutputFormat.MARKDOWN or fmt == "markdown":
            # Markdown: Header sections
            plain_answer = example_data.get("_plain_answer", "")
            reasoning = example_data.get("evidence", {}).get("rationale", "")

            md = f"## Diagnosis\n{plain_answer}\n\n"
            md += f"## Reasoning\n{reasoning}\n\n"
            md += "## Confidence\n95"  # Example confidence
            return md

        else:
            # Default to JSON
            return f"```json\n{json.dumps(example_data, indent=4)}\n```"

    def _parse_formatted_response(self, response: str) -> Dict[str, Any]:
        """Parse response based on output format."""
        import re

        if self.output_format == OutputFormat.PLAIN:
            # Extract FINAL ANSWER and CONFIDENCE from plain text
            answer_match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response, re.IGNORECASE)

            answer = answer_match.group(1).strip() if answer_match else response.strip()
            confidence = int(confidence_match.group(1)) if confidence_match else None

            return {
                "answer": answer,
                "confidence": confidence,
                "reasoning": response,
                "raw": response,
                "parse_success": bool(answer_match),
                "format": "plain",
            }

        if self.output_format == OutputFormat.JSON:
            return self._parse_json(response)
        elif self.output_format == OutputFormat.TOON:
            return self._parse_toon(response)
        elif self.output_format == OutputFormat.TOML:
            return self._parse_toml(response)
        elif self.output_format == OutputFormat.XML:
            return self._parse_xml(response)
        elif self.output_format == OutputFormat.YAML:
            return self._parse_yaml(response)
        elif self.output_format == OutputFormat.MARKDOWN:
            return self._parse_markdown(response)
        else:
            return {
                "answer": response.strip(),
                "reasoning": response,
                "raw": response,
                "parse_success": False,
                "format": self.output_format,
                "error": f"Unknown format: {self.output_format}",
            }

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON response."""
        import re

        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[^{}]*"diagnosis"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return self._parse_failure(response, "json")

        try:
            parsed = json.loads(json_str)
            return self._extract_fields(parsed, response, "json")
        except json.JSONDecodeError:
            return self._parse_failure(response, "json")

    def _parse_toon(self, response: str) -> Dict[str, Any]:
        """Parse TOON (Token-Oriented Object Notation) response."""
        import re

        # Extract TOON block from code fence
        toon_match = re.search(r"```(?:toon)?\s*(.*?)\s*```", response, re.DOTALL)
        content = toon_match.group(1) if toon_match else response

        # Parse key: value pairs
        result = {}
        for line in content.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key and value:
                    # Auto-convert booleans
                    if value.lower() == "true":
                        result[key] = True
                    elif value.lower() == "false":
                        result[key] = False
                    else:
                        result[key] = value

        if "diagnosis" in result:
            return self._extract_fields(result, response, "toon")
        return self._parse_failure(response, "toon")

    def _parse_toml(self, response: str) -> Dict[str, Any]:
        """Parse TOML response."""
        import re

        toml_match = re.search(r"```toml\s*(.*?)\s*```", response, re.DOTALL)
        if not toml_match:
            return self._parse_failure(response, "toml")

        try:
            import tomllib

            parsed = tomllib.loads(toml_match.group(1))
            # Handle [response] section
            if "response" in parsed:
                parsed = parsed["response"]
            return self._extract_fields(parsed, response, "toml")
        except Exception:
            return self._parse_failure(response, "toml")

    def _parse_xml(self, response: str) -> Dict[str, Any]:
        """Parse XML response."""
        import re

        xml_match = re.search(r"```xml\s*(.*?)\s*```", response, re.DOTALL)
        content = xml_match.group(1) if xml_match else response

        try:
            import xmltodict

            def postprocessor(path, key, value):
                if value and isinstance(value, str):
                    if value.lower() == "true":
                        return key, True
                    if value.lower() == "false":
                        return key, False
                return key, value

            parsed = xmltodict.parse(content.strip(), postprocessor=postprocessor)
            # Remove root element wrapper if present (e.g. <response>)
            if len(parsed) == 1:
                key = next(iter(parsed))
                parsed = parsed[key]

            return self._extract_fields(parsed, response, "xml")
        except Exception:
            # Fallback to ElementTree with recursive parsing
            try:
                import xml.etree.ElementTree as ET

                def elem_to_dict(elem):
                    text = elem.text.strip() if elem.text else None
                    # Convert booleans
                    if text:
                        if text.lower() == "true":
                            text = True
                        elif text.lower() == "false":
                            text = False

                    children = list(elem)
                    if not children:
                        return text

                    result = {}
                    for child in children:
                        child_val = elem_to_dict(child)
                        if child.tag in result:
                            if not isinstance(result[child.tag], list):
                                result[child.tag] = [result[child.tag]]
                            result[child.tag].append(child_val)
                        else:
                            result[child.tag] = child_val
                    return result

                root = ET.fromstring(content.strip())
                parsed = elem_to_dict(root)
                # If root returned a dict (nested content), use it directly
                # If root was simple value (not likely for top level), usage might differ
                if isinstance(parsed, dict):
                    pass

                return self._extract_fields(parsed, response, "xml")
            except Exception:
                return self._parse_failure(response, "xml")

    def _parse_yaml(self, response: str) -> Dict[str, Any]:
        """Parse YAML response."""
        import re

        yaml_match = re.search(r"```yaml\s*(.*?)\s*```", response, re.DOTALL)
        if not yaml_match:
            return self._parse_failure(response, "yaml")

        try:
            import yaml

            parsed = yaml.safe_load(yaml_match.group(1))
            return self._extract_fields(parsed, response, "yaml")
        except Exception:
            return self._parse_failure(response, "yaml")

    def _parse_markdown(self, response: str) -> Dict[str, Any]:
        """Parse structured Markdown response."""
        import re

        result = {}
        # Extract sections
        diagnosis_match = re.search(
            r"##\s*Diagnosis\s*\n(.*?)(?=\n##|\Z)", response, re.DOTALL | re.IGNORECASE
        )
        confidence_match = re.search(
            r"##\s*Confidence\s*\n(.*?)(?=\n##|\Z)", response, re.DOTALL | re.IGNORECASE
        )
        reasoning_match = re.search(
            r"##\s*Reasoning\s*\n(.*?)(?=\n##|\Z)", response, re.DOTALL | re.IGNORECASE
        )

        if diagnosis_match:
            result["diagnosis"] = diagnosis_match.group(1).strip()
        if confidence_match:
            try:
                result["confidence"] = int(confidence_match.group(1).strip())
            except ValueError:
                result["confidence"] = 0
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        if "diagnosis" in result:
            return self._extract_fields(result, response, "markdown")
        return self._parse_failure(response, "markdown")

    def _extract_fields(self, parsed: dict, response: str, fmt: str) -> Dict[str, Any]:
        """Extract standard fields from parsed response."""
        steps = parsed.get("step_by_step", [])
        reasoning = "\n".join(steps) if isinstance(steps, list) else parsed.get("reasoning", "")

        # Ensure confidence is integer
        confidence = parsed.get("confidence", 0)
        if isinstance(confidence, str):
            try:
                confidence = int(confidence)
            except ValueError:
                confidence = 0
        elif not isinstance(confidence, int):
            confidence = int(confidence) if confidence else 0

        result = {
            "answer": str(parsed.get("diagnosis", "")),
            "reasoning": reasoning,
            "confidence": confidence,
            "step_by_step": steps,
            "raw": response,
            "parsed_data": parsed,
            "parse_success": True,
            "format": fmt,
        }

        # Merge all parsed fields into top-level result for easy access
        # (excluding standard keys to avoid overwriting processed values)
        for k, v in parsed.items():
            if k not in result:
                result[k] = v

        return result

    def _parse_failure(self, response: str, fmt: str) -> Dict[str, Any]:
        """Return parse failure result."""
        return {
            "answer": response.strip(),
            "reasoning": response,
            "raw": response,
            "parse_success": False,
            "format": fmt,
        }


class BaseExperiment(ABC):
    """Abstract base class for experiments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Experiment name for logging."""
        ...

    @abstractmethod
    def run(
        self,
        backend: "InferenceBackend",
        dataset: Any,
        prompt_strategy: BasePromptStrategy,
        **kwargs,
    ) -> ExperimentResult:
        """
        Run the experiment.

        Args:
            backend: Inference backend (vLLM or Transformers)
            dataset: Dataset to run on
            prompt_strategy: How to construct prompts

        Returns:
            ExperimentResult with metrics and outputs
        """
        ...

    def validate_backend(self, backend: "InferenceBackend") -> None:
        """Check if backend supports required features."""
        pass
