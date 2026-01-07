"""Automatic experiment documentation generator.

Generates EXPERIMENT.md files for every experiment run with:
- Human-readable experiment description
- Research context and questions
- Full configuration
- Reproduction commands
- Results summary
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class ExperimentDocumenter:
    """Generates markdown documentation for experiments."""

    def __init__(self, config: DictConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.start_time = datetime.now()

    def generate_title(self) -> str:
        """Generate human-readable experiment title from config."""
        parts = []

        # Dataset
        dataset_name = self.config.get("dataset", {}).get("name", "unknown")
        parts.append(dataset_name.capitalize())

        # Reasoning mode
        prompt_cfg = self.config.get("prompt", {})
        if prompt_cfg.get("answer_first", False):
            parts.append("Answer-First")
        elif prompt_cfg.get("contrarian", False):
            parts.append("Contrarian")
        else:
            parts.append("Standard")

        # Few-shot
        if not prompt_cfg.get("few_shot", True):
            parts.append("Zero-Shot")

        # Output format
        output_fmt = prompt_cfg.get("output_format", "json")
        if output_fmt != "json":
            parts.append(f"({output_fmt.upper()})")

        return " ".join(parts)

    def infer_research_questions(self) -> list[str]:
        """Infer research questions from configuration."""
        questions = []
        prompt_cfg = self.config.get("prompt", {})

        # Few-shot ablation
        if not prompt_cfg.get("few_shot", True):
            questions.append("Does the model perform well without few-shot examples (zero-shot)?")

        # Reasoning mode
        if prompt_cfg.get("contrarian", False):
            questions.append("Does skeptical/contrarian reasoning improve diagnostic accuracy?")
        elif prompt_cfg.get("answer_first", False):
            questions.append(
                'Does "answer first, then justify" reasoning order affect performance?'
            )

        # Output format
        output_fmt = prompt_cfg.get("output_format", "json")
        if output_fmt != "json":
            questions.append(
                f"How does {output_fmt.upper()} output format affect parsing and accuracy?"
            )

        # Default question if none inferred
        if not questions:
            questions.append("Standard baseline experiment for comparison.")

        return questions

    def generate_reproduction_command(self) -> str:
        """Generate exact CLI command to reproduce experiment."""
        parts = ["python -m cotlab.main"]

        # Add overrides from config
        prompt_cfg = self.config.get("prompt", {})
        dataset_cfg = self.config.get("dataset", {})

        # Prompt selection
        if "name" in prompt_cfg:
            parts.append(f"prompt={prompt_cfg['name']}")

        # Prompt parameters
        if prompt_cfg.get("contrarian", False):
            parts.append("prompt.contrarian=true")
        if prompt_cfg.get("answer_first", False):
            parts.append("prompt.answer_first=true")
        if not prompt_cfg.get("few_shot", True):
            parts.append("prompt.few_shot=false")

        output_fmt = prompt_cfg.get("output_format", "json")
        if output_fmt != "json":
            parts.append(f"prompt.output_format={output_fmt}")

        # Dataset
        if "name" in dataset_cfg:
            parts.append(f"dataset={dataset_cfg['name']}")

        return " \\\n  ".join(parts)

    def create_initial_doc(self) -> str:
        """Create initial experiment documentation (before results)."""
        title = self.generate_title()
        questions = self.infer_research_questions()
        repro_cmd = self.generate_reproduction_command()

        # Configuration summary
        prompt_cfg = self.config.get("prompt", {})
        dataset_cfg = self.config.get("dataset", {})

        reasoning_mode = "Standard"
        if prompt_cfg.get("answer_first", False):
            reasoning_mode = "Answer-First"
        elif prompt_cfg.get("contrarian", False):
            reasoning_mode = "Contrarian (skeptical)"

        few_shot = "Yes" if prompt_cfg.get("few_shot", True) else "No (zero-shot)"
        output_fmt = prompt_cfg.get("output_format", "json").upper()
        dataset_name = dataset_cfg.get("name", "unknown")

        doc = f"""# Experiment: {title}

**Status:** Running
**Started:** {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}

## Research Questions

"""
        for i, question in enumerate(questions, 1):
            doc += f"{i}. {question}\n"

        doc += f"""
## Configuration

**Prompt Strategy:** {prompt_cfg.get("name", "unknown").capitalize()}
**Reasoning Mode:** {reasoning_mode}
**Few-Shot Examples:** {few_shot}
**Output Format:** {output_fmt}
**Dataset:** {dataset_name}

<details>
<summary>Full Configuration (YAML)</summary>

```yaml
{OmegaConf.to_yaml(self.config)}
```
</details>

## Reproduce

```bash
{repro_cmd}
```

## Results

_Results will be added after experiment completes..._
"""

        return doc

    def update_with_results(
        self, results: Optional[Dict[str, Any]] = None, duration_seconds: Optional[float] = None
    ) -> str:
        """Update documentation with results after completion."""
        # Read existing doc
        doc_path = self.output_dir / "EXPERIMENT.md"
        if doc_path.exists():
            doc = doc_path.read_text()
        else:
            doc = self.create_initial_doc()

        # Update status
        doc = doc.replace("**Status:** Running", "**Status:** Completed")

        # Add duration if provided
        if duration_seconds is not None:
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration_str = (
                f"{minutes} minutes {seconds} seconds" if minutes > 0 else f"{seconds} seconds"
            )
            doc = doc.replace(
                f"**Started:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Started:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}  \n**Duration:** {duration_str}",
            )

        # Update results section
        if results:
            results_md = self._format_results(results)
            doc = doc.replace(
                "## Results\n\n_Results will be added after experiment completes..._",
                f"## Results\n\n{results_md}",
            )

        return doc

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results dictionary as markdown."""
        md = ""

        # Basic metrics
        if "accuracy" in results:
            md += f"- **Accuracy:** {results['accuracy']:.1%}\n"
        if "total_samples" in results:
            md += f"- **Samples Processed:** {results['total_samples']}\n"
        if "parse_failures" in results:
            md += f"- **Parse Failures:** {results['parse_failures']}\n"
        if "avg_time" in results:
            md += f"- **Average Time per Sample:** {results['avg_time']:.2f}s\n"

        # Additional metrics
        for key, value in results.items():
            if key not in ["accuracy", "total_samples", "parse_failures", "avg_time"]:
                if isinstance(value, float):
                    md += f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n"
                else:
                    md += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        return md

    def save(self, content: Optional[str] = None) -> Path:
        """Save experiment documentation to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        doc_path = self.output_dir / "EXPERIMENT.md"

        if content is None:
            content = self.create_initial_doc()

        doc_path.write_text(content)
        return doc_path
