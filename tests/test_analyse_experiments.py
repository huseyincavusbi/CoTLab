"""Answer-extraction regexes in analyse_experiments.

The feature branch precompiles the inline regexes (and adds a ``data`` param
to avoid re-reading files). These tests pin the behavior against the literal
patterns that main used inline.
"""

import json
import re
from pathlib import Path

import pytest

import cotlab.analyse_experiments as ae

_MAIN_INLINE_PATTERNS = {
    "boxed": (r"\$?\\boxed\{([^}]+)\}\$?", 0),
    "final_answer": (
        r"(?:final answer|answer)[:\s]*(?:the final answer is\s*)?[:\s]*([^\n$]+)",
        re.IGNORECASE,
    ),
    "trailing_dollar": (r"\$.*$", 0),
    "leading_trailing_stars": (r"^[*\s]+|[*\s]+$", 0),
    "diagnosis": (r"diagnosis[:\s]+([^\n,]+)", re.IGNORECASE),
    "bold": (r"\*\*([^*]+)\*\*", 0),
}

_SAMPLES = [
    "The answer is \\boxed{osteomyelitis}.",
    "Final answer: community-acquired pneumonia.",
    "The final answer is $\\boxed{peptic ulcer}$",
    "Diagnosis: type 2 diabetes mellitus, complicated.",
    "**normal sinus rhythm** on ECG.",
    "answer: cholecystitis\n\nReasoning: ...",
    "Bold **acute appendicitis** and **peritonitis** markers.",
    "The patient's answer is $12\\,000$ cells.",
    "Diagnosis: acute coronary syndrome",
]


@pytest.mark.parametrize("name", sorted(_MAIN_INLINE_PATTERNS))
def test_precompiled_regex_matches_main_inline_pattern(name):
    """Compiled regex behaves identically to main's inline pattern."""
    pattern, flags = _MAIN_INLINE_PATTERNS[name]
    inline = re.compile(pattern, flags)
    compiled = getattr(ae, f"_RE_{name.upper()}")
    for text in _SAMPLES:
        assert compiled.findall(text) == inline.findall(text), (
            f"{name}: {text!r} -> {compiled.findall(text)} vs {inline.findall(text)}"
        )


def test_analyse_experiment_accepts_presupplied_data(tmp_path):
    """The new ``data`` param returns the same result as loading from file."""
    results = {
        "samples": [{"predicted": "A", "ground_truth": "A"}],
        "metrics": {"accuracy": 1.0},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))

    assert ae.analyse_experiment(path) == ae.analyse_experiment(path, data=results)
