#!/usr/bin/env python3
"""
Analyze CoTLab experiment results with improved answer extraction.

Usage:
    python -m cotlab.analyse_experiments <results_dir>
    python -m cotlab.analyse_experiments /path/to/experiment/results
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def extract_answer(text: str) -> str:
    """Extract the final answer/diagnosis from a response."""
    if not text:
        return ""

    text = text.strip().lower()

    # 1. Try to extract from \boxed{...}
    boxed = re.findall(r"\$?\\boxed\{([^}]+)\}\$?", text)
    if boxed:
        return boxed[0].strip().lower()

    # 2. Try to extract from "Final Answer: ..." or "**Final Answer:**"
    final_answer = re.search(
        r"(?:final answer|answer)[:\s]*(?:the final answer is\s*)?[:\s]*([^\n$]+)",
        text,
        re.IGNORECASE,
    )
    if final_answer:
        answer = final_answer.group(1).strip()
        answer = re.sub(r"\$.*$", "", answer).strip()
        answer = re.sub(r"^[*\s]+|[*\s]+$", "", answer)
        if answer:
            return answer.lower()

    # 3. Try to extract from "Diagnosis: ..."
    diagnosis = re.search(r"diagnosis[:\s]+([^\n,]+)", text, re.IGNORECASE)
    if diagnosis:
        return diagnosis.group(1).strip().lower()

    # 4. If response is very short (single word/phrase), use it directly
    words = text.split()
    if len(words) <= 5 and words:
        return words[0].strip("*.,!?\"'").lower()

    # 5. Look for bold text (**diagnosis**)
    bold = re.findall(r"\*\*([^*]+)\*\*", text)
    if bold:
        return bold[-1].strip().lower()

    return text[:50].lower()


def normalize_answer(answer) -> str:
    """Normalize answer for comparison."""
    if not isinstance(answer, str):
        return str(answer).lower().strip() if answer else ""
    answer = answer.lower().strip()
    answer = re.sub(r"^(the\s+|a\s+|an\s+)", "", answer)
    answer = re.sub(r"\s+(disease|syndrome|disorder)$", "", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    return answer.strip()


def answers_match(answer1: str, answer2: str) -> bool:
    """Check if two answers match (fuzzy matching)."""
    a1 = normalize_answer(answer1)
    a2 = normalize_answer(answer2)

    if not a1 or not a2:
        return False

    if a1 == a2:
        return True

    if a1 in a2 or a2 in a1:
        return True

    if a1.split()[0] == a2.split()[0]:
        return True

    return False


def analyse_experiment(results_path: Path) -> Optional[dict]:
    """Analyse a single experiment's results.json file."""
    with open(results_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        return None

    agreements = 0
    correct_cot = 0
    correct_direct = 0

    for sample in samples:
        cot_response = sample.get("cot_response", "") or sample.get("cot_answer", "")
        direct_response = sample.get("direct_response", "") or sample.get("direct_answer", "")
        expected = sample.get("expected_answer", "")

        cot_answer = extract_answer(cot_response)
        direct_answer = extract_answer(direct_response)
        expected_answer = normalize_answer(expected)

        if answers_match(cot_answer, direct_answer):
            agreements += 1

        if answers_match(cot_answer, expected_answer):
            correct_cot += 1
        if answers_match(direct_answer, expected_answer):
            correct_direct += 1

    n = len(samples)
    return {
        "num_samples": n,
        "agreement_rate": agreements / n if n > 0 else 0,
        "cot_accuracy": correct_cot / n if n > 0 else 0,
        "direct_accuracy": correct_direct / n if n > 0 else 0,
        "agreements": agreements,
        "correct_cot": correct_cot,
        "correct_direct": correct_direct,
    }


def analyse_experiments_dir(results_dir: Path) -> list:
    """Analyse all experiments in a directory."""
    all_results = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        results_file = exp_dir / "results.json"
        if not results_file.exists():
            continue

        name = exp_dir.name
        parts = name.split("_")
        if len(parts) >= 3:
            dataset = parts[2]
            prompt = "_".join(parts[3:])
        else:
            dataset = "unknown"
            prompt = name

        metrics = analyse_experiment(results_file)
        if metrics:
            metrics["experiment"] = name
            metrics["dataset"] = dataset
            metrics["prompt"] = prompt
            all_results.append(metrics)

    return all_results


def print_analysis_report(all_results: list, title: str = "Experiment Analysis"):
    """Print a formatted analysis report."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()

    # Group by prompt
    by_prompt = defaultdict(list)
    for r in all_results:
        by_prompt[r["prompt"]].append(r)

    print(f"{'Prompt':<25} {'Agree%':>8} {'CoT Acc':>8} {'Direct Acc':>10} {'Samples':>8}")
    print("-" * 60)

    for prompt in sorted(by_prompt.keys()):
        results = by_prompt[prompt]
        total_samples = sum(r["num_samples"] for r in results)
        total_agree = sum(r["agreements"] for r in results)
        total_cot = sum(r["correct_cot"] for r in results)
        total_direct = sum(r["correct_direct"] for r in results)

        agree_pct = 100 * total_agree / total_samples if total_samples > 0 else 0
        cot_acc = 100 * total_cot / total_samples if total_samples > 0 else 0
        direct_acc = 100 * total_direct / total_samples if total_samples > 0 else 0

        print(
            f"{prompt:<25} {agree_pct:>7.1f}% {cot_acc:>7.1f}% {direct_acc:>9.1f}% {total_samples:>8}"
        )

    print()
    print("=" * 80)
    print("SUMMARY BY DATASET")
    print("=" * 80)

    by_dataset = defaultdict(list)
    for r in all_results:
        by_dataset[r["dataset"]].append(r)

    print(f"{'Dataset':<20} {'Agree%':>8} {'CoT Acc':>8} {'Direct Acc':>10} {'Samples':>8}")
    print("-" * 55)

    for dataset in sorted(by_dataset.keys()):
        results = by_dataset[dataset]
        total_samples = sum(r["num_samples"] for r in results)
        total_agree = sum(r["agreements"] for r in results)
        total_cot = sum(r["correct_cot"] for r in results)
        total_direct = sum(r["correct_direct"] for r in results)

        agree_pct = 100 * total_agree / total_samples if total_samples > 0 else 0
        cot_acc = 100 * total_cot / total_samples if total_samples > 0 else 0
        direct_acc = 100 * total_direct / total_samples if total_samples > 0 else 0

        print(
            f"{dataset:<20} {agree_pct:>7.1f}% {cot_acc:>7.1f}% {direct_acc:>9.1f}% {total_samples:>8}"
        )

    # Overall
    print()
    total_samples = sum(r["num_samples"] for r in all_results)
    total_agree = sum(r["agreements"] for r in all_results)
    total_cot = sum(r["correct_cot"] for r in all_results)
    total_direct = sum(r["correct_direct"] for r in all_results)

    print(f"OVERALL: {total_samples} samples")
    print(f"  - Agreement: {100 * total_agree / total_samples:.1f}%")
    print(f"  - CoT Accuracy: {100 * total_cot / total_samples:.1f}%")
    print(f"  - Direct Accuracy: {100 * total_direct / total_samples:.1f}%")


def export_to_csv(all_results: list, output_path: Path):
    """Export analysis results to CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "experiment",
                "dataset",
                "prompt",
                "num_samples",
                "agreement_rate",
                "cot_accuracy",
                "direct_accuracy",
                "cot_correct",
                "direct_correct",
                "agreements",
            ]
        )

        # Data rows
        for r in all_results:
            writer.writerow(
                [
                    r["experiment"],
                    r["dataset"],
                    r["prompt"],
                    r["num_samples"],
                    f"{r['agreement_rate']:.4f}",
                    f"{r['cot_accuracy']:.4f}",
                    f"{r['direct_accuracy']:.4f}",
                    r["correct_cot"],
                    r["correct_direct"],
                    r["agreements"],
                ]
            )

    print(f"\nResults saved to: {output_path}")


def main():
    import sys

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Default path for development
        results_dir = Path("/Users/huseyin/Documents/CoT/18-41-38_medgemma27b-text-it-vLLM")

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    all_results = analyse_experiments_dir(results_dir)

    if not all_results:
        print(f"No experiment results found in {results_dir}")
        sys.exit(1)

    print_analysis_report(all_results, f"Analysis: {results_dir.name}")

    # Export to CSV
    csv_path = results_dir / "analysis_results.csv"
    export_to_csv(all_results, csv_path)


if __name__ == "__main__":
    main()
