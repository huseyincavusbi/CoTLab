#!/usr/bin/env python3
"""
Analyze JSON Following Capability

Analyzes how well model outputs follow JSON format instructions.

Usage:
    python analyze_json_compliance.py --dir <results_dir>
    python analyze_json_compliance.py --dir <dir1> --dir <dir2> --compare
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def analyze_single_response(response: str) -> Dict[str, Any]:
    """Analyze a single response for JSON compliance."""
    result = {
        "is_empty": False,
        "has_json_block": False,
        "has_curly_braces": False,
        "is_valid_json": False,
        "is_malformed": False,
        "is_pure_text": False,
        "json_content": None,
    }

    if not response or response.strip() == "":
        result["is_empty"] = True
        return result

    # Check for JSON code block
    if "```json" in response.lower():
        result["has_json_block"] = True

    # Check for curly braces
    if "{" in response and "}" in response:
        result["has_curly_braces"] = True

        # Try to extract JSON
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1)
        else:
            brace_match = re.search(r"\{[\s\S]*\}", response)
            json_str = brace_match.group(0) if brace_match else None

        if json_str:
            try:
                parsed = json.loads(json_str)
                result["is_valid_json"] = True
                result["json_content"] = parsed
            except json.JSONDecodeError:
                result["is_malformed"] = True
    else:
        result["is_pure_text"] = True

    return result


def analyze_directory(dir_path: Path) -> Dict[str, Any]:
    """Analyze all results in a directory."""
    stats = {
        "total": 0,
        "valid_json": 0,
        "has_json_block": 0,
        "has_curly_braces": 0,
        "malformed": 0,
        "pure_text": 0,
        "empty": 0,
        "sample_valid": [],
        "sample_malformed": [],
        "sample_pure_text": [],
    }

    results_files = list(dir_path.glob("**/results.json"))

    for f in results_files:
        with open(f) as fp:
            data = json.load(fp)

        for item in data.get("raw_outputs", []):
            stats["total"] += 1
            response = item.get("cot_response", "")

            analysis = analyze_single_response(response)

            if analysis["is_empty"]:
                stats["empty"] += 1
            elif analysis["is_valid_json"]:
                stats["valid_json"] += 1
                stats["has_json_block"] += 1 if analysis["has_json_block"] else 0
                stats["has_curly_braces"] += 1
                if len(stats["sample_valid"]) < 3:
                    stats["sample_valid"].append(response[:200])
            elif analysis["is_malformed"]:
                stats["malformed"] += 1
                stats["has_curly_braces"] += 1
                stats["has_json_block"] += 1 if analysis["has_json_block"] else 0
                if len(stats["sample_malformed"]) < 3:
                    stats["sample_malformed"].append(response[:200])
            elif analysis["is_pure_text"]:
                stats["pure_text"] += 1
                if len(stats["sample_pure_text"]) < 3:
                    stats["sample_pure_text"].append(response[:200])
            elif analysis["has_curly_braces"]:
                stats["has_curly_braces"] += 1

    return stats


def print_report(stats: Dict[str, Any], name: str):
    """Print a formatted report."""
    total = stats["total"]

    def pct(x):
        return (x / total * 100) if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"Total responses: {total}")

    print("\n  JSON Compliance Breakdown:")
    print(f"  {'-' * 50}")
    print(f"    Valid JSON:        {stats['valid_json']:5d} ({pct(stats['valid_json']):5.1f}%)")
    print(
        f"    Has JSON block:    {stats['has_json_block']:5d} ({pct(stats['has_json_block']):5.1f}%)"
    )
    print(
        f"    Has curly braces:  {stats['has_curly_braces']:5d} ({pct(stats['has_curly_braces']):5.1f}%)"
    )
    print(f"    Malformed JSON:    {stats['malformed']:5d} ({pct(stats['malformed']):5.1f}%)")
    print(f"    Pure text only:    {stats['pure_text']:5d} ({pct(stats['pure_text']):5.1f}%)")
    print(f"    Empty:             {stats['empty']:5d} ({pct(stats['empty']):5.1f}%)")

    # Success rate
    success_rate = pct(stats["valid_json"])
    attempt_rate = pct(stats["has_curly_braces"])
    if stats["has_curly_braces"] > 0:
        accuracy = stats["valid_json"] / stats["has_curly_braces"] * 100
    else:
        accuracy = 0

    print("\n  Key Metrics:")
    print(f"  {'-' * 50}")
    print(f"    JSON Success Rate:    {success_rate:5.1f}%")
    print(f"    JSON Attempt Rate:    {attempt_rate:5.1f}%")
    print(f"    Accuracy (when tried): {accuracy:5.1f}%")

    if stats["sample_malformed"]:
        print("\n  Sample Malformed Outputs:")
        for i, sample in enumerate(stats["sample_malformed"][:2], 1):
            clean = sample.replace("\n", " ")[:70]
            print(f"    {i}. {clean}...")


def compare_directories(all_stats: Dict[str, Dict], names: List[str]):
    """Print comparison table."""
    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Model':<25} {'Valid JSON':>12} {'Malformed':>12} {'Pure Text':>12}")
    print("-" * 70)

    for name in names:
        stats = all_stats[name]
        total = stats["total"]

        def pct(x):
            return (x / total * 100) if total > 0 else 0

        print(
            f"{name:<25} {pct(stats['valid_json']):>11.1f}% {pct(stats['malformed']):>11.1f}% {pct(stats['pure_text']):>11.1f}%"
        )

    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze JSON following capability")
    parser.add_argument(
        "--dir",
        action="append",
        required=True,
        help="Directory with results.json files (can specify multiple)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Show comparison table for multiple directories"
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    all_stats = {}
    names = []

    for dir_path in args.dir:
        path = Path(dir_path)
        if not path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        # Extract name from path
        name = path.name
        names.append(name)

        stats = analyze_directory(path)
        all_stats[name] = stats

        print_report(stats, name)

    if args.compare and len(names) > 1:
        compare_directories(all_stats, names)

    if args.output:
        # Remove non-serializable samples for JSON output
        output_stats = {}
        for name, stats in all_stats.items():
            output_stats[name] = {k: v for k, v in stats.items() if not k.startswith("sample_")}

        with open(args.output, "w") as f:
            json.dump(output_stats, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
