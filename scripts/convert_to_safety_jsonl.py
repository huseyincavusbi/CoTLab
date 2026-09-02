#!/usr/bin/env python
"""Convert CoTLab-style dataset files to the SafetyNeuron unified jsonl format.

Output schema (arXiv:2406.14144, THU-KEG/SafetyNeuron reformat_datasets.py):
    {"dataset": str, "id": str, "messages": [{"role": "user"|"assistant", "content": str}, ...]}

Supported inputs:
- jsonl with {"text": ..., "label"/"response"/"answer": ...} rows
- json array of the same shape

Usage:
    python scripts/convert_to_safety_jsonl.py --input data.jsonl \
        --dataset-name medqa --output-dir SafetyNeuron/data/processed/medqa
"""

import argparse
import json
import os
from typing import Any, Dict, List


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def convert(rows: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        text = row.get("text") or row.get("prompt") or row.get("question")
        if not text:
            continue
        target = (
            row.get("response") or row.get("label") or row.get("answer") or row.get("chosen") or ""
        )
        messages = [{"role": "user", "content": str(text)}]
        if target:
            messages.append({"role": "assistant", "content": str(target)})
        out.append(
            {
                "dataset": dataset_name,
                "id": row.get("id") or f"{dataset_name}_{idx}",
                "messages": messages,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input .jsonl or .json file")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = convert(_load_rows(args.input), args.dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset_name}_data.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
