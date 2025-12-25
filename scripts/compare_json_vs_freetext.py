#!/usr/bin/env python3
"""
Compare MedGemma 27B JSON vs Free Text (Non-JSON) Results

Usage:
    python compare_json_vs_freetext.py
"""

from pathlib import Path

import pandas as pd

# Paths to results
JSON_RESULTS = Path("/Users/huseyin/Documents/CoT/outputs/medgemma27b_json/analysis_results.csv")
FREETEXT_RESULTS = Path(
    "/Users/huseyin/Documents/CoT/outputs/outputs_vllm/18-41-38_medgemma27b-text-it-vLLM/analysis_results.csv"
)


def load_and_prepare(path: Path, suffix: str) -> pd.DataFrame:
    """Load CSV and add suffix to metric columns."""
    df = pd.read_csv(path)
    # Rename metric columns with suffix
    for col in ["agreement_rate", "cot_accuracy", "direct_accuracy", "num_samples"]:
        if col in df.columns:
            df[f"{col}_{suffix}"] = df[col]
    return df


def compare_by_prompt(json_df: pd.DataFrame, freetext_df: pd.DataFrame) -> pd.DataFrame:
    """Compare results grouped by prompt strategy."""
    # Exclude radiology dataset from faithfulness comparison
    json_filtered = json_df[~json_df["dataset"].str.contains("radiology", case=False, na=False)]
    freetext_filtered = freetext_df[
        ~freetext_df["dataset"].str.contains("radiology", case=False, na=False)
    ]

    # Group by prompt
    json_grouped = (
        json_filtered.groupby("prompt")
        .agg(
            {
                "cot_accuracy": "mean",
                "direct_accuracy": "mean",
                "agreement_rate": "mean",
                "num_samples": "sum",
            }
        )
        .reset_index()
    )

    freetext_grouped = (
        freetext_filtered.groupby("prompt")
        .agg(
            {
                "cot_accuracy": "mean",
                "direct_accuracy": "mean",
                "agreement_rate": "mean",
                "num_samples": "sum",
            }
        )
        .reset_index()
    )

    # Merge
    merged = json_grouped.merge(
        freetext_grouped, on="prompt", suffixes=("_json", "_freetext"), how="outer"
    )

    # Calculate differences
    merged["cot_diff"] = merged["cot_accuracy_json"] - merged["cot_accuracy_freetext"]
    merged["direct_diff"] = merged["direct_accuracy_json"] - merged["direct_accuracy_freetext"]

    return merged.sort_values("cot_diff", ascending=False)


def compare_by_dataset(json_df: pd.DataFrame, freetext_df: pd.DataFrame) -> pd.DataFrame:
    """Compare results grouped by dataset."""
    # Group by dataset
    json_grouped = (
        json_df.groupby("dataset")
        .agg(
            {
                "cot_accuracy": "mean",
                "direct_accuracy": "mean",
                "agreement_rate": "mean",
                "num_samples": "sum",
            }
        )
        .reset_index()
    )

    freetext_grouped = (
        freetext_df.groupby("dataset")
        .agg(
            {
                "cot_accuracy": "mean",
                "direct_accuracy": "mean",
                "agreement_rate": "mean",
                "num_samples": "sum",
            }
        )
        .reset_index()
    )

    # Merge
    merged = json_grouped.merge(
        freetext_grouped, on="dataset", suffixes=("_json", "_freetext"), how="outer"
    )

    # Calculate differences
    merged["cot_diff"] = merged["cot_accuracy_json"] - merged["cot_accuracy_freetext"]
    merged["direct_diff"] = merged["direct_accuracy_json"] - merged["direct_accuracy_freetext"]

    return merged


def print_table(df: pd.DataFrame, title: str, show_cols: list):
    """Print a formatted table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(
        df[show_cols].to_string(
            index=False, float_format=lambda x: f"{x*100:.1f}%" if abs(x) < 2 else f"{x:.0f}"
        )
    )


def main():
    print("\n" + "=" * 80)
    print("MedGemma 27B: JSON vs Free Text Comparison")
    print("=" * 80)

    # Load data
    json_df = pd.read_csv(JSON_RESULTS)
    freetext_df = pd.read_csv(FREETEXT_RESULTS)

    print(f"\nJSON samples: {json_df['num_samples'].sum()}")
    print(f"Free Text samples: {freetext_df['num_samples'].sum()}")

    # Compare by prompt
    by_prompt = compare_by_prompt(json_df, freetext_df)
    print("\n" + "-" * 80)
    print("COMPARISON BY PROMPT (Excluding Radiology)")
    print("-" * 80)
    print(f"{'Prompt':<25} {'CoT JSON':>10} {'CoT Free':>10} {'Δ CoT':>10} {'JSON Better?':>12}")
    print("-" * 80)
    for _, row in by_prompt.iterrows():
        json_val = row["cot_accuracy_json"] * 100 if pd.notna(row["cot_accuracy_json"]) else 0
        free_val = (
            row["cot_accuracy_freetext"] * 100 if pd.notna(row["cot_accuracy_freetext"]) else 0
        )
        diff = row["cot_diff"] * 100 if pd.notna(row["cot_diff"]) else 0
        better = "✅ Yes" if diff > 0 else "❌ No"
        print(
            f"{row['prompt']:<25} {json_val:>9.1f}% {free_val:>9.1f}% {diff:>+9.1f}% {better:>12}"
        )

    # Compare by dataset
    by_dataset = compare_by_dataset(json_df, freetext_df)
    print("\n" + "-" * 80)
    print("COMPARISON BY DATASET")
    print("-" * 80)
    print(f"{'Dataset':<20} {'CoT JSON':>10} {'CoT Free':>10} {'Δ CoT':>10} {'JSON Better?':>12}")
    print("-" * 80)
    for _, row in by_dataset.iterrows():
        json_val = row["cot_accuracy_json"] * 100 if pd.notna(row["cot_accuracy_json"]) else 0
        free_val = (
            row["cot_accuracy_freetext"] * 100 if pd.notna(row["cot_accuracy_freetext"]) else 0
        )
        diff = row["cot_diff"] * 100 if pd.notna(row["cot_diff"]) else 0
        better = "✅ Yes" if diff > 0 else "❌ No"
        note = " (classification only)" if "radiology" in row["dataset"].lower() else ""
        print(
            f"{row['dataset']:<20} {json_val:>9.1f}% {free_val:>9.1f}% {diff:>+9.1f}% {better:>12}{note}"
        )

    # Overall summary (excluding radiology)
    json_filtered = json_df[~json_df["dataset"].str.contains("radiology", case=False, na=False)]
    freetext_filtered = freetext_df[
        ~freetext_df["dataset"].str.contains("radiology", case=False, na=False)
    ]

    json_cot = (json_filtered["cot_accuracy"] * json_filtered["num_samples"]).sum() / json_filtered[
        "num_samples"
    ].sum()
    freetext_cot = (
        freetext_filtered["cot_accuracy"] * freetext_filtered["num_samples"]
    ).sum() / freetext_filtered["num_samples"].sum()

    json_direct = (
        json_filtered["direct_accuracy"] * json_filtered["num_samples"]
    ).sum() / json_filtered["num_samples"].sum()
    freetext_direct = (
        freetext_filtered["direct_accuracy"] * freetext_filtered["num_samples"]
    ).sum() / freetext_filtered["num_samples"].sum()

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (Excluding Radiology)")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'JSON':>12} {'Free Text':>12} {'Difference':>12}")
    print("-" * 60)
    print(
        f"{'CoT Accuracy':<20} {json_cot*100:>11.1f}% {freetext_cot*100:>11.1f}% {(json_cot-freetext_cot)*100:>+11.1f}%"
    )
    print(
        f"{'Direct Accuracy':<20} {json_direct*100:>11.1f}% {freetext_direct*100:>11.1f}% {(json_direct-freetext_direct)*100:>+11.1f}%"
    )

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    if json_cot > freetext_cot:
        print(f"\n✅ JSON output IMPROVES CoT accuracy by {(json_cot-freetext_cot)*100:.1f}%")
    else:
        print(f"\n❌ JSON output DECREASES CoT accuracy by {(freetext_cot-json_cot)*100:.1f}%")

    # Count which prompts benefit
    benefits = (by_prompt["cot_diff"] > 0).sum()
    total_prompts = len(by_prompt)
    print(f"\n📊 {benefits}/{total_prompts} prompts benefit from JSON output")

    # Top beneficiaries
    print("\n🏆 Top 3 prompts that BENEFIT from JSON:")
    for _, row in by_prompt.head(3).iterrows():
        print(f"   • {row['prompt']}: +{row['cot_diff']*100:.1f}%")

    print("\n⚠️  Top 3 prompts that are HURT by JSON:")
    for _, row in by_prompt.tail(3).iterrows():
        print(f"   • {row['prompt']}: {row['cot_diff']*100:.1f}%")

    print()


if __name__ == "__main__":
    main()
