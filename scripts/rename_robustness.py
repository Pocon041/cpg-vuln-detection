from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_prediction_shift(
    before: dict[str, float],
    after: dict[str, float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    shared = sorted(set(before) & set(after))
    if not shared:
        return {
            "sample_count": 0.0,
            "mean_abs_probability_delta": 0.0,
            "prediction_label_agreement": 0.0,
        }
    deltas = [abs(float(before[item]) - float(after[item])) for item in shared]
    agreements = [
        int(float(before[item]) >= threshold) == int(float(after[item]) >= threshold)
        for item in shared
    ]
    return {
        "sample_count": float(len(shared)),
        "mean_abs_probability_delta": sum(deltas) / len(deltas),
        "prediction_label_agreement": sum(1.0 for value in agreements if value) / len(agreements),
    }


def read_probabilities(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["sample_id"]: float(row["probability"])
            for row in reader
            if row.get("sample_id") and row.get("probability")
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rename robustness prediction shifts")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    summary = summarize_prediction_shift(
        read_probabilities(args.before),
        read_probabilities(args.after),
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
