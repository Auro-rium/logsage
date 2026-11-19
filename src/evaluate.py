"""Evaluate one adapter against a held-out JSONL split; run once per adapter."""
import argparse
import json
from collections import Counter

from src.data import load_and_validate_jsonl
from src.inference import DEFAULT_BASE_MODEL, InferenceOutputError, InputError, LogSageInference
from src.splits import dataset_fingerprint


def evaluate(rows, runtime):
    counts = Counter()
    by_severity = {severity: Counter() for severity in ("low", "medium", "high")}
    for row in rows:
        expected = row["output"]["severity"]
        counts["total"] += 1
        by_severity[expected]["total"] += 1
        try:
            result = runtime.analyze(row["input"], row["instruction"])
        except (InferenceOutputError, InputError):
            counts["invalid_or_oversized"] += 1
            continue
        counts["valid"] += 1
        if result["severity"] == expected:
            counts["severity_correct"] += 1
            by_severity[expected]["correct"] += 1
    total = counts["total"]
    return {"total": total, "valid_outputs": counts["valid"],
            "invalid_or_oversized": counts["invalid_or_oversized"],
            "json_validity_rate": counts["valid"] / total if total else 0,
            "severity_accuracy": counts["severity_correct"] / total if total else 0,
            "severity_recall": {key: value["correct"] / value["total"] if value["total"] else None
                                for key, value in by_severity.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="held-out test JSONL")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_and_validate_jsonl(args.data_path)
    runtime = LogSageInference(base_model=args.base_model, adapter_dir=args.adapter)
    report = {"dataset_sha256": dataset_fingerprint(rows), "base_model": args.base_model,
              "adapter": args.adapter, "metrics": evaluate(rows, runtime)}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
