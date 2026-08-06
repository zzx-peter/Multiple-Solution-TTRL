#!/usr/bin/env python3
"""Summarize the low-cost 3D actor/verifier prompt smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prompt_content(value) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        message = value[0]
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return str(value)


def check_parquet(path: Path, marker: str) -> None:
    import pandas as pd

    frame = pd.read_parquet(path)
    prompt = prompt_content(frame.iloc[0]["prompt"])
    if marker not in prompt:
        raise RuntimeError(f"Expected prompt marker {marker!r} is absent from {path}")
    print(f"Data check: {path} contains {len(frame)} rows and the expected prompt marker.")
    print("First loaded prompt:\n" + prompt)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_actor(validation_dir: Path) -> None:
    initial_path = validation_dir / "0.jsonl"
    if not initial_path.exists():
        raise FileNotFoundError(f"Initial validation dump not found: {initial_path}")
    rows = read_jsonl(initial_path)
    count = len(rows)
    accuracy = sum(float(row.get("acc", 0.0)) for row in rows) / count
    parse_rate = sum(float(row.get("3d_parse_success", 0.0)) for row in rows) / count
    print(f"Actor initial validation: n={count}, accuracy={accuracy:.4f}, parse_rate={parse_rate:.4f}")

    good_rows = [row for row in rows if float(row.get("acc", 0.0))]
    if good_rows:
        row = good_rows[0]
        print(f"\nActor success example:\nINPUT:\n{row.get('input', '')}\nOUTPUT:\n{row.get('output', '')}")

    bad_rows = [row for row in rows if not float(row.get("acc", 0.0))]
    for index, row in enumerate(bad_rows[:3], start=1):
        print(f"\nActor failure example {index}:\nINPUT:\n{row.get('input', '')}\nOUTPUT:\n{row.get('output', '')}")


def summarize_verifier(solution_set_path: Path) -> None:
    state = json.loads(solution_set_path.read_text(encoding="utf-8"))
    verified = []
    for prompt in state.get("solution_sets", []):
        for set_name in ("correct_set", "candidate_set", "wrong_set"):
            for row in prompt.get(set_name, []):
                score = row.get("plausibility")
                if score is not None:
                    verified.append((float(score), bool(row.get("is_ground_truth")), set_name))

    if not verified:
        print("Verifier: no verified modes were found.")
        return

    # The routing policy treats scores <= 0.5 as wrong. Use the same boundary
    # for a simple binary verifier diagnostic, separately from set purity.
    predictions = [score > 0.5 for score, _, _ in verified]
    labels = [label for _, label, _ in verified]
    accuracy = sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)
    positive = sum(labels)
    negative = len(labels) - positive
    true_positive = sum(prediction and label for prediction, label in zip(predictions, labels))
    true_negative = sum((not prediction) and (not label) for prediction, label in zip(predictions, labels))
    predicted_positive = sum(predictions)
    print(
        "Verifier epoch 0: "
        f"modes={len(labels)}, accuracy={accuracy:.4f}, "
        f"TPR={true_positive / positive if positive else 0.0:.4f}, "
        f"TNR={true_negative / negative if negative else 0.0:.4f}, "
        f"precision={true_positive / predicted_positive if predicted_positive else 0.0:.4f}"
    )
    print("Set purity:\n" + json.dumps(state.get("set_mode_correctness", {}), indent=2))
    print("Accepted-this-epoch precision:\n" + json.dumps(state.get("verifier_correct_set_accuracy", {}), indent=2))

    example = state.get("verifier_example")
    if example:
        print("\nSaved verifier example:\n" + json.dumps(example, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--marker", default="Required output skeleton")
    args = parser.parse_args()

    if args.train_parquet:
        check_parquet(args.train_parquet, args.marker)
    summarize_actor(args.dump_dir / "validation")
    summarize_verifier(args.dump_dir / "solution_sets" / "epoch_0.json")


if __name__ == "__main__":
    main()
