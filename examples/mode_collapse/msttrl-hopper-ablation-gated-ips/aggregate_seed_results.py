#!/usr/bin/env python3
"""Aggregate final validation metrics across five no-gated-ips MSTTRL seeds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WORKSPACE = Path("/scratch/Projects/CFP-04/CFP04-CF-024/ophv164")
RESULT_ROOT = WORKSPACE / "result/mode_collapse"
TB_ROOT = WORKSPACE / "code/Multiple-Solution-TTRL/tensorboard_log/mode_collapse"
OUTPUT = WORKSPACE / "result/no_gated_ips_msttrl_seed_summary.json"
SEEDS = (0, 1, 42, 1337, 3447)
TASKS = {
    "causal": {"step": 360, "n": 16},
    "3d": {"step": 120, "n": 27},
    "boolean": {"step": 90, "n": 20},
}


def experiment_name(task: str, seed: int) -> str:
    if task == "causal":
        return f"Qwen3-1.7B-causal-no-gated-ips-msttrl_lambda_0.9_high_0.875_seed_{seed}_2"
    if task == "3d":
        return (
            "Qwen3-1.7B-3d-no-gated-ips-msttrl_lambda_0.9_high_0.875_"
            f"low_0.5_verifiern_8_seed_{seed}"
        )
    return f"Qwen3-1.7B-boolean-no-gated-ips-msttrl_lambda_0.9_high_0.875_seed_{seed}"


def metric_tags(task: str, n: int) -> dict[str, str]:
    return {
        "acc": f"val-core/{task}/acc/mean@{n}",
        "recovery_rate": f"val-core/{task}/{task}_recovery/recovery@{n}/mean",
        "mode_recovery_rate": f"val-core/{task}/mode_recovery_rate/recovery@{n}/mean",
    }


def seed_metrics(task: str, seed: int) -> dict[str, float | str | int]:
    config = TASKS[task]
    name = experiment_name(task, seed)
    checkpoint = (
        RESULT_ROOT
        / name
        / f"global_step_{config['step']}/actor/model_world_size_1_rank_0.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint not found: {checkpoint}")

    log_dir = TB_ROOT / name
    if not log_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard directory not found: {log_dir}")
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    values: dict[str, float | str | int] = {
        "checkpoint": str(checkpoint),
        "global_step": config["step"],
    }
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    anchor_time = checkpoint.stat().st_mtime
    for metric, tag in metric_tags(task, config["n"]).items():
        if tag not in scalar_tags:
            raise KeyError(f"Missing TensorBoard scalar {tag!r} in {log_dir}")
        events = [event for event in accumulator.Scalars(tag) if event.step == config["step"]]
        if not events:
            raise ValueError(f"No step {config['step']} event for {tag!r} in {log_dir}")
        event = min(events, key=lambda item: abs(item.wall_time - anchor_time))
        values[metric] = float(event.value)
    return values


def main() -> None:
    output: dict[str, object] = {
        "metadata": {
            "model": "Qwen3-1.7B",
            "method": "no-gated-ips-msttrl",
            "seeds": list(SEEDS),
            "variance_type": "population variance",
        },
        "tasks": {},
    }
    tasks_output: dict[str, object] = {}
    for task in TASKS:
        per_seed = {str(seed): seed_metrics(task, seed) for seed in SEEDS}
        aggregate = {}
        for metric in ("acc", "recovery_rate", "mode_recovery_rate"):
            values = [float(per_seed[str(seed)][metric]) for seed in SEEDS]
            aggregate[metric] = {
                "mean": statistics.mean(values),
                "variance": statistics.pvariance(values),
            }
        tasks_output[task] = {"seeds": per_seed, "aggregate": aggregate}
    output["tasks"] = tasks_output

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(OUTPUT)

    print(f"Results written to {OUTPUT}")
    for task, task_result in tasks_output.items():
        print(f"\n{task}")
        for metric, values in task_result["aggregate"].items():
            print(f"  {metric}: mean={values['mean']:.8f}, variance={values['variance']:.8f}")


if __name__ == "__main__":
    main()
