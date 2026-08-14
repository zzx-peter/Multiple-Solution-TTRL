#!/usr/bin/env python3
"""Combine fresh validation runs with seed 1337 metrics from the training logs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WORKSPACE = Path("/scratch/Projects/CFP-04/CFP04-CF-024/ophv164")
REPO = WORKSPACE / "code/Multiple-Solution-TTRL"
RESULT_ROOT = WORKSPACE / "result/qwen_seed1337_checkpoint_eval"
ORIGINAL_ROOT = WORKSPACE / "result/mode_collapse"
TB_ROOT = REPO / "tensorboard_log"
TASKS = {"causal": (360, 16), "3d": (120, 27), "boolean": (90, 20)}
METHODS = ("base", "ips-grpo", "ttrl", "msttrl")
SEEDS = (0, 1, 42, 1337, 3447)


def original_experiment(task: str, method: str) -> str:
    if method in {"base", "ips-grpo"}:
        return f"Qwen3-1.7B-{task}-reason-ips-grpo-norm"
    if method == "ttrl":
        return f"Qwen3-1.7B-{task}-reason-ttrl"
    if task == "causal":
        return "Qwen3-1.7B-causal-gated-ips-msttrl_lambda_0.9_high_0.875_seed_1337_2"
    return f"Qwen3-1.7B-{task}-gated-ips-msttrl_lambda_0.9_high_0.875_seed_1337"


def checkpoint_path(task: str, method: str) -> Path:
    if method == "base":
        return WORKSPACE / "model/Qwen3-1.7B"
    final_step, _ = TASKS[task]
    return ORIGINAL_ROOT / original_experiment(task, method) / f"global_step_{final_step}"


def metric_tags(task: str, n: int) -> dict[str, str]:
    return {
        "acc": f"val-core/{task}/acc/mean@{n}",
        "recovery_rate": f"val-core/{task}/{task}_recovery/recovery@{n}/mean",
        "mode_recovery_rate": f"val-core/{task}/mode_recovery_rate/recovery@{n}/mean",
    }


def load_accumulator(log_dir: Path) -> EventAccumulator:
    if not log_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard directory not found: {log_dir}")
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def select_scalar(
    accumulator: EventAccumulator,
    tag: str,
    step: int,
    anchor_time: float | None = None,
) -> tuple[float, float]:
    if tag not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"Missing TensorBoard scalar: {tag}")
    candidates = [event for event in accumulator.Scalars(tag) if event.step == step]
    if not candidates:
        raise ValueError(f"No event for tag={tag!r}, step={step}")
    if anchor_time is None:
        chosen = max(candidates, key=lambda event: event.wall_time)
    else:
        chosen = min(candidates, key=lambda event: abs(event.wall_time - anchor_time))
    return float(chosen.value), float(chosen.wall_time)


def extract_metrics(
    log_dir: Path,
    task: str,
    step: int,
    anchor_time: float | None = None,
) -> tuple[dict[str, float], float]:
    _, n = TASKS[task]
    accumulator = load_accumulator(log_dir)
    values: dict[str, float] = {}
    event_times: list[float] = []
    for name, tag in metric_tags(task, n).items():
        value, event_time = select_scalar(accumulator, tag, step, anchor_time)
        values[name] = value
        event_times.append(event_time)
    return values, statistics.mean(event_times)


def original_seed_result(task: str, method: str) -> dict:
    final_step, _ = TASKS[task]
    source_experiment = original_experiment(task, method)
    step = 0 if method == "base" else final_step

    # A directory can contain events from several restarts. Anchor the selected
    # event to the checkpoint timestamp so seed 1337 comes from the matching run.
    if method == "base":
        anchor_checkpoint = (
            ORIGINAL_ROOT
            / original_experiment(task, "ips-grpo")
            / f"global_step_{final_step}/actor/model_world_size_1_rank_0.pt"
        )
    else:
        anchor_checkpoint = checkpoint_path(task, method) / "actor/model_world_size_1_rank_0.pt"
    if not anchor_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint anchor not found: {anchor_checkpoint}")

    log_dir = TB_ROOT / "mode_collapse" / source_experiment
    metrics, event_time = extract_metrics(log_dir, task, step, anchor_checkpoint.stat().st_mtime)
    return {
        **metrics,
        "source": "original_training_validation",
        "tensorboard_dir": str(log_dir),
        "global_step": step,
        "event_wall_time": event_time,
    }


def fresh_seed_result(task: str, method: str, seed: int) -> dict:
    final_step, _ = TASKS[task]
    step = 0 if method == "base" else final_step
    experiment = f"{method}-{task}-eval-seed-{seed}"
    log_dir = TB_ROOT / "qwen_seed1337_checkpoint_eval" / experiment
    metrics, event_time = extract_metrics(log_dir, task, step)
    trajectory = RESULT_ROOT / f"runs/{task}/{method}/seed_{seed}/validation/{step}.jsonl"
    if not trajectory.is_file():
        raise FileNotFoundError(f"Validation trajectory not found: {trajectory}")
    return {
        **metrics,
        "source": "fresh_checkpoint_validation",
        "tensorboard_dir": str(log_dir),
        "trajectory_file": str(trajectory),
        "global_step": step,
        "event_wall_time": event_time,
    }


def aggregate_seed_results(seed_results: dict[str, dict]) -> dict:
    summary = {}
    for metric in ("acc", "recovery_rate", "mode_recovery_rate"):
        values = [seed_results[str(seed)][metric] for seed in SEEDS]
        summary[metric] = {
            "mean": statistics.mean(values),
            "variance": statistics.pvariance(values),
        }
    return summary


def main() -> None:
    output = {
        "metadata": {
            "model": "Qwen3-1.7B",
            "training_checkpoint_seed": 1337,
            "evaluation_seeds": list(SEEDS),
            "seed_1337_policy": "reused from the matching original training validation log; not rerun",
            "fresh_seed_policy": "sets both data.seed and the vLLM engine seed",
            "variance_type": "population variance across the five evaluation seeds",
            "metrics": {
                "acc": "mean correctness over all sampled answers",
                "recovery_rate": "task recovery accumulated over all n sampled answers",
                "mode_recovery_rate": "mode recovery accumulated over all n sampled answers",
            },
        },
        "tasks": {},
    }

    for task, (final_step, validation_n) in TASKS.items():
        output["tasks"][task] = {}
        for method in METHODS:
            seed_results = {}
            for seed in SEEDS:
                if seed == 1337:
                    result = original_seed_result(task, method)
                else:
                    result = fresh_seed_result(task, method, seed)
                seed_results[str(seed)] = result
            output["tasks"][task][method] = {
                "checkpoint": str(checkpoint_path(task, method)),
                "checkpoint_training_seed": None if method == "base" else 1337,
                "validation_n": validation_n,
                "seeds": seed_results,
                "aggregate": aggregate_seed_results(seed_results),
            }

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_ROOT / "results.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
