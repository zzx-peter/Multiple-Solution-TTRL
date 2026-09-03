"""Training metrics for rewards and advantages assigned to rare valid outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from verl.utils.reward_score import math_reward
from verl.utils.reward_score.boolean import (
    _expression_key,
    _ground_truth_expression,
    _parse_formula,
)
from verl.utils.reward_score.causal import _canonical_graph, _graph_key
from verl.utils.reward_score.mace_math import _normalize_answer
from verl.utils.reward_score.three_d import _canonical_structure, _structure_key


_PREDICTION_KEYS = {
    "causal": "causal_pred",
    "3d": "3d_pred",
    "boolean": "boolean_pred",
    "mace_math": "mace_math_pred",
}
_RULES = {
    "old_rule": 1,
    "new_rule": 2,
}


def new_epoch_totals() -> dict[str, dict[str, float]]:
    """Return an empty prompt-weighted accumulator for one training epoch."""
    return {
        rule: {
            "reward_sum": 0.0,
            "advantage_sum": 0.0,
            "num_prompts": 0.0,
            "num_rare_outcomes": 0.0,
            "num_reinforced_rare_outcomes": 0.0,
        }
        for rule in _RULES
    }


def merge_epoch_totals(
    totals: dict[str, dict[str, float]],
    update: dict[str, dict[str, float]],
) -> None:
    """Accumulate one step without averaging equally across unequal batches."""
    for rule in _RULES:
        for key in totals[rule]:
            totals[rule][key] += update[rule][key]


def format_epoch_metrics(totals: dict[str, dict[str, float]]) -> dict[str, float]:
    """Format an exact equal-prompt epoch average for the training logger."""
    metrics: dict[str, float] = {}
    for rule in _RULES:
        values = totals[rule]
        count = int(values["num_prompts"])
        prefix = f"rare_valid_epoch/{rule}"
        metrics[f"{prefix}/num_prompts"] = float(count)
        metrics[f"{prefix}/num_rare_outcomes"] = float(values["num_rare_outcomes"])
        metrics[f"{prefix}/num_reinforced_rare_outcomes"] = float(
            values["num_reinforced_rare_outcomes"]
        )
        if values["num_rare_outcomes"]:
            metrics[f"{prefix}/reinforced_ratio"] = (
                values["num_reinforced_rare_outcomes"]
                / values["num_rare_outcomes"]
            )
        if count:
            metrics[f"{prefix}/reward"] = values["reward_sum"] / count
            metrics[f"{prefix}/advantage"] = values["advantage_sum"] / count
    return metrics


def compute_rare_valid_metrics(batch) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute prompt-level rare-valid reward/advantage averages for one step.

    A valid outcome is judged against the dataset's original ground truth, not
    against TTRL/CoCoV pseudo labels.  Within each prompt group, every truly
    valid outcome tied at the minimum positive frequency is selected.  Its
    reward and advantage are first averaged over occurrences, then across tied
    outcomes.  The returned step metrics average prompts equally.

    ``old_rule`` requires at least one distinct valid outcome. ``new_rule``
    requires at least two; repeated samples of one outcome still count as one.
    A rare-valid outcome is considered reinforced when its mean actual
    training advantage over all occurrences is strictly greater than zero.
    """
    required_batch_keys = {"token_level_scores", "advantages", "response_mask"}
    missing_batch = required_batch_keys - set(batch.batch.keys())
    if missing_batch:
        raise ValueError(f"rare-valid metrics missing batch fields: {sorted(missing_batch)}")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("rare-valid metrics require batch.non_tensor_batch['uid']")

    batch_size = int(batch.batch["token_level_scores"].shape[0])
    uids = _as_list(batch.non_tensor_batch["uid"], batch_size)
    data_sources = _as_list(batch.non_tensor_batch.get("data_source", [""] * batch_size), batch_size)
    reward_models = _as_list(batch.non_tensor_batch.get("reward_model", [{}] * batch_size), batch_size)
    prediction_columns = {
        key: _as_list(batch.non_tensor_batch.get(key, [""] * batch_size), batch_size)
        for key in set(_PREDICTION_KEYS.values()) | {"pred"}
    }

    rewards = batch.batch["token_level_scores"].sum(dim=-1).detach().float().cpu().numpy()
    response_mask = batch.batch["response_mask"].bool()
    advantages = batch.batch["advantages"].detach().float()
    response_lengths = response_mask.sum(dim=-1)
    scalar_advantages = (
        (advantages * response_mask).sum(dim=-1)
        / response_lengths.clamp_min(1)
    ).cpu().numpy()

    outcomes: list[str] = []
    ground_truths: list[Any] = []
    for index, data_source_value in enumerate(data_sources):
        data_source = str(_unwrap(data_source_value))
        prediction_key = _PREDICTION_KEYS.get(data_source)
        prediction = prediction_columns[prediction_key][index] if prediction_key else prediction_columns["pred"][index]
        prediction = _unwrap(prediction)
        outcomes.append(str(prediction) if prediction not in (None, "") else "<invalid>")

        reward_model = _unwrap(reward_models[index])
        if isinstance(reward_model, dict):
            ground_truths.append(reward_model.get("original_gt", reward_model.get("ground_truth")))
        else:
            ground_truths.append(None)

    indices_by_uid: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        indices_by_uid[str(_unwrap(uid))].append(index)

    totals = new_epoch_totals()
    for indices in indices_by_uid.values():
        group_sources = {str(_unwrap(data_sources[index])) for index in indices}
        if len(group_sources) != 1:
            raise ValueError(f"One rollout group contains multiple data sources: {sorted(group_sources)}")
        data_source = next(iter(group_sources))
        if data_source not in _PREDICTION_KEYS:
            continue

        outcome_counts = Counter(outcomes[index] for index in indices)
        outcome_is_valid: dict[str, bool] = {}
        group_supported = True
        for outcome in outcome_counts:
            occurrence_indices = [index for index in indices if outcomes[index] == outcome]
            decisions = {
                _is_true_valid(data_source, outcome, ground_truths[index])
                for index in occurrence_indices
            }
            if None in decisions:
                group_supported = False
                break
            if len(decisions) != 1:
                raise ValueError(f"Inconsistent true validity for outcome {outcome!r}")
            outcome_is_valid[outcome] = bool(next(iter(decisions)))
        if not group_supported:
            continue

        valid_outcomes = [outcome for outcome, valid in outcome_is_valid.items() if valid]
        if not valid_outcomes:
            continue
        minimum_frequency = min(outcome_counts[outcome] for outcome in valid_outcomes)
        rare_outcomes = [
            outcome for outcome in valid_outcomes if outcome_counts[outcome] == minimum_frequency
        ]

        rare_rewards = []
        rare_advantages = []
        for outcome in rare_outcomes:
            occurrence_indices = [index for index in indices if outcomes[index] == outcome]
            rare_rewards.append(float(np.mean(rewards[occurrence_indices])))
            rare_advantages.append(float(np.mean(scalar_advantages[occurrence_indices])))
        prompt_reward = float(np.mean(rare_rewards))
        prompt_advantage = float(np.mean(rare_advantages))
        num_reinforced_rare_outcomes = sum(
            advantage > 0.0 for advantage in rare_advantages
        )

        for rule, minimum_distinct_valid in _RULES.items():
            if len(valid_outcomes) < minimum_distinct_valid:
                continue
            totals[rule]["reward_sum"] += prompt_reward
            totals[rule]["advantage_sum"] += prompt_advantage
            totals[rule]["num_prompts"] += 1.0
            totals[rule]["num_rare_outcomes"] += float(len(rare_outcomes))
            totals[rule]["num_reinforced_rare_outcomes"] += float(
                num_reinforced_rare_outcomes
            )

    metrics: dict[str, float] = {}
    for rule in _RULES:
        values = totals[rule]
        count = int(values["num_prompts"])
        prefix = f"rare_valid/{rule}"
        metrics[f"{prefix}/num_prompts"] = float(count)
        metrics[f"{prefix}/num_rare_outcomes"] = float(values["num_rare_outcomes"])
        metrics[f"{prefix}/num_reinforced_rare_outcomes"] = float(
            values["num_reinforced_rare_outcomes"]
        )
        if values["num_rare_outcomes"]:
            metrics[f"{prefix}/reinforced_ratio"] = (
                values["num_reinforced_rare_outcomes"]
                / values["num_rare_outcomes"]
            )
        if count:
            metrics[f"{prefix}/reward"] = values["reward_sum"] / count
            metrics[f"{prefix}/advantage"] = values["advantage_sum"] / count
    return metrics, totals


def _as_list(values: Any, expected_length: int) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    values = list(values)
    if len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {len(values)}")
    return values


def _unwrap(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _ground_truth_list(ground_truth: Any) -> list[Any] | None:
    if isinstance(ground_truth, dict):
        values = ground_truth.get("ground_truth_list")
        return list(values) if isinstance(values, (list, tuple)) else None
    if isinstance(ground_truth, (list, tuple)):
        return list(ground_truth)
    return None


def _is_true_valid(data_source: str, outcome: str, ground_truth: Any) -> bool | None:
    if outcome == "<invalid>":
        return False
    references = _ground_truth_list(ground_truth)
    if references is None:
        return None

    if data_source == "causal":
        valid = {
            _graph_key(_canonical_graph(reference))
            for reference in references
            if isinstance(reference, dict)
        }
        return outcome in valid

    if data_source == "3d":
        valid = {
            _structure_key(structure)
            for reference in references
            if (structure := _canonical_structure(reference)) is not None
        }
        return outcome in valid

    if data_source == "boolean":
        variables = ground_truth.get("variables", ["x", "y"]) if isinstance(ground_truth, dict) else ["x", "y"]
        operators = (
            ground_truth.get("operators", ["AND", "OR", "NOT"])
            if isinstance(ground_truth, dict)
            else ["AND", "OR", "NOT"]
        )
        max_depth = ground_truth.get("max_depth") if isinstance(ground_truth, dict) else None
        valid = set()
        for reference in references:
            formula = _ground_truth_expression(reference)
            if formula is None:
                continue
            expression = _parse_formula(
                formula,
                variables=variables,
                operators=operators,
                max_depth=max_depth,
            )
            if expression is not None:
                valid.add(_expression_key(expression))
        return outcome in valid

    if data_source == "mace_math":
        normalized_references = [_normalize_answer(reference) for reference in references]
        return any(math_reward.is_equiv(outcome, reference) for reference in normalized_references)

    return None


__all__ = [
    "compute_rare_valid_metrics",
    "format_epoch_metrics",
    "merge_epoch_totals",
    "new_epoch_totals",
]
