from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.rare_valid_metrics import (
    _is_true_valid,
    compute_rare_valid_metrics,
    format_epoch_metrics,
    merge_epoch_totals,
    new_epoch_totals,
)
from verl.utils.reward_score.boolean import _expression_key, _parse_formula


def _causal_ground_truth(*graphs):
    return {
        "nodes": ["A", "B", "C"],
        "max_edges": 3,
        "ground_truth_list": [
            {"edges": [edge.split("->") for edge in graph.split(",")]}
            for graph in graphs
        ],
    }


def test_rare_valid_metrics_use_original_gt_and_apply_both_rules():
    group_one_outcomes = ["A->B", "A->C", "A->C", "B->A", "", "", "", ""]
    group_two_outcomes = ["A->B", "B->A", "", "", "", "", "", ""]
    outcomes = group_one_outcomes + group_two_outcomes

    rewards = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rewards += [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    scalar_advantages = [-0.5, 1.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0]
    scalar_advantages += [2.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    true_gt_one = _causal_ground_truth("A->B", "A->C")
    true_gt_two = _causal_ground_truth("A->B")
    pseudo_gt = _causal_ground_truth("B->A")
    reward_models = [
        {"ground_truth": pseudo_gt, "original_gt": true_gt_one} for _ in range(8)
    ] + [
        {"ground_truth": pseudo_gt, "original_gt": true_gt_two} for _ in range(8)
    ]

    batch = SimpleNamespace(
        batch={
            "token_level_scores": torch.tensor([[reward, 0.0] for reward in rewards]),
            "advantages": torch.tensor([[advantage, advantage] for advantage in scalar_advantages]),
            "response_mask": torch.ones(16, 2),
        },
        non_tensor_batch={
            "uid": np.array(["group-one"] * 8 + ["group-two"] * 8, dtype=object),
            "data_source": np.array(["causal"] * 16, dtype=object),
            "causal_pred": np.array(outcomes, dtype=object),
            "reward_model": np.array(reward_models, dtype=object),
        },
    )

    metrics, step_totals = compute_rare_valid_metrics(batch)

    # Old rule includes both groups. Their rare-valid (reward, advantage) pairs
    # are (0, -0.5) and (1, 2), respectively.
    assert metrics["rare_valid/old_rule/num_prompts"] == 2
    assert metrics["rare_valid/old_rule/num_rare_outcomes"] == 2
    assert metrics["rare_valid/old_rule/reward"] == pytest.approx(0.5)
    assert metrics["rare_valid/old_rule/advantage"] == pytest.approx(0.75)
    assert metrics["rare_valid/old_rule/num_reinforced_rare_outcomes"] == 1
    assert metrics["rare_valid/old_rule/reinforced_ratio"] == pytest.approx(0.5)

    # New rule excludes group two because it has only one distinct valid outcome.
    assert metrics["rare_valid/new_rule/num_prompts"] == 1
    assert metrics["rare_valid/new_rule/num_rare_outcomes"] == 1
    assert metrics["rare_valid/new_rule/reward"] == pytest.approx(0.0)
    assert metrics["rare_valid/new_rule/advantage"] == pytest.approx(-0.5)
    assert metrics["rare_valid/new_rule/num_reinforced_rare_outcomes"] == 0
    assert metrics["rare_valid/new_rule/reinforced_ratio"] == pytest.approx(0.0)

    epoch_totals = new_epoch_totals()
    merge_epoch_totals(epoch_totals, step_totals)
    merge_epoch_totals(epoch_totals, step_totals)
    epoch_metrics = format_epoch_metrics(epoch_totals)
    assert epoch_metrics["rare_valid_epoch/old_rule/num_prompts"] == 4
    assert epoch_metrics["rare_valid_epoch/old_rule/reward"] == pytest.approx(0.5)
    assert epoch_metrics["rare_valid_epoch/old_rule/num_reinforced_rare_outcomes"] == 2
    assert epoch_metrics["rare_valid_epoch/old_rule/reinforced_ratio"] == pytest.approx(0.5)
    assert epoch_metrics["rare_valid_epoch/new_rule/num_prompts"] == 2
    assert epoch_metrics["rare_valid_epoch/new_rule/advantage"] == pytest.approx(-0.5)
    assert epoch_metrics["rare_valid_epoch/new_rule/num_reinforced_rare_outcomes"] == 0
    assert epoch_metrics["rare_valid_epoch/new_rule/reinforced_ratio"] == pytest.approx(0.0)


def test_true_validity_routes_all_supported_tasks():
    structure_key = "[[[1,0,0],[0,0,0],[0,0,0]]]"
    structure_gt = {
        "ground_truth_list": [
            {"layers": [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]}
        ]
    }
    assert _is_true_valid("3d", structure_key, structure_gt)

    expression_key = _expression_key(_parse_formula("x AND y", max_depth=2))
    expression_gt = {
        "variables": ["x", "y"],
        "operators": ["AND", "OR", "NOT"],
        "max_depth": 2,
        "ground_truth_list": [{"formula": "y AND x"}],
    }
    assert _is_true_valid("boolean", expression_key, expression_gt)

    assert _is_true_valid(
        "mace_math",
        r"\frac{1}{2}",
        {"ground_truth_list": ["0.5"]},
    )
