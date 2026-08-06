import asyncio
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.multiple_solution_ttrl import (
    MultipleSolutionTTRLRewardManager,
    SET_CORRECT,
)


class ToyTokenizer:
    eos_token = "<eos>"

    def decode(self, token_ids, skip_special_tokens=True):
        vocab = {1: "A", 2: "B", 9: "prompt", 0: ""}
        return "".join(vocab.get(int(token), "") for token in token_ids)


def _make_batch(response_ids, confidences):
    response_ids = torch.tensor(response_ids, dtype=torch.long)
    bsz, response_len = response_ids.shape
    prompt_len = 1
    prompts = torch.full((bsz, prompt_len), 9, dtype=torch.long)
    response_mask = response_ids.ne(0)
    attention_mask = torch.cat([torch.ones_like(prompts), response_mask.long()], dim=1)
    rollout_log_probs = torch.zeros_like(response_ids, dtype=torch.float32)
    for i, confidence in enumerate(confidences):
        rollout_log_probs[i, response_mask[i]] = math.log(confidence)

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": response_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask.long(),
            "rollout_log_probs": rollout_log_probs,
        },
        batch_size=bsz,
    )
    non_tensor_batch = {
        "data_source": np.array(["toy"] * bsz, dtype=object),
        "reward_model": np.array([{"ground_truth": None} for _ in range(bsz)], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def _toy_compute_score(data_source, solution_str, ground_truth, extra_info):
    pred = solution_str.strip().lower()
    return {"score": 1.0, "acc": 1.0, "pred": pred}


def _invalid_causal_compute_score(data_source, solution_str, ground_truth, extra_info):
    return {"score": 0.0, "acc": 0.0, "causal_pred": ""}


def _make_manager(alpha=0.5, novelty_lambda=1.0):
    config = OmegaConf.create(
        {
            "msttrl": {
                "plausibility_alpha": alpha,
                "novelty_lambda": novelty_lambda,
            },
            "reward": {"num_workers": 1},
        }
    )
    return MultipleSolutionTTRLRewardManager(config=config, tokenizer=ToyTokenizer(), compute_score=_toy_compute_score)


def test_multiple_solution_ttrl_rewards_novel_modes_then_uses_history():
    manager = _make_manager(alpha=0.5, novelty_lambda=1.0)

    first = asyncio.run(manager.run_batch(_make_batch([[1], [2]], confidences=[0.8, 0.5])))
    assert first[0]["reward_extra_info"]["msttrl_solution_mode"] == "toy:a"
    assert first[0]["reward_extra_info"]["msttrl_final_answer"] == "a"
    assert first[0]["reward_extra_info"]["acc"] == 1.0
    assert first[0]["reward_extra_info"]["pred"] == "a"
    assert first[0]["reward_extra_info"]["msttrl_novelty"] == 1.0
    expected_first_history = 1.0 - math.exp(-1.0)
    expected_first_a = (0.5 * 0.8 + 0.5 * expected_first_history) * 2.0
    expected_first_b = (0.5 * 0.5 + 0.5 * expected_first_history) * 2.0
    assert first[0]["reward_extra_info"]["msttrl_history_count"] == 1.0
    assert first[0]["reward_extra_info"]["msttrl_historical_stability"] == pytest.approx(expected_first_history)
    assert first[0]["reward_score"] == pytest.approx(expected_first_a)
    assert first[1]["reward_score"] == pytest.approx(expected_first_b)

    second = asyncio.run(manager.run_batch(_make_batch([[1]], confidences=[0.25])))
    expected_history = 1.0 - math.exp(-2.0)
    expected_plausibility = 0.5 * 0.25 + 0.5 * expected_history
    assert second[0]["reward_extra_info"]["msttrl_novelty"] == 0.0
    assert second[0]["reward_extra_info"]["msttrl_history_count"] == 2.0
    assert second[0]["reward_extra_info"]["msttrl_historical_stability"] == pytest.approx(expected_history)
    assert second[0]["reward_score"] == pytest.approx(expected_plausibility)


def test_multiple_solution_ttrl_alpha_zero_rewards_first_batch_after_memory_update():
    manager = _make_manager(alpha=0.0, novelty_lambda=1.0)

    first = asyncio.run(manager.run_batch(_make_batch([[1], [1]], confidences=[0.8, 0.5])))

    expected_history = 1.0 - math.exp(-2.0)
    expected_reward = expected_history * 2.0
    assert first[0]["reward_extra_info"]["msttrl_history_count"] == 2.0
    assert first[0]["reward_extra_info"]["msttrl_novelty"] == 1.0
    assert first[0]["reward_score"] == pytest.approx(expected_reward)
    assert first[1]["reward_score"] == pytest.approx(expected_reward)


def test_multiple_solution_ttrl_invalid_modes_do_not_enter_memory_bank():
    config = OmegaConf.create({"msttrl": {"plausibility_alpha": 0.0, "novelty_lambda": 1.0}, "reward": {"num_workers": 1}})
    manager = MultipleSolutionTTRLRewardManager(
        config=config,
        tokenizer=ToyTokenizer(),
        compute_score=_invalid_causal_compute_score,
    )
    data = _make_batch([[1], [2]], confidences=[0.8, 0.5])
    data.non_tensor_batch["data_source"] = np.array(["causal"] * len(data), dtype=object)

    outputs = asyncio.run(manager.run_batch(data))
    state = manager.get_state()

    assert outputs[0]["reward_extra_info"]["msttrl_solution_mode"] == "causal:<invalid>"
    assert outputs[0]["reward_extra_info"]["msttrl_history_count"] == 0.0
    assert outputs[0]["reward_score"] == 0.0
    assert state["solution_sets"] == []
    assert state["total_count"] == 0


def test_multiple_solution_ttrl_requires_rollout_log_probs():
    manager = _make_manager()
    data = _make_batch([[1]], confidences=[0.8])
    data.batch.pop("rollout_log_probs")

    with pytest.raises(ValueError, match="calculate_log_probs=True"):
        asyncio.run(manager.run_batch(data))


def test_multiple_solution_ttrl_run_single_allows_agent_loop_placeholder():
    manager = _make_manager()
    data = _make_batch([[1]], confidences=[0.8])
    data.batch.pop("rollout_log_probs")

    output = asyncio.run(manager.run_single(data))
    assert output == {"reward_score": 0.0, "reward_extra_info": {}}


def test_set_mode_correctness_statistics_reports_each_set_purity():
    solution_sets = [
        {
            "correct_set": [{"is_ground_truth": True}, {"is_ground_truth": False}],
            "candidate_set": [{"is_ground_truth": True}],
            "wrong_set": [{"is_ground_truth": False}, {"is_ground_truth": False}],
        },
        {
            "correct_set": [{"is_ground_truth": True}],
            "candidate_set": [],
            "wrong_set": [{"is_ground_truth": True}],
        },
    ]

    statistics = MultipleSolutionTTRLRewardManager._set_mode_correctness_statistics(solution_sets)

    assert statistics["correct_set"] == {
        "num_modes": 3,
        "num_correct_modes": 2,
        "num_incorrect_modes": 1,
        "correct_mode_ratio": pytest.approx(2 / 3),
        "incorrect_mode_ratio": pytest.approx(1 / 3),
    }
    assert statistics["candidate_set"] == {
        "num_modes": 1,
        "num_correct_modes": 1,
        "num_incorrect_modes": 0,
        "correct_mode_ratio": 1.0,
        "incorrect_mode_ratio": 0.0,
    }
    assert statistics["wrong_set"] == {
        "num_modes": 3,
        "num_correct_modes": 1,
        "num_incorrect_modes": 2,
        "correct_mode_ratio": pytest.approx(1 / 3),
        "incorrect_mode_ratio": pytest.approx(2 / 3),
    }


def test_set_mode_correctness_statistics_uses_none_for_empty_sets():
    statistics = MultipleSolutionTTRLRewardManager._set_mode_correctness_statistics([])

    for set_statistics in statistics.values():
        assert set_statistics["num_modes"] == 0
        assert set_statistics["num_correct_modes"] == 0
        assert set_statistics["num_incorrect_modes"] == 0
        assert set_statistics["correct_mode_ratio"] is None
        assert set_statistics["incorrect_mode_ratio"] is None


def test_self_reconsider_boundary_modes_builds_prompt_and_moves_only_valid_modes():
    manager = _make_manager()
    prompt_key = "prompt-key"
    manager.prompt_bank[prompt_key] = """
    Perturb(A) -> A:0 B:1 C:0 D:0
    Perturb(B) -> A:0 B:0 C:0 D:0
    """
    manager.data_source_bank[prompt_key] = "causal"
    manager.observation_bank[prompt_key] = [
        "Perturb(A) -> A:0 B:1 C:0 D:0",
        "Perturb(B) -> A:0 B:0 C:0 D:0",
    ]
    manager.solution_sets[prompt_key][SET_CORRECT] = {
        "causal:A->B",
        "causal:A->B, B->C",
        "causal:C->D",
        "causal:D->A",
    }

    messages = manager._self_reconsider_boundary_modes(prompt_key)
    assert "Perturb(A) -> A:0 B:1 C:0 D:0" in messages[1]["content"]
    assert "A->B, B->C" in messages[1]["content"]
    assert "Select the 2 most suspicious" in messages[1]["content"]

    moved = manager._self_reconsider_boundary_modes(
        prompt_key,
        """
Suspicious Mode: C->D
Suspicious Mode: hallucinated->mode
not the required format: D->A
Suspicious Mode: C->D
Suspicious Mode: D->A
Suspicious Mode: A->B
""",
    )

    assert moved == ["causal:C->D", "causal:D->A"]
    assert manager.solution_sets[prompt_key][SET_CORRECT] == {
        "causal:A->B",
        "causal:A->B, B->C",
    }
    assert manager.solution_sets[prompt_key]["candidate"] == {
        "causal:C->D",
        "causal:D->A",
    }
    assert manager.get_state()["self_reconsider_failure_count"] == 1

    manager.reset_epoch_stats()
    assert manager.get_state()["self_reconsider_failure_count"] == 0

    manager.solution_sets[prompt_key][SET_CORRECT] = {
        "causal:A->B",
        "causal:C->D",
        "causal:D->A",
    }
    manager.solution_sets[prompt_key]["candidate"].clear()
    manager._self_reconsider_boundary_modes(prompt_key, "")
    assert manager.get_state()["self_reconsider_failure_count"] == 1

    manager.solution_sets[prompt_key][SET_CORRECT] = {
        "causal:A->B",
        "causal:C->D",
        "causal:D->A",
    }
    manager.solution_sets[prompt_key]["candidate"].clear()
    manager._self_reconsider_boundary_modes(
        prompt_key,
        "Suspicious Mode: C->D\nSuspicious Mode: D->A",
    )
    assert manager.get_state()["self_reconsider_failure_count"] == 2
    assert manager.solution_sets[prompt_key]["candidate"] == {"causal:C->D"}


def test_three_d_solution_mode_and_verifier_prompt_are_routed_by_task():
    manager = MultipleSolutionTTRLRewardManager.__new__(MultipleSolutionTTRLRewardManager)
    structure_key = "[[[1,0,0],[0,0,0],[0,0,0]],[[1,0,0],[0,0,0],[0,0,0]]]"
    solution_mode = manager._solution_mode("", "3d", {"3d_pred": structure_key})
    prompt = """
Maximum allowed height: 3 layers.

### OBSERVATION
Top View (1=block visible, 0=empty):
1 0 0
0 0 0
0 0 0
"""

    messages = manager._verification_messages(
        prompt,
        solution_mode,
        "3d",
        {"grid_size": 3, "max_height": 3, "max_blocks": 3},
    )

    assert solution_mode == f"3d:{structure_key}"
    assert "3D block structures" in messages[0]["content"]
    assert "Top View (1=block visible, 0=empty):" in messages[1]["content"]
    assert "Layer 2:" in messages[1]["content"]
    assert "projected Top View is exactly Layer 1" in messages[1]["content"]
    assert "Mismatches:" in messages[1]["content"]
    assert "if and only if" in messages[1]["content"]
    assert "Compare all nine cells" in messages[1]["content"]
    assert "at most 3 blocks in total" not in messages[1]["content"]
    assert "causal graph" not in messages[1]["content"]


def test_verifier_routing_uses_data_source_not_solution_mode_or_prompt_text():
    manager = MultipleSolutionTTRLRewardManager.__new__(MultipleSolutionTTRLRewardManager)
    prompt = """
Perturb(A) -> A:0 B:1
Top View (1=block visible, 0=empty):
1 0 0
0 0 0
0 0 0
"""

    messages = manager._verification_messages(prompt, "3d:<invalid>", "causal")

    assert "causal-graph" in messages[0]["content"]
    assert "Perturb(A) -> A:0 B:1" in messages[1]["content"]
    assert "3D block structure" not in messages[1]["content"]


def test_invalid_three_d_mode_still_uses_three_d_verifier():
    manager = MultipleSolutionTTRLRewardManager.__new__(MultipleSolutionTTRLRewardManager)
    prompt = """
Top View (1=block visible, 0=empty):
1 0 0
0 0 0
0 0 0
"""

    messages = manager._verification_messages(
        prompt,
        "anything:<invalid>",
        "3d",
        {"grid_size": 3, "max_height": 3, "max_blocks": 3},
    )

    assert "3D block structures" in messages[0]["content"]
    assert "Top View (1=block visible, 0=empty):" in messages[1]["content"]
    assert "Candidate structure:\n<invalid>" in messages[1]["content"]


def test_three_d_ground_truth_uses_canonical_structures():
    manager = MultipleSolutionTTRLRewardManager.__new__(MultipleSolutionTTRLRewardManager)
    ground_truth = {
        "ground_truth_list": [
            {
                "grid_size": 3,
                "layers": [
                    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
            }
        ],
        "grid_size": 3,
        "max_height": 3,
    }
    formatted = manager._format_ground_truth(ground_truth, "3d")
    assert formatted["num_structures"] == 1
