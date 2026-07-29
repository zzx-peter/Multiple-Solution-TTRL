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


def test_msttrl_prior_block_from_correct_set():
    manager = _make_manager()
    observations = ["Perturb(A) -> A:0 B:1"]
    prompt_key = manager._prompt_key_from_observations(observations)
    manager.solution_sets[prompt_key][SET_CORRECT] = {"causal:A->B", "causal:C->D"}
    prior = manager.prior_block_for_observations(observations)
    assert prior == "Graph: A->B\nGraph: C->D"


def test_msttrl_inject_prior_into_prompt_text():
    manager = _make_manager()
    observations = ["Perturb(A) -> A:0 B:1"]
    prompt_key = manager._prompt_key_from_observations(observations)
    manager.solution_sets[prompt_key][SET_CORRECT] = {"causal:A->B"}

    base_prompt = """
    Observations:
    Perturb(A) -> A:0 B:1

    Task:
    Output a graph.
    """.strip()
    updated = manager.inject_prior_into_messages([{"role": "user", "content": base_prompt}])[0]["content"]
    assert "Prior predictions (do not repeat if avoidable):" in updated
    assert "Graph: A->B" in updated
    assert updated.index("Graph: A->B") < updated.index("Task:")


def test_msttrl_prior_defaults_to_none_for_unknown_prompt():
    manager = _make_manager()
    base_prompt = """
    Observations:
    Perturb(A) -> A:0 B:1

    Task:
    Output a graph.
    """.strip()
    updated = manager.inject_prior_into_messages([{"role": "user", "content": base_prompt}])[0]["content"]
    assert "Prior predictions (do not repeat if avoidable):" in updated
    assert "None" in updated


def test_self_reconsider_boundary_modes_builds_prompt_and_moves_only_valid_modes():
    manager = _make_manager()
    prompt_key = "prompt-key"
    manager.prompt_bank[prompt_key] = """
    Perturb(A) -> A:0 B:1 C:0 D:0
    Perturb(B) -> A:0 B:0 C:0 D:0
    """
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
