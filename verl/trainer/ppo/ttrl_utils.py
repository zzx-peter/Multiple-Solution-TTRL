# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from collections import Counter
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.causal import _graph_key, _parse_prediction
from verl.utils.reward_score.three_d import _parse_prediction as _parse_3d_prediction
from verl.utils.reward_score.three_d import _structure_key


def select_top_k_per_prompt(data: DataProto, n_votes_per_prompt: int, n_samples_per_prompt: int) -> DataProto:
    """Keep the first training samples after using all samples for voting."""
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    assert n_samples_per_prompt <= n_votes_per_prompt, "n_samples_per_prompt must be <= n_votes_per_prompt"

    indices = []
    for start in range(0, len(data), n_votes_per_prompt):
        indices.extend(range(start, start + n_samples_per_prompt))
    return data[torch.tensor(indices, dtype=torch.long)]


def apply_original_gt(batch: DataProto) -> DataProto:
    """Restore original ground truth saved by apply_ttrl_gt."""
    for i in range(len(batch)):
        reward_model = batch[i].non_tensor_batch["reward_model"]
        if "original_gt" in reward_model:
            reward_model["ground_truth"] = reward_model["original_gt"]
    return batch


def apply_ttrl_gt(
    batch: DataProto,
    gen_batch_output: DataProto,
    n_votes_per_prompt: int,
    tokenizer,
    num_ans: int = 1,
) -> DataProto:
    """Replace ground truth with a majority-vote pseudo label for each prompt."""
    assert len(gen_batch_output) % n_votes_per_prompt == 0, (
        "gen_batch_output length must be divisible by n_votes_per_prompt"
    )
    assert num_ans >= 1, "num_ans must be >= 1"
    num_prompts = len(gen_batch_output) // n_votes_per_prompt
    assert len(batch) == num_prompts, "batch length must match number of prompt groups"

    responses = _decode_responses(gen_batch_output, tokenizer)
    majority_gts = []
    majority_keys = []
    majority_ratios = []

    for prompt_idx in range(num_prompts):
        start = prompt_idx * n_votes_per_prompt
        original_gt = batch[prompt_idx].non_tensor_batch["reward_model"]["ground_truth"]
        data_source = str(batch[prompt_idx].non_tensor_batch.get("data_source", ""))
        prompt_outputs = responses[start : start + n_votes_per_prompt]
        pseudo_gt, majority_key, majority_ratio = _majority_vote(
            data_source,
            prompt_outputs,
            original_gt,
            num_ans=num_ans,
        )
        majority_gts.append(pseudo_gt)
        majority_keys.append(majority_key)
        majority_ratios.append(majority_ratio)

    for i in range(num_prompts):
        reward_model = batch[i].non_tensor_batch["reward_model"]
        original_gt = reward_model.get("original_gt", reward_model["ground_truth"])
        reward_model["original_gt"] = original_gt
        reward_model["ground_truth"] = majority_gts[i]
        reward_model["majority_gt"] = majority_keys[i]

    batch.non_tensor_batch["majority_gt"] = np.array(majority_keys, dtype=object)
    batch.non_tensor_batch["majority_ratio"] = np.array(majority_ratios, dtype=np.float32)
    return batch


def _majority_vote(
    data_source: str,
    model_outputs: list[str],
    original_gt: Any,
    num_ans: int = 1,
) -> tuple[dict[str, Any], list[str], float]:
    if data_source == "causal":
        return _majority_vote_causal(model_outputs, original_gt, num_ans=num_ans)
    if data_source == "3d":
        return _majority_vote_3d(model_outputs, original_gt, num_ans=num_ans)
    raise NotImplementedError(f"TTRL majority voting is not implemented for data_source={data_source!r}")


def _decode_responses(data: DataProto, tokenizer) -> list[str]:
    prompt_len = data.batch["prompts"].shape[-1]
    response_ids = data.batch["responses"]
    valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

    responses = []
    for i in range(len(data)):
        valid_len = valid_response_lengths[i].item()
        valid_response_ids = response_ids[i][:valid_len]
        responses.append(tokenizer.decode(valid_response_ids, skip_special_tokens=True))
    return responses


def _majority_vote_causal(
    model_outputs: list[str],
    original_gt: Any,
    num_ans: int = 1,
) -> tuple[dict[str, Any], list[str], float]:
    nodes = original_gt.get("nodes") if isinstance(original_gt, dict) else None
    max_edges = original_gt.get("max_edges") if isinstance(original_gt, dict) else None

    predictions = []
    for output in model_outputs:
        prediction = _parse_prediction(output, nodes=nodes, max_edges=max_edges)
        if prediction is not None:
            predictions.append(prediction)
    if not predictions:
        return _make_causal_ground_truth([], original_gt), [], 0.0

    top_predictions = Counter(predictions).most_common(num_ans)
    majority_ratio = top_predictions[0][1] / len(model_outputs)
    majority_edges_list = [edges for edges, _ in top_predictions]
    return (
        _make_causal_ground_truth(majority_edges_list, original_gt),
        [_graph_key(edges) for edges in majority_edges_list],
        majority_ratio,
    )


def _make_causal_ground_truth(
    edges_list: list[tuple[tuple[str, str], ...]],
    original_gt: Any,
) -> dict[str, Any]:
    pseudo_gt = dict(original_gt) if isinstance(original_gt, dict) else {}
    pseudo_gt["ground_truth_list"] = [
        {"edges": [[src, dst] for src, dst in edges]}
        for edges in edges_list
    ]
    return pseudo_gt


def _majority_vote_3d(
    model_outputs: list[str],
    original_gt: Any,
    num_ans: int = 1,
) -> tuple[dict[str, Any], list[str], float]:
    grid_size = int(original_gt.get("grid_size", 3)) if isinstance(original_gt, dict) else 3
    max_height = original_gt.get("max_height") if isinstance(original_gt, dict) else None

    predictions = []
    for output in model_outputs:
        prediction = _parse_3d_prediction(output, grid_size=grid_size, max_height=max_height)
        if prediction is not None:
            predictions.append(prediction)
    if not predictions:
        return _make_3d_ground_truth([], original_gt), [], 0.0

    top_predictions = Counter(predictions).most_common(num_ans)
    majority_ratio = top_predictions[0][1] / len(model_outputs)
    majority_structures = [structure for structure, _ in top_predictions]
    return (
        _make_3d_ground_truth(majority_structures, original_gt),
        [_structure_key(structure) for structure in majority_structures],
        majority_ratio,
    )


def _make_3d_ground_truth(
    structures: list[tuple[tuple[tuple[int, ...], ...], ...]],
    original_gt: Any,
) -> dict[str, Any]:
    pseudo_gt = dict(original_gt) if isinstance(original_gt, dict) else {}
    pseudo_gt["ground_truth_list"] = [
        {
            "grid_size": int(pseudo_gt.get("grid_size", 3)),
            "layers": [[list(row) for row in layer] for layer in structure],
        }
        for structure in structures
    ]
    return pseudo_gt
