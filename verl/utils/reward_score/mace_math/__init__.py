# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Multi-answer MACE Math reward built on verl's MATH equivalence rules."""

from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score import math_reward


def _normalize_answer(answer: Any) -> str:
    answer = str(answer)
    try:
        return math_reward.strip_string(answer)
    except Exception:
        return answer.strip()


def _parse_prediction(solution_str: str) -> Optional[str]:
    boxed = math_reward.last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        return _normalize_answer(math_reward.remove_boxed(boxed))
    except Exception:
        return None


def _get_ground_truth_list(ground_truth: Any) -> list[Any]:
    if isinstance(ground_truth, dict):
        return list(ground_truth.get("ground_truth_list", []))
    if isinstance(ground_truth, (list, tuple)):
        return list(ground_truth)
    return [ground_truth] if ground_truth is not None else []


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """Score the last boxed answer against every valid MACE reference answer."""
    references = _get_ground_truth_list(ground_truth)
    valid_answers = {_normalize_answer(reference) for reference in references}
    prediction = _parse_prediction(solution_str)
    if prediction is None:
        return {
            "score": format_score,
            "acc": format_score,
            "mace_math_parse_success": 0.0,
            "mace_math_valid": 0.0,
            "mace_math_pred": "",
            "mace_math_n_ground_truths": len(valid_answers),
        }

    # Reuse the same normalization/equivalence behavior as verl's MATH reward.
    is_valid = any(math_reward.is_equiv(prediction, reference) for reference in valid_answers)
    result_score = score if is_valid else format_score
    return {
        "score": result_score,
        "acc": result_score,
        "mace_math_parse_success": 1.0,
        "mace_math_valid": float(is_valid),
        "mace_math_pred": prediction,
        "mace_math_n_ground_truths": len(valid_answers),
    }


__all__ = ["compute_score"]
