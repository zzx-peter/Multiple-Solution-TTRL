# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CoCoV reward manager for the causal, 3D, and Boolean TTRL tasks.

The routing in this module follows the released TTRL-CoCoV implementation:

* route A (high consistency): keep the Stage-1 majority pseudo label;
* route M (middle consistency): keep the majority label without verification;
* route B1 (low consistency with a verified candidate): use the most frequent
  candidate receiving strictly more than half of the verifier votes;
* route B2 (low consistency without a verified candidate): keep the majority
  label for diagnostics, but suppress the Stage-1 policy advantage.

Only the task-specific verifier prompts are shared with MSTTRL.  CoCoV does not
use MSTTRL's memory bank or its correct/candidate/wrong solution sets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.multiple_solution_ttrl import (
    MultipleSolutionTTRLRewardManager,
)


def _cfg_value(config, key: str, default: Any) -> Any:
    cocov_config = config.get("cocov", {})
    return cocov_config.get(key, default)


def classify_consistency(
    majority_rate: float,
    high_threshold: float = 0.6,
    low_threshold: float = 0.4,
) -> str:
    """Return CoCoV's high/middle/low consistency bucket."""
    if majority_rate >= high_threshold:
        return "high"
    if majority_rate <= low_threshold:
        return "low"
    return "middle"


@register("cocov")
class CoCoVRewardManager(MultipleSolutionTTRLRewardManager):
    """Stateful two-pass reward manager implementing TTRL-CoCoV routing."""

    requires_single_worker = True
    supports_agent_reward_loop = False

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        # The parent supplies task parsers, prompt construction, and the
        # DataProto builder.  All MSTTRL routing methods are overridden below.
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
        self.method = "cocov"
        self.verifier_n = int(_cfg_value(config, "verifier_n", 8))
        self.high_threshold = float(_cfg_value(config, "high_threshold", 0.6))
        self.low_threshold = float(_cfg_value(config, "low_threshold", 0.4))
        self.high_max_candidates = int(_cfg_value(config, "high_max_candidates", 3))
        self.low_max_candidates = int(_cfg_value(config, "low_max_candidates", 5))
        self.high_temperature = float(_cfg_value(config, "high_temperature", 1.0))
        self.low_temperature = float(_cfg_value(config, "low_temperature", 0.6))
        self.top_p = float(_cfg_value(config, "top_p", 0.85))
        self.max_new_tokens = int(_cfg_value(config, "max_new_tokens", 2048))
        self.rollout_response_length = int(config.actor_rollout_ref.rollout.response_length)
        self.fallback_mode = str(_cfg_value(config, "fallback_mode", "no_update_both"))

        if self.verifier_n < 1:
            raise ValueError("cocov.verifier_n must be at least 1.")
        if self.low_threshold > self.high_threshold:
            raise ValueError("cocov.low_threshold must be <= cocov.high_threshold.")
        if self.fallback_mode not in {"no_update_second", "no_update_both"}:
            raise ValueError(
                "cocov.fallback_mode must be 'no_update_second' or 'no_update_both'."
            )

        self.pending_group_state: dict[str, dict[str, Any]] = {}
        self.pending_scores: dict[tuple[str, str], float] = {}
        self.pending_verifier_prompts: list[list[dict[str, str]]] = []

    async def prepare_verification_batch(self, data):
        """Extract Stage-1 candidates and construct CoCoV verifier rollouts."""
        rows = await self._prepare_rows(data)
        rows_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_prompt[row["prompt_key"]].append(row)

        self.pending_rows = rows
        self.pending_group_state = {}
        self.pending_scores = {}

        prompts: list[list[dict[str, str]]] = []
        prompt_keys: list[str] = []
        solution_modes: list[str] = []
        sampling_params: list[dict[str, Any]] = []

        for prompt_key, group_rows in rows_by_prompt.items():
            valid_modes = [
                row["solution_mode"]
                for row in group_rows
                if not self._is_invalid_mode(row["solution_mode"])
            ]
            counts = Counter(valid_modes)
            majority_mode, majority_count = counts.most_common(1)[0] if counts else (None, 0)
            majority_rate = majority_count / max(1, len(group_rows))
            bucket = classify_consistency(
                majority_rate,
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
            )

            max_candidates = self.high_max_candidates if bucket == "high" else self.low_max_candidates
            candidates = counts.most_common(max_candidates) if bucket != "middle" else []
            route = {"high": "A", "middle": "M", "low": "B2"}[bucket]
            self.pending_group_state[prompt_key] = {
                "bucket": bucket,
                "route": route,
                "counts": counts,
                "majority_mode": majority_mode,
                "majority_rate": majority_rate,
                "num_votes": len(group_rows),
                "selected_mode": majority_mode,
                "selected_consistency": majority_rate,
                "zero_advantage": bucket == "low" and self.fallback_mode == "no_update_both",
            }

            if bucket == "middle":
                continue

            temperature = self.high_temperature if bucket == "high" else self.low_temperature
            for solution_mode, _frequency in candidates:
                messages = self._verification_messages(
                    self.prompt_bank[prompt_key],
                    solution_mode,
                    self.data_source_bank[prompt_key],
                    self.ground_truth_bank[prompt_key],
                )
                for _ in range(self.verifier_n):
                    prompts.append(messages)
                    prompt_keys.append(prompt_key)
                    solution_modes.append(solution_mode)
                    sampling_params.append(
                        {
                            "temperature": temperature,
                            "top_p": self.top_p,
                            # The mode-collapse baselines keep the actor's 1024
                            # token response limit.  Preserve CoCoV's 2048-token
                            # default while respecting that shared model limit.
                            "max_tokens": min(self.max_new_tokens, self.rollout_response_length),
                        }
                    )

        self.pending_verifier_prompts = prompts
        verification_batch = self._build_generation_batch(prompts)
        if verification_batch is not None:
            # AgentLoopWorker consumes this internal field as a per-sample
            # sampling override and does not expose it to task agent loops.
            verification_batch.non_tensor_batch["__sampling_params__"] = np.array(
                sampling_params, dtype=object
            )
            verification_batch.non_tensor_batch["__skip_reward__"] = np.ones(
                len(verification_batch), dtype=bool
            )

        bucket_counts = Counter(state["bucket"] for state in self.pending_group_state.values())
        return {
            "batch": verification_batch,
            "prompt_keys": prompt_keys,
            "solution_modes": solution_modes,
            "num_modes": len({(key, mode) for key, mode in zip(prompt_keys, solution_modes, strict=True)}),
            "num_samples": len(prompts),
            "high_groups": bucket_counts["high"],
            "middle_groups": bucket_counts["middle"],
            "low_groups": bucket_counts["low"],
        }

    def apply_verification_results(
        self,
        prompt_keys: list[str],
        solution_modes: list[str],
        verifier_outputs: list[str],
    ) -> dict[str, Any]:
        """Aggregate verifier votes and finish CoCoV's A/M/B1/B2 routing."""
        if not (len(prompt_keys) == len(solution_modes) == len(verifier_outputs)):
            raise ValueError("Verifier result lengths do not match.")

        votes: dict[tuple[str, str], list[bool]] = defaultdict(list)
        parse_fail_count = 0
        for prompt_key, solution_mode, output in zip(
            prompt_keys, solution_modes, verifier_outputs, strict=True
        ):
            parsed = self._parse_verification_result(output)
            if parsed is None:
                parse_fail_count += 1
            else:
                votes[(prompt_key, solution_mode)].append(parsed)

        true_count = 0
        false_count = 0
        self.pending_scores = {}
        for key, results in votes.items():
            positives = sum(results)
            true_count += positives
            false_count += len(results) - positives
            self.pending_scores[key] = positives / self.verifier_n

        for prompt_key, state in self.pending_group_state.items():
            if state["bucket"] != "low":
                continue

            verified_candidates = []
            for mode, frequency in state["counts"].most_common(self.low_max_candidates):
                positives = sum(votes.get((prompt_key, mode), []))
                # The released code uses the configured verification budget as
                # the denominator, so parse failures do not lower the majority bar.
                if positives > self.verifier_n / 2:
                    verified_candidates.append((mode, frequency, positives))

            if verified_candidates:
                # CoCoV's low-consistency "majority" strategy prioritizes the
                # Stage-1 frequency and uses verifier positives as the tie-break.
                verified_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
                selected_mode, selected_count, _ = verified_candidates[0]
                state["selected_mode"] = selected_mode
                state["selected_consistency"] = selected_count / max(1, state["num_votes"])
                state["route"] = "B1"
                state["zero_advantage"] = False

        route_counts = Counter(state["route"] for state in self.pending_group_state.values())
        return {
            "num_modes": len({(key, mode) for key, mode in zip(prompt_keys, solution_modes, strict=True)}),
            "num_samples": len(verifier_outputs),
            "true_count": true_count,
            "false_count": false_count,
            "parse_fail_count": parse_fail_count,
            "route_a": route_counts["A"],
            "route_m": route_counts["M"],
            "route_b1": route_counts["B1"],
            "route_b2": route_counts["B2"],
        }

    def _build_outputs(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs = []
        for row in rows:
            state = self.pending_group_state.get(row["prompt_key"])
            if state is None:
                raise RuntimeError("CoCoV rewards requested without a prepared verification batch.")

            is_selected = (
                state["selected_mode"] is not None
                and row["solution_mode"] == state["selected_mode"]
                and not self._is_invalid_mode(row["solution_mode"])
            )
            reward = float(is_selected)
            outputs.append(
                {
                    "reward_score": reward,
                    "reward_extra_info": {
                        "cocov_reward": reward,
                        "cocov_answer_type": 0 if is_selected else 1,
                        "cocov_consistency": float(state["selected_consistency"]),
                        "cocov_majority_rate": float(state["majority_rate"]),
                        "cocov_route": state["route"],
                        "cocov_zero_advantage": float(state["zero_advantage"]),
                        "cocov_solution_mode": row["solution_mode"],
                        "cocov_selected_mode": state["selected_mode"] or "<invalid>",
                        **row["task_reward_extra_info"],
                    },
                }
            )
        return outputs
