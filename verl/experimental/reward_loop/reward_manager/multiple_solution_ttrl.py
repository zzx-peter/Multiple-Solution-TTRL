# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy
import hashlib
import inspect
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

VERIFIER_SYSTEM_PROMPT = "You are a rigorous verifier for causal-graph candidates."

# The parser below looks for a "Verification Result" line, so keep the verifier
# prompt explicit and stable across sampled verification attempts.
VERIFIER_USER_TEMPLATE = """Verify whether the candidate causal graph explains the perturbation observations.

Semantics:
- When a node is perturbed, the perturbed node is 0.
- A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.
- All other nodes are 0.

Nodes: A, B, C, D

Constraint: The graph should have at most 6 edges.

Reasoning example:
An edge `X->Y` points from the upstream cause X to the downstream effect Y.
Follow arrows only in this left-to-right direction; do not reverse an edge.
For candidate graph `A->B, B->D`, B and D are descendants of A, while C is not.
Therefore, perturbing A gives A:0, B:1, C:0, D:1, so
`Perturb(A) -> A:0 B:1 C:0 D:1` is consistent. In contrast,
`Perturb(A) -> A:0 B:0 C:0 D:1` is inconsistent because B is reachable from A.

Perturbation observations:
{observations}

Candidate graph:
{candidate}

Reason through reachability for every listed perturbation. Judge only this candidate; multiple candidates may be valid.

First provide a reasoning paragraph. Then end your response with exactly one line:
Verification Result: True
or
Verification Result: False
"""

SET_CORRECT = "correct"
SET_CANDIDATE = "candidate"
SET_WRONG = "wrong"
VERIFIER_MAX_PROBLEM_CHARS = 2048
PRIOR_SECTION_HEADER = "Prior predictions (do not repeat if avoidable):"
PRIOR_SECTION_PATTERN = re.compile(
    r"\n\s*Prior predictions \(do not repeat if avoidable\):\s*\n.*?(?=\n\s*Task:|\n\s*Diversity rule:|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _read_cfg(config, key: str, default: Any) -> Any:
    msttrl_cfg = config.get("msttrl", {})
    if key in msttrl_cfg:
        return msttrl_cfg[key]
    reward_kwargs = config.reward.get("reward_kwargs", {})
    return reward_kwargs.get(key, default)


@register("multiple_solution_ttrl")
@register("msttrl")
class MultipleSolutionTTRLRewardManager(RewardManagerBase):
    """Label-free multiple-solution TTRL reward manager.

    For each prompt, it maintains three disjoint sets of solution modes:
    high-confidence correct, candidate, and high-confidence wrong. New modes
    and retained candidates are routed by self-verification pass rate.
    """

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
        super().__init__(config, tokenizer, compute_score)
        if config.reward.num_workers != 1:
            raise ValueError("MultipleSolutionTTRLRewardManager requires reward.num_workers=1.")

        self.compute_score = compute_score or default_compute_score
        self.method = str(_read_cfg(config, "method", "msttrl"))
        if self.method not in {"msttrl", "verifier"}:
            raise ValueError(f"Unknown msttrl.method={self.method!r}. Expected 'msttrl' or 'verifier'.")
        self.verifier_n = int(_read_cfg(config, "verifier_n", 4))
        self.high_threshold = float(_read_cfg(config, "high_threshold", 0.75))
        self.low_threshold = float(_read_cfg(config, "low_threshold", 0.25))
        self.inject_prior_predictions = bool(_read_cfg(config, "inject_prior_predictions", False))
        self.verifier_max_prompt_length = min(
            int(config.data.max_prompt_length),
            int(config.actor_rollout_ref.rollout.prompt_length),
        )
        # Keep verifier prompt formatting aligned with actor rollout prompts,
        # e.g. Qwen3's `enable_thinking=False`.
        self.apply_chat_template_kwargs = dict(config.data.get("apply_chat_template_kwargs", {}))
        if self.low_threshold > self.high_threshold:
            raise ValueError("msttrl.low_threshold must be <= msttrl.high_threshold.")
        # Counts how often each prompt emits each valid solution mode across
        # training batches. This is used for inspection and boundary-mode moves.
        self.memory_bank: dict[str, Counter[str]] = defaultdict(Counter)
        # Stores prompt text by prompt hash so later verifier prompts can be
        # built from compact prompt keys carried through the rollout batch.
        self.prompt_bank: dict[str, str] = {}
        # Stores compact debug-only problem fields for JSON dumps.
        self.observation_bank: dict[str, list[str]] = {}
        self.ground_truth_bank: dict[str, dict[str, Any]] = {}
        # Counts valid generated modes and modes sent to verifier since the
        # last solution-set dump. These are mode-level efficiency counters.
        self.valid_mode_count_bank: dict[str, int] = defaultdict(int)
        self.verified_mode_count_bank: dict[str, int] = defaultdict(int)
        # The three disjoint per-prompt sets from the MSTTRL note:
        # S_t(correct), C_t(candidate), and W_t(wrong).
        self.solution_sets: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {SET_CORRECT: set(), SET_CANDIDATE: set(), SET_WRONG: set()}
        )
        # Latest verifier pass rate p_t(z) for candidate modes.
        self.plausibility_bank: dict[str, dict[str, float]] = defaultdict(dict)
        # Last summed reward per prompt; increases trigger a boundary refresh.
        self.last_reward_sum: dict[str, float] = {}
        # Two-stage handshake with the trainer: prepare_verification_batch()
        # extracts and caches rollout rows, apply_verification_results() routes
        # modes, and run_batch() then reuses the same rows to emit rewards.
        self.pending_rows: list[dict[str, Any]] | None = None
        self.pending_verify_modes: dict[str, list[str]] = {}
        self.pending_scores: dict[tuple[str, str], float] = {}
        self.pending_verifier_prompts: list[list[dict[str, str]]] = []
        # Epoch-level debug data for modes that verifier routing places into
        # correct_set, plus one concrete verifier request/response example.
        self.epoch_verifier_correct_count = 0
        self.epoch_verifier_correct_ground_truth_count = 0
        self.epoch_verifier_example: dict[str, Any] | None = None

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        # During normal MSTTRL training the verifier path prepares rows first.
        # Reusing them here avoids decoding the same rollout twice and avoids a
        # second memory update for the same responses.
        if self.pending_rows is not None:
            rows = self.pending_rows
            self.pending_rows = None
        else:
            rows = await self._prepare_rows(data)
        outputs = self._build_outputs(rows)
        # Validation batches should be scored against the current memory only;
        # they must not change solution sets or future training rewards.
        if self.method == "msttrl" and not any(row.get("validate", False) for row in rows):
            self._maintain_sets(rows, outputs)
        return outputs

    async def prepare_verification_batch(self, data: DataProto) -> dict[str, Any]:
        # Stage 1 of MSTTRL reward computation: extract solution modes from the
        # rollout batch, choose unresolved modes to verify, and build a new
        # generation batch whose responses will be True/False verifier votes.
        rows = await self._prepare_rows(data)
        generated_modes = self._count_valid_modes(rows)
        if self.method == "verifier":
            verify_modes = {prompt_key: sorted(counter) for prompt_key, counter in generated_modes.items() if counter}
        else:
            verify_modes = self._modes_to_verify(rows, generated_modes=generated_modes)
        for prompt_key, counter in generated_modes.items():
            self.valid_mode_count_bank[prompt_key] += len(counter)
        for prompt_key, modes in verify_modes.items():
            self.verified_mode_count_bank[prompt_key] += len(modes)
        self.pending_rows = rows
        self.pending_verify_modes = verify_modes
        self.pending_scores = {}

        prompts = []
        prompt_keys = []
        solution_modes = []
        for prompt_key, modes in verify_modes.items():
            for mode in modes:
                for _ in range(self.verifier_n):
                    # Multiple sampled verifier responses estimate p_t(z) as a
                    # pass rate instead of relying on a single brittle judgment.
                    messages = self._verification_messages(self.prompt_bank[prompt_key], mode)
                    prompts.append(messages)
                    prompt_keys.append(prompt_key)
                    solution_modes.append(mode)
        self.pending_verifier_prompts = prompts

        return {
            "batch": self._build_generation_batch(prompts),
            "prompt_keys": prompt_keys,
            "solution_modes": solution_modes,
            "num_modes": sum(len(modes) for modes in verify_modes.values()),
            "num_samples": len(prompts),
        }

    def apply_verification_results(
        self,
        prompt_keys: list[str],
        solution_modes: list[str],
        verifier_outputs: list[str],
    ) -> dict[str, Any]:
        # Stage 2 of MSTTRL reward computation: parse verifier generations back
        # into votes, aggregate them per (prompt, solution mode), and update the
        # correct/candidate/wrong sets before rewards are assembled.
        if len(prompt_keys) != len(solution_modes) or len(solution_modes) != len(verifier_outputs):
            raise ValueError("Verifier result lengths do not match.")

        votes: dict[tuple[str, str], list[bool]] = defaultdict(list)
        parse_fail_count = 0
        for prompt_key, solution_mode, text in zip(prompt_keys, solution_modes, verifier_outputs, strict=True):
            parsed = self._parse_verification_result(text)
            if parsed is None:
                # Unparseable outputs are counted for metrics and excluded from
                # the pass-rate denominator for this mode.
                parse_fail_count += 1
                continue
            votes[(prompt_key, solution_mode)].append(parsed)

        self.pending_scores = {}
        true_count = 0
        false_count = 0
        for key, results in votes.items():
            true_votes = sum(1 for result in results if result)
            false_votes = len(results) - true_votes
            true_count += true_votes
            false_count += false_votes
            self.pending_scores[key] = true_votes / len(results) if results else 0.0
            prompt_key, solution_mode = key
            self.plausibility_bank[prompt_key][solution_mode] = self.pending_scores[key]

        if self.epoch_verifier_example is None and verifier_outputs:
            first_prompt = self.pending_verifier_prompts[0] if self.pending_verifier_prompts else []
            first_key = (prompt_keys[0], solution_modes[0])
            self.epoch_verifier_example = {
                "prompt_key": prompt_keys[0],
                "solution_mode": solution_modes[0],
                "verifier_prompt": first_prompt,
                "verifier_output": verifier_outputs[0],
                "parsed_result": self._parse_verification_result(verifier_outputs[0]),
                "pass_rate": self.pending_scores.get(first_key),
            }

        # Modes with no parsed votes are absent from pending_scores. MSTTRL routes
        # them as score 0.0; verifier-only reward reads the same default directly.
        if self.method == "msttrl":
            self._route_pending_modes()
        return {
            "num_modes": len(self.pending_scores),
            "num_samples": len(verifier_outputs),
            "true_count": true_count,
            "false_count": false_count,
            "parse_fail_count": parse_fail_count,
        }

    async def _prepare_rows(self, data: DataProto) -> list[dict[str, Any]]:
        if "rollout_log_probs" not in data.batch:
            raise ValueError(
                "Multiple-solution TTRL requires rollout log probabilities. "
                "Set actor_rollout_ref.rollout.calculate_log_probs=True."
            )

        rows = [await self._extract_row(data, i) for i in range(len(data))]
        for row in rows:
            self.prompt_bank[row["prompt_key"]] = row["prompt_text"]
            self.observation_bank[row["prompt_key"]] = row["observations"]
            self.ground_truth_bank[row["prompt_key"]] = row["ground_truth"]

        # A "solution mode" is the compact answer/mode identity used by the
        # memory bank; invalid extraction markers never enter the mode counts.
        batch_counts = self._count_valid_modes(rows)
        for row in rows:
            row["batch_count"] = batch_counts[row["prompt_key"]][row["solution_mode"]]

        # Training updates the memory bank before verification so current modes
        # are visible in dumps and boundary-selection counts. Validation is read-only.
        if not data.meta_info.get("validate", False):
            for prompt_key, counter in batch_counts.items():
                self.memory_bank[prompt_key].update(counter)

        return rows

    def _build_outputs(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs = []
        for row in rows:
            prompt_memory = self.memory_bank[row["prompt_key"]]
            is_invalid_mode = self._is_invalid_mode(row["solution_mode"])
            history_count = 0 if is_invalid_mode else prompt_memory[row["solution_mode"]]
            confidence = row["confidence"]
            if self.method == "verifier":
                # Verifier baseline: every generated valid mode is rewarded by
                # its verifier pass rate, without S/C/W set routing.
                set_name = "verifier"
                plausibility = 0.0 if is_invalid_mode else self.pending_scores.get(
                    (row["prompt_key"], row["solution_mode"]), 0.0
                )
                reward = plausibility
            else:
                # Reward follows the three-set rule: correct -> 1, candidate ->
                # verifier plausibility, wrong/unknown/invalid -> 0.
                set_name, plausibility = self._mode_state(row["prompt_key"], row["solution_mode"])
                reward = self._mode_reward(set_name, plausibility)

            outputs.append(
                {
                    "reward_score": reward,
                    "reward_extra_info": {
                        "msttrl_reward": reward,
                        "msttrl_plausibility": plausibility,
                        "msttrl_confidence": confidence,
                        "msttrl_history_count": float(history_count),
                        "msttrl_set": set_name,
                        "msttrl_solution_mode": row["solution_mode"],
                        "msttrl_final_answer": row["final_answer"],
                        "msttrl_prompt_key": row["prompt_key"],
                        **row["task_reward_extra_info"],
                    },
                }
            )

        for output in outputs:
            prompt_key = output["reward_extra_info"]["msttrl_prompt_key"]
            output["reward_extra_info"]["msttrl_memory_size"] = float(len(self.memory_bank[prompt_key]))

        return outputs

    def _maintain_sets(self, rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> None:
        # Track reward movement per prompt. When the current batch's total reward
        # is higher than the previous time this prompt appeared, reopen boundary
        # modes for candidate verification.
        reward_sums: dict[str, float] = defaultdict(float)
        for row, output in zip(rows, outputs, strict=True):
            reward_sums[row["prompt_key"]] += float(output["reward_score"])

        for prompt_key, reward_sum in reward_sums.items():
            prev_sum = self.last_reward_sum.get(prompt_key)
            self.last_reward_sum[prompt_key] = reward_sum
            if prev_sum is None:
                continue
            self._reconsider_boundary_modes(prompt_key)

    def _reconsider_boundary_modes(self, prompt_key: str) -> None:
        sets = self.solution_sets[prompt_key]
        counter = self.memory_bank[prompt_key]
        if len(sets[SET_CORRECT]) > 1:
            # Least-observed accepted mode is the weakest member of S_t.
            mode = min(sets[SET_CORRECT], key=lambda item: (counter.get(item, 0), item))
            sets[SET_CORRECT].remove(mode)
            sets[SET_CANDIDATE].add(mode)
        if len(sets[SET_WRONG]) > 1:
            # Most-observed rejected mode is the strongest candidate for being a
            # repeatedly generated alternative that deserves another check.
            mode = max(sets[SET_WRONG], key=lambda item: (counter.get(item, 0), item))
            sets[SET_WRONG].remove(mode)
            sets[SET_CANDIDATE].add(mode)

    def _modes_to_verify(
        self,
        rows: list[dict[str, Any]],
        generated_modes: dict[str, Counter[str]] | None = None,
    ) -> dict[str, list[str]]:
        # U_t(x) = retained candidates union newly generated modes, excluding
        # modes already locked into the high-confidence correct or wrong sets.
        if generated_modes is None:
            generated_modes = self._count_valid_modes(rows)
        verify_modes = {}
        for prompt_key in {row["prompt_key"] for row in rows}:
            sets = self.solution_sets[prompt_key]
            modes = set(sets[SET_CANDIDATE]) | set(generated_modes[prompt_key])
            modes.difference_update(sets[SET_CORRECT])
            modes.difference_update(sets[SET_WRONG])
            if modes:
                verify_modes[prompt_key] = sorted(modes)
        return verify_modes

    @staticmethod
    def _count_valid_modes(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            if not MultipleSolutionTTRLRewardManager._is_invalid_mode(row["solution_mode"]):
                counts[row["prompt_key"]][row["solution_mode"]] += 1
        return counts

    def _route_pending_modes(self) -> None:
        # Route each verified mode using the configured low/high thresholds:
        # score >= high -> S_{t+1}, score <= low -> W_{t+1}, middle -> C_{t+1}.
        for prompt_key, modes in self.pending_verify_modes.items():
            sets = self.solution_sets[prompt_key]
            next_candidates = set()
            for mode in modes:
                score = self.pending_scores.get((prompt_key, mode), 0.0)
                sets[SET_CANDIDATE].discard(mode)
                if score >= self.high_threshold:
                    sets[SET_CORRECT].add(mode)
                    sets[SET_WRONG].discard(mode)
                    self.epoch_verifier_correct_count += 1
                    self.epoch_verifier_correct_ground_truth_count += int(
                        self._is_ground_truth_mode(prompt_key, mode)
                    )
                elif score <= self.low_threshold:
                    sets[SET_WRONG].add(mode)
                    sets[SET_CORRECT].discard(mode)
                else:
                    next_candidates.add(mode)
            next_candidates.difference_update(sets[SET_CORRECT])
            next_candidates.difference_update(sets[SET_WRONG])
            sets[SET_CANDIDATE] = next_candidates

    def _mode_state(self, prompt_key: str, solution_mode: str) -> tuple[str, float]:
        # Unknown modes are treated as wrong until they survive the verifier and
        # get explicitly routed into candidate or correct.
        if self._is_invalid_mode(solution_mode):
            return SET_WRONG, 0.0
        sets = self.solution_sets[prompt_key]
        if solution_mode in sets[SET_CORRECT]:
            return SET_CORRECT, 1.0
        if solution_mode in sets[SET_CANDIDATE]:
            return SET_CANDIDATE, self.plausibility_bank[prompt_key].get(solution_mode, 0.0)
        return SET_WRONG, 0.0

    @staticmethod
    def _mode_reward(set_name: str, plausibility: float) -> float:
        if set_name == SET_CORRECT:
            return 1.0
        if set_name == SET_CANDIDATE:
            return plausibility
        return 0.0

    def get_state(self) -> dict[str, Any]:
        # Returned state is used by trainer-side JSON dumps for inspecting how
        # the solution-mode memory and the three routing sets evolve over epochs.
        solution_sets = []
        total_modes = 0
        total_count = 0
        for prompt_key in sorted(self.memory_bank):
            counter = self.memory_bank[prompt_key]
            mode_counts = {mode: int(count) for mode, count in sorted(counter.items())}
            sets = self.solution_sets[prompt_key]
            total_modes += len(mode_counts)
            total_count += int(sum(mode_counts.values()))
            solution_sets.append(
                {
                    "observations": self.observation_bank.get(prompt_key, []),
                    "ground_truth": self.ground_truth_bank.get(prompt_key, {}),
                    "valid_mode_count": int(self.valid_mode_count_bank.get(prompt_key, 0)),
                    "verified_mode_count": int(self.verified_mode_count_bank.get(prompt_key, 0)),
                    "correct_set": self._set_rows(prompt_key, SET_CORRECT),
                    "candidate_set": self._set_rows(prompt_key, SET_CANDIDATE),
                    "wrong_set": self._set_rows(prompt_key, SET_WRONG),
                    "num_correct": len(sets[SET_CORRECT]),
                    "num_candidate": len(sets[SET_CANDIDATE]),
                    "num_wrong": len(sets[SET_WRONG]),
                }
            )

        return {
            "solution_sets": solution_sets,
            "num_prompts": len(solution_sets),
            "total_modes": total_modes,
            "total_count": total_count,
            "total_valid_modes": sum(item["valid_mode_count"] for item in solution_sets),
            "total_verified_modes": sum(item["verified_mode_count"] for item in solution_sets),
            "total_correct": sum(item["num_correct"] for item in solution_sets),
            "total_candidate": sum(item["num_candidate"] for item in solution_sets),
            "total_wrong": sum(item["num_wrong"] for item in solution_sets),
            "verifier_correct_set_accuracy": {
                "num_modes": self.epoch_verifier_correct_count,
                "num_ground_truth_modes": self.epoch_verifier_correct_ground_truth_count,
                "accuracy": (
                    self.epoch_verifier_correct_ground_truth_count / self.epoch_verifier_correct_count
                    if self.epoch_verifier_correct_count
                    else None
                ),
            },
            "verifier_example": self.epoch_verifier_example,
        }

    def reset_epoch_stats(self) -> dict[str, Any]:
        self.valid_mode_count_bank.clear()
        self.verified_mode_count_bank.clear()
        self.epoch_verifier_correct_count = 0
        self.epoch_verifier_correct_ground_truth_count = 0
        self.epoch_verifier_example = None
        return {}

    def _set_rows(self, prompt_key: str, set_name: str) -> list[dict[str, Any]]:
        counter = self.memory_bank[prompt_key]
        return [
            {
                "solution_mode": mode,
                "count": int(counter.get(mode, 0)),
                "plausibility": self.plausibility_bank[prompt_key].get(mode),
                "is_ground_truth": self._is_ground_truth_mode(prompt_key, mode),
            }
            for mode in sorted(self.solution_sets[prompt_key][set_name])
        ]

    def _is_ground_truth_mode(self, prompt_key: str, solution_mode: str) -> bool:
        ground_truth = self.ground_truth_bank.get(prompt_key, {})
        graphs = set(ground_truth.get("graphs", [])) if isinstance(ground_truth, dict) else set()
        if not graphs or self._is_invalid_mode(solution_mode):
            return False
        if ":" not in solution_mode:
            return False
        _, candidate = solution_mode.split(":", 1)
        return candidate in graphs

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        if "rollout_log_probs" not in data.batch:
            return {"reward_score": 0.0, "reward_extra_info": {}}
        return (await self.run_batch(data[-1:]))[0]

    async def _extract_row(self, data: DataProto, index: int) -> dict[str, Any]:
        # Decode one rollout row into the prompt key, response text, confidence,
        # task-specific reward metadata, and solution-mode identity used by MSTTRL.
        prompt_ids = data.batch["prompts"][index]
        response_ids = data.batch["responses"][index]
        attention_mask = data.batch["attention_mask"][index]
        prompt_len = prompt_ids.shape[-1]

        valid_prompt_len = int(attention_mask[:prompt_len].sum().item())
        valid_prompt_ids = prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[:0]
        prompt_key = self._prompt_key(valid_prompt_ids)
        prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        observations = self._extract_observations(prompt_text)
        if self.method == "msttrl" and observations:
            # Keep bank keys stable even when actor prompts include prior predictions.
            prompt_key = self._prompt_key_from_observations(observations)

        response_mask = data.batch.get("response_mask", None)
        if response_mask is not None:
            valid_response_mask = response_mask[index].bool()
            valid_response_len = int(valid_response_mask.sum().item())
        else:
            valid_response_len = int(attention_mask[prompt_len:].sum().item())
            valid_response_mask = torch.zeros_like(response_ids, dtype=torch.bool)
            valid_response_mask[:valid_response_len] = True

        valid_response_ids = response_ids[:valid_response_len]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        if getattr(self.tokenizer, "eos_token", None) and response_str.endswith(self.tokenizer.eos_token):
            response_str = response_str[: -len(self.tokenizer.eos_token)]

        log_probs = data.batch["rollout_log_probs"][index]
        confidence = self._confidence(log_probs, valid_response_mask)
        data_source = data.non_tensor_batch.get("data_source", [""] * len(data))[index]
        reward_model = data.non_tensor_batch.get("reward_model", [{}] * len(data))[index]
        ground_truth = reward_model.get("ground_truth", None) if isinstance(reward_model, dict) else None
        task_reward_extra_info = await self._task_reward_extra_info(data, index, data_source, response_str, ground_truth)
        solution_mode = self._solution_mode(response_str, data_source, task_reward_extra_info)

        return {
            "prompt_key": prompt_key,
            "prompt_text": prompt_text,
            "observations": observations,
            "ground_truth": self._format_ground_truth(ground_truth),
            "solution_mode": solution_mode,
            "final_answer": self._final_answer(response_str, data_source, task_reward_extra_info),
            "confidence": confidence,
            "validate": data.meta_info.get("validate", False),
            "task_reward_extra_info": task_reward_extra_info,
        }

    @staticmethod
    def _prompt_key(valid_prompt_ids: torch.Tensor) -> str:
        prompt_token_key = ",".join(str(int(token)) for token in valid_prompt_ids.detach().cpu().tolist())
        return hashlib.sha1(prompt_token_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _prompt_key_from_observations(observations: list[str]) -> str:
        payload = "\n".join(line.strip() for line in observations)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _confidence(log_probs: torch.Tensor, response_mask: torch.Tensor) -> float:
        valid_log_probs = log_probs[response_mask]
        if valid_log_probs.numel() == 0:
            return 0.0
        mean_log_prob = torch.clamp(valid_log_probs.float().mean(), min=-60.0, max=0.0)
        return float(torch.exp(mean_log_prob).item())

    async def _task_reward_extra_info(
        self,
        data: DataProto,
        index: int,
        data_source: Any,
        response_str: str,
        ground_truth: Any,
    ) -> dict[str, Any]:
        if self.compute_score is None:
            return {}

        extra_info = data.non_tensor_batch.get("extra_info", [{}] * len(data))[index]
        extra_info = dict(extra_info) if isinstance(extra_info, dict) else {}
        extra_info["num_turns"] = data.non_tensor_batch.get("__num_turns__", [None] * len(data))[index]
        extra_info["rollout_reward_scores"] = data.non_tensor_batch.get("reward_scores", [{}] * len(data))[index]

        result = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            return dict(result)
        score = float(result)
        return {"score": score, "acc": score}

    @staticmethod
    def _solution_mode(response_str: str, data_source: Any, task_reward_extra_info: dict[str, Any]) -> str:
        if str(data_source) == "causal" and "causal_pred" in task_reward_extra_info:
            pred = task_reward_extra_info.get("causal_pred")
            return "Graph:" + (str(pred) if pred else "<invalid>")
        if "pred" in task_reward_extra_info:
            pred = task_reward_extra_info.get("pred")
            if pred is not None:
                return f"{data_source}:{pred}"
        return f"{data_source}:" + response_str.strip()

    @staticmethod
    def _extract_observations(prompt_text: str) -> list[str]:
        return re.findall(r"Perturb\([^)]+\)\s*->\s*[^\n]+", prompt_text or "")

    @staticmethod
    def _graph_from_solution_mode(solution_mode: str) -> str:
        if ":" in solution_mode:
            return solution_mode.split(":", 1)[1]
        return solution_mode

    @classmethod
    def _format_candidate_graph(cls, solution_mode: str) -> str:
        graph = cls._graph_from_solution_mode(solution_mode)
        if not graph or graph == "<invalid>":
            return graph or "<invalid>"
        if graph == "No edges":
            return "No edges"
        edges = [part.strip() for part in graph.split(",") if part.strip()]
        return ", ".join(edges)

    @classmethod
    def format_prior_block(cls, correct_modes: set[str] | list[str]) -> str:
        """Format correct_set modes into the HypoSpace prior-hypothesis block."""
        lines = []
        for mode in sorted(correct_modes):
            graph = cls._graph_from_solution_mode(mode)
            if not graph or graph == "<invalid>":
                continue
            if graph == "No edges":
                lines.append("Graph: No edges")
            else:
                edges = [part.strip() for part in graph.split(",") if part.strip()]
                lines.append("Graph: " + ", ".join(edges))
        return "\n".join(lines) if lines else "None"

    def prior_block_for_observations(self, observations: list[str]) -> str:
        if not observations:
            return "None"
        prompt_key = self._prompt_key_from_observations(observations)
        return self.format_prior_block(self.solution_sets[prompt_key][SET_CORRECT])

    @classmethod
    def inject_prior_into_prompt_text(cls, prompt_text: str, prior_block: str) -> str:
        """Insert or replace the prior-prediction section after Observations."""
        prior_section = f"\n    {PRIOR_SECTION_HEADER}\n    {prior_block}\n"
        if PRIOR_SECTION_PATTERN.search(prompt_text):
            return PRIOR_SECTION_PATTERN.sub(prior_section, prompt_text, count=1)

        for marker in ("\n    Task:", "\nTask:", "\n    Diversity rule:", "\nDiversity rule:"):
            idx = prompt_text.find(marker)
            if idx != -1:
                return prompt_text[:idx] + prior_section + prompt_text[idx:]
        return prompt_text + prior_section

    def inject_prior_into_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.method != "msttrl" or not self.inject_prior_predictions:
            return messages

        updated = copy.deepcopy(messages)
        for message in updated:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            observations = self._extract_observations(content)
            prior_block = self.prior_block_for_observations(observations)
            message["content"] = self.inject_prior_into_prompt_text(content, prior_block)
        return updated

    @staticmethod
    def _format_ground_truth(ground_truth: Any) -> dict[str, Any]:
        if not isinstance(ground_truth, dict):
            return {}

        ground_truth_list = ground_truth.get("ground_truth_list", [])
        graphs = []
        for graph in ground_truth_list:
            edges = graph.get("edges", []) if isinstance(graph, dict) else []
            canonical_edges = sorted({(str(src), str(dst)) for src, dst in edges})
            if canonical_edges:
                graphs.append(",".join(f"{src}->{dst}" for src, dst in canonical_edges))
            else:
                graphs.append("No edges")

        return {
            "nodes": ground_truth.get("nodes"),
            "max_edges": ground_truth.get("max_edges"),
            "num_graphs": len(graphs),
            "graphs": graphs,
        }

    @staticmethod
    def _is_invalid_mode(solution_mode: str) -> bool:
        return solution_mode.endswith(":<invalid>")

    def _build_generation_batch(self, raw_prompts: list[list[dict[str, str]]]) -> DataProto | None:
        if not raw_prompts:
            return None

        raw_prompt_ids = []
        for prompt in raw_prompts:
            try:
                # Prefer model-native chat formatting for verifier prompts.
                ids = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    **self.apply_chat_template_kwargs,
                )
            except Exception:
                # Some tokenizers do not expose chat templates; fall back to a
                # plain role-prefixed prompt that the rollout backend can still sample.
                text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt) + "\nassistant:"
                ids = self.tokenizer.encode(text, add_special_tokens=True)
            raw_prompt_ids.append(ids)

        max_prompt_length = self.verifier_max_prompt_length
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        position_ids = []
        for ids in raw_prompt_ids:
            # Match verl's left-padded prompt convention for rollout generation.
            ids = ids[-max_prompt_length:]
            pad_len = max_prompt_length - len(ids)
            input_ids.append([pad_token_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))
            position_ids.append([0] * pad_len + list(range(len(ids))))

        batch = TensorDict(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
            },
            batch_size=len(raw_prompts),
        )
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "raw_prompt_ids": np.array(raw_prompt_ids, dtype=object),
                "raw_prompt": np.array(raw_prompts, dtype=object),
            },
            meta_info={},
        )

    def _verification_messages(self, prompt_text: str, solution_mode: str) -> list[dict[str, str]]:
        observations = self._extract_observations(prompt_text[-VERIFIER_MAX_PROBLEM_CHARS:])
        return [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VERIFIER_USER_TEMPLATE.format(
                    observations="\n".join(observations) if observations else "No observations found.",
                    candidate=self._format_candidate_graph(solution_mode),
                ),
            },
        ]

    @staticmethod
    def _parse_verification_result(text: str) -> bool | None:
        # Prefer explicit verifier conclusions, allowing minor format drift like
        # "verification result is: False" while still using the last conclusion.
        normalized_text = text or ""
        matches = re.findall(
            r"verification\s+result(?:\s+is)?\s*[:=\-]?\s*(true|false)\b",
            normalized_text,
            flags=re.IGNORECASE,
        )
        if matches:
            return matches[-1].lower() == "true"

        # Conservative fallback: only trust a bare True/False if it appears on
        # the final non-empty line, so reasoning text does not pollute the vote.
        lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
        if lines:
            final_match = re.fullmatch(r"(?:answer\s*[:=\-]?\s*)?(true|false)[.!]?", lines[-1], flags=re.IGNORECASE)
            if final_match is not None:
                return final_match.group(1).lower() == "true"

        print(f"No verification result found in: {text}")
        return None

    @staticmethod
    def _final_answer(response_str: str, data_source: Any, task_reward_extra_info: dict[str, Any]) -> str:
        if str(data_source) == "causal" and "causal_pred" in task_reward_extra_info:
            pred = task_reward_extra_info.get("causal_pred")
            return str(pred) if pred else "<invalid>"
        if "pred" in task_reward_extra_info:
            pred = task_reward_extra_info.get("pred")
            if pred is not None:
                return str(pred)
        return response_str.strip()
