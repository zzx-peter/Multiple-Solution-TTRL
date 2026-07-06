# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import inspect
import math
from collections import Counter, defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


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

    It assigns r_t(z)=P_t(z)(1+lambda N_t(z)) where P_t mixes response
    confidence and memory-bank stability, and N_t rewards solution modes
    that were unseen before the current reward batch.
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
        self.alpha = float(_read_cfg(config, "plausibility_alpha", 0.5))
        self.alpha = min(max(self.alpha, 0.0), 1.0)
        self.novelty_lambda = float(_read_cfg(config, "novelty_lambda", 1.0))
        self.acceptance_threshold = float(_read_cfg(config, "acceptance_threshold", 0.0))
        self.memory_bank: dict[str, Counter[str]] = defaultdict(Counter)
        self.prompt_bank: dict[str, str] = {}

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        if "rollout_log_probs" not in data.batch:
            raise ValueError(
                "Multiple-solution TTRL requires rollout log probabilities. "
                "Set actor_rollout_ref.rollout.calculate_log_probs=True."
            )

        rows = [await self._extract_row(data, i) for i in range(len(data))]
        for row in rows:
            self.prompt_bank[row["prompt_key"]] = row["prompt_text"]

        batch_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            if self._is_invalid_mode(row["solution_mode"]):
                continue
            batch_counts[row["prompt_key"]][row["solution_mode"]] += 1

        for prompt_key, counter in batch_counts.items():
            self.memory_bank[prompt_key].update(counter)

        outputs = []
        accepted_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            prompt_memory = self.memory_bank[row["prompt_key"]]
            is_invalid_mode = self._is_invalid_mode(row["solution_mode"])
            history_count = 0 if is_invalid_mode else prompt_memory[row["solution_mode"]]
            prior_count = (
                0 if is_invalid_mode else history_count - batch_counts[row["prompt_key"]][row["solution_mode"]]
            )
            confidence = row["confidence"]
            historical_stability = 1.0 - math.exp(-history_count)
            novelty = 1.0 if prior_count == 0 else 0.0
            plausibility = self.alpha * confidence + (1.0 - self.alpha) * historical_stability
            reward = plausibility * (1.0 + self.novelty_lambda * novelty)

            if not is_invalid_mode and reward >= self.acceptance_threshold:
                accepted_counts[row["prompt_key"]][row["solution_mode"]] += 1

            outputs.append(
                {
                    "reward_score": reward,
                    "reward_extra_info": {
                        "msttrl_reward": reward,
                        "msttrl_plausibility": plausibility,
                        "msttrl_confidence": confidence,
                        "msttrl_historical_stability": historical_stability,
                        "msttrl_novelty": novelty,
                        "msttrl_history_count": float(history_count),
                        "msttrl_solution_mode": row["solution_mode"],
                        "msttrl_final_answer": row["final_answer"],
                        "msttrl_prompt_key": row["prompt_key"],
                        **row["task_reward_extra_info"],
                    },
                }
            )

        for prompt_key, counter in batch_counts.items():
            rejected_counts = counter - accepted_counts[prompt_key]
            self.memory_bank[prompt_key].subtract(rejected_counts)
            self.memory_bank[prompt_key] += Counter()

        for output in outputs:
            prompt_key = output["reward_extra_info"]["msttrl_prompt_key"]
            output["reward_extra_info"]["msttrl_memory_size"] = float(len(self.memory_bank[prompt_key]))

        return outputs

    def get_state(self) -> dict[str, Any]:
        memory_bank = []
        for prompt_key in sorted(self.memory_bank):
            counter = self.memory_bank[prompt_key]
            mode_counts = {mode: int(count) for mode, count in sorted(counter.items())}
            memory_bank.append(
                {
                    "prompt_key": prompt_key,
                    "prompt": self.prompt_bank.get(prompt_key, ""),
                    "mode_counts": mode_counts,
                    "num_modes": len(mode_counts),
                    "total_count": int(sum(mode_counts.values())),
                }
            )

        return {
            "memory_bank": memory_bank,
            "num_prompts": len(memory_bank),
            "total_modes": sum(item["num_modes"] for item in memory_bank),
            "total_count": sum(item["total_count"] for item in memory_bank),
        }

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        if "rollout_log_probs" not in data.batch:
            return {"reward_score": 0.0, "reward_extra_info": {}}
        return (await self.run_batch(data[-1:]))[0]

    async def _extract_row(self, data: DataProto, index: int) -> dict[str, Any]:
        prompt_ids = data.batch["prompts"][index]
        response_ids = data.batch["responses"][index]
        attention_mask = data.batch["attention_mask"][index]
        prompt_len = prompt_ids.shape[-1]

        valid_prompt_len = int(attention_mask[:prompt_len].sum().item())
        valid_prompt_ids = prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[:0]
        prompt_key = self._prompt_key(valid_prompt_ids)
        prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

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
            "solution_mode": solution_mode,
            "final_answer": self._final_answer(response_str, data_source, task_reward_extra_info),
            "confidence": confidence,
            "task_reward_extra_info": task_reward_extra_info,
        }

    @staticmethod
    def _prompt_key(valid_prompt_ids: torch.Tensor) -> str:
        prompt_token_key = ",".join(str(int(token)) for token in valid_prompt_ids.detach().cpu().tolist())
        return hashlib.sha1(prompt_token_key.encode("utf-8")).hexdigest()

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
            return "causal:" + (str(pred) if pred else "<invalid>")
        if "pred" in task_reward_extra_info:
            pred = task_reward_extra_info.get("pred")
            if pred is not None:
                return f"{data_source}:{pred}"
        return f"{data_source}:" + response_str.strip()

    @staticmethod
    def _is_invalid_mode(solution_mode: str) -> bool:
        return solution_mode.endswith(":<invalid>")

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
