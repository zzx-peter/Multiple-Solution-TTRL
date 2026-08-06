#!/bin/bash
set -euo pipefail

prompt_tag="${1:-v1}"
repo_root="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/code/Multiple-Solution-TTRL"
train_path="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/data/hypespace/3d_reason/train.parquet"
test_path="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/data/hypespace/3d_reason/test.parquet"
dump_dir="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/prompt_smoke_3d/${prompt_tag}"
main_model="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/model/Qwen3-1.7B"

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

cd "${repo_root}"

python3 - <<PY
import pandas as pd
path = "${train_path}"
prompt = pd.read_parquet(path).iloc[0]["prompt"][0]["content"]
assert "Required output skeleton" in prompt, "The regenerated actor prompt is not loaded from parquet."
print("Confirmed regenerated actor prompt in", path)
print(prompt)
PY

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gated_ips_grpo \
    +algorithm.gated_ips_grpo_lambda=0.9 \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    data.train_files="['${train_path}']" \
    data.val_files="['${test_path}']" \
    data.train_max_samples=16 \
    data.val_max_samples=29 \
    data.train_batch_size=16 \
    +data.seed=1337 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.reward_fn_key=data_source \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${main_model}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.num_workers=1 \
    +msttrl.enable=True \
    +msttrl.high_threshold=0.875 \
    +msttrl.low_threshold=0.5 \
    +msttrl.verifier_n=8 \
    +msttrl.dump_dir="${dump_dir}" \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["tensorboard"]' \
    trainer.val_before_train=True \
    trainer.project_name=prompt_smoke_3d \
    trainer.experiment_name="${prompt_tag}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir="${dump_dir}/validation" \
    trainer.default_local_dir="${dump_dir}"

python3 examples/mode_collapse/msttrl-hopper/analyze_3d_prompt_smoke.py \
    "${dump_dir}" --train-parquet "${train_path}" \
    --marker "Required output skeleton"
