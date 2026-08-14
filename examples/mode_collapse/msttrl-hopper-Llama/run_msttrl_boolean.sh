#!/bin/bash
set -x

gated_ips_grpo_lambda="${1:-0.9}"
if (( $# > 0 )); then
    shift
fi

high_threshold="${1:-0.875}"
if (( $# > 0 )); then
    shift
fi

seed="${1:-1337}"
if (( $# > 0 )); then
    shift
fi

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="3ad98c1a4043033c1a4ea70737ebb314a2f09ba0"
export PYTHONPATH=/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/code/Multiple-Solution-TTRL:$PYTHONPATH

boolean_path="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/data/hypespace/boolean_reason/train.parquet"
boolean_test_path="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/data/hypespace/boolean_reason/test.parquet"

train_files="['$boolean_path']"
test_files="['$boolean_test_path']"
clip_ratio_low=0.2
clip_ratio_high=0.28

n_gpus_per_node=1
tensor_model_parallel_size=1
micro_batch_size=4
gpu_memory_utilization=0.5
offload=True

project_name="mode_collapse"
experiment_name="Llama-3.2-3B-Instruct-boolean-gated-ips-msttrl_lambda_${gated_ips_grpo_lambda}_high_${high_threshold}_seed_${seed}"
msttrl_dump_dir="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/${project_name}/${experiment_name}"

main_model=/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/model/Llama-3.2-3B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gated_ips_grpo \
    +algorithm.gated_ips_grpo_lambda=${gated_ips_grpo_lambda} \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=16 \
    +data.seed=${seed} \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.reward_fn_key=data_source \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${main_model} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.n=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.num_workers=1 \
    +msttrl.enable=True \
    +msttrl.high_threshold=${high_threshold} \
    +msttrl.low_threshold=0.5 \
    +msttrl.verifier_n=8 \
    +msttrl.dump_dir=${msttrl_dump_dir} \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb","tensorboard"]' \
    trainer.val_before_train=True \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 \
    trainer.log_val_generations=10 \
    trainer.default_local_dir=/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/${project_name}/${experiment_name} \
    "$@"
