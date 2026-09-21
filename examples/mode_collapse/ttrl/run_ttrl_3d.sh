#!/bin/bash
set -x

ttrl_num_ans="${TTRL_NUM_ANS:-1}"

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTHONPATH=/scratch/ophv164/code/verl-v0.8.0:$PYTHONPATH
three_d_path="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/data/hypespace/3d_reason/train.parquet"
three_d_test_path="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/data/hypespace/3d_reason/test.parquet"

train_files="['$three_d_path']"
test_files="['$three_d_test_path']"

clip_ratio_low=0.2
clip_ratio_high=0.28

n_gpus_per_node=1

tensor_model_parallel_size=1
mirco_bacth_size=4
gpu_memory_utilization=0.5
offload=True

project_name="mode_collapse"
experiment_name="Qwen3-1.7B-3d-reason-ttrl"
seed=1337

main_model=/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/model/Qwen3-1.7B

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ips_grpo \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mirco_bacth_size} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${mirco_bacth_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=27 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mirco_bacth_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.num_workers=1 \
    +ttrl.enable=True \
    +ttrl.num_ans=${ttrl_num_ans} \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["tensorboard","wandb"]' \
    trainer.val_before_train=True \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.log_val_generations=10 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/${project_name}/${experiment_name} \
    "$@"
