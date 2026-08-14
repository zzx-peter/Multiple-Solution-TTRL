#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <causal|3d|boolean> <base|ips-grpo|ttrl|msttrl> <eval-seed>" >&2
    exit 2
fi

task="$1"
method="$2"
eval_seed="$3"

repo_root="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/code/Multiple-Solution-TTRL"
result_root="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/qwen_seed1337_checkpoint_eval"
checkpoint_root="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/mode_collapse"

case "${task}" in
    causal)
        final_step=360
        ;;
    3d)
        final_step=120
        ;;
    boolean)
        final_step=90
        ;;
    *)
        echo "Unknown task: ${task}" >&2
        exit 2
        ;;
esac

case "${method}" in
    base)
        train_script="${repo_root}/examples/mode_collapse/ips-grpo/run_ips_grpo_${task}.sh"
        resume_mode="disable"
        checkpoint=""
        output_step=0
        ;;
    ips-grpo)
        train_script="${repo_root}/examples/mode_collapse/ips-grpo/run_ips_grpo_${task}.sh"
        resume_mode="resume_path"
        checkpoint="${checkpoint_root}/Qwen3-1.7B-${task}-reason-ips-grpo-norm/global_step_${final_step}"
        output_step="${final_step}"
        ;;
    ttrl)
        train_script="${repo_root}/examples/mode_collapse/ttrl/run_ttrl_${task}.sh"
        resume_mode="resume_path"
        checkpoint="${checkpoint_root}/Qwen3-1.7B-${task}-reason-ttrl/global_step_${final_step}"
        output_step="${final_step}"
        ;;
    msttrl)
        train_script="${repo_root}/examples/mode_collapse/msttrl-hopper/run_msttrl_${task}.sh"
        resume_mode="resume_path"
        if [[ "${task}" == "causal" ]]; then
            experiment="Qwen3-1.7B-causal-gated-ips-msttrl_lambda_0.9_high_0.875_seed_1337_2"
        else
            experiment="Qwen3-1.7B-${task}-gated-ips-msttrl_lambda_0.9_high_0.875_seed_1337"
        fi
        checkpoint="${checkpoint_root}/${experiment}/global_step_${final_step}"
        output_step="${final_step}"
        ;;
    *)
        echo "Unknown method: ${method}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${train_script}" ]]; then
    echo "Training script does not exist: ${train_script}" >&2
    exit 1
fi
if [[ "${resume_mode}" == "resume_path" && ! -f "${checkpoint}/actor/model_world_size_1_rank_0.pt" ]]; then
    echo "Checkpoint does not exist: ${checkpoint}" >&2
    exit 1
fi

run_name="${method}-${task}-eval-seed-${eval_seed}"
run_dir="${result_root}/runs/${task}/${method}/seed_${eval_seed}"
mkdir -p "${run_dir}/validation" "${run_dir}/runtime"

overrides=(
    "data.seed=${eval_seed}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${eval_seed}"
    "trainer.val_only=True"
    "trainer.val_before_train=True"
    "trainer.resume_mode=${resume_mode}"
    "trainer.del_local_ckpt_after_load=False"
    "trainer.project_name=qwen_seed1337_checkpoint_eval"
    "trainer.experiment_name=${run_name}"
    'trainer.logger=["tensorboard"]'
    "trainer.validation_data_dir=${run_dir}/validation"
    "trainer.default_local_dir=${run_dir}/runtime"
    "trainer.log_val_generations=0"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
)

if [[ "${resume_mode}" == "resume_path" ]]; then
    overrides+=("trainer.resume_from_path=${checkpoint}")
fi

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
cd "${repo_root}"

echo "task=${task} method=${method} eval_seed=${eval_seed} output_step=${output_step}"
if [[ "${method}" == "msttrl" ]]; then
    bash "${train_script}" 0.9 0.875 1337 "${overrides[@]}"
else
    bash "${train_script}" "${overrides[@]}"
fi

