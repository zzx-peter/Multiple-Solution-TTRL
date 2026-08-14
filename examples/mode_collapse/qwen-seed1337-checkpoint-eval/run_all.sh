#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
result_root="/scratch/Projects/CFP-04/CFP04-CF-024/ophv164/result/qwen_seed1337_checkpoint_eval"

tasks=(causal 3d boolean)
methods=(base ips-grpo ttrl msttrl)
seeds=(0 1 42 3447)

for method in "${methods[@]}"; do
    for task in "${tasks[@]}"; do
        case "${task}" in
            causal) final_step=360 ;;
            3d) final_step=120 ;;
            boolean) final_step=90 ;;
        esac
        if [[ "${method}" == "base" ]]; then
            output_step=0
        else
            output_step="${final_step}"
        fi

        for seed in "${seeds[@]}"; do
            run_dir="${result_root}/runs/${task}/${method}/seed_${seed}"
            trajectory="${run_dir}/validation/${output_step}.jsonl"
            mkdir -p "${run_dir}"
            if [[ -s "${trajectory}" ]]; then
                echo "SKIP completed: task=${task} method=${method} seed=${seed}"
                continue
            fi
            echo "START task=${task} method=${method} seed=${seed}"
            bash "${script_dir}/run_eval.sh" "${task}" "${method}" "${seed}" \
                >"${run_dir}/run.log" 2>&1
            if [[ ! -s "${trajectory}" ]]; then
                echo "Missing validation output after successful command: ${trajectory}" >&2
                exit 1
            fi
            echo "DONE  task=${task} method=${method} seed=${seed}"
        done
    done
done

python3 "${script_dir}/aggregate_results.py"

