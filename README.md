# Multiple-Solution TTRL (MSTTRL)

MSTTRL (Multiple-Solution Test-Time Reinforcement Learning) is built on verl for reasoning tasks
where one prompt can have multiple valid solutions. The method samples multiple responses, uses
the model to verify candidate solutions, and keeps a per-prompt memory of solution modes. The verifier plausibility (pass rate) becomes the training reward, allowing test-time learning to preserve and
explore multiple valid solutions.

## The MSTTRL method

### 1. Sampling and solution normalization

For each prompt, the policy generates multiple responses. Task-specific parsers extract and
normalize the final answer into a solution mode. Repeated occurrences of the same mode increase
that prompt's history count. Responses that cannot be parsed are excluded from the solution sets.

The implementation supports causal graphs, Boolean expressions, 3D structures, and MACE-MATH
multi-answer tasks.

### 2. Self-verification

MSTTRL formats the prompt constraints and a candidate solution as a verifier prompt. The current
policy samples the verifier response `verifier_n` times and parses
`Verification Result: True` or `Verification Result: False`.

For a solution mode, its plausibility `p` is the fraction of valid verifier votes that are True.
Unparseable verifier responses are excluded from the denominator. If every verifier response is
unparseable, `p` is set to zero. The default is eight verifier samples per solution mode.

### 3. Three solution sets and cross-step memory

Each prompt owns three mutually exclusive sets. The verifier plausibility determines both the set
and the reward:

| Set | Routing condition | Training reward |
| --- | --- | --- |
| `correct` | `p >= high_threshold` | `1` |
| `candidate` | `low_threshold < p < high_threshold` | `p` |
| `wrong` | `p <= low_threshold` | `0` |

The defaults are `high_threshold = 0.875` and `low_threshold = 0.5`. Invalid or unknown solution
modes receive zero reward.

When a prompt appears again, MSTTRL verifies newly generated modes and retained `candidate`
modes. Modes already classified as `correct` or `wrong` reuse their stored classification, which
reduces repeated verification while continuing to explore new solutions. The memory stores verifier judgments and classifications; a self-verifier can still make an incorrect classification.

### 4. Policy update and verl integration

The reward attached to each response is passed to the policy update. Select IPS-GRPO, Gated-IPS-GRPO, or `gated_history_ips_grpo` with the `algorithm.adv_estimator` configuration key. Only the history-aware variant uses MSTTRL history metadata.
The training entry point remains `python3 -m verl.trainer.main_ppo`.

MSTTRL-specific changes in this repository include:

- A stateful reward manager in
  `verl/experimental/reward_loop/reward_manager/multiple_solution_ttrl.py` for parsing solution
  modes, maintaining memory, routing sets, and building rewards.
- A two-stage rollout path: generate candidate responses, generate verifier responses, aggregate
  verifier votes, then compute the training rewards.
- Per-sample verifier rollout parameters and reward skipping for verifier-only samples.
- Diagnostic state containing solution sets, plausibility, history counts, memory size, and verifier
  examples.
- Rare-valid metrics and reward metadata for analyzing uncommon but valid solutions.

MSTTRL runtime requirements and the default verifier setting are:

~~~text
reward.num_workers=1
actor_rollout_ref.rollout.calculate_log_probs=True
+msttrl.enable=True
+msttrl.verifier_n=8
~~~

### Experiment layout

- `examples/mode_collapse/msttrl-hopper/`: causal, Boolean, and 3D MSTTRL experiments.
- `examples/mode_collapse/MACE-MATH*/`: MACE-MATH experiments.
- `examples/mode_collapse/msttrl-hopper-ablation-*/`: candidate, memory, and Gated-IPS ablations.

The scripts expect external parquet files, model checkpoints, and output directories. Large data and
models are not stored in Git; mount them into the container. Provide `WANDB_API_KEY` through the
container environment instead of hard-coding it in a script.

## Quick Start with the verl Docker environment

The host must have Docker, NVIDIA Container Toolkit, and a working `nvidia-smi`. Replace the
image name if you already have a compatible verl image.

### 1. Create the container

Run the following from the repository root:

~~~bash
IMAGE=verlai/verl:vllm017.latest
CONTAINER=msttrl-verl

docker pull $IMAGE
docker create --runtime=nvidia --gpus all --net=host --shm-size=10g \
  --cap-add=SYS_ADMIN -v "$PWD":/workspace/verl \
  --name $CONTAINER $IMAGE sleep infinity
docker start $CONTAINER
docker exec -it $CONTAINER bash
~~~

For an existing container:

~~~bash
docker start msttrl-verl
docker exec -it msttrl-verl bash
~~~

### 2. Install this repository

Inside the container:

~~~bash
cd /workspace/verl
pip3 install --no-deps -e .
python3 -c "import verl; print(verl.__file__)"
~~~

The verl image already provides CUDA, PyTorch, and the vLLM/SGLang dependencies. Using
`--no-deps` avoids replacing those versions. Install extra backend dependencies only when the
selected image does not already provide them.

### 3. Prepare data, model, and logging

The repository does not include large datasets or model checkpoints. Mount them into the
container and set paths such as:

~~~bash
export MODEL_PATH=/models/Qwen3-1.7B
export TRAIN_FILE=/data/hypespace/causal_reason/train.parquet
export TEST_FILE=/data/hypespace/causal_reason/test.parquet
export OUTPUT_DIR=/workspace/verl/outputs/msttrl
export WANDB_API_KEY=your_key
~~~

The example tasks use `data_source` values such as `causal`, `boolean`, `3d`, and `mace_math`.
If memory is limited, reduce `rollout.n`, `verifier_n`, `gpu_memory_utilization`, or the batch
size first.

### 4. Run MSTTRL

Edit the model, data, and output paths in
`examples/mode_collapse/msttrl-hopper/run_msttrl_causal.sh`, then run:

~~~bash
cd /workspace/verl
bash examples/mode_collapse/msttrl-hopper/run_msttrl_causal.sh \
  0.9 0.875 1337 8
~~~

The script accepts four positional arguments: two reserved arguments, `seed`, and `verifier_n`. The current causal script uses `ips_grpo`; the first two arguments are stored for compatibility but are not passed to the training configuration. The third argument sets `+data.seed`, and the fourth sets `+msttrl.verifier_n`. Override the thresholds by appending `+msttrl.high_threshold=...` and/or `+msttrl.low_threshold=...`. The essential MSTTRL settings are:

~~~bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=ips_grpo \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  reward.num_workers=1 \
  +msttrl.enable=True \
  +msttrl.verifier_n=8 \
  +msttrl.high_threshold=0.875 \
  +msttrl.low_threshold=0.5 \
  +msttrl.dump_dir=/workspace/verl/outputs/msttrl
~~~

The full experiment scripts also set model, dataset, optimizer, FSDP, batch, and logging
parameters.

### 5. Inspect outputs

verl writes checkpoints and normal logs under `trainer.default_local_dir`. MSTTRL diagnostic
snapshots under `msttrl.dump_dir` contain solution sets, memory statistics, and verifier examples.

Run the focused CPU tests inside a container with the required dependencies:

~~~bash
python3 -m pytest \
  tests/experimental/reward_loop/test_multiple_solution_ttrl_on_cpu.py \
  tests/trainer/ppo/test_ttrl_utils_on_cpu.py \
  tests/trainer/ppo/test_rare_valid_metrics_on_cpu.py -q
~~~

## Code entry points

~~~text
verl/experimental/reward_loop/reward_manager/multiple_solution_ttrl.py  MSTTRL reward manager
verl/trainer/ppo/core_algos.py                                         IPS/Gated-IPS advantages
verl/trainer/ppo/rare_valid_metrics.py                                 rare-valid metrics
verl/trainer/ppo/ray_trainer.py                                        verifier integration
verl/utils/reward_score/mace_math/                                     MACE-MATH reward
examples/mode_collapse/msttrl-hopper/                                  MSTTRL experiments
~~~