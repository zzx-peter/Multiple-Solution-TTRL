# Multiple-Solution TTRL (MSTTRL)

MSTTRL（Multiple-Solution Test-Time Reinforcement Learning）基于 verl，面向同一道题存在多个有效答案的推理任务。方法通过多次采样发现不同解，使用模型自验证估计每个解的可信度，再结合跨步记忆生成训练奖励，目标是在测试时学习中保留多种有效解。

## MSTTRL 方法

### 1. 采样与解的归一化

对每个问题生成多条回答，通过任务解析器提取并归一化最终答案，得到不同的 solution mode。同一种解的重复出现会累积到该问题的历史计数中；无法解析的回答不进入解集合。

当前支持 causal graph、Boolean expression、3D structure 和 MACE-MATH 多答案任务。

### 2. 多次自验证

将问题中的约束和候选解组成验证提示，使用当前策略模型对每个待验证的解采样 `verifier_n` 次，解析 `Verification Result: True/False`，以有效投票中 True 的比例作为可信度 `p`。

无法解析的验证结果不计入分母；一个解的所有验证结果都无法解析时，按 `p = 0` 处理。默认每个解验证 8 次。验证判断可能有误，集合中的名称表示模型的置信判断。

### 3. 三类解集合与跨步记忆

为每个问题维护三个互斥集合，并根据验证可信度分配奖励：

| 解集合 | 分配条件 | 训练奖励 |
| --- | --- | --- |
| `correct` | `p >= high_threshold` | `1` |
| `candidate` | `low_threshold < p < high_threshold` | `p` |
| `wrong` | `p <= low_threshold` | `0` |

默认阈值为 `high_threshold = 0.875`、`low_threshold = 0.5`。无法解析或尚未归类的解奖励为 0。

当问题再次出现在训练批次中时，待验证集合由本次新解和历史 `candidate` 组成；已经进入 `correct` 或 `wrong` 的解复用已有判断。这样可以持续探索新解，并将验证开销集中到尚未确定的解上。记忆中的已确定判断也可能保留自验证错误。

### 4. 策略更新与 verl 集成

将每条回答所属解的奖励送入策略更新。仓库提供 IPS-GRPO、Gated-IPS-GRPO 和使用历史计数的 `gated_history_ips_grpo`，具体选择由 `algorithm.adv_estimator` 控制。

本仓库围绕 MSTTRL 对 verl 增加了以下实现：

- 状态化 reward manager：管理解的归一化、历史计数、三类集合与奖励。
- 两阶段训练流程：候选回答生成后，再生成验证回答，聚合验证结果后计算训练奖励。
- 诊断输出：记录解集合、可信度、历史出现次数、memory size 和 verifier 样例；另提供 rare-valid 指标分析稀有有效解。

训练入口为 `python3 -m verl.trainer.main_ppo`。状态化 reward manager 要求 `reward.num_workers=1`，并开启 `actor_rollout_ref.rollout.calculate_log_probs=True`。通过 `+msttrl.enable=True` 启用方法。

### 实验目录

- examples/mode_collapse/msttrl-hopper/：MSTTRL causal、Boolean、3D；
- examples/mode_collapse/MACE-MATH*/：MACE-MATH；
- examples/mode_collapse/msttrl-hopper-ablation-*/：消融实验。

脚本默认读取外部 parquet、模型和输出目录，请按机器修改路径。大体积数据和模型不放入
Git；运行时通过容器挂载。WANDB_API_KEY 应通过容器环境变量提供。

## Quick Start：verl Docker

宿主机需要 Docker、NVIDIA Container Toolkit 和可用的 nvidia-smi。镜像可以替换为你已有
的 verl 镜像。

### 1. 创建容器

在仓库根目录执行：

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

以后进入已有容器：

~~~bash
docker start msttrl-verl
docker exec -it msttrl-verl bash
~~~

### 2. 安装仓库

在容器内：

~~~bash
cd /workspace/verl
pip3 install --no-deps -e .
python3 -c "import verl; print(verl.__file__)"
~~~

verl Docker 镜像已经提供 CUDA、PyTorch、vLLM/SGLang 等依赖；使用 --no-deps 可避免重装
依赖导致版本变化。

### 3. 准备数据和模型

当前 main 不包含大体积数据和模型。把 parquet 和模型挂载到容器后设置：

~~~bash
export MODEL_PATH=/models/Qwen3-1.7B
export TRAIN_FILE=/data/hypespace/causal_reason/train.parquet
export TEST_FILE=/data/hypespace/causal_reason/test.parquet
export OUTPUT_DIR=/workspace/verl/outputs/msttrl
export WANDB_API_KEY=你的_key
~~~

examples 使用的 data_source 包括 causal、boolean、3d 和 mace_math。显存不足时，优先
降低 rollout.n、verifier_n、gpu_memory_utilization 或 batch size。

### 4. 运行 MSTTRL

先将 examples/mode_collapse/msttrl-hopper/run_msttrl_causal.sh 中的模型、数据和输出路径
替换为容器内路径，然后执行：

~~~bash
cd /workspace/verl
bash examples/mode_collapse/msttrl-hopper/run_msttrl_causal.sh \
  0.9 0.875 1337 8
~~~

当前 causal 脚本使用 `ips_grpo`，后两个位置参数分别设置 seed 和 verifier_n；前两个为保留参数，当前脚本未将它们传入训练配置。修改阈值时可在命令末尾追加 `+msttrl.high_threshold=0.875`。以下仅列出 MSTTRL 核心配置，完整运行还需脚本中的模型、数据和 batch 等参数：

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

### 5. 检查输出

verl 的 trainer.default_local_dir 保存 checkpoint 和日志；MSTTRL 的 msttrl.dump_dir 保存
solution sets、memory 统计和 verifier 样例。在已安装依赖的 verl 容器中运行相关 CPU 测试：

~~~bash
python3 -m pytest tests/experimental/reward_loop/test_multiple_solution_ttrl_on_cpu.py tests/trainer/ppo/test_ttrl_utils_on_cpu.py tests/trainer/ppo/test_rare_valid_metrics_on_cpu.py -q
~~~

## 代码入口

~~~text
verl/experimental/reward_loop/reward_manager/multiple_solution_ttrl.py  MSTTRL
verl/trainer/ppo/core_algos.py                                         训练算法
verl/trainer/ppo/rare_valid_metrics.py                                 稀有有效解
verl/trainer/ppo/ray_trainer.py                                       verifier 集成
verl/utils/reward_score/mace_math/                                     MACE-MATH reward
examples/mode_collapse/                                                 实验脚本
~~~

## 参考

- verl: https://github.com/verl-project/verl
- TTRL: https://arxiv.org/abs/2504.16084
- 仓库：https://github.com/zzx-peter/Multiple-Solution-TTRL
