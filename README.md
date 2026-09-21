# Multiple-Solution TTRL (MSTTRL)

这是一个基于 verl 的 TTRL 研究仓库。当前 main 以 ttrl-cocov 分支为基础，包含 MSTTRL、
TTRL-CoCoV、普通 TTRL 以及 mode-collapse 实验。运行目标是 verl 的 Docker 环境。

## MSTTRL 修改内容

### 状态化 multiple-solution reward

核心文件：verl/experimental/reward_loop/reward_manager/multiple_solution_ttrl.py

- 为每个 prompt 维护 solution memory，并维护 correct、candidate、wrong 三个互斥集合。
- 对新 mode 和候选 mode 进行多次 verifier rollout，按 high_threshold / low_threshold
  的 pass rate 路由。
- 高置信度解得到正奖励，低置信度解得到零奖励，中间区域保留为 candidate。
- 复用历史 mode，记录 solution_mode、final_answer、history_count、集合计数、置信度和
  memory size，并将诊断结果写入 msttrl.dump_dir。
- 支持 causal graph、Boolean expression、3D structure 和 MACE-MATH 多答案任务。

MSTTRL 必须满足：

~~~text
reward.num_workers=1
actor_rollout_ref.rollout.calculate_log_probs=True
+msttrl.enable=True
+msttrl.verifier_n=8
~~~

### 二阶段 verifier 和 CoCoV

verl/trainer/ppo/ray_trainer.py、verl/experimental/agent_loop/agent_loop.py 和
verl/trainer/main_ppo_sync.py 增加了 verifier rollout 数据流：第一阶段生成候选，第二阶段
使用独立 sampling 参数验证，并用 __skip_reward__ 跳过普通 reward。

verl/experimental/reward_loop/reward_manager/cocov.py 实现 CoCoV 的 A/M/B1/B2 路由：

- A：高一致性，保留 majority pseudo-label；
- M：中等一致性，直接使用 majority label；
- B1：低一致性但有通过 verifier 的候选，选择通过验证的候选；
- B2：低一致性且无通过候选，保留诊断信息并抑制对应 advantage。

### 算法、reward 和指标

- 新增 pass_grpo_penalized，结合 pass@k 与长度/多样性惩罚；
- 扩展 IPS-GRPO、Gated-IPS-GRPO 和 history-aware advantage；
- 新增 MACE-MATH 多参考答案 reward；
- 新增 rare-valid metrics；
- 所有训练仍从 python3 -m verl.trainer.main_ppo 进入。

### 实验目录

- examples/mode_collapse/msttrl-hopper/：MSTTRL causal、Boolean、3D；
- examples/mode_collapse/ttrl-cocov/：CoCoV causal、Boolean、3D；
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

位置参数依次是脚本保留的 lambda 参数、high_threshold、seed 和 verifier_n；当前 causal 脚本使用 ips_grpo，lambda 参数主要用于兼容已有实验命名。核心配置为：

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

运行 CoCoV：

~~~bash
bash examples/mode_collapse/ttrl-cocov/run_ttrl_causal.sh
~~~

CoCoV 参数包括 +cocov.verifier_n、+cocov.high_threshold、+cocov.low_threshold、
+cocov.high_max_candidates 和 +cocov.low_max_candidates。

### 5. 检查输出

verl 的 trainer.default_local_dir 保存 checkpoint 和日志；MSTTRL 的 msttrl.dump_dir 保存
solution sets、memory 统计和 verifier 样例。CPU 指标测试：

~~~bash
python3 -m pytest tests/experimental/reward_loop/test_multiple_solution_ttrl_on_cpu.py tests/trainer/ppo/test_ttrl_utils_on_cpu.py tests/trainer/ppo/test_rare_valid_metrics_on_cpu.py -q
~~~

## 代码入口

~~~text
verl/experimental/reward_loop/reward_manager/multiple_solution_ttrl.py  MSTTRL
verl/experimental/reward_loop/reward_manager/cocov.py                 CoCoV
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
