
# 🚀 Running verL on Lambda Cloud (1 GPU Version)

## Introduction
This project demonstrates how to **run verl** (Volcano Engine Reinforcement Learning for LLMs) on **Lambda Cloud** with a minimal setup using **only 1 A100 GPU**.

The goal is to **explore the minimum resource requirements**, **training costs**, and **performance baselines** for reinforcement learning fine-tuning of LLMs using verl.

> ✨ Later we will extend this to 4 and 8 GPUs for comparison!

---

## Environment Setup

| Item               | Setting                          |
|--------------------|----------------------------------|
| Instance Type      | Lambda Cloud A100 (40GB VRAM)    |
| OS                 | Ubuntu 20.04                     |
| Python Version     | 3.10                             |
| CUDA Version       | 12.2                             |
| Frameworks         | verl, vLLM>=0.8.3, Ray           |

---

## Installation

```bash
# Create environment
conda create -n verl python=3.10
conda activate verl

# Install verl
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .

# Install vLLM
pip install vllm==0.8.3

# Install flash-attn
pip install flash-attn --no-build-isolation

# (Optional) Install Ray
pip install "ray[default]"
```

---

## Quickstart: PPO Training on GSM8K (1 GPU)

### Step 1: Prepare Dataset

```bash
cd examples/data_preprocess
python3 gsm8k.py --local_dir ~/data/gsm8k
```

---

### Step 2: Run PPO Training

Training script (`scripts/run_ppo_1gpu.sh`):

```bash
#!/bin/bash
set -x

python3 -m verl.trainer.main_ppo     data.train_files=~/data/gsm8k/train.parquet     data.val_files=~/data/gsm8k/test.parquet     data.train_batch_size=256     data.max_prompt_length=512     data.max_response_length=512     actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct     actor_rollout_ref.actor.optim.lr=1e-6     actor_rollout_ref.actor.ppo_mini_batch_size=64     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8     actor_rollout_ref.model.enable_gradient_checkpointing=True     actor_rollout_ref.actor.fsdp_config.param_offload=True     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True     actor_rollout_ref.rollout.tensor_model_parallel_size=1     actor_rollout_ref.rollout.name=vllm     actor_rollout_ref.rollout.gpu_memory_utilization=0.5     critic.model.path=Qwen/Qwen2.5-0.5B-Instruct     critic.model.enable_gradient_checkpointing=True     critic.ppo_micro_batch_size_per_gpu=8     trainer.logger=['console']     trainer.val_before_train=False     trainer.total_epochs=3
```

✅ This command trains PPO for **3 epochs** with **1 A100 GPU**.

---

## Results (1 GPU A100)

| Metric                     | Value                                    |
|-----------------------------|------------------------------------------|
| Training Throughput         | ~65 tokens/sec (small batch)             |
| GPU Memory Usage            | ~32GB VRAM used                         |
| Cost (Lambda Cloud)         | ~$1.1/hour (A100 hourly rate)            |
| PPO Training Time (3 Epochs) | ~3.5 hours                              |
| Total Cost                  | ~$4                                     |

---

## Observations on 1 GPU

- **Feasible**: verl can run smoothly on a single A100 without Out-Of-Memory (OOM) errors.
- **Memory optimization critical**: Gradient checkpointing and parameter offloading are essential.
- **Training speed**: Reasonable for basic testing but slower compared to multi-GPU setups.

---

## What's Next? 🔥

We plan to extend this guide:

| Version | Status        | Plan                                        |
|---------|---------------|---------------------------------------------|
| 1 GPU   | ✅ Completed   | This README (first version)                 |
| 4 GPUs  | 🔜 Coming Soon | Higher throughput, cost scaling analysis   |
| 8 GPUs  | 🔜 Coming Soon | Best throughput, resource scaling analysis |

Stay tuned for updates!

---

## Folder Structure

```text
verl-lambda-cloud-guide/
├── README.md
├── images/
│    └── training_1gpu.png   # Screenshot of training logs (coming soon)
├── scripts/
│    └── run_ppo_1gpu.sh     # Script to run PPO with 1 GPU
```

---

## Acknowledgements
- **verl**: [GitHub - volcengine/verl](https://github.com/volcengine/verl)
- **vLLM**: [GitHub - vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Lambda Cloud**: [Lambda Labs](https://lambdalabs.com/)

---
