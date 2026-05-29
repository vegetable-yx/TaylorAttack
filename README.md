# TaylorAttack

Reproduction code for the MLP weight-recovery attack against Taylor-approximation-based model protection. The attack exploits published Taylor coefficients to algebraically recover protected weight matrices with near-perfect accuracy.

## Installation

Please use the provided Docker image (requires NVIDIA GPU).

**1. Set up the project directory**

```bash
mkdir TaylorMLP
cd TaylorMLP
curl -L https://anonymous.4open.science/api/repo/TaylorAttack-1BDB/zip -o TaylorAttack.zip
unzip TaylorAttack.zip
rm TaylorAttack.zip
chmod +x TaylorAttack/scripts/*
```

**2. Build the Docker image**

```bash
export HF_TOKEN="your_huggingface_token"
docker build -f TaylorAttack/Dockerfile -t taylorattack:sglang-v0.5.12 .
```

**3. Start the container**

```bash
HF_TOKEN="$HF_TOKEN" TaylorAttack/scripts/docker_run.sh bash
```

All subsequent `taylormlp` commands are run **inside the container** (working directory `/workspace/TaylorMLP`). The suite scripts (`run_attack_suite.sh`, `run_kl_suite.sh`) are run **on the host** from the `TaylorMLP/` directory — they manage Docker internally.

## Protection & Weight Recovery

The core attack protects an MLP using TaylorMLP, then algebraically recovers the original weights from the published Taylor coefficients.

### Step 1: Download models and datasets

```bash
TaylorAttack/scripts/taylormlp download --config TaylorAttack/configs/default.yaml
```

### Step 2: Collect activation statistics

```bash
TaylorAttack/scripts/taylormlp calibrate \
  --config TaylorAttack/configs/default.yaml \
  --model qwen35_9b \
  --output TaylorAttack/results/calibration/qwen35_9b_hidden_states.pt
```

### Step 3: Run the attack

```bash
TaylorAttack/scripts/taylormlp attack \
  --config TaylorAttack/configs/default.yaml \
  --model qwen35_9b \
  --stats-file TaylorAttack/results/calibration/qwen35_9b_hidden_states.pt \
  --output TaylorAttack/results/qwen35_9b_attack \
  --save-recovered-model
```

Writes `attack_metrics.json` and (optionally) the recovered model weights to the output directory.

### Full suite (all models)

Run from the repo root (outside the container):

```bash
TaylorAttack/scripts/run_attack_suite.sh
```

## KL Divergence

Measures how closely the recovered model's output distribution matches the original, using KL divergence over evaluation prompts.

```bash
TaylorAttack/scripts/taylormlp kl \
  --config TaylorAttack/configs/default.yaml \
  --original-model Qwen/Qwen3.5-9B \
  --recovered-model TaylorAttack/results/qwen35_9b_attack/recovered_model \
  --model-label qwen35_9b \
  --benchmark math500 \
  --benchmark gpqa_diamond
```

Writes `kl_divergence.md` and per-benchmark `*_kl.json` to the output directory.

> **Note:** KL evaluation loads both models simultaneously. If you encounter CUDA OOM, try reducing `--sglang-mem-fraction-static` (default: 0.70).

### Full suite (all models and datasets)

Run from the repo root (outside the container):

```bash
TaylorAttack/scripts/run_kl_suite.sh
```

## Configuration

All models, datasets, and protection parameters are defined in `TaylorAttack/configs/default.yaml`.

## Reference

[Taylor Unswift: Secured Weight Release for Large Language Models via Taylor Expansion](https://aclanthology.org/2024.emnlp-main.393/) (Wang et al., EMNLP 2024)

Original implementation: [https://github.com/guanchuwang/taylor-unswift](https://github.com/guanchuwang/taylor-unswift)
