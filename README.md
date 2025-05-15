# 🧠 Reasoning Activation in LLMs via Small Model Transfer

This repository contains the codebase and evaluation pipeline for our NeurIPS 2025 submission: **Reasoning Activation in LLMs via Small Model Transfer**. We propose a decoding-time method, **RAST**, that activates reasoning capabilities in large language models by transferring logit-level adjustments from smaller RL-tuned models.

## 🚀 Project Overview

Instead of directly fine-tuning large models with reinforcement learning (RL), our method applies reasoning corrections derived from small-scale RL-tuned models to the output logits of larger models during inference—yielding reasoning performance gains without retraining.

---

## 📂 Directory Structure

- `analysis/` – Scripts for result analysis and visualization.
- `datasets/` – Benchmark datasets used in our experiments.
- `modeling/` – Model loading and interface wrappers.
- `scripts/` – Auxiliary scripts for launching experiments.
- `simplerl_math_eval/` – Evaluation tools for math reasoning datasets.
- `vlm/` – Vision-language model tools (if applicable to ablations).
- `environment.yaml` – Conda environment file for reproducibility.
- `requirements.txt` – Python package dependencies.
- `utils.py` – Utility functions used throughout the codebase.
- `vllm` - Core implementation that implements RAST with vLLM framework for faster inference.

---

## 🧪 Usage

### 1. Setup Environment

```bash
conda env create -f environment.yaml
conda activate reasoners
```

### 2. Install vllm in editable mode to apply the changes

```bash
cd vllm
pip install -e .
```
Note that this step requires vllm==0.6.3, as specified in requirements.txt

### 3. Run Inference

Take Olympiad Bench as an example, the inference scripts is shown in the folloiwng:

```bash
cd vllm
export CUDA_VISIBLE_DEVICES=1,2,3,4

python tests/contrast_decode/run.py \
    --dataset Olympiad \
    --base_model Qwen/Qwen2.5-14B \
    --positive_model hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
    --negative_model Qwen/Qwen2.5-1.5B \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.6 \
    --cd_decoding_alpha 1.0 \
    --decoding_temperature 1.0 \
    --num_runs 32
```
