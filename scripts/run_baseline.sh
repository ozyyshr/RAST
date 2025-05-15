export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
export CUDA_VISIBLE_DEVICES=5,6

python run_vllm.py \
    --dataset MATH \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.75 \
    --decoding_temperature 1.0 \
    --num_runs 1