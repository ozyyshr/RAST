cd vllm
export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
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

