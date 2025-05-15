cd vllm
export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
export CUDA_VISIBLE_DEVICES=1,2,3,4
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.3 \
#     --decoding_temperature 1.0 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --dataset MATH \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-7B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.6 \
#     --cd_decoding_alpha 1.0 \
#     --decoding_temperature 1.0 \
#     --num_runs 8

python tests/contrast_decode/run.py \
    --dataset MATH \
    --base_model Qwen/Qwen2.5-14B \
    --positive_model hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo \
    --negative_model Qwen/Qwen2.5-7B \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.6 \
    --cd_decoding_alpha 1.0 \
    --decoding_temperature 1.0 \
    --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --positive_model1 hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
#     --negative_model1 Qwen/Qwen2.5-7B \
#     --positive_model2 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
#     --negative_model2 Qwen/Qwen2.5-1.5B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.3 \
#     --decoding_temperature 1.0 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.4 \
#     --decoding_temperature 1.1 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.6 \
#     --decoding_temperature 0.9 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.6 \
#     --decoding_temperature 1.1 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.8 \
#     --decoding_temperature 1.0 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.9 \
#     --decoding_temperature 1.0 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.5 \
#     --decoding_temperature 1.3 \
#     --num_runs 10

# python tests/contrast_decode/run.py \
#     --base_model Qwen/Qwen2.5-32B \
#     --positive_model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
#     --negative_model Qwen/Qwen2.5-14B \
#     --tensor_parallel_size 4 \
#     --gpu_memory_utilization 0.8 \
#     --cd_decoding_alpha 0.5 \
#     --decoding_temperature 1.4 \
#     --num_runs 10