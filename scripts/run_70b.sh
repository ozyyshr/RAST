cd vllm
export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
export HF_TOKEN='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'

python tests/contrast_decode/run.py \
    --dataset Olympiad \
    --base_model /shared/data3/xzhong23/models/Llama-3.1-70B-Instruct \
    --positive_model hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo \
    --negative_model /shared/data3/xzhong23/models/Llama-3.1-8B \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.7 \
    --cd_decoding_alpha 1.0 \
    --decoding_temperature 1.0 \
    --num_runs 5

python tests/contrast_decode/run.py \
    --dataset MATH \
    --base_model /shared/data3/xzhong23/models/Llama-3.1-70B-Instruct \
    --positive_model hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo \
    --negative_model /shared/data3/xzhong23/models/Llama-3.1-8B \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.7 \
    --cd_decoding_alpha 1.0 \
    --decoding_temperature 1.0 \
    --num_runs 5

python tests/contrast_decode/run.py \
    --dataset GSM8K \
    --base_model /shared/data3/xzhong23/models/Llama-3.1-70B-Instruct \
    --positive_model hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo \
    --negative_model /shared/data3/xzhong23/models/Llama-3.1-8B \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.7 \
    --cd_decoding_alpha 1.0 \
    --decoding_temperature 1.0 \
    --num_runs 5



