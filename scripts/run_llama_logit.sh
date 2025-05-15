# size=7
# export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
# echo "Results dir: results/MATH/Distilled-qwen-${size}B-direct"
# CUDA_VISIBLE_DEVICES=2,3 python -m run_logit \
#     --data_dir MATH \
#     --save_dir results/MATH/Distilled-qwen-${size}B-direct \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --eval_batch_size 1


size=8
export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
echo "Results dir: results/MATH/Distilled-llama-${size}B-CD"
CUDA_VISIBLE_DEVICES=2,3 python -m run_logit \
    --data_dir MATH \
    --save_dir results/MATH/Distilled-llama-${size}B-CD \
    --base_model_name_or_path  /shared/data/bowenj4/ChemRAG/models/Llama-3.1-${size}B-Instruct\
    --expert_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --do_sample \
    --eval_batch_size 1

# Model Families:
# 1. Llama-3.1-8B-Instruct /shared/data/bowenj4/ChemRAG/models/Llama-3.1-${size}B-Instruct
# 1.1 deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# 1.2 hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo

# 2. Qwen/Qwen2.5-Math-7B
# 2.1 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# 2.2 hkust-nlp/Qwen-2.5-7B-SimpleRL-Zero
# 2.3 hkust-nlp/Qwen-2.5-Math-7B-SimpleRL