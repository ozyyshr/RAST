# Contrastive decoding on MATH dataset
# Llama-3.1-8B-Instruct -> DeepSeek-R1-Distill-Llama-8B
# Expert temp 1.0, base temp 0.6

size=8
# export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
CUDA_VISIBLE_DEVICES=2,3 python -m run_logit \
    --data_dir MATH \
    --save_dir results/MATH/Llama-3.1-8B-CD \
    --base_model_name_or_path  meta-llama/Llama-3.1-8B-Instruct\
    --expert_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --do_sample \
    --eval_batch_size 4

# Contrastive decoding on AIME dataset
# Llama-3.1-8B-Instruct -> DeepSeek-R1-Distill-Llama-8B
# Expert temp 1.0, base temp 0.6

CUDA_VISIBLE_DEVICES=0,1 python -m run_logit \
    --data_dir AIME \
    --save_dir results/AIME/Llama-3.1-8B-CD \
    --base_model_name_or_path  meta-llama/Llama-3.1-8B-Instruct\
    --expert_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --do_sample \
    --eval_batch_size 4