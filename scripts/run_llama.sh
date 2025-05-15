# export CUDA_VISIBLE_DEVICES=6,7
# size=8
# echo "Results dir: results/aime/llama-${size}B"
# echo $CUDA_VISIBLE_DEVICES
# python -m run_logit \
#     --data_dir Maxwell-Jia/AIME_2024 \
#     --save_dir results/aime/llama-${size}B \
#     --model_name_or_path /shared/data/bowenj4/ChemRAG/models/Llama-3.1-${size}B-Instruct \
#     --eval_batch_size 1

export CUDA_VISIBLE_DEVICES=6,7
size=8
echo "Results dir: results/aime/llama-${size}B"
echo $CUDA_VISIBLE_DEVICES
python -m run_logit \
    --data_dir Maxwell-Jia/AIME_2024 \
    --save_dir results/aime/llama-${size}B \
    --base_model_name_or_path /shared/data/bowenj4/ChemRAG/models/Llama-3.1-${size}B-Instruct \
    --expert_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-${size}B \
    --eval_batch_size 1

# /shared/data/bowenj4/ChemRAG/models/Llama-3.1-8B-Instruct
# deepseek-ai/DeepSeek-R1-Distill-Llama-${size}B