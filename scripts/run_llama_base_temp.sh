export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
CUDA_VISIBLE_DEVICES=2,3 python -m run_eval \
    --data_dir MATH \
    --save_dir results/MATH/qwen2.5-1.5B_greedy \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --eval_batch_size 4


CUDA_VISIBLE_DEVICES=0,1 python -m run_eval \
    --data_dir MATH \
    --save_dir results/MATH/qwen2.5-7B_greedy \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --eval_batch_size 4