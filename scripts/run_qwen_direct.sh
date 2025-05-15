export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
echo "Results dir: results/MATH/dexperts-7B-direct-1"
CUDA_VISIBLE_DEVICES=6,7 python -m run_eval \
    --data_dir MATH \
    --save_dir results/MATH/dexperts-7B-direct-1 \
    --base_model_name_or_path Qwen/Qwen2.5-7B \
    --expert_model_name_or_path hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
    --antiexpert_model_name_or_path Qwen/Qwen2.5-1.5B \
    --do_sample \
    --eval_batch_size 4

CUDA_VISIBLE_DEVICES=6,7 python -m run_eval \
    --data_dir MATH \
    --save_dir results/MATH/dexperts-7B-direct-2 \
    --base_model_name_or_path Qwen/Qwen2.5-7B \
    --expert_model_name_or_path hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
    --antiexpert_model_name_or_path Qwen/Qwen2.5-1.5B \
    --do_sample \
    --eval_batch_size 4