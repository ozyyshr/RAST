# Evaluating DExperts with rl-zero expert
export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
echo "Results dir: results/AIME/dexperts-7B-chat-1"
CUDA_VISIBLE_DEVICES=2,3 python -m run_eval \
    --data_dir AIME \
    --save_dir results/AIME/dexperts-7B-chat-1 \
    --base_model_name_or_path Qwen/Qwen2.5-7B \
    --expert_model_name_or_path hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
    --antiexpert_model_name_or_path Qwen/Qwen2.5-1.5B \
    --do_sample \
    --use_chat_format \
    --eval_batch_size 4