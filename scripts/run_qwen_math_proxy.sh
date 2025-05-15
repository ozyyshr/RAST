export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
echo "Results dir: results/MATH/ensemble"
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m run_eval \
    --data_dir MATH \
    --save_dir results/MATH/ensemble \
    --base_model_name_or_path Qwen/Qwen2.5-32B \
    --expert_model_name_or_path hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
    --antiexpert_model_name_or_path Qwen/Qwen2.5-7B \
    --expert_model1 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
    --antiexpert_model1 Qwen/Qwen2.5-1.5B \
    --do_sample \
    --eval_batch_size 1