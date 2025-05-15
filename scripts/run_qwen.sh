# Contrastive decoding on MATH dataset
# Qwen/Qwen2.5-1.5B -> hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo
# Expert temp 1.0, base temp 0.6

size=8
export HF_HOME=/shared/data3/siruo2/hf_checkpoints/
CUDA_VISIBLE_DEVICES=2 python -m run_eval \
    --data_dir MATH \
    --save_dir results/MATH/ensemble \
    --base_model_name_or_path  Qwen/Qwen2.5-1.5B\
    --expert_model_name_or_path hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \
    --do_sample \
    --eval_batch_size 1