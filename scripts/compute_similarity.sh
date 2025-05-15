export CUDA_VISIBLE_DEVICES=1,2,3,4 

python analysis/similarity.py \
    --base_model1 Qwen/Qwen2.5-7B \
    --base_model2 Qwen/Qwen2.5-14B \
    --RL_model1 hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
    --RL_model2 hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo

python analysis/similarity.py \
    --base_model1 Qwen/Qwen2.5-7B \
    --base_model2 Qwen/Qwen2.5-1.5B \
    --RL_model1 hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
    --RL_model2 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo

python analysis/similarity.py \
    --base_model1 Qwen/Qwen2.5-7B \
    --base_model2 Qwen/Qwen2.5-32B \
    --RL_model1 hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
    --RL_model2 hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo

python analysis/similarity.py \
    --base_model1 Qwen/Qwen2.5-14B \
    --base_model2 Qwen/Qwen2.5-32B \
    --RL_model1 hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
    --RL_model2 hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo

python analysis/similarity.py \
    --base_model1 Qwen/Qwen2.5-14B \
    --base_model2 Qwen/Qwen2.5-1.5B \
    --RL_model1 hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
    --RL_model2 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo

python analysis/similarity.py \
    --base_model1 Qwen/Qwen2.5-32B \
    --base_model2 Qwen/Qwen2.5-1.5B \
    --RL_model1 hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo \
    --RL_model2 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo