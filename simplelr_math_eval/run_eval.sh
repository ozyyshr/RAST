#Qwen2.5-Math-Instruct Series
PROMPT_TYPE="mathstral"

#Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="1"
export HF_HOME="/shared/data3/siruo2/hf_checkpoints"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="Qwen2.5-1.5B-Math-mathstral"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR 0.0 16384 1.0