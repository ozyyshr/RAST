import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")


# ## compute for base model
# res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/baseline/GSM8K/hkust-nlp-Qwen-2.5-32B-SimpleRL-Zoo_temp_1.0"
# tokens_all = 0
# for i in range(32):
#     with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
#         data = [json.loads(line) for line in f]
#         # get the length of the model_output
#         tokens = 0
#         for j in range(len(data)):
#             tokens += len(tokenizer(data[j]['model_output'])['input_ids'])
#     tokens_all += tokens

# print(tokens_all / (32*1319))

# res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/baseline/GSM8K/hkust-nlp-Qwen-2.5-14B-SimpleRL-Zoo_temp_1.0"
# tokens_all = 0
# for i in range(32):
#     with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
#         data = [json.loads(line) for line in f]
#         # get the length of the model_output
#         tokens = 0
#         for j in range(len(data)):
#             tokens += len(tokenizer(data[j]['model_output'])['input_ids'])
#     tokens_all += tokens

# print(tokens_all / (32*1319))

# res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/baseline/GSM8K/hkust-nlp-Qwen-2.5-7B-SimpleRL-Zoo_temp_1.0"
# tokens_all = 0
# for i in range(32):
#     with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
#         data = [json.loads(line) for line in f]
#         # get the length of the model_output
#         tokens = 0
#         for j in range(len(data)):
#             tokens += len(tokenizer(data[j]['model_output'])['input_ids'])
#     tokens_all += tokens

# print(tokens_all / (32*1319))

# ### compute 32b  +7b

# res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/GSM8K/14B_1.5B_alpha_1.0_temp_1.0"

# tokens_all = 0
# for i in range(32):
#     with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
#         data = [json.loads(line) for line in f]
#         # get the length of the model_output
#         tokens = 0
#         for j in range(len(data)):
#             tokens += len(tokenizer(data[j]['model_output'])['input_ids'])
#     tokens_all += tokens

# print(tokens_all / (32*1319))


### compute 32b + 14b

res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/GSM8K/32B_1.5B_alpha_1.0_temp_1.0"

tokens_all = 0
for i in range(13):
    with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        # get the length of the model_output
        tokens = 0
        for j in range(len(data)):
            tokens += len(tokenizer(data[j]['model_output'])['input_ids'])
    tokens_all += tokens

print(tokens_all / (12*1319))