from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import torch.nn.functional as F
from torch import Tensor
import random

# from sklearn.metrics.pairwise import cosine_similarity

model_kwargs = {
        'device_map': 'auto',
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
    }
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B", **model_kwargs)
positive_model = AutoModelForCausalLM.from_pretrained("hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo", **model_kwargs)
negative = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B", **model_kwargs)

base_model.eval()

def compute_percentage(prompt, answer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(base_model.device)
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(base_model.device)

    cnt = 0
    total_len = answer_ids.shape[1]

    for i in range(answer_ids.shape[1] - 1):
        cur_input_ids = torch.cat((input_ids, answer_ids[:, :i]), dim=1)

        try:
            with torch.no_grad():
                # get the next token id from base_model using greedy decoding
                logits = base_model(cur_input_ids).logits
                next_token_logits = logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs, dim=-1)
                if next_token_id == answer_ids[:, i]:
                    cnt += 1

        except:
            print(f"Error in {i}", flush=True)
            return cnt / (i+1)
    print(f"cnt = {cnt}, total_len = {total_len}", flush=True)
    return cnt / total_len

def compute_percentage_delta(prompt, answer, i):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(base_model.device)
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(base_model.device)

    cnt = 0
    total_len = answer_ids.shape[1]

    for i in range(answer_ids.shape[1] - 1):
        cur_input_ids = torch.cat((input_ids, answer_ids[:, :i]), dim=1)

        try:
            with torch.no_grad():
                # get the next token id from base_model using greedy decoding
                base_logits = base_model(cur_input_ids).logits[:, -1, :]
                positive_logits = positive_model(cur_input_ids).logits[:, -1, :]
                negative_logits = negative(cur_input_ids).logits[:, -1, :]
                # get the next token id from positive_model using greedy decoding
                next_token_logits = base_logits + i * (positive_logits - negative_logits)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs, dim=-1)
                # next_token_logits = logits[:, -1, :]
                # next_token_probs = F.softmax(next_token_logits, dim=-1)
                # next_token_id = torch.argmax(next_token_probs, dim=-1)

                if next_token_id == answer_ids[:, i]:
                    cnt += 1

        except:
            print(f"Error in {i}", flush=True)
            return cnt / (i+1)
    print(f"cnt = {cnt}, total_len = {total_len}", flush=True)
    return cnt / total_len
            

for i in [1.4, 1.5]:
# for i in [1.5, 1.0, 0.5]:
    print(f"processing lambda = {i}", flush=True)

    with open(f"/shared/data3/siruo2/ContrastiveReasoning/results/qwen-32b-chat-greedy/prediction-0.jsonl", 'r') as f:
    # with open(f"/shared/data3/siruo2/ContrastiveReasoning/results/MATH/32B_14B_alpha_{i}_temp_0.0/prediction-0.jsonl", 'r') as f:
        data = []
        for line in f:
            item = json.loads(line)
            data.append(item)
    # random select 100 items from data
    data = random.sample(data, 10)
    avg = 0
    for idx, dp in enumerate(data):
        print(f"processing {idx} / {len(data)}", flush=True)
        percentage = compute_percentage_delta(dp['question'], dp['model_output'], i)
        # percentage = compute_percentage(dp['question'], dp['model_output'])
        avg += percentage
        print(avg, flush=True) # 11.79
    print(avg/10, flush=True)