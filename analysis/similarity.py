from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# with open('/shared/data3/siruo2/ContrastiveReasoning/results/qwen-32b-14b-rl-chat/prediction-0.jsonl', 'r') as f:
#     data = []
#     for line in f:
#         item = json.loads(line)
#         data.append(item)
# with open('/shared/data3/siruo2/ContrastiveReasoning/results/qwen-32b-1.5b-rl-direct/prediction-0.jsonl', 'r') as f:
#     data2 = []
#     for line in f:
#         item = json.loads(line)
#         data2.append(item)

# for i in range(len(data)):
#     print(data[i]['model_output'])
#     print("==============================")
#     print(data2[i]['model_output'])
#     input()


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# with open("./analysis/logit_magnitude.jsonl", "r") as f:
#     data = []
#     for line in f:
#         item = json.loads(line)
#         data.append(item)
# for item in data:
#     token = tokenizer.convert_ids_to_tokens(item['token_id']).replace("Ġ", "_")
#     print(token, item['delta_logit']/(item['delta_logit']+item['base_logit']))

def compute_cosine_similarity(input_ids, answer_ids):

    cosine_average = 0
    for i in range(answer_ids.shape[1]):
        # try:
        input_ids = torch.cat((input_ids, answer_ids[:, :i]), dim=1)
        with torch.no_grad():
            base_model1.eval()
            RL_model1.eval()
            input_ids_b1 = input_ids.to(base_model1.device)
            input_ids_rl1 = input_ids.to(RL_model1.device)
            model1_logit_diff = RL_model1(input_ids=input_ids_rl1, return_dict=True).logits[..., -1, :] - base_model1(input_ids=input_ids_b1, return_dict=True).logits[..., -1, :]

            base_model2.eval()
            RL_model2.eval()
            input_ids_b2 = input_ids.to(base_model2.device)
            input_ids_rl2 = input_ids.to(RL_model2.device)
            model2_logit_diff = RL_model2(input_ids=input_ids_rl2, return_dict=True).logits[..., -1, :] - base_model2(input_ids=input_ids_b2, return_dict=True).logits[..., -1, :]
            # model2_logit_diff = base_model2(input_ids=input_ids_b2, return_dict=True).logits[..., -1, :]
        # calculate cosine similarity
        model1_logit_diff = model1_logit_diff.squeeze(0)
        model2_logit_diff = model2_logit_diff.squeeze(0)
        model1_logit_diff = model1_logit_diff.cpu().numpy()
        model2_logit_diff = model2_logit_diff.cpu().numpy()

        # truncate model1_logit_diff and model2_logit_diff to the same length
        min_length = min(model1_logit_diff.shape[0], model2_logit_diff.shape[0])
        model1_logit_diff = model1_logit_diff[:min_length]
        model2_logit_diff = model2_logit_diff[:min_length]
        similarity = cosine_similarity(model1_logit_diff.reshape(1, -1), model2_logit_diff.reshape(1, -1))
        similarity = similarity[0][0]
        
        cosine_average += similarity
        print(cosine_average)
            # print("Cosine similarity: ", similarity, " | ", tokenizer.decode(answer_ids[:, i][0], skip_special_tokens=True))
        # except Exception as e:
        #     break
    
    return cosine_average/(i+1)

### KL_divergence computation
### ---------------------------
# with open("./analysis/logit_distribution.jsonl", "r") as f:
#     data = []
#     for line in f:
#         item = json.loads(line)
#         data.append(item)

# for item in data:
#     # generate logits using the base model
#     with torch.no_grad():
#         base_model.eval()
#         outputs = base_model(input_ids=input_ids, return_dict=True)
#         logits = outputs.logits[..., -1, :] + 1e-6
#         base_logit = F.log_softmax(logits, dim=-1) 
#     # generate logits using the RL model
#     with torch.no_grad():
#         RL_model.eval()
#         outputs = RL_model(input_ids=input_ids, return_dict=True)
#         logits = outputs.logits[..., -1, :] + 1e-6
#         RL_logit = F.log_softmax(logits, dim=-1) 

#     logit = torch.Tensor(item['logit']) + 1e-6
#     M = F.softmax(logit, dim=-1).cuda()
#     base_kl_value = F.kl_div(base_logit, M, reduction="none").mean(-1) *1e9
#     rl_kl_value = F.kl_div(RL_logit, M, reduction='none').mean(-1) * 1e9

#     # concatenate the current input_ids with item['token_id']
#     input_ids = torch.cat((input_ids, torch.tensor([[item['token_id']]]).cuda()), dim=1)
#     with open("./analysis/kl_distribution.jsonl", "a") as f:
#         token_info = {
#             'token_id': tokenizer.convert_ids_to_tokens(item['token_id']).replace("Ġ", "_"),
#             'base_kl': base_kl_value.item(),
#             'rl_kl': rl_kl_value.item(),
#         }
#         f.write(json.dumps(token_info) + "\n")

# input()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--base_model1", type=str, default="Qwen/Qwen2.5-7B")
    args.add_argument("--RL_model1", type=str, default="hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo")
    args.add_argument("--base_model2", type=str, default="Qwen/Qwen2.5-1.5B")
    args.add_argument("--RL_model2", type=str, default="hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo")

    args = args.parse_args()

    model_kwargs = {
        'device_map': 'auto',
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
    }
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
    base_model1 = AutoModelForCausalLM.from_pretrained(args.base_model1, **model_kwargs)
    RL_model1 = AutoModelForCausalLM.from_pretrained(args.RL_model1, **model_kwargs)
    base_model2 = AutoModelForCausalLM.from_pretrained(args.base_model2, **model_kwargs)
    RL_model2 = AutoModelForCausalLM.from_pretrained(args.RL_model2, **model_kwargs)

    with open('/shared/data3/siruo2/ContrastiveReasoning/results/qwen-32b-rl/prediction-1.jsonl', 'r') as f:
        data = []
        for line in f:
            item = json.loads(line)
            data.append(item)

    with open('./analysis/similarity.jsonl', 'a') as f:
    
        for i in range(len(data[:50])):

            input_prompt = data[i]['question']
            input_ids = tokenizer(input_prompt)["input_ids"]
            input_ids = torch.tensor(input_ids).unsqueeze(0)

            answer = data[i]['model_output']
            answer_ids = tokenizer(answer)["input_ids"]
            answer_ids = torch.tensor(answer_ids).unsqueeze(0)

            cosine_simlarity = compute_cosine_similarity(input_ids, answer_ids)

            f.write(json.dumps({
                'data_id': i,
                'base_model1': args.base_model1,
                'RL_model1': args.RL_model1,
                'base_model2': args.base_model2,
                'RL_model2': args.RL_model2,
                'cosine_similarity': cosine_simlarity
            }) + "\n")

