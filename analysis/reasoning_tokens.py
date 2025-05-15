import json
import spacy
from collections import Counter, defaultdict
from tqdm import trange

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Base (lemmatized) tokens for each reasoning behavior
REASONING_TOKENS = {
    "branching": {
        "alternatively", "another", "try", "suppose", "consider",
        "different", "assume", "option"
    },
    "backtracking": {
        "however", "but", "mistake", "error", "contradiction",
        "wrong", "revisit", "actually", "again", "flaw"
    },
    "self_verification": {
        "check", "verify", "confirm", "satisfy", "plug", "back",
        "substitute", "ensure", "validate", "test"
    }
}

def count_reasoning_tokens_lemma(outputs):
    """
    Count the frequency of lemmatized reasoning tokens in model outputs.

    Args:
        outputs (List[str]): A list of generated texts from a model.

    Returns:
        dict: Dictionary with per-token and per-category counts.
    """
    total_counts = {k: Counter() for k in REASONING_TOKENS}
    global_counts = defaultdict(int)

    for doc in nlp.pipe(outputs, batch_size=32):
        for token in doc:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower()
            for behavior, token_set in REASONING_TOKENS.items():
                if lemma in token_set:
                    total_counts[behavior][lemma] += 1
                    global_counts[behavior] += 1

    return {
        "per_token_counts": total_counts,
        "total_counts": dict(global_counts)
    }

## AIME dataset
num_problem = 675

# res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/baseline/Olympiad/Qwen-Qwen2.5-32B_temp_0.0"
# print(res_dir)
# # read all the jsonl files in the directory

# counts = {
#     "branching": 0,
#     "backtracking": 0,
#     "self_verification": 0
# }
# for i in trange(1):

#     with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
#         data = [json.loads(line) for line in f]
#         # get the model_output
#         outputs = [d['model_output'] for d in data]
#         # count the reasoning tokens
#         count_tokens = count_reasoning_tokens_lemma(outputs)
#         counts['branching'] += count_tokens['total_counts']['branching']
#         counts['backtracking'] += count_tokens['total_counts']['backtracking']
#         counts['self_verification'] += count_tokens['total_counts']['self_verification']
# print(counts)
# print(counts['branching'] / (1*num_problem))
# print(counts['backtracking'] / (1*num_problem))
# print(counts['self_verification'] / (1*num_problem))

print("===================================")

res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/Olympiad/32B_14B_alpha_1.0_temp_1.0"
print(res_dir)
# read all the jsonl files in the directory

counts = {
    "branching": 0,
    "backtracking": 0,
    "self_verification": 0
}
for i in trange(32):

    with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        # get the model_output
        outputs = [d['model_output'] for d in data]
        # count the reasoning tokens
        count_tokens = count_reasoning_tokens_lemma(outputs)
        counts['branching'] += count_tokens['total_counts']['branching']
        counts['backtracking'] += count_tokens['total_counts']['backtracking']
        counts['self_verification'] += count_tokens['total_counts']['self_verification']
print(counts)
print(counts['branching'] / (32*num_problem))
print(counts['backtracking'] / (32*num_problem))
print(counts['self_verification'] / (32*num_problem))

print("===================================")

res_dir = "/shared/data3/siruo2/ContrastiveReasoning/results/baseline/Olympiad/hkust-nlp-Qwen-2.5-32B-SimpleRL-Zoo_temp_1.0"
print(res_dir)
# read all the jsonl files in the directory

counts = {
    "branching": 0,
    "backtracking": 0,
    "self_verification": 0
}
for i in trange(32):

    with open(f"{res_dir}/prediction-{i}.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        # get the model_output
        outputs = [d['model_output'] for d in data]
        # count the reasoning tokens
        count_tokens = count_reasoning_tokens_lemma(outputs)
        counts['branching'] += count_tokens['total_counts']['branching']
        counts['backtracking'] += count_tokens['total_counts']['backtracking']
        counts['self_verification'] += count_tokens['total_counts']['self_verification']
print(counts)
print(counts['branching'] / (32*num_problem))
print(counts['backtracking'] / (32*num_problem))
print(counts['self_verification'] / (32*num_problem))