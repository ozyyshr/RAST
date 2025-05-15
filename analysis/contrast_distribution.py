import json
import re

with open('./datasets/MATH_sampled.jsonl') as f:
    dataset = [json.loads(line) for line in f]

with open('./results/direct_inference/output_math.jsonl') as f:
    expert_data = [json.loads(line) for line in f]


def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match.group(1)
    return None

expert_greater_than_base = {}
base_greater_than_expert = {}
expert_base = {}

count = 0
count_exp = 0
count_base = 0
count_near = 0
count_none = 0

with open('./results/MATH/Qwen-1.5B-CD/tokens_stats.json') as f:
    all_tokens = json.load(f)

tokens = {}

for k, v in all_tokens.items():
    if v[0] <=5:
        continue
    # maintain two digits after the decimal point
    tokens[k]= [round(v[1]/v[0]*100,2), round(v[2]/v[0]*100,2), round(v[3]/v[0]*100,2)]

# filter tokens with item[1][0] being the max
expert_tokens = {k: v for k, v in tokens.items() if v[0] >= max(v[1], v[2])}
# sort tokens by values
expert_tokens = {k: v for k, v in sorted(expert_tokens.items(), key=lambda item: item[1][0], reverse=True)}
# get top 100 items
print(list(expert_tokens.items())[:100])
# sort tokens by values
base_tokens = {k: v for k, v in tokens.items() if v[1] >= max(v[0], v[2])}
# sort tokens by values
base_tokens = {k: v for k, v in sorted(base_tokens.items(), key=lambda item: item[1][1], reverse=True)}
# get top 100 items
print(list(base_tokens.items())[:100])
# sort tokens by values
near_tokens = {k: v for k, v in tokens.items() if v[2] >= max(v[1], v[0])}
# sort tokens by values
near_tokens = {k: v for k, v in sorted(near_tokens.items(), key=lambda item: item[1][2], reverse=True)}
# get top 100 items
print(list(near_tokens.items())[:100])



all_tokens = {} # for each token, maintain a list recording the number it appears in expert, base and near

for i in range(40):

    with open(f'./results/MATH/Qwen-1.5B-CD/MATH-{i}.jsonl') as f:
        data = [json.loads(line) for line in f]

        ground_truth = dataset[i]['answer']
        difficulty_level = dataset[i]['level']

        prediction = "".join([item['next_token'] for item in data])

        extracted_pred = extract_boxed_answer(prediction)

        for data_item in data:
            selected_token = data_item['next_token']
            if selected_token not in all_tokens.keys():
                all_tokens[selected_token] = [1, 0, 0, 0]
            else:
                all_tokens[selected_token][0] += 1

            expert_tokens = data_item['raw_expert_greater_than_base'].keys()

            # for t in expert_tokens:
            #     if t not in expert_greater_than_base:
            #         expert_greater_than_base[t] = 0
            #     expert_greater_than_base[t] += 1
            
            base_tokens = data_item['raw_base_greater_than_expert'].keys()
            # for t in base_tokens:
            #     if t not in base_greater_than_expert:
            #         base_greater_than_expert[t] = 0
            #     base_greater_than_expert[t] += 1

            near_tokens = data_item['raw_expert_close_to_base'].keys()
            # for t in near_tokens:
            #     if t not in expert_base:
            #         expert_base[t] = 0
            #     expert_base[t] += 1

            if selected_token in expert_tokens:
                all_tokens[selected_token][1] += 1
            elif selected_token in base_tokens:
                all_tokens[selected_token][2] += 1
            elif selected_token in near_tokens:
                all_tokens[selected_token][3] += 1
            

            # if len(expert_tokens) == 0 and len(base_tokens) == 0 and len(near_tokens) == 0:
            #     count_none += 1
            
            # if data_item['next_token'] in expert_tokens:
            #     count_exp += 1
            # if data_item['next_token'] in base_tokens:
            #     print(data_item['next_token'].encode('utf-8'))
            #     count_base += 1
            # if data_item['next_token'] in near_tokens:
            #     count_near += 1
            # count += 1

with open('./results/MATH/Qwen-1.5B-CD/tokens_stats.json', 'w') as f:
    # dump all_toens to a json file
    json.dump(all_tokens, f)
print(len(all_tokens))



# sort expert_greater_than_base by values
expert_greater_than_base = {k: v for k, v in sorted(expert_greater_than_base.items(), key=lambda item: item[1], reverse=True)}
base_greater_than_expert = {k: v for k, v in sorted(base_greater_than_expert.items(), key=lambda item: item[1], reverse=True)}
expert_base = {k: v for k, v in sorted(expert_base.items(), key=lambda item: item[1], reverse=True)}
# get top 100 items
print(list(expert_greater_than_base.items())[:100])
print("------------------------")
print(list(base_greater_than_expert.items())[:100])
print("------------------------")
print(list(expert_base.items())[:100])

print(count_exp)
print(count_base)
print(count_near)
print(count_none)
print(count)
        


        # for item in data:
        #     contrast = item['expert_logits'][0] - item['base_logits'][0]
        #     if contrast <=6.0 and contrast >= .0:
        #         dicts[item['next_token']] = contrast
        #         # if item['next_token'] not in dicts.keys():
        #         #     dicts[item['next_token']] = 1
        #         # else:
        #         #     dicts[item['next_token']] += 1
# print(len(dicts))


# def get_accuracy(answer, pred):

    




# sort dicts by values
# dicts = {k: v for k, v in sorted(dicts.items(), key=lambda item: item[1], reverse=True)}
# print top 50 items
# print(list(dicts.items())[:50])
# plot distribution in dicts
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(len(dicts))
# plt.bar(x, dicts.values())
# plt.xticks(x, dicts.keys(), rotation=90)
# plt.tight_layout()
# plt.show()
# plt.savefig('contrast_distribution.png')