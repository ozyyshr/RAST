import random
import json

with open('./datasets/test.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

level_1 = []
level_2 = []
level_3 = []
level_4 = []
level_5 = []

for item in data:
    if item['level'] == 1:
        level_1.append(item)
    elif item['level'] == 2:
        level_2.append(item)
    elif item['level'] == 3:
        level_3.append(item)
    elif item['level'] == 4:
        level_4.append(item)
    elif item['level'] == 5:
        level_5.append(item)

def get_random_sample(data, num_samples):
    return random.sample(data, num_samples)

level_1_sample = get_random_sample(level_1, 10)
level_2_sample = get_random_sample(level_2, 10)
level_3_sample = get_random_sample(level_3, 10)
level_4_sample = get_random_sample(level_4, 10)
level_5_sample = get_random_sample(level_5, 10)

items_total = level_1_sample + level_2_sample + level_3_sample + level_4_sample + level_5_sample

random.shuffle(items_total)

with open('./datasets/MATH_sampled.jsonl', 'w') as f:
    for item in items_total:
        f.write(json.dumps(item) + '\n')