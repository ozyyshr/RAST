import torch
from transformers import pipeline
import json
import datasets

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load dataset from "MATH_sampled.jsonl"
dataset = []
with open("./datasets/MATH_sampled.jsonl", "r") as f:
    for line in f:
        dataset.append(json.loads(line))

llama_31 = "meta-llama/Llama-3.1-8B-Instruct" # <-- llama 3.1
llama_32 = "meta-llama/Llama-3.2-3B-Instruct" # <-- llama 3.2
llama_ds = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

instruction = "Please reason step by step, and put your final answer within \boxed{}. Avoid overthinking (uncessary verifications or thinking) or underthinking (constantly changing ideas without fully exploring them)."

for item in dataset:
    problem = item["problem"]

    prompt = [
        {"role": "user", "content": problem+ "\n"+instruction},
    ]

    generator = pipeline(model=llama_ds, device=device, torch_dtype=torch.bfloat16)
    generation = generator(
        prompt,
        do_sample=True,
        temperature=0.6,
        max_new_tokens=128000,
    )
    outputs = {
        "unique_id": item["unique_id"],
        "problem": problem,
        "generated_text": generation[0]["generated_text"],
        "answer": item["answer"],
        "level": item["level"],
    }

    with open("./inference_res/output_math_prompt.jsonl", "a") as f:
        f.write(json.dumps(outputs) + "\n")


# Generation:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant, that responds as a pirate.'},
#   {'role': 'user', 'content': "What's Deep Learning?"},
#   {'role': 'assistant', 'content': "Yer lookin' fer a treasure trove o'
#             knowledge on Deep Learnin', eh? Alright then, listen close and
#             I'll tell ye about it.\n\nDeep Learnin' be a type o' machine
#             learnin' that uses neural networks"}
# ]