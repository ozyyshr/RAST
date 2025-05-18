import time
import json
from vllm import LLM, SamplingParams
import os
import argparse
import datasets

def main(args):

    base_model = args.base_model
    positive_model = args.positive_model
    negative_model = args.negative_model
    positive_model1 = args.positive_model1
    negative_model1 = args.negative_model1
    positive_model2 = args.positive_model2
    negative_model2 = args.negative_model2

    if args.dataset == "MATH":
        # with open("../datasets/MATH_test.jsonl") as fin:
        #     hf_dataset = []
        #     for line in fin:
        #         hf_dataset.append(json.loads(line))
        hf_dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif args.dataset == "AIME":
        hf_dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif args.dataset == "Minerva":
        hf_dataset = datasets.load_dataset("math-ai/minervamath", split="test")
    elif args.dataset == "AMC":
        hf_dataset = datasets.load_dataset("math-ai/amc23", split="test")
    elif args.dataset == 'Olympiad':
        with open("../datasets/olympiad.jsonl") as fin:
            hf_dataset = []
            for line in fin:
                hf_dataset.append(json.loads(line))
    elif args.dataset == 'GSM8K':
        with open("../datasets/gsm8k.jsonl") as fin:
            hf_dataset = []
            for line in fin:
                hf_dataset.append(json.loads(line))

    worker = LLM(
        model=base_model,
        cd_positive_model=positive_model,
        cd_negative_model=negative_model,
        cd_positive_model1 = positive_model1,
        cd_negative_model1 = negative_model1,
        cd_positive_model2 = positive_model2,
        cd_negative_model2 = negative_model2,
        cd_decoding_alpha=args.cd_decoding_alpha,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    use_chat = False

    if use_chat:
        prompt_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt_template = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    
    if args.dataset == "MATH" or args.dataset == "AIME":
        prompts = [prompt_template.format(input=example["problem"]) for example in hf_dataset]
    elif args.dataset == "Minerva" or args.dataset == "AMC" or args.dataset == 'Olympiad' or args.dataset == 'GSM8K':
        prompts = [prompt_template.format(input=example["question"]) for example in hf_dataset]


    sampling_params = SamplingParams(
        temperature=args.decoding_temperature, 
        top_p=args.top_p,
        max_tokens=16384,
        n=1,
        stop=["</s>", "<|im_end|>", "<|endoftext|>", "assistant", "user", "_end", "_start"],
        stop_token_ids=[151645, 151643],
    )
    
    base_scale = args.base_model.split("-")[-1]
    rl_scale = args.negative_model.split("-")[-1]

    ### run several times and save them inside a folder
    for i in range(args.num_runs):
        os.makedirs(f"../results/{args.dataset}/{base_scale}_{rl_scale}_alpha_{args.cd_decoding_alpha}_temp_{args.decoding_temperature}", exist_ok=True)
        outputs = worker.generate(prompts, sampling_params, use_tqdm=True)
        with open(f"../results/{args.dataset}/{base_scale}_{rl_scale}_alpha_{args.cd_decoding_alpha}_temp_{args.decoding_temperature}/prediction-{i}.jsonl", "a") as f:
            if args.dataset == "MATH":
                for i, output in enumerate(outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    label = hf_dataset[i]['answer']
                    data_item = {
                        "question": prompt,
                        "model_output": generated_text,
                        "answer": label,
                        "difficulty": hf_dataset[i]['level'],
                    }
                    f.write(json.dumps(data_item) + "\n")
            elif args.dataset == "Olympiad":
                for i, output in enumerate(outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    label = hf_dataset[i]['final_answer'][0]
                    data_item = {
                        "question": prompt,
                        "model_output": generated_text,
                        "answer": label,
                    }
                    f.write(json.dumps(data_item) + "\n")
            else:
                for i, output in enumerate(outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    label = hf_dataset[i]['answer']
                    data_item = {
                        "question": prompt,
                        "model_output": generated_text,
                        "answer": label,
                    }
                    f.write(json.dumps(data_item) + "\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run LLM with contrastive decoding")
    parser.add_argument("--dataset", type=str, required=True, choices=["MATH", "AIME", "Minerva", "AMC", "Olympiad", "GSM8K"], help="Dataset to use")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--positive_model", type=str, required=True, help="Positive model name")
    parser.add_argument("--negative_model", type=str, required=True, help="Negative model name")
    parser.add_argument("--positive_model1", type=str, default=None, help="Positive model name 1")
    parser.add_argument("--negative_model1", type=str, default=None, help="Negative model name 1")
    parser.add_argument("--positive_model2", type=str, default=None, help="Positive model name 2")
    parser.add_argument("--negative_model2", type=str, default=None, help="Negative model name 2")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--cd_decoding_alpha", type=float, default=1.0, help="Decoding alpha for contrastive decoding")
    parser.add_argument("--decoding_temperature", type=float, default=1.0, help="Decoding temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for generating outputs")

    args = parser.parse_args()

    main(args)