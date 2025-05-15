import argparse
import os
import re
import json
import random
import evaluate
import datasets
from utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    load_logit_model_and_tokenizer,
    load_base_model_and_tokenizer,
    dynamic_import_function,
    ensure_dir
)
import wandb

wandb.init(project='reasoning_model_logits_RLZero')

exact_match = evaluate.load("exact_match")

def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match.group(1)
    return None


def main(args):
    random.seed(42)

    print("Loading data...")
    if args.data_dir == "AIME":
        test_data = datasets.load_dataset("Maxwell-Jia/AIME_2024",split="train")
    elif args.data_dir == "MATH":
        test_data = []
        with open('./datasets/MATH_test.jsonl') as fin:
            for line in fin:
                test_data.append(json.loads(line))
    # test_data = []
    # with open(os.path.join(args.data_dir, "test.jsonl")) as fin:
    #     for line in fin:
    #         example = json.loads(line)
    #         test_data.append({
    #             "question": example["question"],
    #             "answer": example["answer"].split("####")[1].strip()
    #         })

    # some numbers are in the `x,xxx` format, and we want to remove the comma
    # for example in test_data:
    #     example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
    #     assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = random.sample(test_data, args.max_examples)

    ensure_dir(args.save_dir)

    if args.use_chat_format:
        prompt_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt_template = "{input}\n\nPlease reason step by step, and put your final answer within \boxed{{}}."

    prompts = []
    # chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in test_data:
        if args.data_dir == "MATH":
            prompt = prompt_template.format(input=example["problem"])
        elif args.data_dir == "AIME":
            prompt = prompt_template.format(input=example["Problem"])
        # if args.use_chat_format:
        #     messages = [{"role": "user", "content": prompt}]
        #     prompt = chat_formatting_function(messages, add_bos=False)
        #     if prompt[-1] in ["\n", " "]:
        #         prompt += "Answer:"
        #     else:
        #         prompt += " Answer:"
        # else:
        #     prompt += "\nAnswer:"
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_base_model_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.antiexpert_model_name_or_path:
        print("Loading Proxy tuning model and tokenizer...")
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            args.antiexpert_model_name_or_path,
            args.expert_model1,
            args.antiexpert_model1,
            chat_response_prefix="Answer:",
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_logit_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            chat_response_prefix="Answer:",
            reasoning_response_prefix="<think>",
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    
    print(f"Generation starts with do_sample={args.do_sample}, temperature={args.temperature}")  

    outputs = generate_completions(
        args,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=16384,
        temperature=args.temperature,
        batch_size=args.eval_batch_size,
        do_sample=args.do_sample,
    )

    ori_outputs = [i for i in outputs]

    outputs = [extract_boxed_answer(o) for o in outputs]

    # predictions = []
    # for output in outputs:
    #     # replace numbers like `x,xxx` with `xxxx`
    #     output = re.sub(r"(\d),(\d)", r"\1\2", output)
    #     numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    #     if numbers:
    #         predictions.append(numbers[-1])
    #     else:
    #         predictions.append(output)

    print("Calculating accuracy...")
    # targets = [example["answer"] for example in test_data]

    # em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    # print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["problem"] if args.data_dir == "MATH" else example["Problem"],
        "answer": example["answer"] if args.data_dir == "MATH" else example["Answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, ori_outputs, outputs)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump({
    #         "exact_match": em_score
    #     }, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gsm"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="If given, we will sample from the model."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for sampling."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--antiexpert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--expert_model1",
        type=str,
        default=None
    )
    parser.add_argument(
        "--antiexpert_model1",
        type=str,
        default=None
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    main(args)