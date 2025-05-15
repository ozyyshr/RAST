import random
import os
import argparse
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="minerva_math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--res_dir", default="./results", type=str)
    parser.add_argument("--num_questions", default=100, type=int)
    parser.add_argument("--evaluation_mode", default="average", type=str, choices=["average", "majority", "pass"])
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen-boxed", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=16348, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    # if not os.path.exists(output_dir):
    #     output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def majority_eval(args, fname):
    # load model
    data_list = args.data_names.split(",")
    need_eval_data_list = []
    if not args.overwrite:
        for data_name in data_list:
            out_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
            out_file =  f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}.jsonl"
            out_metric_json = out_file.replace(".jsonl", f"_metrics.json")
            
            if os.path.exists(out_metric_json):
                print(f"Skipping {data_name} because {out_metric_json} already exists.")
                continue
            else:
                need_eval_data_list.append(data_name)
    
        if len(need_eval_data_list) == 0:
            print("All datasets already evaluated. Exiting.")
            exit(0)
        data_list = need_eval_data_list

    # infer & eval
    return main_majority(args.data_names, args, fname)


def main_majority(data_name, args, fname):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    # if args.apply_chat_template:
    #     input_prompts = [
    #         tokenizer.apply_chat_template(
    #             [{"role": "user", "content": prompt.strip()}],
    #             tokenize=False,
    #             add_generation_prompt=True,
    #         )
    #         for prompt in input_prompts
    #     ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    elif "deepseek" in args.prompt_type:
        stop_words.extend(["\nProblem", "User:", "Assistant:", "</answer>", "</s>"])
    elif "qwen" in args.prompt_type:
        stop_words.extend(["assistant", "user", "_end", "_start"])
    elif "abel" in args.prompt_type:
        stop_words.extend(["Question:", "Answer:"])

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        outputs = []
        with open(fname, 'r') as f:
            # Read the JSONL file line by line
            for line in f:
                # Parse each line as a JSON object
                data = json.loads(line)
                outputs.append((data['model_output'], None))
                
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), (output, finish_reason) in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query, finish_reason))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query, finish_reason))
            # elif "boxed" not in output and output.endswith("```"):
            #     program = extract_program(query)
            #     remain_prompts.append((i, query))
            #     remain_codes.append(program)
            else:
                end_prompts.append((i, query, finish_reason))

        # execute the remain prompts
        # assert len(remain_codes)==0
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            # assert False
            i, query, finish_reason = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query,finish_reason)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    finish_reasons = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt, finish_reason = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
        finish_reasons.append(finish_reason)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        finish_reason_list = finish_reasons[i * args.n_sampling : (i + 1) * args.n_sampling]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports, "finish_reason": finish_reason_list  })
        all_samples.append(sample)

    return all_samples

def setup(args, fname):
    # load model
    data_list = args.data_names.split(",")
    need_eval_data_list = []
    if not args.overwrite:
        for data_name in data_list:
            out_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
            out_file =  f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}.jsonl"
            out_metric_json = out_file.replace(".jsonl", f"_metrics.json")
            
            if os.path.exists(out_metric_json):
                print(f"Skipping {data_name} because {out_metric_json} already exists.")
                continue
            else:
                need_eval_data_list.append(data_name)
    
        if len(need_eval_data_list) == 0:
            print("All datasets already evaluated. Exiting.")
            exit(0)
        data_list = need_eval_data_list

    # infer & eval
    results = []
    for data_name in data_list:
        results.append(main(data_name, args, fname))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
    return results[0]["acc"]


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(data_name, args, fname):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    # if args.apply_chat_template:
    #     input_prompts = [
    #         tokenizer.apply_chat_template(
    #             [{"role": "user", "content": prompt.strip()}],
    #             tokenize=False,
    #             add_generation_prompt=True,
    #         )
    #         for prompt in input_prompts
    #     ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    elif "deepseek" in args.prompt_type:
        stop_words.extend(["\nProblem", "User:", "Assistant:", "</answer>", "</s>"])
    elif "qwen" in args.prompt_type:
        stop_words.extend(["assistant", "user", "_end", "_start"])
    elif "abel" in args.prompt_type:
        stop_words.extend(["Question:", "Answer:"])

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        outputs = []
        with open(fname, 'r') as f:
            # Read the JSONL file line by line
            for line in f:
                # Parse each line as a JSON object
                data = json.loads(line)
                outputs.append((data['model_output'], None))
                
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), (output, finish_reason) in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query, finish_reason))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query, finish_reason))
            # elif "boxed" not in output and output.endswith("```"):
            #     program = extract_program(query)
            #     remain_prompts.append((i, query))
            #     remain_codes.append(program)
            else:
                end_prompts.append((i, query, finish_reason))

        # execute the remain prompts
        # assert len(remain_codes)==0
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            # assert False
            i, query, finish_reason = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query,finish_reason)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    finish_reasons = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt, finish_reason = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
        finish_reasons.append(finish_reason)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        finish_reason_list = finish_reasons[i * args.n_sampling : (i + 1) * args.n_sampling]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports, "finish_reason": finish_reason_list  })
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json, _ = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json

def random_majority_answer(answers):
    preds = [str(ans["pred"]) for ans in answers]
    count = Counter(preds)
    majority_pred, _ = count.most_common(1)[0]
    
    # Filter original answers with majority pred
    majority_answers = [ans for ans in answers if str(ans["pred"]) == majority_pred]
    
    return random.choice(majority_answers)


if __name__ == "__main__":
    args = parse_args()
    # set_seed(args.seed) # comment this out for differnet ks, since we want one for sampling and average

    res_dir = args.res_dir

    # evaluation_mode = 'majority' # 'average@k' or 'majority@k' or 'pass@k'
    evaluation_mode = args.evaluation_mode

    if evaluation_mode == 'average':

        results = []
        for i in range(1):
            res = setup(args, fname=f"{res_dir}/prediction-{i}.jsonl")
            results.append(res)
        mean = np.mean(results)
        std = np.std(results)
        print(f"mean: {mean}, std: {std}")
    
    elif evaluation_mode == 'majority':
        if args.k == 32:
            count_num = 1
        else:
            count_num = 10
        score = 0
        for _ in range(count_num):
            results = []
            random_integers = random.sample(range(32), args.k)

            for i in random_integers:
                res = majority_eval(args, fname=f"{res_dir}/prediction-{i}.jsonl")
                results.append(res)
            
            all_samples = []
            for i in range(len(results[0])): # num of problems
                temp_res = []
                for j in range(len(results)): # number of inference runs
                    temp_res.append(results[j][i])
                
                # find the majority answer in temp_res, according to the key "pred"
                all_samples.append(random_majority_answer(temp_res))
            
            # calculate the accuracy
            all_samples, result_json, _ = evaluate(
                samples=all_samples,
                data_name=args.data_names,
                prompt_type=args.prompt_type,
                execute=True,
            )
            print(result_json)
            score += result_json['acc']
        print(f"score: {score/count_num}")

    elif evaluation_mode == 'pass':
        if args.k == 32:
            count_num = 1
        else:
            count_num = 10
        score = 0
        for _ in range(count_num):
            results = []
            matrix = [0] * args.num_questions
            random_integers = random.sample(range(32), args.k)

            for i in random_integers:
                res = majority_eval(args, fname=f"{res_dir}/prediction-{i}.jsonl")
            
                # calculate the accuracy
                all_samples, result_json, acc_sample = evaluate(
                    samples=res,
                    data_name=args.data_names,
                    prompt_type=args.prompt_type,
                    execute=True,
                )
                for i in range(len(acc_sample)):
                    matrix[i] += acc_sample[i]
            score += sum([1 for i in matrix if i > 0])/len(matrix)
        print(f"score: {score/count_num}")