import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import os

def load_log(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def compute_average_entropy_per_log(log_dir):
    """
    Compute average entropy metrics for each .jsonl log file in a directory.

    Args:
        log_dir: directory with .jsonl log files
        output_path: where to save the summary TSV file
    """
    stats_1 = {"expert": 0, "base": 0, "cd": 0, "diff": 0}
    stats_2 = {"expert": 0, "base": 0, "cd": 0, "diff": 0}
    stats_3 = {"expert": 0, "base": 0, "cd": 0, "diff": 0}
    stats_4 = {"expert": 0, "base": 0, "cd": 0, "diff": 0}
    stats_5 = {"expert": 0, "base": 0, "cd": 0, "diff": 0}

    log_files = [f"MATH-{i}.jsonl" for i in range(50)]
    with open("./datasets/MATH_sampled.jsonl", 'r', encoding='utf-8') as f:
        ori_data = [json.loads(line) for line in f]
    difficulty = [ori_data[i]['level'] for i in range(50)]

    for log_ids, log_file in enumerate(log_files):
        path = os.path.join(log_dir, log_file)
        with open(path, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]

        if not logs:
            continue

        expert = [log["entropy_expert_raw"] for log in logs]
        base = [log["entropy_base_raw"] for log in logs]
        cd = [log["entropy_cd"] for log in logs]
        diff = [e - b for e, b in zip(expert, base)]

        avg_expert = np.mean(expert)
        avg_base = np.mean(base)
        avg_cd = np.mean(cd)
        avg_diff = np.mean(diff)
        print(avg_base, avg_expert, avg_cd, avg_diff)

        if difficulty[log_ids] == 1:
            stats_1["expert"] += avg_expert
            stats_1["base"] += avg_base
            stats_1["cd"] += avg_cd
            stats_1["diff"] += avg_diff
        elif difficulty[log_ids] == 2:
            stats_2["expert"] += avg_expert
            stats_2["base"] += avg_base
            stats_2["cd"] += avg_cd
            stats_2["diff"] += avg_diff
        elif difficulty[log_ids] == 3:
            stats_3["expert"] += avg_expert
            stats_3["base"] += avg_base
            stats_3["cd"] += avg_cd
            stats_3["diff"] += avg_diff
        elif difficulty[log_ids] == 4: 
            stats_4["expert"] += avg_expert
            stats_4["base"] += avg_base
            stats_4["cd"] += avg_cd
            stats_4["diff"] += avg_diff
        elif difficulty[log_ids] == 5:
            stats_5["expert"] += avg_expert
            stats_5["base"] += avg_base
            stats_5["cd"] += avg_cd
            stats_5["diff"] += avg_diff

    print(stats_1)
    print(stats_2)
    print(stats_3)
    print(stats_4)
    print(stats_5)


def plot_entropy(logs, title='Entropy Over Decoding Steps'):
    steps = [log['decoding_step'] for log in logs]
    exp_ent = [log['entropy_expert_raw'] for log in logs]
    base_ent = [log['entropy_base_raw'] for log in logs]
    cd_ent = [log['entropy_cd'] for log in logs]

    plt.plot(steps, exp_ent, label='Expert')
    plt.plot(steps, base_ent, label='Base')
    plt.plot(steps, cd_ent, label='CD')
    plt.xlabel('Step')
    plt.ylabel('Entropy')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_kl_div(logs, title='KL Divergence (Expert || Base)'):
    steps = [log['decoding_step'] for log in logs]
    kl_divs = [log['kl_div_expert_base_raw'] for log in logs]

    plt.plot(steps, kl_divs, label='KL(expert || base)', color='purple')
    plt.xlabel('Step')
    plt.ylabel('KL Divergence')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_topk_overlap(logs, title='Top-k Overlap (Raw)'):
    steps = [log['decoding_step'] for log in logs]
    overlap = [log['top_k_overlap_raw'] for log in logs]

    plt.plot(steps, overlap, label='Top-k Overlap', color='green')
    plt.xlabel('Step')
    plt.ylabel('# of Shared Tokens in Top-k')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

def summarize_contrastive_tokens(logs):
    token_counts = defaultdict(int)
    for log in logs:
        for token in log.get('raw_expert_greater_than_base', {}):
            token_counts[token] += 1
    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
    print(f"\n🔎 Tokens frequently favored by expert over base (raw):")
    for token, count in sorted_tokens[:10]:
        print(f"{token!r}: {count} steps")

def find_entropy_spikes(logs, threshold=0.4):
    print(f"\n⚠️  Steps with large expert–base entropy differences (>|{threshold}|):")
    for log in logs:
        diff = log['entropy_diff']
        if abs(diff) > threshold:
            print(f"Step {log['decoding_step']}: ΔEntropy = {diff:.3f}, Token = {log['next_token']!r}")

def analyze_entropy_tokens(
    logs,
    top_frac=0.1,
    output_dir="entropy_analysis_logs"
):
    """
    Analyze entropy-related signals and save top/bottom percentile tokens.

    Args:
        logs: list of decoding step logs (from JSONL)
        top_frac: float, fraction of tokens to consider top/bottom
        output_dir: where to save the token logs
    """
    os.makedirs(output_dir, exist_ok=True)

    signals = {
        # 'entropy_expert_raw': [],
        'entropy_base_raw': [],
        # 'entropy_cd': [],
        # 'entropy_diff': [],
    }
    tokens = []

    for log in logs:
        token = log['next_token'].strip()
        tokens.append(token)

        # signals['entropy_expert_raw'].append(log['entropy_expert_raw'])
        signals['entropy_base_raw'].append(log['entropy_base_raw'])
        # signals['entropy_cd'].append(log['entropy_cd'])
        # signals['entropy_diff'].append(
        #     log['entropy_expert_raw'] - log['entropy_base_raw']
        # )

    for signal_name, values in signals.items():
        values = np.array(values)
        token_value_pairs = list(zip(tokens, values))

        n = len(token_value_pairs)
        top_n = max(1, int(top_frac * n))
        sorted_by_val = sorted(token_value_pairs, key=lambda x: x[1], reverse=True)

        top_tokens = sorted_by_val[:top_n]
        bottom_tokens = sorted_by_val[-top_n:]

        return top_tokens, bottom_tokens

        # Save each set to file
        # def save_list_to_file(data, name):
        #     path = os.path.join(output_dir, f"{signal_name}_{name}.txt")
        #     with open(path, "w", encoding="utf-8") as f:
        #         for token, val in data:
        #             f.write(f"{token}\t{val:.4f}\n")

        # save_list_to_file(top_tokens, "top10")
        # save_list_to_file(bottom_tokens, "bottom10")

        # print(f"[✓] Saved {signal_name} top/bottom 10% tokens to '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", help="./results/MATH/Distilled-llama-8B-CD/MATH-0.jsonl")
    args = parser.parse_args()

    log_files = [f"./results/MATH/Distilled-llama-8B-CD/MATH-{i}.jsonl" for i in range(50)]
    top = {}
    bottom = {}
    for log_file in log_files:
        logs = load_log(log_file)
        top_tokens, bottom_tokens = analyze_entropy_tokens(logs)
        for k,v in top_tokens:
            if k not in top.keys():
                top[k] = 0
            top[k] += 1
        for k,v in bottom_tokens:
            if k not in bottom.keys():
                bottom[k] = 0
            bottom[k] += 1
    top = sorted(top.items(), key=lambda x: -x[1])
    bottom = sorted(bottom.items(), key=lambda x: -x[1])
    with open("./entropy_analysis_logs/base_raw_overall_top.txt", "w") as f:
        for k,v in top:
            f.write(f"{k}\t{v}\n")
    with open("./entropy_analysis_logs/base_raw_overall_bottom.txt", "w") as f:
        for k,v in bottom:
            f.write(f"{k}\t{v}\n")
    # compute_average_entropy_per_log("./results/MATH/Distilled-llama-8B-CD/")

    
    input()

    plot_entropy(logs)
    plot_topk_overlap(logs)
    summarize_contrastive_tokens(logs)
    find_entropy_spikes(logs)

if __name__ == "__main__":
    main()