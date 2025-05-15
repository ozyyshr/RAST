import torch
import tqdm
import os
from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)
import json
from torch import Tensor
import torch.nn.functional as F
import wandb

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

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.inference_mode()
def generate_completions(
    args,
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=0.95,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts['input_ids']
        attention_mask = tokenized_prompts['attention_mask']

        if model.device.type == "cuda":
            if isinstance(batch_input_ids, dict):
                for k in batch_input_ids:
                    batch_input_ids[k] = batch_input_ids[k].cuda()
                    attention_mask[k] = attention_mask[k].cuda()
            else:
                batch_input_ids = batch_input_ids.cuda()
                attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            save_logit_dir=f'{args.save_dir}/MATH-{i}.jsonl',
            **generation_kwargs,
        )

        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(batch_input_ids, dict):
            batch_input_ids = batch_input_ids['llama']

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations
        print(batch_generations)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

def load_base_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    system_prompt: str = None,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.basemodel import BaseModel

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    model = BaseModel(
        base_model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer

def load_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    convert_to_half=False,
    use_fast_tokenizer=True,
    padding_side="left",
):

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if convert_to_half:
        model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    return model, tokenizer


def add_pad_token(tokenizer, padding_side="left"):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer

def load_logit_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    device_map: str = "auto",
    system_prompt: str = None,
    chat_response_prefix: str = None,
    reasoning_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.logitmodel import LogitsModel

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
    }

    tokenizer = AutoTokenizer.from_pretrained(expert_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    model = LogitsModel(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        chat_response_prefix=chat_response_prefix,
        reasoning_response_prefix=reasoning_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def load_dexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: str = None,
    expert_model1: str = None,
    antiexpert_model1: str = None,
    device_map: str = "auto",
    system_prompt: str = None,
    alpha: float = 0.5,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.dexperts import DExpertsLlama

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
    }

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)
    if not antiexpert_model_name_or_path:
        antiexpert_model_name_or_path = 'meta-llama/Llama-2-7b-hf'
    
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    expert_tokenizer = AutoTokenizer.from_pretrained(expert_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    
    # inverse_base_vocab = {v: k for k, v in base_tokenizer.vocab.items()}
    # inverse_expert_vocab = {v: k for k, v in expert_tokenizer.vocab.items()}
    # for i in range(151646):
    #     assert inverse_base_vocab[i] == inverse_expert_vocab[i], f"vocab mismatch at {i}: {inverse_base_vocab[i]} vs {inverse_expert_vocab[i]}"
        
    # input()

    print("loading base model from ", base_model_name_or_path)
    print("loading expert model from ", expert_model_name_or_path)
    print("loading antiexpert model from ", antiexpert_model_name_or_path)
    print("loading expert1 from ", expert_model1)
    print("loading antiexpert model 1 from ", antiexpert_model1)

    model = DExpertsLlama(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        antiexpert_model_name_or_path=antiexpert_model_name_or_path,
        expert_model1=expert_model1,
        antiexpert_model1=antiexpert_model1,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function


def compute_entropy(logits):
    """
    Calculate the entropy H(T_{i,j}) using tensor operations.
    
    Args:
        logits (Tensor): Logits for the expert model.
    
    Returns:
        tuple: entropy of model probabilities
    """

    probs = F.softmax(logits, dim=-1).squeeze(0)

    # Compute entropy using tensor operations
    entropy_exp = -torch.sum(probs * torch.log2(probs + 1e-6))

    return entropy_exp.item()


def log_decoding_step(
    step: int,
    tokenizer,
    next_tokens: torch.Tensor,
    base_logits_raw: torch.Tensor,
    expert_logits_raw: torch.Tensor,
    base_logits_warped: torch.Tensor,
    expert_logits_warped: torch.Tensor,
    cd_logits: torch.Tensor,
    save_logit_dir: str,
    cur_sample: str,
    threshold_pos: float = 0.1,
    threshold_close: float = 0.0001,
    top_k: int = 20,
):
    assert next_tokens.shape[0] == 1, "Only single-batch supported for now"

    def softmax_probs(logits): return F.softmax(logits, dim=-1)
    def entropy(p): return -torch.sum(p * torch.log2(p + 1e-6)).item()

    # --- Softmax ---
    base_probs_raw = softmax_probs(base_logits_raw)
    expert_probs_raw = softmax_probs(expert_logits_raw)
    base_probs_filt = softmax_probs(base_logits_warped)
    expert_probs_filt = softmax_probs(expert_logits_warped)
    cd_probs = softmax_probs(cd_logits)

    # --- Entropies ---
    log_entry = {
        'decoding_step': step,
        'next_token': tokenizer.decode(next_tokens[0]),
        'entropy_base_raw': entropy(base_probs_raw),
        'entropy_expert_raw': entropy(expert_probs_raw),
        'entropy_base_filtered': entropy(base_probs_filt),
        'entropy_expert_filtered': entropy(expert_probs_filt),
        'entropy_cd': entropy(cd_probs),
    }

    # --- Top-k Tokens + Prob Diff (Raw) ---
    top_vals, top_idxs = torch.topk(cd_probs, top_k)
    top_tokens = [tokenizer.decode(idx) for idx in top_idxs[0]]
    base_probs_top = base_probs_raw[0, top_idxs[0]].tolist()
    expert_probs_top = expert_probs_raw[0, top_idxs[0]].tolist()
    prob_diff_top = [e - b for e, b in zip(expert_probs_top, base_probs_top)]

    log_entry.update({
        'top_k_tokens': top_tokens,
        'top_k_cd_probs': top_vals[0].tolist(),
        'base_probs_top': base_probs_top,
        'expert_probs_top': expert_probs_top,
        'prob_diff_top': prob_diff_top,
    })

    # --- Top-k Overlap (Raw) ---
    top_k_expert = torch.topk(expert_probs_raw, top_k).indices[0].tolist()
    top_k_base = torch.topk(base_probs_raw, top_k).indices[0].tolist()
    top_k_overlap = list(set(top_k_expert).intersection(top_k_base))
    log_entry['top_k_overlap_raw'] = top_k_overlap

    # --- Contrastive Categories (Raw + Filtered) ---
    def get_token_diff_dict(probs_exp, probs_base, label):
        diff = probs_exp - probs_base
        result = {
            f'{label}_expert_greater_than_base': {},
            f'{label}_base_greater_than_expert': {},
            f'{label}_expert_close_to_base': {},
        }
        mask_exp = (diff > threshold_pos)[0]
        mask_base = (diff < -threshold_pos)[0]
        mask_close = (diff.abs() < threshold_close)[0] & (probs_exp[0] > 0) & (probs_base[0] > 0)

        for name, mask in [
            ('expert_greater_than_base', mask_exp),
            ('base_greater_than_expert', mask_base),
            ('expert_close_to_base', mask_close),
        ]:
            indices = torch.nonzero(mask, as_tuple=False).squeeze()
            if indices.numel() == 0:
                continue
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
            for idx in indices:
                token = tokenizer.decode(idx)
                result[f'{label}_{name}'][token] = diff[0, idx].item()
        return result

    # Raw categories
    log_entry.update(get_token_diff_dict(expert_probs_raw, base_probs_raw, label='raw'))

    # Filtered categories
    log_entry.update(get_token_diff_dict(expert_probs_filt, base_probs_filt, label='filtered'))

    wandb_log = {
        f'{cur_sample}/step': step,
        f'{cur_sample}/entropy/expert_raw': log_entry['entropy_expert_raw'],
        f'{cur_sample}/entropy/base_raw': log_entry['entropy_base_raw'],
        f'{cur_sample}/entropy/contrastive': log_entry['entropy_cd'],
        f'{cur_sample}/overlap/top_k_raw': log_entry['top_k_overlap_raw'],
    }
    wandb.log(wandb_log)

    # --- Save to file ---
    with open(save_logit_dir, 'a+', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
