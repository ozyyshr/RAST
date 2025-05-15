from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from text_generation.generation_logit_process import *
from utils import top_k_top_p_filtering, compute_entropy, log_decoding_step
from collections import defaultdict
import time
import json
import wandb

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class EntrPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, relative_top: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        relative_top = float(relative_top)
        if relative_top < 0 or relative_top > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {relative_top}")

        self.relative_top = relative_top
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[:, self.min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(self.relative_top)
        # print(min_thresh, probs_thresh)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = self.filter_value
        return scores_normalized
    
    
class LogitsModel:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 0.5,
        chat_response_prefix: str = None,
        reasoning_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        **chat_response_prefix:** For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        **reasoning_response_prefix:** For deepseek-distilled models, it can be helpful for the response
        to start with a certain prefix <think> to avoid vocab mismatch issue.
        """

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path, **model_kwargs
        )

        self.base.eval()
        self.expert.eval()

        self.tokenizer = tokenizer
        self.beta = alpha
        self.truncate_ratio = 0.1
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix
        self.reasoning_response_prefix = reasoning_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = True if 'base' in base_model_name_or_path.lower() else False
        # Deepseek experts need different formatting
        self.use_reasoning_format_for_expert = True if 'reasoning' in expert_model_name_or_path.lower() else False

        if self.use_chat_format_for_expert:
            # chat_prefix goes before the query, and chat_suffix goes after it
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"
            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

        if self.use_reasoning_format_for_expert:
            # prompt end with reasoning response prefix
            if self.reasoning_response_prefix:
                self.reasoning_suffix = f" {reasoning_response_prefix}"

    def forward(
        self,
        base_inputs,
        expert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs

    def _get_tokenized_reasoning_inputs(self, input_ids):
        """Decode input_ids and encode again to insert reasoning formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "<think>") from the prompt if it's already there
        if self.reasoning_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.reasoning_response_prefix in p:
                    p = p.replace(self.reasoning_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        reasoning_prompts = [f'{p} {self.reasoning_suffix}' for p in cleaned_prompts]
        print('Deepseek reasoning model prompt', flush=True)
        print(reasoning_prompts[0], flush=True)
        reasoning_inputs = self.tokenizer(
            reasoning_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        reasoning_inputs.input_ids = reasoning_inputs.input_ids.to(self.device)
        reasoning_inputs.attention_mask = reasoning_inputs.attention_mask.to(self.device)

        return reasoning_inputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to insert chat formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "Answer:") from the prompt if it's already there
        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        print('Chat model prompt', flush=True)
        print(chat_prompts[0], flush=True)
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        save_logit_dir = None,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()

        # prepare inputs for base model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            base_input_ids = chat_inputs.input_ids.to(input_ids.device)
            base_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            base_input_ids = input_ids.to(input_ids.device)
        
        # prepare inputs for reasoning model
        if self.use_reasoning_format_for_expert:
            reasoning_inputs = self._get_tokenized_reasoning_inputs(input_ids)
            input_ids = reasoning_inputs.input_ids.to(input_ids.device)
            expert_kwargs['attention_mask'] = reasoning_inputs.attention_mask
        else:
            input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)
        
        # prepare logit warper for teacher and student
        logits_warper = EntrPLogitsWarper(relative_top=self.truncate_ratio)
        logits_warper_student = TemperatureLogitsWarper(temperature=0.6)
        logits_warper_expert_temp = TemperatureLogitsWarper(temperature=0.6)

        cur_sample = save_logit_dir.split('/')[-1].split('.')[0]

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(base_input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(input_ids, **expert_kwargs)

            # forward pass to get next token logits
            base_outputs, expert_outputs = self.forward(
                base_inputs, expert_inputs, return_dict=True
            )

            # warp logits
            ## filtering in expert model
            base_next_token_logits_ori = base_outputs.logits[..., -1, :] # (batch_size, vocab_size)
            base_next_token_logits = logits_warper_student(input_ids, base_next_token_logits_ori)
            expert_next_token_logits_ori = expert_outputs.logits[..., -1, :]
            # expert_next_token_logits_temp = logits_warper_expert_temp(input_ids, expert_next_token_logits_ori)
            expert_next_token_logits = logits_warper(input_ids, expert_next_token_logits_ori)

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]

            # Logit difference
            next_token_logits = (
                (1 + self.beta) * expert_next_token_logits - self.beta * base_next_token_logits
            )

            # # pre-process logits
            # if logits_processor:
            #     next_token_logits = logits_processor(input_ids, next_token_logits)

            # # warp logits
            # if temperature != 1.0:
            #     next_token_logits = next_token_logits / temperature

            # if top_p < 1.0:
            #     next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            log_decoding_step(
                step=step,
                tokenizer=self.tokenizer,
                next_tokens=next_tokens,
                base_logits_raw=base_next_token_logits_ori,
                expert_logits_raw=expert_next_token_logits_ori,
                base_logits_warped=base_next_token_logits,
                expert_logits_warped=expert_next_token_logits,
                cd_logits=next_token_logits,
                save_logit_dir=save_logit_dir,
                cur_sample=cur_sample,
            )

            # pad next tokens
            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'logit': next_token_logits,
                    'base': base_next_token_logits,
                    'expert': expert_next_token_logits,
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

            # update model inputs for next step
            ## skip special token <think> and </think> in base_input_ids
            # if next_tokens[0] != self.tokenizer.convert_tokens_to_ids('<think>') and next_tokens[0] != self.tokenizer.convert_tokens_to_ids('</think>'):
            #     base_input_ids = torch.cat([base_input_ids, next_tokens[:, None]], dim=-1)
            #     base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            # else:
            if next_tokens[0] == self.tokenizer.convert_tokens_to_ids('<think>') or next_tokens[0] == self.tokenizer.convert_tokens_to_ids('</think>'):
                base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs, 
                                                                       is_concat=False, 
                                                                       clear_kv_cache=True)
            elif next_tokens[0] != self.tokenizer.convert_tokens_to_ids('<think>') and next_tokens[0] != self.tokenizer.convert_tokens_to_ids('</think>'):
                base_input_ids = torch.cat([base_input_ids, next_tokens[:, None]], dim=-1)
                base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            else:
                raise ValueError('Invalid token')

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
        

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
        is_concat: bool = True, 
        clear_kv_cache: bool = False
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        if clear_kv_cache:
            kwargs["past_key_values"] = None

        # update attention mask
        if "attention_mask" in kwargs and is_concat:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs