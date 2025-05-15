from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from utils import top_k_top_p_filtering
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import numpy as np
import json


B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
class BaseModel:
    def __init__(
        self,
        base_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )

        self.base.eval()

        self.tokenizer = tokenizer
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = True if 'instruct' in base_model_name_or_path.lower() else False

        if self.use_chat_format_for_expert:
            # chat_prefix goes before the query, and chat_suffix goes after it
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"

            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

    def forward(
        self,
        base_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)

        return base_outputs

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
        save_logit_dir=None,
        **kwargs
    ):
        base_kwargs = kwargs.copy()

        # prepare inputs for base model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            input_ids = chat_inputs.input_ids.to(input_ids.device)
            base_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)

            # generate starts!
            base_outputs = self.forward(base_inputs, return_dict=True)

            base_next_token_logits = base_outputs.logits[..., -1, :] # (batch_size, vocab_size)

            base_probs = F.softmax(base_next_token_logits, dim=-1)
            # top_50_values_base, top_50_indices_base = torch.topk(base_probs, 50)
            # top_50_tokens_base = [self.tokenizer.decode(idx) for idx in top_50_indices_base.tolist()[0]]

            # ### write distributions
            # with open(save_logit_dir, 'a+') as f:
            #     w_dict = {
            #         'decoding_step': step,
            #         'tokens': top_50_tokens_base,
            #         'probabilities': top_50_values_base.tolist()[0]
            #     }
            #     line = json.dumps(w_dict, ensure_ascii=False)
            #     f.write(line+'\n') 

            # pre-process logits
            if logits_processor:
                base_next_token_logits = logits_processor(input_ids, base_next_token_logits)

            # warp logits
            if temperature != 1.0:
                base_next_token_logits = base_next_token_logits / temperature
            
            if top_p < 1.0:
                base_next_token_logits = top_k_top_p_filtering(base_next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                base_probs = F.softmax(base_next_token_logits, dim=-1)
                base_next_tokens = torch.multinomial(base_probs, num_samples=1).squeeze(1)
            else:
                base_next_tokens = torch.argmax(base_next_token_logits, dim=-1)

            base_next_tokens = (
                base_next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'base': base_next_token_logits,
                }
                analysis_data = self.update_analysis_data(analysis_data, base_next_tokens, next_token_logits_dict)

            # update model inputs for next step
            input_ids = torch.cat([input_ids, base_next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                base_next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
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
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs