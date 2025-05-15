from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import json
import torch.nn.functional as F
from vllm.config import (
    ContrastiveDecodingConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler,
    SpecDecodeStochasticBaseSampler,
)
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler,
)
from vllm.sequence import (
    VLLM_INVALID_TOKEN_ID,
    CompletionSequenceGroupOutput,
    ExecuteModelRequest,
    HiddenStates,
    SequenceGroupMetadata,
    get_all_seq_ids_and_request_ids,
)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.contrast_decode.contrast_model_runner import ContrastModelRunner

from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


def create_contrastive_worker(*args, **kwargs) -> "ContrastiveDecodeWorker":
    assert "contrastive_decoding_config" in kwargs
    contrastive_decoding_config: ContrastiveDecodingConfig = kwargs.get(
        "contrastive_decoding_config"
    )
    assert contrastive_decoding_config is not None

    contrastive_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = ContrastModelRunner
    base_worker = Worker(*args, **kwargs)

    contrastive_worker_kwargs.update(
        parallel_config=contrastive_decoding_config.parallel_config,
    )

    contrastive_decode_worker = ContrastiveDecodeWorker.create_worker(
        base_worker=base_worker,
        worker_kwargs=contrastive_worker_kwargs,
        positive_model_config=contrastive_decoding_config.positive_model_config,
        negative_model_config=contrastive_decoding_config.negative_model_config,
        positive_model_config1=contrastive_decoding_config.positive_model_config1,
        negative_model_config1=contrastive_decoding_config.negative_model_config1,
        positive_model_config2=contrastive_decoding_config.positive_model_config2,
        negative_model_config2=contrastive_decoding_config.negative_model_config2,
        sampler_alpha=contrastive_decoding_config.sampler_alpha,
    )

    return contrastive_decode_worker


class ContrastiveDecodeWorker(LoraNotSupportedWorkerBase):

    @classmethod
    def create_worker(
        cls,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_model_config: Optional[ModelConfig],
        negative_model_config: Optional[ModelConfig],
        positive_model_config1: Optional[ModelConfig],
        negative_model_config1: Optional[ModelConfig],
        positive_model_config2: Optional[ModelConfig],
        negative_model_config2: Optional[ModelConfig],
        sampler_alpha: float,
    ) -> "ContrastiveDecodeWorker":
        """
        Create a ContrastiveDecodeWorker from the given arguments.
        """
        assert (
            positive_model_config is not None or negative_model_config is not None
        ), "Either positive_model_config or negative_model_config must be specified."

        if positive_model_config is None:
            positive_worker = None
        else:
            positive_worker_kwargs = worker_kwargs.copy()
            positive_worker_kwargs.update(
                model_config=positive_model_config,
            )
            positive_worker = MultiStepWorker(**positive_worker_kwargs)

        if negative_model_config is None:
            negative_worker = None
        else:
            negative_worker_kwargs = worker_kwargs.copy()
            negative_worker_kwargs.update(
                model_config=negative_model_config,
            )
            negative_worker = MultiStepWorker(**negative_worker_kwargs)
        
        if positive_model_config1 is None:
            positive_worker1 = None
        else:
            positive_worker_kwargs1 = worker_kwargs.copy()
            positive_worker_kwargs1.update(
                model_config=positive_model_config1,
            )
            positive_worker1 = MultiStepWorker(**positive_worker_kwargs1)
        
        if negative_model_config1 is None:
            negative_worker1 = None
        else:
            negative_worker_kwargs1 = worker_kwargs.copy()
            negative_worker_kwargs1.update(
                model_config=negative_model_config1,
            )
            negative_worker1 = MultiStepWorker(**negative_worker_kwargs1)
        
        if positive_model_config2 is None:
            positive_worker2 = None
        else:
            positive_worker_kwargs2 = worker_kwargs.copy()
            positive_worker_kwargs2.update(
                model_config=positive_model_config2,
            )
            positive_worker2 = MultiStepWorker(**positive_worker_kwargs2)
        
        if negative_model_config2 is None:
            negative_worker2 = None
        else:
            negative_worker_kwargs2 = worker_kwargs.copy()
            negative_worker_kwargs2.update(
                model_config=negative_model_config2,
            )
            negative_worker2 = MultiStepWorker(**negative_worker_kwargs2)

        # decode_sampler = ContrastiveSampler(
        #     alpha=sampler_alpha,
        # )

        return cls(
            base_worker=base_worker,
            worker_kwargs=worker_kwargs,
            positive_worker=positive_worker,
            negative_worker=negative_worker,
            positive_worker1=positive_worker1,
            negative_worker1=negative_worker1,
            positive_worker2=positive_worker2,
            negative_worker2=negative_worker2,
            sampler_alpha=sampler_alpha,
            # decode_sampler=decode_sampler,
        )

    def __init__(
        self,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_worker: Optional[WorkerBase],
        negative_worker: Optional[WorkerBase],
        positive_worker1: Optional[WorkerBase],
        negative_worker1: Optional[WorkerBase],
        positive_worker2: Optional[WorkerBase],
        negative_worker2: Optional[WorkerBase],
        sampler_alpha: float,
        # decode_sampler: ContrastiveSamplerBase,
    ):
        self.base_worker = base_worker
        self.worker_kwargs = worker_kwargs
        self.positive_worker = positive_worker
        self.negative_worker = negative_worker
        self.positive_worker1 = positive_worker1
        self.negative_worker1 = negative_worker1
        self.positive_worker2 = positive_worker2
        self.negative_worker2 = negative_worker2
        # self.decode_sampler = decode_sampler
        self.sampler_alpha = sampler_alpha
        self.sampler = Sampler()

    def init_device(self) -> None:
        self.base_worker.init_device()
        if self.positive_worker is not None:
            self.positive_worker.init_device()
        if self.negative_worker is not None:
            self.negative_worker.init_device()
        if self.positive_worker1 is not None:
            self.positive_worker1.init_device()
        if self.negative_worker1 is not None:
            self.negative_worker1.init_device()
        if self.positive_worker2 is not None:
            self.positive_worker2.init_device()

        self.base_worker.load_model()
        if self.positive_worker is not None:
            self.positive_worker.load_model()
        if self.negative_worker is not None:
            self.negative_worker.load_model()
        if self.positive_worker1 is not None:
            self.positive_worker1.load_model()
        if self.negative_worker1 is not None:
            self.negative_worker1.load_model()
        if self.positive_worker2 is not None:
            self.positive_worker2.load_model()
        if self.negative_worker2 is not None:
            self.negative_worker2.load_model()

        # self._metrics.init_gpu_tensors(self.rank)
        # self.decode_sampler.init_gpu_tensors(self.rank)

    def load_model(self, *args, **kwargs):
        pass

    def get_cache_block_size_bytes(self) -> int:
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.

        This is done by profiling the base model (which is typically the
        larger of the two). Then the total memory which would be used by the
        base model KV is divided evenly between the positive and negative model KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = self.base_worker.determine_num_available_blocks()
        positive_num_gpu_blocks = 0
        negative_num_gpu_blocks = 0
        positive_num_gpu_blocks1 = 0
        negative_num_gpu_blocks1 = 0
        positive_num_gpu_blocks2 = 0
        negative_num_gpu_blocks2 = 0

        if self.positive_worker is not None:
            positive_num_gpu_blocks, positive_num_cpu_blocks = self.positive_worker.determine_num_available_blocks()
        if self.negative_worker is not None:
            negative_num_gpu_blocks, negative_num_cpu_blocks = self.negative_worker.determine_num_available_blocks()

        if self.positive_worker1 is not None:
            positive_num_gpu_blocks1, positive_num_cpu_blocks1 = self.positive_worker1.determine_num_available_blocks()
        if self.negative_worker1 is not None:
            negative_num_gpu_blocks1, negative_num_cpu_blocks1 = self.negative_worker1.determine_num_available_blocks()

        if self.positive_worker2 is not None:
            positive_num_gpu_blocks2, positive_num_cpu_blocks2 = self.positive_worker2.determine_num_available_blocks()
        if self.negative_worker2 is not None:
            negative_num_gpu_blocks2, negative_num_cpu_blocks2 = self.negative_worker2.determine_num_available_blocks()

        logger.info(f"Num GPU blocks: {num_gpu_blocks}, {positive_num_gpu_blocks}, {negative_num_gpu_blocks}, {positive_num_gpu_blocks1}, {negative_num_gpu_blocks1}, {positive_num_gpu_blocks2}, {negative_num_gpu_blocks2}")
        divident = 3
        if self.positive_worker1 is not None:
            divident += 2
        if self.positive_worker2 is not None:
            divident += 2
        return max(num_gpu_blocks, positive_num_gpu_blocks, negative_num_gpu_blocks, positive_num_gpu_blocks1, negative_num_gpu_blocks1, positive_num_gpu_blocks2, negative_num_gpu_blocks2) // divident, num_cpu_blocks
    
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache engine of the scorer and proposer workers.
        """
        self.base_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                            num_cpu_blocks=num_cpu_blocks)
        if self.positive_worker is not None:
            self.positive_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        if self.negative_worker is not None:
            self.negative_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        if self.positive_worker1 is not None:
            self.positive_worker1.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        if self.negative_worker1 is not None:
            self.negative_worker1.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        if self.positive_worker2 is not None:
            self.positive_worker2.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        if self.negative_worker2 is not None:
            self.negative_worker2.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
    
    @torch.inference_mode()
    def execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform contrastive decoding on the input batch."""
        # print(f"Running current rank {self.rank}, driver rank {self._driver_rank}: {str(execute_model_req)[:10]}")

        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []
        
        if execute_model_req is None:
            """
            This signals that there's no more requests to process for now.
            All workers are running infinite loop with broadcast_tensor_dict,
            and it stops the loop when the driver broadcasts an empty input.
            Send an empty input to notify all other workers to stop their
            execution loop.
            """
            broadcast_tensor_dict({}, src=0)
            return []
        
        # print(f"#2 Running current rank {self.rank}, driver rank {self._driver_rank}")
        disable_all_contrastive_decoding = (
            self._should_disable_all_contrastive_decoding(execute_model_req)
        )
        boardcast_dict = dict(
            disable_all_contrastive_decoding=disable_all_contrastive_decoding,
        )
        # print(f"#3 Running current rank {self.rank}, driver rank {self._driver_rank}") 
        broadcast_tensor_dict(boardcast_dict, src=self._driver_rank)
        # print(f"#4 Running current rank {self.rank}, driver rank {self._driver_rank}")

        if disable_all_contrastive_decoding:
            return self._run_no_contrastive_decoding(execute_model_req)
        # print(f"#5 Running current rank {self.rank}, driver rank {self._driver_rank}")
        return self._run_contrastive_decoding(execute_model_req)

    def _should_disable_all_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> bool:
        """
        Determine if all contrastive decoding should be disabled.
        """
        # TODO: Implement this
        return False
    
    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop to perform speculative decoding
        in parallel worker."""
        while self._run_non_driver_rank():
            pass
    
    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True if there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank
        data = broadcast_tensor_dict(src=self._driver_rank)
        # print(f"Running non-driver rank {self.rank} driver rank {self._driver_rank} data: {data}")
        if not data:
            return False

        self.base_worker.execute_model()

        if self.positive_worker is not None:
            self.positive_worker.execute_model()

        if self.negative_worker is not None:
            self.negative_worker.execute_model()
        
        if self.positive_worker1 is not None:
            self.positive_worker1.execute_model()
        
        if self.negative_worker1 is not None:
            self.negative_worker1.execute_model()

        if self.positive_worker2 is not None:
            self.positive_worker2.execute_model()

        if self.negative_worker2 is not None:
            self.negative_worker2.execute_model()

        return True

    def _run_no_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model without contrastive decoding.
        """
        sampler_output = self.base_worker.execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        sampler_output.sampled_token_ids = None
        sampler_output.sampled_token_probs = None
        sampler_output.logprobs = None
        sampler_output.logits = None
        return [sampler_output]

    def _run_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model with contrastive decoding.
        """
        # print(f"#6 Running current rank {self.rank}, driver rank {self._driver_rank}")
        base_sampler_output = self.base_worker.execute_model(execute_model_req)
        # print(f"#7 Running current rank {self.rank}, driver rank {self._driver_rank}")
        if self.positive_worker is not None:
            positive_sampler_output = self.positive_worker.execute_model(execute_model_req)
            # print(f"#8 Running current rank {self.rank}, driver rank {self._driver_rank}")
        else:
            positive_sampler_output = []
        if self.negative_worker is not None:
            negative_sampler_output = self.negative_worker.execute_model(execute_model_req)
            # print(f"#9 Running current rank {self.rank}, driver rank {self._driver_rank}")
        else:
            negative_sampler_output = []
        
        if self.positive_worker1 is not None:
            positive_sampler_output1 = self.positive_worker1.execute_model(execute_model_req)
            # print(f"#10 Running current rank {self.rank}, driver rank {self._driver_rank}")
        else:
            positive_sampler_output1 = []

        if self.negative_worker1 is not None:
            negative_sampler_output1 = self.negative_worker1.execute_model(execute_model_req)
            # print(f"#11 Running current rank {self.rank}, driver rank {self._driver_rank}")
        else:
            negative_sampler_output1 = []
        
        if self.positive_worker2 is not None:
            positive_sampler_output2 = self.positive_worker2.execute_model(execute_model_req)
            # print(f"#12 Running current rank {self.rank}, driver rank {self._driver_rank}")
        else:
            positive_sampler_output2 = []

        if self.negative_worker2 is not None:  
            negative_sampler_output2 = self.negative_worker2.execute_model(execute_model_req)
            # print(f"#13 Running current rank {self.rank}, driver rank {self._driver_rank}")
        else:
            negative_sampler_output2 = []

        generators = self.base_worker.model_runner.get_generators(
            execute_model_req.finished_requests_ids)
        
        input_tokens_tensor, seq_lens, query_lens = self._prepare_input_tensors(
            execute_model_req.seq_group_metadata_list,
        )

        sampling_metadata = SamplingMetadata.prepare(
            execute_model_req.seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            self.base_worker.model_runner.pin_memory,
            generators,
        )

        contrastive_sampler_output = self._create_contrastive_sampler_output(
            sampling_metadata,
            base_sampler_output,
            positive_sampler_output,
            negative_sampler_output,
            positive_sampler_output1,
            negative_sampler_output1,
            positive_sampler_output2,
            negative_sampler_output2,
        )
        return contrastive_sampler_output

    def _create_contrastive_sampler_output(
        self,
        sampling_metadata: SamplingMetadata,
        base_sampler_output: List[SamplerOutput],
        positive_sampler_output: List[SamplerOutput],
        negative_sampler_output: List[SamplerOutput],
        positive_sampler_output1: List[SamplerOutput],
        negative_sampler_output1: List[SamplerOutput],
        positive_sampler_output2: List[SamplerOutput],
        negative_sampler_output2: List[SamplerOutput],
    ) -> List[SamplerOutput]:
        """
        Create a contrastive sampler output.
        """
        # Sample the next token.
        logits = base_sampler_output[0].logits
        # Align different logits shapes caused by tokenizer
        if len(positive_sampler_output) > 0:
            positive_logits = positive_sampler_output[0].logits
            negative_logits = negative_sampler_output[0].logits
        
        if len(positive_sampler_output1) > 0:
            positive_logits1 = positive_sampler_output1[0].logits
            negative_logits1 = negative_sampler_output1[0].logits

        if len(positive_sampler_output2) > 0:
            positive_logits2 = positive_sampler_output2[0].logits
            negative_logits2 = negative_sampler_output2[0].logits

        if len(positive_sampler_output1) > 0 and len(positive_sampler_output2) > 0:
            smallest_shape = min(logits.shape[-1], positive_logits.shape[-1], positive_logits1.shape[-1], positive_logits2.shape[-1])
        elif len(positive_sampler_output1) > 0:
            smallest_shape = min(logits.shape[-1], positive_logits.shape[-1], positive_logits1.shape[-1])
        else:
            smallest_shape = min(logits.shape[-1], positive_logits.shape[-1])

        if logits.shape[-1] != smallest_shape:
            flag = True
            logits = logits[:,:smallest_shape]
        else:
            flag = False
        
        positive_logits = positive_logits[:,:smallest_shape]
        negative_logits = negative_logits[:,:smallest_shape]
        if len(positive_sampler_output1) > 0:
            positive_logits1 = positive_logits1[:,:smallest_shape]
            negative_logits1 = negative_logits1[:,:smallest_shape]
        if len(positive_sampler_output2) > 0:
            positive_logits2 = positive_logits2[:,:smallest_shape]
            negative_logits2 = negative_logits2[:,:smallest_shape]
            # logits = logits[:,:positive_logits.shape[-1]]
            # if len(positive_sampler_output1) > 0 and positive_logits1.shape[-1] < logits.shape[-1]:
            #     positive_logits1 = torch.cat([positive_logits1, torch.full((positive_logits1.shape[0], 128), float('-inf'), device=self.device)], dim=-1)
            #     negative_logits1 = torch.cat([negative_logits1, torch.full((negative_logits1.shape[0], 128), float('-inf'), device=self.device)], dim=-1)

        # base_logits = logits.clone()

        if self.positive_worker and self.positive_worker1 and self.positive_worker2:
            next_token_logits = (
                logits + 
                self.sampler_alpha * (positive_logits - negative_logits) +
                self.sampler_alpha * (positive_logits1 - negative_logits1) +
                self.sampler_alpha * (positive_logits2 - negative_logits2)
            )

        elif self.positive_worker1 and self.positive_worker:
            next_token_logits = (
                logits + 
                0.4 * (positive_logits - negative_logits) +
                0.1 * (positive_logits1 - negative_logits1)
            )
        
        else:
            next_token_logits = (
                logits + 
                self.sampler_alpha * (positive_logits - negative_logits)
            )

        #     logits = logits + self.sampler_alpha * positive_logits1
        # if self.negative_worker1:
        #     logits = logits - self.sampler_alpha * negative_logits1

        # if self.positive_worker2:
        #     logits = logits + self.sampler_alpha * positive_logits2
        # if self.negative_worker2:
        #     logits = logits - self.sampler_alpha * negative_logits2

        # logits = logits / (1 + self.sampler_alpha)

        if flag:
            # pad float('-inf') to the logits
            next_token_logits = torch.cat([next_token_logits, torch.full((next_token_logits.shape[0], 128), float('-inf'), device=self.device)], dim=-1)

        output: SamplerOutput = self.base_worker.model_runner.model.sample(
            logits=next_token_logits,
            sampling_metadata=sampling_metadata,
        )
        # selected_token_id = output.outputs[0].samples[0].output_token
        # delta_logit = self.sampler_alpha * (positive_logits - negative_logits)[:, selected_token_id].item()
        # base_logit = base_logits[:, selected_token_id].item()

        # with open("/shared/data3/siruo2/ContrastiveReasoning/analysis/logit_distribution.jsonl", "a") as f:
        #     data_item = {
        #         "token_id": selected_token_id,
        #         "logit": logits.tolist(),
        #     }
        #     f.write(json.dumps(data_item) + "\n")
        return [output]

    def _prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        if not seq_group_metadata_list:
            return torch.empty(0, device=self.device), [], []

        input_tokens: List[int] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            for seq_data in seq_group_metadata.seq_data.values():
                seq_data_len = seq_data.get_len()
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                    seq_len = min(
                        seq_data_len,
                        context_len + seq_group_metadata.token_chunk_size)
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                    seq_lens.append(seq_len)
                    input_tokens.extend(tokens)
                    query_lens.append(seq_len - context_len)
                else:
                    seq_lens.append(seq_data_len)
                    input_tokens.append(seq_data.get_last_token_id())
                    query_lens.append(1)

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        return input_tokens_tensor, seq_lens, query_lens
    
    @cached_property
    def vocab_size(self) -> int:
        return self.base_worker.vocab_size

    @property
    def rank(self) -> int:
        return self.base_worker.rank

    @property
    def device(self) -> torch.device:
        return self.base_worker.device

    @property
    def _driver_rank(self) -> int:
        return 0
