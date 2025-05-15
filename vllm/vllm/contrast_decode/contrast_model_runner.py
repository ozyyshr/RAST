from typing import Optional, List

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.sequence import SequenceGroupMetadata
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)

class ContrastModelRunner(ModelRunner):
    """
    Specialized model runner for contrastive decoding.
    """

    def __init__(self,
                 model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 scheduler_config: SchedulerConfig,
                 device_config: DeviceConfig,
                 cache_config: CacheConfig,
                 load_config: LoadConfig,
                 lora_config: Optional[LoRAConfig],
                 kv_cache_dtype: Optional[str] = "auto",
                 is_driver_worker: bool = False,
                 prompt_adapter_config: Optional[PromptAdapterConfig] = None,
                 return_logits: bool = False,
                 observability_config: Optional[ObservabilityConfig] = None):
        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            return_logits=return_logits,
            observability_config=observability_config,
        )

    # TODO: Skip the sampling process as we will sample in the contrastive decode worker.