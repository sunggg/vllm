"""Utilities for selecting and loading models."""
from typing import Type, Callable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import initialize_dummy_weights
from vllm.model_executor.layers.sampler import TvmSampler

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "RWForCausalLM": FalconForCausalLM,
}


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")

"""
def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    model = model_class(model_config.hf_config)
    if model_config.use_dummy_weights:
        model = model.cuda()
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                           model_config.use_np_weights)
        model = model.cuda()
    return model.eval()
"""


def get_model(model_config: ModelConfig) -> Callable:
    # TVM deps
    import tvm
    from tvm import relax
    from mlc_llm import utils

    dist_home = f"/home/ubuntu/mlc-llm/dist"
    dist = "llama-2-7b-chat-hf-q0f16"
    #dist = "llama-2-7b-chat-hf-q4f16_1"
    target = "cuda"
    artifact_home = dist_home + f"/{dist}/"
    lib_path = artifact_home + f"{dist}-{target}.so"
    tvm_ex = tvm.runtime.load_module(lib_path)
    tvm_device = tvm.device(target)
    vm = relax.VirtualMachine(tvm_ex, tvm_device)
    const_params = utils.load_params(artifact_home, tvm_device)

    class Model:
        def __init__(self) -> None:
            self.tot_seq_len = 0
            self.kv_cache = vm["create_kv_cache"]()
            self.sampler = TvmSampler(model_config.hf_config.vocab_size)

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches,  #: List[KVCache],
            input_metadata,  #: InputMetadata,
            cache_events,  #: Optional[List[torch.cuda.Event]],
        ) -> torch.Tensor:
            # Change input_ids to tvm format
            seqlen = torch.count_nonzero(input_ids)
            input_ids = input_ids[:seqlen]
            input_ids = torch.unsqueeze(input_ids, 0).to(torch.int32)
            input_ids = tvm.nd.from_dlpack(input_ids)

            self.tot_seq_len += input_ids.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([self.tot_seq_len])
            if input_ids.shape[1] > 1:
                logits, kv_cache = vm["prefill"](
                    input_ids, seq_len_shape, self.kv_cache, const_params
                )
            else:
                logits, kv_cache = vm["decode"](
                    input_ids, seq_len_shape, self.kv_cache, const_params
                )
            self.kv_cache = kv_cache

            logits = torch.from_dlpack(logits)
            logits = torch.squeeze(logits, 0)

            next_tokens = self.sampler(logits, input_metadata)

            return next_tokens

    model = Model()
    return model.forward
