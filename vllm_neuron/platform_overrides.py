# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Optional

from vllm.config import ParallelConfig
from vllm.entrypoints.openai.serving_engine import (AnyRequest, OpenAIServing,
                                                    TextTokensPrompt)
from vllm.entrypoints.renderer import CompletionRenderer
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Skip attention head divisibility check
def skip_verify_with_parallel_config(
    self,
    parallel_config: "ParallelConfig",
) -> None:
    logger.info(
        "Skipping attention head divisibility check for Neuron platform")
    if parallel_config.distributed_executor_backend == "external_launcher":
        assert self.seed is not None, (
            "Seed must be set when using external launcher backend to "
            "make sure sampling results are the same across workers.")

    if parallel_config.enable_expert_parallel:
        self._verify_with_expert_parallelism()

    pipeline_parallel_size = parallel_config.pipeline_parallel_size
    if pipeline_parallel_size > 1:
        if not self.registry.is_pp_supported_model(self.architectures):
            raise NotImplementedError(
                "Pipeline parallelism is not supported for this model. "
                "Supported models implement the `SupportsPP` interface.")

        if self.use_async_output_proc:
            self.use_async_output_proc = False


def skip_verify_quantization(self):
    pass


def skip_verify_cuda_graph(self):
    pass


def changed_get_and_verify_max_len(self, max_model_len: int):
    # NOTE: Don't use HF config values like sliding_window
    # to impact max_model_len validation when on Neuron.
    if self.spec_target_max_model_len is not None:
        return self.spec_target_max_model_len
    return max_model_len


# OpenAI API Server is the most popular at the moment.
# The below is used for chat/completions
original_validate_input = OpenAIServing._validate_input


def changed_validate_input(  # pragma: no cover
    self,
    request: AnyRequest,
    input_ids: list[int],
    input_text: str,
) -> TextTokensPrompt:
    if self.model_config.max_prompt_length is not None:
        token_num = len(input_ids)
        if token_num > self.model_config.max_prompt_length:
            raise ValueError(
                f"This model's maximum prompt length is "
                f"{self.model_config.max_prompt_length} tokens. However, "
                f"your request has {token_num} prompt tokens. Please reduce "
                "the length of the prompt.")

    return original_validate_input(self, request, input_ids, input_text)


original_create_tokens_prompt = CompletionRenderer._create_tokens_prompt


def changed_create_tokens_prompt(  # pragma: no cover
    self,
    token_ids: list[int],
    max_length: Optional[int] = None,
    cache_salt: Optional[str] = None,
    prompt: Optional[str] = None,
) -> EngineTokensPrompt:
    if self.model_config.max_prompt_length is not None:
        if len(token_ids) > self.model_config.max_prompt_length:
            raise ValueError(
                f"This model's maximum prompt length is {self.model_config.max_prompt_length} tokens. "
                f"However, your request has {len(token_ids)} prompt tokens. "
                "Please reduce the length of the prompt.")

    return original_create_tokens_prompt(self, token_ids, max_length,
                                         cache_salt, prompt)
