# SPDX-License-Identifier: Apache-2.0
import logging
import os
from dataclasses import dataclass
from test.utils.fsx_utils.model_path import resolve_model_dir
from test.utils.lora_manager import LoRAAdapterManager

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["VLLM_USE_V1"] = "1"


@dataclass
class OfflineCfg:
    name: str
    model: str
    lora_ckpt_dict: dict
    max_loras: int
    max_cpu_loras: int = 0
    tp_degree: int = 32
    batch_size: int = 4
    max_model_len: int = 128
    override_neuron_config: dict = None
    enable_lora: bool = True
    random_adapter_ids: bool = True
    traced_models_dir: str = "/home/ubuntu/traced_models/"
    fsx: bool = False
    base_model_quantized: bool = False
    quantized_checkpoints_path: str = None


def multi_lora_offline_inference(
    model,
    lora_ckpt_dict,
    max_loras,
    traced_models_dir,
    max_cpu_loras=0,
    batch_size=2,
    tp_degree=32,
    max_model_len=1024,
    random_adapter_ids=False,
    enable_lora=True,
    dynamic_multi_lora=False,
    fsx=False,
    base_model_quantized=False,
):
    """
    We assume all LoRA adapter checkpoints are saved in a directory lora_adapters/
    and each LoRA checkpoint is a folder under lora_adapters/.
    Each LoRA folder has two files, adapter_config.json and adapter_model.safetensors,
    following the LoRA adapter checkpoint format in Hugging Face.
    """
    model_name = model.split("/")[-1]
    traced_model_path = os.path.join(traced_models_dir, f"{model_name}-multi-lora")
    os.environ["NEURON_COMPILED_ARTIFACTS"] = traced_model_path
    lora_adapter_manager = LoRAAdapterManager(lora_ckpt_dict)

    sampling_params = SamplingParams(max_tokens=128, top_k=1, temperature=0.0)
    override_neuron_config = {
        "skip_warmup": True,
        "lora_ckpt_json": lora_adapter_manager.get_lora_json_file(),
    }
    if base_model_quantized:
        quantization_config = {
            "quantized": True,
            "quantized_checkpoints_path": os.path.join(
                traced_model_path, "model_quant.pt"
            ),
            "quantization_type": "per_channel_symmetric",
            "enable-bucketing": True,
        }
        override_neuron_config.update(quantization_config)
    if fsx:
        model, _ = resolve_model_dir(model)
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_degree,
        max_num_seqs=batch_size,
        max_model_len=max_model_len,
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras,
        additional_config={"override_neuron_config": override_neuron_config},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    prompts = ["The president of the United States is"] * batch_size
    lora_reqs = []
    selected_adapter_ids = lora_adapter_manager.select_lora_adapter_ids(
        batch_size,
        random_adapter_ids=random_adapter_ids,
        dynamic_multi_lora=dynamic_multi_lora,
    )
    for i in range(batch_size):
        # lora_int_id and lora_path are not needed when using Neuron. Using placeholder values.
        lora_reqs.append(
            LoRARequest(f"{selected_adapter_ids[i]}", lora_int_id=1, lora_path=" ")
        )
        logging.info(
            f"Request {i} uses LoRA adapter {selected_adapter_ids[i]} for prompt: {prompts[i]!r}"
        )
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_reqs)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info(
            f"\n[Prompt {i + 1}]\n{prompt!r}\n[Generated]\n{generated_text!r}\n"
        )
        assert generated_text.strip(), (
            f"Output {i + 1} was empty for prompt: {prompt!r}"
        )


def multi_lora_offline_inference_config(config):
    logger.info(f"Running offline inference test with config: {config}")
    return multi_lora_offline_inference(
        model=config.model,
        lora_ckpt_dict=config.lora_ckpt_dict,
        max_loras=config.max_loras,
        max_cpu_loras=config.max_cpu_loras,
        batch_size=config.batch_size,
        tp_degree=config.tp_degree,
        max_model_len=config.max_model_len,
        random_adapter_ids=config.random_adapter_ids,
        enable_lora=config.enable_lora,
        traced_models_dir=config.traced_models_dir,
        fsx=config.fsx,
        base_model_quantized=config.base_model_quantized,
    )
