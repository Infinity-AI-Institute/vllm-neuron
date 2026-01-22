# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import random

from huggingface_hub import hf_hub_download


class LoRAAdapterManager:
    """
    This class manages LoRA adapters by downloading them from Hugging Face Hub,
    maintaining a mapping of adapter IDs to their checkpoint paths, and writing
    this information to a JSON file.
    args:
        lora_ckpt_dict (dict): A dictionary containing:
            - "lora-ckpt-dir": Directory to store LoRA adapters.
            - "lora-ckpt-paths": A dictionary mapping IDs of adapters in HBM to LoRA adapter model IDs from HuggingFace Hub.
            - "lora-ckpt-paths-cpu" (optional): A dictionary mapping IDs of adapters in CPU memory to LoRA adapter model IDs from HuggingFace Hub.
        lora_json_filename (str): The name of the JSON file to write the adapter information to.
    """

    def __init__(self, lora_ckpt_dict, lora_json_filename="adapters.json"):
        self.lora_ckpt_dict = lora_ckpt_dict
        self.lora_json_filename = lora_json_filename

        self.lora_adapters_dir = lora_ckpt_dict["lora_ckpt_dir"]
        self.lora_ckpt_paths = lora_ckpt_dict["lora_ckpt_paths"]
        self.lora_ckpt_paths_cpu = lora_ckpt_dict.get("lora_ckpt_paths_cpu", {})

        os.makedirs(self.lora_adapters_dir, exist_ok=True)
        self.lora_json_file = os.path.join(
            self.lora_adapters_dir, self.lora_json_filename
        )
        self.download_lora_adapters()

    def _download_lora_adapter(self, lora_repo_id):
        lora_adapter_dir = lora_repo_id.split("/")[-1]
        lora_path = os.path.join(self.lora_adapters_dir, lora_adapter_dir)
        if os.path.exists(lora_path):
            logging.info(
                f"LoRA adapter {lora_repo_id} already exists in {lora_path}. Skip downloading."
            )
            return lora_adapter_dir

        os.makedirs(lora_path, exist_ok=True)
        hf_hub_download(
            repo_id=lora_repo_id, filename="adapter_config.json", local_dir=lora_path
        )
        hf_hub_download(
            repo_id=lora_repo_id,
            filename="adapter_model.safetensors",
            local_dir=lora_path,
        )
        logging.info(f"LoRA adapter {lora_repo_id} is downloaded to {lora_path}.")
        return lora_adapter_dir

    def _download_lora_adapters(self, lora_ckpt_paths):
        for adapter_id, lora_repo_id in lora_ckpt_paths.items():
            lora_adapter_dir = self._download_lora_adapter(lora_repo_id)
            lora_ckpt_paths[adapter_id] = lora_adapter_dir

    def write_to_json(self, filename, data):
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def download_lora_adapters(self):
        self._download_lora_adapters(self.lora_ckpt_paths)
        self._download_lora_adapters(self.lora_ckpt_paths_cpu)
        data = {
            "lora-ckpt-dir": self.lora_adapters_dir,
            "lora-ckpt-paths": self.lora_ckpt_paths,
            "lora-ckpt-paths-cpu": self.lora_ckpt_paths_cpu,
        }
        self.write_to_json(self.lora_json_file, data)

    def get_lora_json_file(self):
        return self.lora_json_file

    def get_lora_adapter_ids(self, cpu_only=False):
        if not cpu_only:
            return list(self.lora_ckpt_paths.keys())
        return list(self.lora_ckpt_paths_cpu.keys())

    def get_lora_adapters_dir(self):
        return self.lora_adapters_dir

    def select_lora_adapter_ids(
        self, num, random_adapter_ids=False, dynamic_multi_lora=False
    ):
        adapter_ids = (
            self.get_lora_adapter_ids(cpu_only=True)
            if dynamic_multi_lora
            else self.get_lora_adapter_ids()
        )
        return (
            random.choices(adapter_ids, k=num)
            if random_adapter_ids
            else adapter_ids[0:num]
        )
