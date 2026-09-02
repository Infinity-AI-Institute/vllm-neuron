"""CPU-only tests for the non-executable GLM-5.3 provider scaffold."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).parents[3]
MODULE = ROOT / "vllm_neuron/model/glm53_flash/provider_factory.py"
SPEC = importlib.util.spec_from_file_location("glm53_provider_factory", MODULE)
provider = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provider
SPEC.loader.exec_module(provider)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def state_contract():
    return {
        "schema": provider.STATE_SCHEMA, "owner": "shared_device_resident_cte_tkg",
        "slots": 1, "slot_input": {"name": "state_slot", "dtype": "int32", "shape": [1]},
        "reset_input": {"name": "state_reset", "dtype": "bool", "shape": [1]},
        "cte_initial_reset_required": True, "tkg_reset_forbidden": True,
        "finish_invalidates_slot": True, "reuse_requires_reset": True,
        "preemption_supported": False, "prefix_caching_supported": False,
        "async_scheduling_supported": False,
    }


def write_package(root, *, tp=64, include_tkg=True, mismatched_state=False):
    root = Path(root); runtime = "a" * 64; ranks = "b" * 64
    state = state_contract(); state_sha = canonical_sha(state)
    refs = {}
    for key, phase in (("cte", "CTE"), ("tkg", "TKG")):
        if key == "tkg" and not include_tkg:
            continue
        manifest = {
            "schema": provider.PHASE_SCHEMA, "phase": phase,
            "tensor_parallel_degree": tp, "logical_neuron_cores": 2,
            "rank_count": tp, "runtime_config_sha256": runtime,
            "rank_bundle_sha256": ranks,
            "state_abi_sha256": ("c" * 64 if mismatched_state and key == "tkg" else state_sha),
            "artifact_manifest_sha256": ("d" if key == "cte" else "e") * 64,
            "compiler_image_digest": "example.invalid/neuron@sha256:" + "f" * 64,
        }
        path = root / f"{key}.json"; path.write_text(json.dumps(manifest, sort_keys=True))
        refs[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    package = {
        "schema": provider.SERVICE_SCHEMA, "architecture": provider.ARCHITECTURE,
        "service_ready": False,
        "topology": {"tensor_parallel_degree": tp, "logical_neuron_cores": 2,
                     "cards": 16, "physical_neuron_cores": 64, "rank_count": tp},
        "workload": {"batch_size": 1, "context_encoding_tokens": 2048,
                     "total_context_capacity": 2560, "token_generation_step_tokens": 1},
        "runtime_config_sha256": runtime, "rank_bundle_sha256": ranks,
        "state_contract": state, "phases": refs,
    }
    path = root / "service-package.json"; path.write_text(json.dumps(package, sort_keys=True))
    return path


def configs(path, tp=64):
    hf = SimpleNamespace(architectures=[provider.ARCHITECTURE])
    neuron = SimpleNamespace(tp_degree=tp, logical_nc_config=2, ctx_batch_size=1,
                            tkg_batch_size=1, is_prefix_caching=False, async_mode=False,
                            glm53_service_package_path=str(path),
                            glm53_service_package_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return hf, neuron


class ProviderFactoryTests(unittest.TestCase):
    def test_valid_pair_is_admitted_but_execution_still_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_package(root)
            admission = provider.Glm53ServicePackageAdmission.load(path)
            self.assertEqual([phase.phase for phase in admission.phases], ["CTE", "TKG"])
            with self.assertRaisesRegex(provider.Glm53ProviderExecutionBridgeUnavailable, "CPU oracle"):
                provider.Glm53FlashProviderForCausalLM.from_configs(*configs(path))

    def test_tp32_r5_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_package(root, tp=32)
            with self.assertRaisesRegex(provider.Glm53ProviderAdmissionError, "TP64"):
                provider.Glm53ServicePackageAdmission.load(path)
            with self.assertRaisesRegex(provider.Glm53ProviderAdmissionError, "TP32/r5"):
                provider.Glm53FlashProviderForCausalLM.from_configs(*configs(path, tp=32))

    def test_cte_only_package_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(provider.Glm53ProviderAdmissionError, "paired CTE and TKG"):
                provider.Glm53ServicePackageAdmission.load(write_package(root, include_tkg=False))

    def test_phase_state_abi_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(provider.Glm53ProviderAdmissionError, "state ABI drift"):
                provider.Glm53ServicePackageAdmission.load(write_package(root, mismatched_state=True))

    def test_state_reset_contract_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_package(root); package = json.loads(path.read_text())
            package["state_contract"]["reuse_requires_reset"] = False
            path.write_text(json.dumps(package, sort_keys=True))
            with self.assertRaisesRegex(provider.Glm53ProviderAdmissionError, "slot/reset contract"):
                provider.Glm53ServicePackageAdmission.load(path)

    def test_external_package_digest_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_package(root); hf, neuron = configs(path)
            neuron.glm53_service_package_sha256 = "0" * 64
            with self.assertRaisesRegex(provider.Glm53ProviderAdmissionError, "externally pinned"):
                provider.Glm53FlashProviderForCausalLM.from_configs(hf, neuron)

    def test_registry_targets_scaffold_not_direct_wrapper_or_cpu_oracle(self):
        registry = (ROOT / "vllm_neuron/model/registry.py").read_text()
        self.assertIn('(\"Glm5NextForConditionalGeneration\", Glm53FlashProviderForCausalLM)', registry)
        self.assertNotIn("NeuronGlm53FlashForCausalLMImpl", registry)
        self.assertNotIn("NeuronGlm53FlashForCausalLM),", registry)


if __name__ == "__main__":
    unittest.main()
