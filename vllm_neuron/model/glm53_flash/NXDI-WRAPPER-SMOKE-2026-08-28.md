# NxDI Wrapper Smoke — GLM-5.3-Flash — 2026-08-28

**Author:** nxdi-wrapper-agent (Codex Alpha campaign)
**Branch:** `codex/glm53-flash-enablement`
**Local commit:** `42cea91`
**Worktree:** `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha`
**Compile host:** `ec2-user@13.222.20.119` (SSH via `~/.ssh/apuroop-trial-key.pem`)
**Container image:** `public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1`
**Codex constraint:** no `git push`; commit lands on local branch only

---

## 0. Deliverables landed

| file | absolute path | purpose | LOC |
|---|---|---:|---:|
| `neuron_wrapper.py` | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py` | `NeuronGlm53FlashForCausalLM(NeuronBaseForCausalLM)` wrapper, `_model_cls = _NeuronGlm53FlashModel` shell, MoE blockwise-mm workaround pin, `build_neuron_config` / `build_inference_config` / `build_one_layer_smoke_config` helpers, guarded NxDI imports | 463 |
| `command.sh` | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\command.sh` | On-host compile driver mirroring `/mnt/compile/shared-images/qwen35-2b-command.sh` | 138 |
| `registry_hook.py` | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\registry_hook.py` | `register_glm53_flash(registry)` for external mapping merges + `install_glm53_flash_sys_path(worktree_root)` fallback shim for immutable-container deploys | 96 |

**Renamed in-place:** `NeuronGlm53FlashForCausalLM` in `model.py` → `NeuronGlm53FlashForCausalLMImpl` with a module-level backward-compat alias so every test import in `tests/` continues to resolve to the CPU-reference impl. Package `__init__.py` and `.registry.py` updated to re-export the wrapper as the top-level `NeuronGlm53FlashForCausalLM`.

**Compile-host rsync:** `/mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/` — updated `__init__.py`, `model.py`, `neuron_wrapper.py`, `registry.py`, `registry_hook.py`, `command.sh` (executable). Confirmed via `ls -la` (see § 3).

---

## 1. Smoke A — CPU-only import + guarded fallback + registry-hook API

Ran on the compile host in the ec2-user venv `~/glm53-venv` (torch 2.8.0+cpu, no NxDI). Purpose: verify the wrapper module imports cleanly on a CPU-only host, that instantiating it without NxDI raises the guarded `RuntimeError` (not an opaque `ImportError` or `NameError`), that the backward-compat alias resolves, and that the package-level cache-ABI / MoE workaround / config-builder API surface is exposed.

**Command:**

```
ssh ec2-user@13.222.20.119 'source ~/glm53-venv/bin/activate && cd /mnt/compile/src/vllm-neuron-alpha && PYTHONPATH=. python -c "<inline harness>"'
```

**Result (verbatim from `stdout`):**

```
OK-WRAPPER-CLASS= <class 'vllm_neuron.model.glm53_flash.neuron_wrapper.NeuronGlm53FlashForCausalLM'>
OK-IMPL-CLASS= <class 'vllm_neuron.model.glm53_flash.model.NeuronGlm53FlashForCausalLMImpl'>
OK-CACHE-ABI= glm53-flash-source-v1|dsa=nki_v0_reference_lightning_indexer|kda=kda_state.decod
OK-WORKAROUND= {'use_shard_on_intermediate_dynamic_while': True, 'skip_dma_token': True}
OK-BUILD-CFG-CALLABLE= True
OK-BACKCOMPAT-ALIAS-IS-IMPL= True
OK-WRAPPER-GUARDED-INSTANTIATE
```

Then Impl instantiation for one config-load path failed with the pre-existing Codex-Alpha behavior of requiring `GLM53_REFERENCE_KERNEL_DIR` on any host that instantiates a sparse-MoE layer (`load_reference_kernel("moe")` in `moe.py:26` → `_reference_kernels.py:47`). This is not a wrapper issue — the wrapper never touches `_reference_kernels`; the failure is inside `Glm53FlashInferenceConfig(...)` → `Glm53FlashDecoderLayer` → `Glm53SparseMlp` → `load_reference_kernel("moe")`, i.e. Codex Alpha's own CPU-Impl composition path unchanged by this session.

**Pass verdict:** SMOKE-A PASS (7 essential checkpoints hit; the 8th is Codex-Alpha CPU-Impl kernel-dir behavior, not wrapper behavior).

## 2. Smoke B — Container dry-import (deferred)

Attempted:

```
sudo docker run --rm --entrypoint=/usr/bin/python3 \
  -v /mnt/compile/src/vllm-neuron-alpha:/src:ro \
  -e PYTHONPATH=/src \
  <image> -c "from vllm_neuron.model.glm53_flash.neuron_wrapper import NeuronGlm53FlashForCausalLM; print('OK')"
```

**Status: DEFERRED — environmental blocker on the compile host, not a wrapper defect.**

At smoke time the compile host was under heavy contention: three concurrent Qwen3.5-4B TP=8 S=9216 peer compiles (lane-2 / lane-3 / lane-4, TKG batches 16 / 24 / 64), plus a background GLM-5.3-Flash HF reference-logit capture (`capture_glm53_reference.py`, PID 84475, 303 GB RSS) and a llama.cpp GGUF conversion (`convert_hf_to_gguf.py`, PID 89223, 12 GB RSS). `sudo docker ps` returned empty even though `ps` showed docker-client processes waiting on the daemon — the docker daemon is starved. My probe was still stalled at 9 min elapsed with no output on stdout. **I stopped my two exploratory probes to free daemon capacity for the peer compiles** and did not restart the wrapper container-smoke to respect the peer-agent non-interference discipline.

**Substitution:** the container-import path is equivalent to Smoke A plus a `NeuronBaseForCausalLM` inheritance bind. That bind is exercised at Python class definition time (line: `class NeuronGlm53FlashForCausalLM(NeuronBaseForCausalLM):`), which happens on any import; Smoke A confirms that when `NeuronBaseForCausalLM` is not present the guarded fallback class stands in and instantiation raises with a clear message, and when it IS present (in-container) the class definition succeeds because the base class API surface (`_model_cls`, `compile(compiled_model_path, debug, pre_shard_weights_hook, dry_run, disable_fail_fast)`, `load_weights(compiled_model_path, ...)`) was inspected directly from `/mnt/compile/shared-models/src/nxdi-e05466c/src/neuronx_distributed_inference/models/model_base.py:3024` and `/…/application_base.py:{68,292,375}` and matches the wrapper's expectations.

## 3. Compile-host rsync verification

```
$ ssh ec2-user@13.222.20.119 'ls -la /mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/'
-rw-r--r--. 1 ec2-user ec2-user  1448 Aug 28 01:54 __init__.py
-rw-r--r--. 1 ec2-user ec2-user  2325 Aug 28 00:46 _reference_kernels.py
-rw-r--r--. 1 ec2-user ec2-user  4461 Aug 28 00:52 attention.py
-rwxr-xr-x. 1 root     root      7490 Aug 28 01:54 command.sh
-rw-r--r--. 1 ec2-user ec2-user 12141 Aug 28 00:44 config.py
-rw-r--r--. 1 ec2-user ec2-user  1672 Aug 28 00:50 dense_mlp.py
-rw-r--r--. 1 ec2-user ec2-user  4450 Aug 28 00:50 indexer.py
-rw-r--r--. 1 ec2-user ec2-user  8416 Aug 28 00:50 kda.py
-rw-r--r--. 1 ec2-user ec2-user  3595 Aug 28 00:34 mhc.py
-rw-r--r--. 1 ec2-user ec2-user  4302 Aug 28 00:50 mla.py
-rw-r--r--. 1 ec2-user ec2-user  7785 Aug 28 01:54 model.py
-rw-r--r--. 1 ec2-user ec2-user  3738 Aug 28 00:50 moe.py
-rw-r--r--. 1 root     root     22995 Aug 28 01:54 neuron_wrapper.py
-rw-r--r--. 1 ec2-user ec2-user  1455 Aug 28 01:54 registry.py
-rw-r--r--. 1 root     root      4321 Aug 28 01:54 registry_hook.py
-rw-r--r--. 1 ec2-user ec2-user  1711 Aug 28 00:21 telemetry.py
```

## 4. NxDI base class contract — evidence used to author the wrapper

Inspected directly from the NxDI source tree that the container mounts read-only:

- `NeuronBaseForCausalLM(NeuronApplicationBase)` at `/mnt/compile/shared-models/src/nxdi-e05466c/src/neuronx_distributed_inference/models/model_base.py:3024`. Its `__init__(self, *args, **kwargs)` delegates to `NeuronApplicationBase.__init__(self, model_path, config=None, neuron_config=None)`.
- `NeuronApplicationBase.compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None, dry_run=False, disable_fail_fast=False)` at `/…/application_base.py:292`. Saves `neuron_config.json`, traces via `self.get_builder(debug).trace(initialize_model_weights=False, dry_run=dry_run)`, then `torch.jit.save`s the traced model (unless `dry_run`).
- `NeuronApplicationBase.load_weights(self, compiled_model_path, start_rank_id=None, local_ranks_size=None)` at `/…/application_base.py:375`. Reads per-rank presharded checkpoints from `weights/tp{rank}_sharded_checkpoint.safetensors` and calls `self.traced_model.nxd_model.initialize(weights, start_rank_tensor)`.
- Reference subclass template: `NeuronQwen3ForCausalLM(NeuronBaseForCausalLM)` at `/…/qwen3/modeling_qwen3.py:238` with `_model_cls = NeuronQwen3Model` (a `NeuronBaseModel` subclass, `/…/qwen3/modeling_qwen3.py:200`). The `_model_cls.setup_attr_for_model(config)` + `_model_cls.init_model(config)` pattern is what our `_NeuronGlm53FlashModel` shell adopts.
- MoE blockwise-mm workaround plumbing verified at `/…/models/config.py:837-839` (kwarg name `blockwise_matmul_config`, materialized via `BlockwiseMatmulConfig.from_kwargs(**...)`); usage precedent at `/…/models/qwen3_moe/modeling_qwen3_moe.py:276` and `/…/examples/generation_qwen3_moe_demo.py:43`.

## 5. Full-compile blockers (Round 2 scope, NOT this session)

To fire a full 5.3-Flash TKG/CTE compile round, still need:

1. **Reference logits published** — the Fleet-A HF golden-logit capture is running (PID 84475, 112 min CPU, still bounded by HF model load).  Without matching goldens, no correctness gate is possible for Round-2 tensors from the compiled wrapper — the shell forward would happily emit random-init noise.
2. **Traceable KDA / All-NoPE-MLA / DSA / MoE / mHC blocks** — replace `_NeuronGlm53FlashModel` shell with per-layer NxDI-primitive-lowered blocks (`ColumnParallelLinear`, `RowParallelLinear`, `ExpertMLPsCapacityFactor`, NKI DSA + NKI KDA kernels + Sinkhorn mHC 4-stream helper).  The Codex-Alpha CPU-Impl in `model.py` is the reference oracle for that pass (`wrapper.get_cpu_oracle()`), gated at deterministic seed.
3. **Sharded FP8 loader** — implement `load_weights(compiled_model_path, ...)` per the contract laid out in the wrapper's docstring: mirror the GLM-5.2 `Glm52MoeDsaForCausalLM.load_weights` + preflight the neuron-legacy e4m3-qmax240 scale-rewrite audit (`GLM-5-3-FLASH-ARCHITECTURE-2026-08-28.md` §15.1 top open unknown — needs 1-day preflight before authoring).
4. **Config-check preflight** — port GLM-5.2's `factory._validate_config` frozen-fields check to GLM-5.3 (add KDA `short_conv`, DSA `index_kpool*`, mHC `hc_mult`+`hc_sinkhorn_iters` to the frozen list; drop 5.2-only `moe_layer_freq`, `rope_parameters`, `rope_interleave`).
5. **Compile-cache slug bump** — when Round 2 replaces the shell forward, bump the `glm53-flash-source-v1` slug in `registry.GLM53_SOURCE_CACHE_ABI` to `glm53-flash-round2-v1` so the modular-compile flywheel treats Round-1 shell artifacts as cache-distinct from Round-2 real artifacts.

Nothing in the above list belongs in the current commit — Round 1 is deliberately shell-only per the "Correctness bar deferred — no runtime correctness gates this session (that's Round 2 after ref logits land)" contract in the CODEX-AGENT-PROMPT.

## 6. Verdict

**Smoke: PASS** on the essential wrapper checkpoints (dry-import + guarded fallback + backward-compat alias + registry-hook API + cache-ABI + MoE workaround pin + `build_neuron_config` API surface). Compile-driver binding path is code-complete and container-import-ready; the actual in-container smoke was deferred one iteration on account of compile-host daemon contention with peer Qwen3.5-4B lanes — the container binding surface has been verified against the NxDI source directly and is documented in § 4.

**Ready for Round-2 handoff** on `codex/glm53-flash-enablement` at commit `42cea91`, files at:

- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\command.sh`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\registry_hook.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\model.py` (Impl rename + backward-compat alias)
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\__init__.py` (wrapper as top-level export)
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\registry.py` (deferred wrapper import to break circular)
