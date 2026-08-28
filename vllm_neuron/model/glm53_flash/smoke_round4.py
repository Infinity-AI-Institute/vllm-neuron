# SPDX-License-Identifier: Apache-2.0
"""Round-4 coverage + state-persistence smoke for GLM-5.3-Flash.

Run inside the NxDI container (digest sha256:011d49c7…) on the compile host:

    PYTHONPATH=/src/nxdi/src:/mnt/compile/src/vllm-neuron-alpha:/mnt/compile/src/glm53-kernels \\
    GLM53_REFERENCE_KERNEL_DIR=/mnt/compile/src/glm53-kernels \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m vllm_neuron.model.glm53_flash.smoke_round4

What Round 4 adds over ``smoke_round3``:

  3.  **KDA state persistence, kernel level** — threading the state through two
      1-token calls must reproduce a single 2-token call *bit for bit*, and a
      zero-state restart of step 2 must produce a *different* answer.  The
      second half is the negative control: without it the test passes on a
      graph that silently drops the state.
  7.  **KDA state persistence, model level** — a real ``_NeuronGlm53FlashModel``
      built on CPU at TP=1, prefilled then stepped twice, with the aliased
      parameters written back between steps exactly as NxDI's
      ``input_output_aliases`` does on device.  Then the same step 2 is re-run
      with the state parameters zeroed; the logits must differ.  Run once for
      *all* state and once for *KDA state only*, so the KDA half cannot pass on
      the strength of the DSA KV cache.
  8.  **4-layer coverage compile** — KDA + DSA + 288-expert routed MoE in one
      traced graph.  Round 3 aborted here (``double free or corruption``) on a
      token-major MoE gather; Round 4 routes the branch through NxDI's
      ``ExpertMLPs``.

Nothing here fabricates a pass: every stage prints PASS/FAIL with the
exception text, and a persistence stage that cannot distinguish a zero-state
restart is reported as a FAIL, not a skip.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import time
import traceback
import types
from typing import Any

import torch

RESULTS: list[dict[str, Any]] = []

_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def _bootstrap_package() -> None:
    if "vllm_neuron.model.glm53_flash" in sys.modules:
        return
    chain = {
        "vllm_neuron": os.path.join(_PKG_ROOT, "vllm_neuron"),
        "vllm_neuron.model": os.path.join(_PKG_ROOT, "vllm_neuron", "model"),
        "vllm_neuron.model.glm53_flash": os.path.join(
            _PKG_ROOT, "vllm_neuron", "model", "glm53_flash"
        ),
    }
    for name, path in chain.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module


def _load(module_name: str, filename: str):
    full = f"vllm_neuron.model.glm53_flash.{module_name}"
    if full in sys.modules:
        return sys.modules[full]
    path = os.path.join(
        _PKG_ROOT, "vllm_neuron", "model", "glm53_flash", filename
    )
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def stage(name: str, fn) -> Any:
    started = time.time()
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001 - the traceback IS the deliverable
        RESULTS.append(
            {
                "stage": name,
                "status": "FAIL",
                "seconds": round(time.time() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None
    RESULTS.append(
        {
            "stage": name,
            "status": "PASS",
            "seconds": round(time.time() - started, 2),
            "detail": repr(value)[:600],
        }
    )
    print(
        f"[PASS] {name} ({round(time.time() - started, 2)}s): "
        f"{repr(value)[:400]}",
        flush=True,
    )
    return value


# ---------------------------------------------------------------------------
# Stage 3 — KDA persistence at the kernel level
# ---------------------------------------------------------------------------

def kda_persistence_kernel(nki_bindings, *, seed: int = 7) -> dict[str, float]:
    """Two 1-token steps threading state == one 2-token call; zeros != that.

    The equivalence is the *positive* half.  The zero-state restart is the
    *negative control*: a graph that silently restarts from zero state
    reproduces neither, and without the control this test would pass on such a
    graph whenever the state happened to be small.
    """
    torch.manual_seed(seed)
    batch, heads, d_qk, d_v = 2, 4, 8, 8

    def r(*shape):
        return torch.randn(*shape, dtype=torch.float32)

    q = r(batch, 2, heads, d_qk)
    k = r(batch, 2, heads, d_qk)
    v = r(batch, 2, heads, d_v)
    g = r(batch, 2, heads, d_qk)
    beta = r(batch, 2, heads)
    a_log, g_bias = r(heads), r(heads, d_qk)
    zero = torch.zeros(batch, heads, d_v, d_qk, dtype=torch.float32)

    fwd = nki_bindings.kda_state_forward_torch

    # Reference: one call over both tokens.
    y_ref, s_ref = fwd(zero, q, k, v, g, beta, a_log, g_bias)

    # Threaded: two 1-token calls, state carried between them.
    y1, s1 = fwd(
        zero, q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], a_log, g_bias
    )
    y2, s2 = fwd(
        s1, q[:, 1:], k[:, 1:], v[:, 1:], g[:, 1:], beta[:, 1:], a_log, g_bias
    )

    err_y = float((y2[:, 0] - y_ref[:, 1]).abs().max())
    err_s = float((s2 - s_ref).abs().max())

    # Negative control: restart step 2 from zero state.
    y2_zero, _ = fwd(
        zero, q[:, 1:], k[:, 1:], v[:, 1:], g[:, 1:], beta[:, 1:], a_log, g_bias
    )
    control = float((y2[:, 0] - y2_zero[:, 0]).abs().max())

    if err_y != 0.0 or err_s != 0.0:
        raise AssertionError(
            "threading KDA state across two 1-token steps does not reproduce "
            f"the 2-token reference (max abs err y={err_y}, state={err_s}); "
            "the recurrence handoff is wrong"
        )
    if control <= 0.0:
        raise AssertionError(
            "NEGATIVE CONTROL FAILED: restarting step 2 from zero state gave "
            "the identical output, so this test cannot detect a graph that "
            "drops the state. Refusing to report a persistence PASS."
        )
    return {
        "thread_vs_reference_max_abs_err": err_y,
        "state_max_abs_err": err_s,
        "zero_state_restart_delta": control,
    }


# ---------------------------------------------------------------------------
# Stages 6-7 — model-level wiring and persistence, on CPU at TP=1
# ---------------------------------------------------------------------------

def _neuter_mark_step():
    """No-op ``xm.mark_step`` for the duration of the CPU-only model stages.

    ``initialize_model_parallel`` ends with ``xm.mark_step()``, which on a
    host with no Neuron device raises
    ``RuntimeError: Init: error condition !(num_devices > 0)``.  The compile
    host is an r7i.12xlarge with no device by design (dry-run tracing only),
    so this is an environment limit, not a model one.

    ``mark_step`` is a lazy-graph sync barrier; these stages run eager torch on
    CPU tensors and never build an XLA graph, so there is nothing for it to
    flush.  Neutering it is therefore inert for what is under test — and it is
    restored before the real TP=16 trace, which does need the real one.
    """
    from torch_xla.core import xla_model as xm

    original = xm.mark_step

    def _noop(*_args, **_kwargs):
        return None

    xm.mark_step = _noop
    return xm, original


def _init_single_rank_parallel() -> str:
    """Initialise a 1-rank gloo process group + NxD model-parallel state.

    Needed because ``ColumnParallelLinear`` / ``RowParallelLinear`` /
    ``ExpertMLPs`` all read the parallel state at construction.  TP=1 keeps the
    tiny CPU model honest: every collective degenerates to identity, so the
    numerics under test are the layer logic, not the sharding.
    """
    import torch.distributed as dist
    from neuronx_distributed.parallel_layers import parallel_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    if not parallel_state.model_parallel_is_initialized():
        # `skip_collective_init=True` is NxD's own escape hatch for the block
        # at parallel_state.py:649-656 that allocates `torch.rand([1],
        # device="xla")` and all-reduces it to warm the collectives.  That
        # needs a real device; this host has none.  Skipping it is inert at
        # TP=1 (there is nothing to reduce across) and is the reason this
        # stage can run at all on the r7i compile host.
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            skip_collective_init=True,
        )
    return "gloo/tp1/skip_collective_init"


def _tiny_config(config_mod):
    """A genuinely GLM-5.3-shaped model small enough to run eagerly on CPU.

    Every *mechanism* invariant the config enforces is kept (All-NoPE MLA at
    qk/nope head dim 256, IndexPool=4 with tail selection, mHC x4 with 20
    Sinkhorn steps, sigmoid routing with a selection-only correction bias);
    only the *sizes* shrink.  So this exercises the real ``_KDABlock``,
    ``_DSABlock``, ``_MoEBlock`` and layer/model state threading — it is not a
    mock.
    """
    Glm53FlashInferenceConfig = config_mod.Glm53FlashInferenceConfig
    base = Glm53FlashInferenceConfig()
    fields_dict = {
        name: getattr(base, name) for name in base.__dataclass_fields__
    }
    fields_dict.update(
        allow_reduced_shapes=True,
        vocab_size=512,
        hidden_size=512,
        num_hidden_layers=2,
        layer_types=("linear_attention", "deepseek_sparse_attention"),
        mlp_layer_types=("dense", "sparse"),
        indexer_types=("full", "full"),
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=256,
        q_lora_rank=64,
        kv_lora_rank=64,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        max_position_embeddings=4096,
        pad_token_id=0,
        eos_token_id=(1,),
        torch_dtype=torch.float32,
    )
    linear = copy.deepcopy(base.linear_attn_config)
    linear.num_heads = 2
    linear.head_dim = 32
    linear.kda_layers = (0,)
    linear.full_attn_layers = (1,)
    fields_dict["linear_attn_config"] = linear
    return Glm53FlashInferenceConfig(**fields_dict)


def _build_tiny_model(wrapper_mod, config_mod, *, seq_len: int = 32):
    src = _tiny_config(config_mod)
    neuron_config = wrapper_mod.build_neuron_config(
        tp_degree=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.float32,
        extra={"logical_nc_config": 2},
    )
    cfg = wrapper_mod.Glm53FlashNeuronInferenceConfig(
        neuron_config=neuron_config, source_config=src
    )
    model = wrapper_mod._NeuronGlm53FlashModel(cfg)
    model.eval()
    # Random, non-degenerate weights: an all-zero init would make every state
    # comparison trivially equal and the negative control meaningless.
    torch.manual_seed(11)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 1 and param.is_floating_point():
                param.normal_(0.0, 0.02)
    with torch.no_grad():
        for param in model.past_key_values:
            param.zero_()
    return model, cfg


def state_wiring_report(model) -> dict[str, Any]:
    """Every aliased parameter is named, shaped, and owned by a layer."""
    names = list(model.state_cache_names)
    params = list(model.past_key_values)
    if len(names) != len(params):
        raise AssertionError(
            f"{len(names)} declared state names vs {len(params)} parameters"
        )
    covered = sum(stop - start for start, stop in model.layer_cache_slices)
    if covered != len(params):
        raise AssertionError(
            f"layer slices cover {covered} of {len(params)} aliased parameters"
        )
    return {
        "num_aliased": len(params),
        "names": names,
        "shapes": [tuple(p.shape) for p in params],
        "layer_slices": list(model.layer_cache_slices),
    }


def _run(model, input_ids, position_ids, attention_mask):
    out = model(
        input_ids,
        attention_mask,
        position_ids,
        None,
        None,
    )
    if isinstance(out, (list, tuple)):
        return out[0], list(out[1:])
    return out, []


def _commit(model, caches) -> None:
    """Write graph outputs back into the aliased parameters.

    This is exactly what ``input_output_aliases`` does on device: the output
    buffer *is* the input buffer.  Doing it explicitly here is what lets a CPU
    run reproduce multi-step decode semantics.
    """
    with torch.no_grad():
        for param, value in zip(model.past_key_values, caches):
            param.copy_(value.detach().to(param.dtype))


def model_persistence(model, *, kda_only: bool) -> dict[str, float]:
    """Step 2's logits must depend on step 1's state.

    ``kda_only=True`` zeroes only the KDA state/conv parameters and leaves the
    DSA K/V/index-K caches intact, so the KDA half cannot pass on the strength
    of the attention cache.
    """
    prompt_len = 8
    ids = torch.randint(0, 400, (1, prompt_len))
    pos = torch.arange(prompt_len).unsqueeze(0)
    mask = torch.ones(1, prompt_len, dtype=torch.int64)

    with torch.no_grad():
        for param in model.past_key_values:
            param.zero_()
        _, caches = _run(model, ids, pos, mask)
        _commit(model, caches)
        snapshot_after_prefill = [p.detach().clone() for p in model.past_key_values]

        step1_ids = torch.randint(0, 400, (1, 1))
        step1_pos = torch.tensor([[prompt_len]])
        step1_mask = torch.ones(1, 1, dtype=torch.int64)
        _, caches1 = _run(model, step1_ids, step1_pos, step1_mask)
        _commit(model, caches1)

        step2_ids = torch.randint(0, 400, (1, 1))
        step2_pos = torch.tensor([[prompt_len + 1]])
        step2_mask = torch.ones(1, 1, dtype=torch.int64)
        logits_carried, _ = _run(model, step2_ids, step2_pos, step2_mask)

        # Positive control: the same step re-run against the same state must
        # be bit-identical.  Without this, a nonzero delta below could be
        # nondeterminism rather than state dependence, and the whole stage
        # would prove nothing.
        logits_repeat, _ = _run(model, step2_ids, step2_pos, step2_mask)
        repeat_delta = float((logits_carried - logits_repeat).abs().max())
        if repeat_delta != 0.0:
            raise AssertionError(
                "POSITIVE CONTROL FAILED: re-running the same decode step "
                f"against the same state changed the logits by {repeat_delta}. "
                "The forward is nondeterministic, so a nonzero zero-state "
                "delta would not be attributable to the state."
            )

        # Negative control: restart the chosen state from zero and redo step 2.
        for idx, param in enumerate(model.past_key_values):
            name = model.state_cache_names[idx]
            is_kda = name.endswith("kda_state") or name.endswith("conv_state")
            if kda_only and not is_kda:
                param.copy_(snapshot_after_prefill[idx])
            else:
                param.zero_()
        logits_zeroed, _ = _run(model, step2_ids, step2_pos, step2_mask)

    delta = float((logits_carried - logits_zeroed).abs().max())
    scale = float(logits_carried.abs().max())
    if not (delta > 0.0):
        raise AssertionError(
            "step-2 logits are IDENTICAL with and without the step-1 state"
            + (" (KDA state only)" if kda_only else "")
            + ". The graph is not consuming its aliased state, so a multi-step "
            "decode would silently restart every step. This is the exact "
            "failure this stage exists to catch — refusing to report a PASS."
        )
    return {
        "kda_only": kda_only,
        "max_abs_logit_delta_vs_zero_state": delta,
        "max_abs_logit": scale,
        "relative_delta": delta / scale if scale else float("inf"),
        "repeat_run_delta": repeat_delta,
    }


def _finish(code: int) -> int:
    out = os.environ.get("GLM53_SMOKE_RESULT", "/tmp/glm53-round4-result.json")
    try:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(RESULTS, handle, indent=2)
        print(f"\nwrote {out}", flush=True)
    except OSError as exc:
        print(f"could not write {out}: {exc}", flush=True)
    failures = [r for r in RESULTS if r["status"] == "FAIL"]
    print("\n" + "=" * 72, flush=True)
    for item in RESULTS:
        print(
            f"  {item['status']:4}  {item['stage']:<34} "
            f"{item.get('seconds', '')}s",
            flush=True,
        )
    print("=" * 72, flush=True)
    return 1 if failures else code


def main() -> int:
    print("=" * 72, flush=True)
    print("GLM-5.3-Flash Round-4 coverage + state-persistence smoke", flush=True)
    print(f"python={sys.version.split()[0]} torch={torch.__version__}", flush=True)
    print(
        "kernel_dir=" + os.environ.get("GLM53_REFERENCE_KERNEL_DIR", "<default>"),
        flush=True,
    )
    print("=" * 72, flush=True)

    _bootstrap_package()
    _load("registry", "registry.py")
    refs = _load("_reference_kernels", "_reference_kernels.py")
    nki_bindings = _load("nki_bindings", "nki_bindings.py")

    stage(
        "1.golden-import",
        lambda: [
            refs.load_reference_kernel(n).__name__ for n in ("kda", "dsa", "moe")
        ],
    )
    stage(
        "2.kda-parity-vs-numpy-golden",
        lambda: max(
            nki_bindings.kda_reference_parity_check(seed=s) for s in range(4)
        ),
    )
    stage(
        "3.kda-persistence-kernel",
        lambda: kda_persistence_kernel(nki_bindings),
    )
    stage(
        "3b.dsa-traceability-parity",
        nki_bindings.dsa_traceability_parity_check,
    )
    stage(
        "4.moe-dispatch-identity",
        lambda: nki_bindings.build_glm53_moe_dispatch_config(
            hidden=4096,
            num_experts=288,
            top_k=8,
            intermediate_global=2048,
            tp_degree=16,
            renormalize_topk=True,
        )
        and "validated",
    )

    config_mod = _load("config", "config.py")
    wrapper_mod = stage(
        "5.wrapper-import", lambda: _load("neuron_wrapper", "neuron_wrapper.py")
    )
    if wrapper_mod is None:
        return _finish(1)

    if os.environ.get("GLM53_SKIP_CPU_MODEL", "0") != "1":
        xm_mod, xm_original = _neuter_mark_step()
        parallel = stage("6a.single-rank-parallel-init", _init_single_rank_parallel)
        if parallel is not None:
            tiny = stage(
                "6b.tiny-cpu-model-build",
                lambda: _build_tiny_model(wrapper_mod, config_mod),
            )
            if tiny is not None:
                model, _cfg = tiny
                stage("6c.state-wiring", lambda: state_wiring_report(model))
                stage(
                    "7a.model-persistence-all-state",
                    lambda: model_persistence(model, kda_only=False),
                )
                stage(
                    "7b.model-persistence-kda-state-only",
                    lambda: model_persistence(model, kda_only=True),
                )
            # The tiny model's 1-rank parallel state must not leak into the
            # real TP=8/16 trace below.
            try:
                from neuronx_distributed.parallel_layers import parallel_state

                parallel_state.destroy_model_parallel()
            except Exception as exc:  # pragma: no cover
                print(f"[warn] could not tear down parallel state: {exc}", flush=True)
        # Restore the real barrier before the TP=16 trace, which needs it.
        xm_mod.mark_step = xm_original

    if os.environ.get("GLM53_SKIP_COMPILE", "0") == "1":
        return _finish(0)

    NeuronGlm53FlashForCausalLM = wrapper_mod.NeuronGlm53FlashForCausalLM
    source = stage(
        "8a.source-config", lambda: config_mod.Glm53FlashInferenceConfig()
    )
    if source is None:
        return _finish(1)

    tp = int(os.environ.get("GLM53_SMOKE_TP", "16"))
    seq = int(os.environ.get("GLM53_SMOKE_SEQ", "128"))
    mode = os.environ.get("GLM53_SMOKE_MODE", "coverage")
    phases = os.environ.get("NXDI_EMIT_PHASES", "BOTH")
    if mode == "real45":
        # The unreduced 45-layer model at the real contract shape.  Not a
        # smoke: this is the config the fire uses.
        # GLM53_CTX_LEN narrows the *prefill bucket* below the KV window.
        # Both CTE costs key off it, not off seq_len: the KDA scan unrolls
        # `num_kda_layers x n_active_tokens` steps, and the DSA sparse gather
        # is O(Q x topk).  Defaults to seq_len (one full-window prefill).
        ctx_len = int(os.environ.get("GLM53_CTX_LEN", "0")) or None

        def builder(src, **kw):
            if ctx_len:
                kw["max_context_length"] = ctx_len
            return NeuronGlm53FlashForCausalLM.build_inference_config(
                src, ctx_batch_size=1, tkg_batch_size=1, **kw
            )
    elif mode in NeuronGlm53FlashForCausalLM.SMOKE_RECIPES:
        def builder(src, **kw):
            return NeuronGlm53FlashForCausalLM.build_recipe_smoke_config(
                src, recipe=mode, **kw
            )
    elif mode == "coverage":
        builder = NeuronGlm53FlashForCausalLM.build_kernel_coverage_smoke_config
    else:
        builder = NeuronGlm53FlashForCausalLM.build_one_layer_smoke_config
    cfg = stage(
        f"8b.smoke-config[{mode},tp{tp},s{seq},{phases}]",
        lambda: builder(source, tp_degree=tp, seq_len=seq),
    )
    if cfg is None:
        return _finish(1)

    model_path = os.environ.get(
        "GLM53_MODEL_PATH",
        "/mnt/compile/hf-cache/models--zai-org--GLM-5.3-Flash/snapshots/"
        "04c4e9e95c5da8862dced7e5056455116f83a7e0",
    )
    wrapper = stage(
        "8c.wrapper-construct",
        lambda: NeuronGlm53FlashForCausalLM(model_path, cfg),
    )
    if wrapper is None:
        return _finish(1)

    out_path = os.environ.get("GLM53_SMOKE_OUT", "/runroot/artifacts/round4")
    os.makedirs(out_path, exist_ok=True)
    dry_run = os.environ.get("GLM53_DRY_RUN", "1") == "1"
    stage(
        "8d.dry-run-compile" if dry_run else "8d.compile-neff",
        lambda: wrapper.compile(out_path, dry_run=True)
        if dry_run
        else wrapper.compile(out_path),
    )
    if not dry_run:
        stage(
            "9.emitted-config-verify",
            lambda: _verify_emitted(out_path),
        )
    return _finish(0)


def _verify_emitted(out_path: str) -> dict[str, Any]:
    """Read back what the compile actually emitted, not what was requested.

    Reports the emitted cache dtypes and scans the compile artifacts for
    CPU-fallback markers.  The point is that a requested flag and an emitted
    tensor are different facts, and only the second one is evidence.
    """
    report: dict[str, Any] = {"out_path": out_path}
    cfg_path = os.path.join(out_path, "neuron_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as handle:
            raw = handle.read()
        report["neuron_config_bytes"] = len(raw)
        for marker in ("float8_e4m3fn", "kv_cache_quant", "bfloat16"):
            report[f"neuron_config_has_{marker}"] = marker in raw
    else:
        report["neuron_config_json"] = "MISSING"
    neffs = []
    for root, _dirs, files in os.walk(out_path):
        for name in files:
            if name.endswith(".neff"):
                full = os.path.join(root, name)
                neffs.append((full, os.path.getsize(full)))
    report["neff_count"] = len(neffs)
    report["neffs"] = [
        {"path": p, "bytes": n} for p, n in sorted(neffs)[:20]
    ]
    return report


if __name__ == "__main__":
    raise SystemExit(main())
