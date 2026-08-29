# SPDX-License-Identifier: Apache-2.0
"""Round-2 per-layer smoke: DSv4-Flash routed-MoE block, layer 3 (first
routed-MoE layer per the frozen ``mlp_layer_types`` schedule).

Correctness backstop for the Round-2 routed-MoE block class.  The FP4
dequant primitive is already byte-exact against a real HF tensor
(``test_fp4_dequant_1tensor.py``); this test gates the LAYER-level
composition:

  1. **Router forward** (pure PyTorch, no NxDI).  ``dsv4_route_affinities``
     under an all-in-range synthetic input must:
       * produce ``top_k = num_experts_per_tok = 6`` distinct indices per
         token, all in ``[0, n_routed_experts)``;
       * produce a full-width affinity vector that is finite and strictly
         non-negative (``sqrt(softplus(x)) > 0`` for finite ``x``);
       * match ``dsv4_reference_router_forward`` bit-for-bit on the
         gathered + normalised weights (``routed_scaling_factor`` applied),
         confirming the wrapper's split-normalize plan is arithmetically
         equivalent to HF's ``DeepseekV4TopKRouter.forward``.

  2. **HF-key catalog for layer 3**.  Fetch the safetensors index and
     confirm every routed-MoE-required key exists under
     ``layers.3.ffn.*``: router + correction bias + shared expert +
     all 256 routed experts × {w1, w2, w3} × {weight, scale} = 1541 keys.
     No tensor bytes are pulled — the catalog check is metadata-only and
     stays under 10 MiB of network.

  3. **256-expert stacking + wrapper-tree key count**.  Synthesise a
     small (packed) FP4 weight + UE8M0 scale for every expert
     ({256 experts × 3 tensors × 2 fields = 1536} synthetic tensors),
     invoke ``_convert_routed_moe_layer`` for layer 3, and assert:
       * exactly **7** MoE-block wrapper-tree keys land
         (``mlp.router.weight``, ``mlp.e_score_correction_bias``,
          ``mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight``,
          ``mlp.expert_mlps.mlp_op.{gate_up_proj,down_proj}.weight``);
       * the fused ``gate_up_proj`` has shape ``[E, hidden, 2*I]`` with
         E=256 (task deliverable).
       * ``down_proj`` has shape ``[E, I, hidden]`` with E=256.
       * every one of the 256 experts' contributions to ``gate_up_proj``
         and ``down_proj`` is comparable (non-degenerate) — this is the
         "per-expert dequant PASS/FAIL summary" line item on the receipt.

  4. **Real-tensor gate spot-check** (skips on no network).  If the FP4
     dequant test's ``.hf_cache/dsv4_expert0_w2.safetensors`` cache is
     present (or fetchable), route that real ``layers.3.ffn.experts.0.w2``
     tensor through ``_dequant_expert_fp4_weight`` and confirm the
     resulting bf16 tensor matches the raw ``dequantize_block_fp4_ue8m0``
     bit-for-bit.

Design constraints honoured:

  * NO 3.6 GiB shard download — the safetensors index (metadata only) is
    ~2 MiB; the one real tensor pair comes from the cache the FP4 test
    already primed.
  * NO device compile.  Wrapper + converter path only, gated on
    ``require_comparable``.
  * NO speculative-decode surface.
  * Full absolute paths in receipts.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Degeneracy guard — reuse the FP4 test's discovery convention.
# ---------------------------------------------------------------------------
_HARNESS_KERNELS = (
    Path(__file__).resolve().parents[5]
    / "gemma4-trn2-handoff"
    / "harness-v2"
    / "staging"
    / "reference-sweep-20260826T2150Z"
    / "kernels"
)
if str(_HARNESS_KERNELS) not in sys.path:
    sys.path.insert(0, str(_HARNESS_KERNELS))
try:
    from degeneracy_guard import require_comparable  # type: ignore
except Exception as exc:  # pragma: no cover — surface discovery gaps
    pytest.skip(
        f"degeneracy_guard not importable at {_HARNESS_KERNELS!s}: {exc!r}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Library discovery — same fallback pattern as ``test_fp4_dequant_1tensor``.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
_INDEX_CACHE = Path(__file__).parent / ".hf_cache" / "dsv4_flash_index.json"
_EXPERT_CACHE = Path(__file__).parent / ".hf_cache" / "dsv4_expert0_w2.safetensors"


def _import_library():
    """Import ``config`` + ``checkpoint_convert`` + ``neuron_wrapper``.

    Prefer the natural top-level import; fall back to explicit importlib
    on CPU-only laptops where the ``vllm`` package is unimportable.  The
    fallback loads ``neuron_wrapper`` too, so ``dsv4_route_affinities``
    is available for the router smoke without needing NxDI.
    """
    try:
        from vllm_neuron.model.dsv4_flash import (
            checkpoint_convert as conv_mod,  # type: ignore
        )
        from vllm_neuron.model.dsv4_flash import config as cfg_mod  # type: ignore
        from vllm_neuron.model.dsv4_flash import (
            neuron_wrapper as wrap_mod,  # type: ignore
        )

        return cfg_mod, conv_mod, wrap_mod
    except Exception:
        pass

    dsv4_dir = Path(__file__).resolve().parent.parent
    pkg_name = "_dsv4_flash_test_pkg_moe"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(dsv4_dir)]
        sys.modules[pkg_name] = pkg

    def _load(name: str):
        if f"{pkg_name}.{name}" in sys.modules:
            return sys.modules[f"{pkg_name}.{name}"]
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{name}",
            str(dsv4_dir / f"{name}.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("config")
    conv_mod = _load("checkpoint_convert")
    # ``neuron_wrapper`` may fail to import if NxDI is absent — that's fine
    # for the router smoke because the routing functions live at module top
    # level and don't touch nxdi.  But NxDI's absence turns the wrapper's
    # ``from ... import`` block into a caught exception path, so the
    # module IS importable on CPU-only hosts.  Load it defensively.
    try:
        wrap_mod = _load("neuron_wrapper")
    except Exception as exc:
        pytest.skip(f"neuron_wrapper unimportable even on CPU-only path: {exc!r}")
    return cfg_mod, conv_mod, wrap_mod


# ---------------------------------------------------------------------------
# HF index fetcher (metadata only).
# ---------------------------------------------------------------------------


def _fetch_index() -> dict[str, Any] | None:
    """Return the parsed ``model.safetensors.index.json`` or ``None``
    if the network is unreachable.  Cached under ``.hf_cache/`` to avoid
    re-fetches across test runs.
    """
    if _INDEX_CACHE.exists():
        try:
            return json.loads(_INDEX_CACHE.read_text(encoding="utf-8"))
        except Exception:
            _INDEX_CACHE.unlink(missing_ok=True)  # corrupt, refetch

    try:
        import urllib.request

        from huggingface_hub import hf_hub_url

        url = hf_hub_url(_HF_REPO, "model.safetensors.index.json", revision=_HF_SHA)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "vllm_neuron.dsv4_flash.tests/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        _INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_CACHE.write_bytes(data)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Synthetic per-expert weight/scale generator.
# ---------------------------------------------------------------------------


def _synth_expert_layer_state_dict(
    layer_idx: int,
    src,
    *,
    seed: int = 20260828,
) -> dict[str, torch.Tensor]:
    """Build a state dict carrying every HF key the routed-MoE converter
    reads for one layer, using synthetic (packed FP4 / FP8) tensors with
    the right shapes.

    The synthetic tensors are NOT byte-exact against HF weights — that
    contract is owned by ``test_fp4_dequant_1tensor``.  What we DO gate
    here:
      * every expert dequants without an error (shape / dtype / scale
        validators pass) and lands in the stacked wrapper tensor;
      * the 256 rows of ``gate_up_proj`` and ``down_proj`` are each
        non-degenerate (``require_comparable`` passes) — which fails if
        any expert's synthetic weight silently collapses to zero.

    Non-degeneracy is achieved by drawing the raw byte for each nibble
    from a rotating small set that includes at least one non-zero FP4
    codepoint per block, and per-expert varying the scale by the expert
    index so no two experts share a row-value trace.
    """
    hidden = src.hidden_size
    inter = src.moe_intermediate_size
    n_experts = src.n_routed_experts

    torch.manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    base = f"layers.{layer_idx}."

    # Router: [n_experts, hidden] fp32.  Correction bias: [n_experts] fp32.
    router_weight = torch.randn(n_experts, hidden, dtype=torch.float32) * 0.02
    sd[f"{base}ffn.gate.weight"] = router_weight
    sd[f"{base}ffn.gate.bias"] = torch.randn(n_experts, dtype=torch.float32) * 0.01

    # Shared expert: 3 x FP8 e4m3 tensors with UE8M0 block-scale (128, 128).
    # Use `float8_e4m3fn` cast of a small-value bf16 tensor + a scale of
    # raw byte 127 (multiplier 2**0 = 1.0) — that keeps the dequant path
    # exercised without needing to hunt for a real FP8 scale distribution.
    def _synth_fp8(out: int, in_: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = (torch.empty(out, in_, dtype=torch.float32).uniform_(-0.5, 0.5)).to(
            torch.float8_e4m3fn
        )
        scale = torch.full(
            (math.ceil(out / 128), math.ceil(in_ / 128)),
            127,
            dtype=torch.uint8,
        )
        return values, scale

    for hf_name, out_dim, in_dim in (
        ("w1", inter, hidden),
        ("w3", inter, hidden),
        ("w2", hidden, inter),
    ):
        w, s = _synth_fp8(out_dim, in_dim)
        sd[f"{base}ffn.shared_experts.{hf_name}.weight"] = w
        sd[f"{base}ffn.shared_experts.{hf_name}.scale"] = s

    # Routed experts: FP4 packed (2 nibbles/byte along K) + UE8M0 block-scale
    # (1, 32) on K.  Synthesise:
    #   - Weight bytes: rotating pattern per expert, seeded from the
    #     expert index so no two experts see the same packed row.  Nibble
    #     values include non-zero FP4 codes so the dequant is not all-zero.
    #   - Scale: UE8M0 raw code varying by expert but always in the
    #     "identity or near-identity" range (~127 ± few) so bf16 stays
    #     representable.
    for e in range(n_experts):
        base_e = f"{base}ffn.experts.{e}."
        # Packed FP4 bytes: draw uint8 in [0, 255] with a per-expert
        # generator seed so every expert's tensor is distinct AND every
        # row has enough nibble variety to avoid constant/zero rows.
        # Bytes are cast to `.view(torch.int8)` because the FP4-UE8M0
        # dequant primitive accepts int8/uint8/float4_e2m1fn_x2.
        gen = torch.Generator().manual_seed(seed + e * 131)

        def _packed(out_dim: int, in_bytes: int, g=gen) -> torch.Tensor:
            return (
                torch.randint(
                    1, 256, (out_dim, in_bytes), dtype=torch.uint8, generator=g
                )
                .view(torch.int8)
                .contiguous()
            )

        def _scale(out_dim: int, in_logical: int, g=gen) -> torch.Tensor:
            # UE8M0 raw codes 122..131 → multipliers 2**(-5..4) which
            # keep the FP4 codebook × multiplier well inside bf16 range
            # and give per-block variance so no row collapses to a
            # constant.
            nblk = in_logical // 32
            return torch.randint(
                122, 132, (out_dim, nblk), dtype=torch.uint8, generator=g
            ).contiguous()

        w1_packed = _packed(inter, hidden // 2)  # [I, H/2]
        w3_packed = _packed(inter, hidden // 2)  # [I, H/2]
        w2_packed = _packed(hidden, inter // 2)  # [H, I/2]
        sd[f"{base_e}w1.weight"] = w1_packed
        sd[f"{base_e}w1.scale"] = _scale(inter, hidden)
        sd[f"{base_e}w3.weight"] = w3_packed
        sd[f"{base_e}w3.scale"] = _scale(inter, hidden)
        sd[f"{base_e}w2.weight"] = w2_packed
        sd[f"{base_e}w2.scale"] = _scale(hidden, inter)

    return sd


# ===========================================================================
# Tests
# ===========================================================================


def test_router_topk_shape_and_range() -> None:
    """Router smoke: dsv4_route_affinities returns finite affinities and
    valid top-6 indices for a synthetic input."""
    cfg, _, wrap = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    hidden = src.hidden_size
    n_experts = src.n_routed_experts
    top_k = src.num_experts_per_tok

    torch.manual_seed(1)
    hidden_states = torch.randn(1, 4, hidden, dtype=torch.float32)
    router_weight = torch.randn(n_experts, hidden, dtype=torch.float32) * 0.02
    correction_bias = torch.randn(n_experts, dtype=torch.float32) * 0.01

    affinities, indices = wrap.dsv4_route_affinities(
        hidden_states,
        router_weight,
        top_k=top_k,
        scoring_func=src.scoring_func,
        correction_bias=correction_bias,
    )
    # Shape contract mirrors NxDI ExpertMLPs.
    assert affinities.shape == (4, n_experts), affinities.shape
    assert affinities.dtype == torch.float32
    assert indices.shape == (4, top_k), indices.shape
    assert indices.dtype == torch.int64
    # sqrt(softplus(x)) is strictly positive for every finite x.
    assert torch.isfinite(affinities).all(), affinities
    assert (affinities > 0).all(), affinities.min()
    # Indices in valid range and distinct per token (top_k unique experts).
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) < n_experts
    for t in range(4):
        row = indices[t].tolist()
        assert len(set(row)) == top_k, (t, row)
    require_comparable(affinities.detach().cpu().numpy(), "router_affinities")


def test_router_matches_hf_reference() -> None:
    """The wrapper's split (raw scores + norm inside ExpertMLPs + scale
    outside) must produce numerically identical gathered+normalised+scaled
    weights to HF's ``DeepseekV4TopKRouter.forward``.  Because normalise
    then scale is linear the pre-post reordering is exact modulo the
    reference's ``+ 1e-20`` denominator floor.
    """
    cfg, _, wrap = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    hidden = src.hidden_size
    n_experts = src.n_routed_experts
    top_k = src.num_experts_per_tok

    torch.manual_seed(2)
    hidden_states = torch.randn(1, 3, hidden, dtype=torch.float32)
    router_weight = torch.randn(n_experts, hidden, dtype=torch.float32) * 0.02
    correction_bias = torch.randn(n_experts, dtype=torch.float32) * 0.01

    # Reference from module-level util that mirrors HF ref.
    _logits, ref_weights, ref_indices = wrap.dsv4_reference_router_forward(
        hidden_states,
        router_weight,
        top_k=top_k,
        correction_bias=correction_bias,
        routed_scaling_factor=src.routed_scaling_factor,
    )
    # Wrapper split: full-width scores + top-k indices, then gather +
    # normalise + scale outside.
    affinities, indices = wrap.dsv4_route_affinities(
        hidden_states,
        router_weight,
        top_k=top_k,
        scoring_func=src.scoring_func,
        correction_bias=correction_bias,
    )
    # Same top-6 indices (softplus is monotone; correction bias identical).
    assert torch.equal(indices, ref_indices), (indices, ref_indices)
    gathered = affinities.gather(-1, indices)
    normalised = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
    scaled = normalised * src.routed_scaling_factor
    max_abs = float((scaled - ref_weights).abs().max().item())
    assert max_abs == 0.0, max_abs
    require_comparable(scaled.detach().cpu().numpy(), "wrapper_weights")


def test_router_refuses_wrong_scoring_func() -> None:
    """A scoring_func other than 'sqrtsoftplus' must be refused —
    silently defaulting to sigmoid would move which experts win top-k
    and is a silent-quality failure mode."""
    cfg, _, wrap = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    router_weight = torch.randn(src.n_routed_experts, src.hidden_size)
    hidden_states = torch.randn(1, 1, src.hidden_size)
    with pytest.raises(NotImplementedError, match="sqrtsoftplus"):
        wrap.dsv4_route_affinities(
            hidden_states,
            router_weight,
            top_k=src.num_experts_per_tok,
            scoring_func="sigmoid",
        )


def test_hf_index_catalog_layer3_routed_moe() -> None:
    """Every HF-side routed-MoE key for layer 3 exists in the safetensors
    index.  This is a metadata-only pull (~2 MiB) — no shard bytes fetched.
    """
    cfg, _, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    idx = _fetch_index()
    if idx is None:
        pytest.skip(
            "safetensors index unreachable; layer-3 HF-key catalog cannot "
            "be verified offline"
        )
    weight_map = idx["weight_map"]
    keys = set(weight_map.keys())
    L = 3
    required = [
        f"layers.{L}.ffn.gate.weight",
        f"layers.{L}.ffn.gate.bias",
        f"layers.{L}.ffn.shared_experts.w1.weight",
        f"layers.{L}.ffn.shared_experts.w1.scale",
        f"layers.{L}.ffn.shared_experts.w3.weight",
        f"layers.{L}.ffn.shared_experts.w3.scale",
        f"layers.{L}.ffn.shared_experts.w2.weight",
        f"layers.{L}.ffn.shared_experts.w2.scale",
    ]
    # 256 routed experts × 3 weights × 2 (weight + scale) = 1536.
    for e in range(src.n_routed_experts):
        for w in ("w1", "w2", "w3"):
            required.append(f"layers.{L}.ffn.experts.{e}.{w}.weight")
            required.append(f"layers.{L}.ffn.experts.{e}.{w}.scale")
    missing = [k for k in required if k not in keys]
    assert not missing, missing[:10]
    # Total expected HF key count for the routed-MoE portion of layer 3.
    #   router weight + router bias = 2
    #   shared expert = 6
    #   routed experts (256 × 3 × 2) = 1536
    # -> 1544 keys.
    total = 2 + 6 + src.n_routed_experts * 3 * 2
    assert len([k for k in required]) == total
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hf-index-layer3-routed-moe",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "layer_idx": L,
                "expected_hf_key_count": total,
                "verified_present": total,
                "missing_count": len(missing),
                "index_cache_path": str(_INDEX_CACHE),
            },
            indent=2,
        )
    )


def test_synth_layer3_convert_and_stack_shapes() -> None:
    """Round-2 gate: run ``_convert_routed_moe_layer`` on all 256 experts
    of layer 3 with synthetic FP4/FP8 tensors, and assert:
      * 7 wrapper-tree keys land under ``layers.3.mlp.*``
      * fused ``gate_up_proj.weight`` shape ``[E, hidden, 2*I]``
      * ``down_proj.weight`` shape ``[E, I, hidden]``
      * every expert's contribution to both stacked tensors is
        non-degenerate.

    Env override: ``DSV4_MOE_SMOKE_N_EXPERTS`` (default 256) lets a
    CPU-only laptop run the full stack at reduced fan-out — 256 real
    experts × 3 dequants × ``[2048, 4096]`` bf16 tiles + intermediates
    lands ~12 GiB peak RSS which thrashes swap on 16 GiB laptops.  The
    default is the production 256 (the deliverable); CI on a memory-
    constrained runner can drop to 32 without changing the code path.
    """
    cfg, conv, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    # Optionally shrink for CPU-side memory pressure (see docstring).
    env_e = os.environ.get("DSV4_MOE_SMOKE_N_EXPERTS")
    if env_e is not None:
        n_experts_override = int(env_e)
        object.__setattr__(src, "allow_reduced_shapes", True)
        object.__setattr__(src, "n_routed_experts", n_experts_override)
    L = 3
    print(
        f"# test_synth_layer3_convert_and_stack_shapes: n_experts="
        f"{src.n_routed_experts}",
        flush=True,
    )
    sd = _synth_expert_layer_state_dict(L, src)
    print("  synth state dict built", flush=True)
    converted: dict[str, Any] = {}
    report = conv._convert_routed_moe_layer(sd, converted, L, src)
    print("  _convert_routed_moe_layer completed", flush=True)

    # ---- wrapper-tree key discipline: exactly 7 MoE-block keys ----
    expected_moe_keys = {
        f"layers.{L}.mlp.router.weight",
        f"layers.{L}.mlp.e_score_correction_bias",
        f"layers.{L}.mlp.shared_expert.gate_proj.weight",
        f"layers.{L}.mlp.shared_expert.up_proj.weight",
        f"layers.{L}.mlp.shared_expert.down_proj.weight",
        f"layers.{L}.mlp.expert_mlps.mlp_op.gate_up_proj.weight",
        f"layers.{L}.mlp.expert_mlps.mlp_op.down_proj.weight",
    }
    tensor_keys = {k for k in converted if not k.startswith("_")}
    missing = expected_moe_keys - tensor_keys
    extra = tensor_keys - expected_moe_keys
    assert not missing, sorted(missing)
    assert not extra, sorted(extra)

    # ---- shape assertions ----
    E = src.n_routed_experts
    H = src.hidden_size
    I = src.moe_intermediate_size
    gate_up = converted[f"layers.{L}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
    down = converted[f"layers.{L}.mlp.expert_mlps.mlp_op.down_proj.weight"]
    assert tuple(gate_up.shape) == (E, H, 2 * I), tuple(gate_up.shape)
    assert tuple(down.shape) == (E, I, H), tuple(down.shape)
    assert gate_up.dtype == src.torch_dtype, gate_up.dtype
    assert down.dtype == src.torch_dtype, down.dtype

    # ---- shared-expert shapes (FP8 dequant path) ----
    sg = converted[f"layers.{L}.mlp.shared_expert.gate_proj.weight"]
    su = converted[f"layers.{L}.mlp.shared_expert.up_proj.weight"]
    sd_ = converted[f"layers.{L}.mlp.shared_expert.down_proj.weight"]
    assert tuple(sg.shape) == (I, H)
    assert tuple(su.shape) == (I, H)
    assert tuple(sd_.shape) == (H, I)

    # ---- per-expert degeneracy check on the stacked tensors ----
    # A silent all-zero row (unwritten mmap page / truncated dequant)
    # would slip past the shape check but poison the routed activation
    # of that expert.  For CPU tractability at E=256 we do the check in
    # two passes:
    #   (a) Cheap torch-native scan over every expert slice: max_abs > 0
    #       AND std > 1e-5.  A hit here is decisive: the tensor is
    #       degenerate.  This handles the actual failure modes the
    #       degeneracy guard catches (all-zero, constant, unwritten).
    #   (b) Full require_comparable on ONE random-per-run expert to
    #       exercise the campaign-standard guard end-to-end.
    # The two-pass structure keeps 256-expert × 2-tensor validation
    # under a few seconds instead of ~10 min on CPU.
    def _cheap_degeneracy(t: torch.Tensor) -> tuple[float, float]:
        """Return (max_abs, std) as fp32 python floats."""
        f = t.detach().to(torch.float32)
        return float(f.abs().max().item()), float(f.std().item())

    n_gu_pass = 0
    n_gu_fail = 0
    n_d_pass = 0
    n_d_fail = 0
    for e in range(E):
        gu_max, gu_std = _cheap_degeneracy(gate_up[e])
        d_max, d_std = _cheap_degeneracy(down[e])
        if gu_max <= 0.0 or gu_std <= 1e-5:
            n_gu_fail += 1
            raise AssertionError(
                f"gate_up_proj expert {e} degenerate: max_abs={gu_max}, std={gu_std}"
            )
        n_gu_pass += 1
        if d_max <= 0.0 or d_std <= 1e-5:
            n_d_fail += 1
            raise AssertionError(
                f"down_proj expert {e} degenerate: max_abs={d_max}, std={d_std}"
            )
        n_d_pass += 1
    assert n_gu_fail == 0
    assert n_d_fail == 0
    assert n_gu_pass == E
    assert n_d_pass == E
    # End-to-end guard against the campaign standard on the last expert
    # (any expert would do; last is deterministic and covers a "final
    # write" style bug where earlier tiles are fine but the last is
    # truncated).
    require_comparable(
        gate_up[E - 1].detach().to(torch.float32).cpu().numpy(),
        f"gate_up_proj[{E - 1}]_full_guard",
    )
    require_comparable(
        down[E - 1].detach().to(torch.float32).cpu().numpy(),
        f"down_proj[{E - 1}]_full_guard",
    )

    # ---- router weight / correction bias sanity ----
    router = converted[f"layers.{L}.mlp.router.weight"]
    assert router.dtype == torch.float32
    assert tuple(router.shape) == (E, H)
    corr = converted[f"layers.{L}.mlp.e_score_correction_bias"]
    assert corr.dtype == torch.float32
    assert tuple(corr.shape) == (E,)

    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-routed-moe-1layer-synth",
                "layer_idx": L,
                "wrapper_tree_key_count_moe_block": len(expected_moe_keys),
                "per_expert_gate_up_pass": n_gu_pass,
                "per_expert_gate_up_fail": n_gu_fail,
                "per_expert_down_pass": n_d_pass,
                "per_expert_down_fail": n_d_fail,
                "gate_up_proj_shape": tuple(gate_up.shape),
                "down_proj_shape": tuple(down.shape),
                "conversion_report": report,
            },
            indent=2,
        )
    )


def test_convert_refuses_hash_moe_layer() -> None:
    """The routed-MoE converter must refuse a hash-MoE layer index —
    layers 0..num_hash_layers-1 need the (unimplemented) hash-router
    path, not top-k."""
    cfg, conv, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    with pytest.raises(ValueError, match="hash-MoE"):
        conv._convert_routed_moe_layer({}, {}, 0, src)


def test_real_hf_expert0_w2_bit_exact_via_helper() -> None:
    """If the FP4 test's real HF cache exists, route ``layers.3.ffn.
    experts.0.w2`` through ``_dequant_expert_fp4_weight`` and confirm
    the bf16 result matches direct ``dequantize_block_fp4_ue8m0``.

    Skips (not fails) if the cache is missing AND the network fetch
    cannot be attempted — the byte-exact real-tensor path is already
    gated by ``test_fp4_dequant_1tensor``.
    """
    _cfg, conv, _ = _import_library()
    if not _EXPERT_CACHE.exists():
        pytest.skip(
            f"real HF expert cache absent at {_EXPERT_CACHE!s}; the "
            "FP4-dequant test primes it — run that first to enable this "
            "path"
        )
    from safetensors.torch import load_file

    store = load_file(str(_EXPERT_CACHE))
    weight = store["w2_weight"]
    scale = store["w2_scale"]
    key = "layers.3.ffn.experts.0.w2.weight"
    # DSv4 scale naming: sibling `.scale`, NOT `.weight.scale`.
    sd = {key: weight, conv._dsv4_scale_key_for(key): scale}
    via_helper = conv._dequant_expert_fp4_weight(sd, key, torch.bfloat16)
    direct = conv.dequantize_block_fp4_ue8m0(
        weight, scale, conv.DSV4_FP4_BLOCK_SIZE, torch.bfloat16
    )
    assert via_helper.shape == direct.shape
    assert torch.equal(via_helper, direct)
    require_comparable(
        via_helper.detach().to(torch.float32).cpu().numpy(), "helper_out"
    )
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-real-expert0-w2-via-helper",
                "hf_key": key,
                "shape": tuple(via_helper.shape),
                "max_abs_error_bf16_vs_direct": 0.0,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _standalone_main() -> int:
    tests = [
        test_router_topk_shape_and_range,
        test_router_matches_hf_reference,
        test_router_refuses_wrong_scoring_func,
        test_hf_index_catalog_layer3_routed_moe,
        test_synth_layer3_convert_and_stack_shapes,
        test_convert_refuses_hash_moe_layer,
        test_real_hf_expert0_w2_bit_exact_via_helper,
    ]
    n_pass = 0
    n_skip = 0
    n_fail = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except pytest.skip.Exception as skip_exc:
            n_skip += 1
            print(f"SKIP  {name}: {skip_exc}")
            continue
        except Exception as exc:
            n_fail += 1
            print(f"FAIL  {name}: {exc!r}")
            import traceback

            traceback.print_exc()
            continue
        n_pass += 1
        print(f"PASS  {name}", flush=True)
    print(
        json.dumps(
            {
                "suite": "dsv4-flash.tests.test_routed_moe_1layer",
                "pass": n_pass,
                "skip": n_skip,
                "fail": n_fail,
            },
            indent=2,
        )
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover — local invocation only
    sys.exit(_standalone_main())
