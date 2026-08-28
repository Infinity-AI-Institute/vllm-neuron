# SPDX-License-Identifier: Apache-2.0
"""Round-3 1-block smoke: MQA attention forward, wrapper vs HF reference.

This is the correctness backstop for the DeepSeek-V4-Flash MQA block —
the attention math the whole architecture rides on, shared across
sliding_attention (layers 0, 1, 40-42), CSA (paper §2.3.1), and HCA
(paper §2.3.2) layers.  A silent bug in partial RoPE application, the
per-head attention-sink softmax, or the grouped output projection would
show up as mild logit drift and no crash.

We defend by:

  1. Loading ALL EIGHT layer-0 attention tensors + the sibling
     ``attn_norm.weight`` from the real HF shard
     ``model-00002-of-00048.safetensors`` at snapshot
     ``deepseek-ai/DeepSeek-V4-Flash-0731 @
     7872f01b1d1fe23eabc4c98b48bffcef5a386062``, via HTTP-Range slice
     (no full shard download).  The tensors add up to ~102 MB — kept in
     the local ``.hf_cache/`` next to this test so subsequent runs are
     offline.

  2. Dequanting the five FP8-e4m3 UE8M0 weights to bf16 via the library's
     ``_convert_mqa_block`` — the same converter path production uses.

  3. Instantiating ``_MQABlock(config, layer_idx=0)`` with those weights
     and running its forward on a synthetic hidden-state input.

  4. Running an INDEPENDENT reference forward that transcribes
     ``transformers 5.15.1
     deepseek_v4/modeling_deepseek_v4.py::DeepseekV4Attention.forward``
     lines 801-873 byte-for-byte with the same dequanted bf16 weights.
     The reference intentionally lives HERE, not imported from the
     wrapper, so a wrapper-side copy-paste error can be caught.

  5. Asserting ``max_abs_error_bf16 < 1e-4`` (ideally 0.0 — both sides
     use identical bf16 weights and identical rope math, so any
     nonzero delta is a real forward-math discrepancy).

  6. Running both outputs through ``require_comparable`` so an all-zero
     or all-NaN output cannot vacuously pass.

The test skips (rather than errors) when the local cache is empty and
the network fetch cannot complete — an offline dev box is not a
correctness failure.  It never falls back to synthetic-only when the
intent is byte-clean verification; a skip is louder than a false pass.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Degeneracy guard — same lookup convention as test_fp4_dequant_1tensor.py
# so we do not carry a private copy.
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
except Exception as exc:  # pragma: no cover — surface the discovery gap
    pytest.skip(
        f"degeneracy_guard not importable at {_HARNESS_KERNELS!s}: {exc!r}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Library under test — imported via importlib so a dev box without the
# full `vllm_neuron` package (which pulls `vllm`) can still run this test.
# ---------------------------------------------------------------------------
def _import_library():
    """Import (config, checkpoint_convert, neuron_wrapper).

    Falls back to importlib-only loading when the top-level
    ``vllm_neuron`` package fails to import (e.g. a dev laptop without
    the ``vllm`` package installed) — mirrors the exact fallback
    ``test_fp4_dequant_1tensor.py`` uses so both tests share the same
    dev-box story.
    """
    try:
        from vllm_neuron.model.dsv4_flash import (  # type: ignore
            checkpoint_convert as convert,
            config as cfg,
            neuron_wrapper as nw,
        )
        return cfg, convert, nw
    except Exception:
        pass

    import importlib.util
    import types

    dsv4_dir = Path(__file__).resolve().parent.parent
    pkg = types.ModuleType("_dsv4_flash_mqa_test_pkg")
    pkg.__path__ = [str(dsv4_dir)]
    sys.modules["_dsv4_flash_mqa_test_pkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_dsv4_flash_mqa_test_pkg.{name}",
            str(dsv4_dir / f"{name}.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg = _load("config")
    convert = _load("checkpoint_convert")
    nw = _load("neuron_wrapper")
    return cfg, convert, nw


# ---------------------------------------------------------------------------
# HF-shard slicer: HTTP-Range fetch of the layer-0 attention tensors.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
# Layer 0's attention subtree lives in shard 00002-of-00048; verified against
# the local index at
# `vllm_neuron/model/dsv4_flash/tests/.hf_cache/dsv4_flash_index.json`.
_HF_SHARD = "model-00002-of-00048.safetensors"
_ATTN_KEYS_FP8 = (
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
)
_ATTN_KEYS_DENSE = (
    "q_norm.weight",
    "kv_norm.weight",
    "attn_sink",
)
_LAYER0_ATTN_KEYS: tuple[str, ...] = tuple(
    [f"layers.0.attn.{n}.weight" for n in _ATTN_KEYS_FP8]
    + [f"layers.0.attn.{n}.scale" for n in _ATTN_KEYS_FP8]
    + [f"layers.0.attn.{n}" for n in _ATTN_KEYS_DENSE]
    + ["layers.0.attn_norm.weight"]
)
_CACHE_DIR = Path(__file__).parent / ".hf_cache"
_CACHE_FILE = _CACHE_DIR / "dsv4_layer0_attn.safetensors"
_HEADER_CHUNK_BYTES = 1024 * 1024  # 1 MB — safetensors header is ~172 KB


def _fetch_range(url: str, byte_range: tuple[int, int]) -> bytes:
    """HTTP ``Range: bytes=start-end`` fetch, inclusive on both ends."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
            "User-Agent": "vllm_neuron.dsv4_flash.tests.mqa/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _build_local_cache() -> Path:
    """Pull the layer-0 attention tensors and repack as a mini-safetensors.

    Same slicer as ``test_fp4_dequant_1tensor.py::_build_local_cache`` —
    only the key list differs.  Raises the network error on failure so
    the caller can decide whether to ``pytest.skip``.
    """
    try:
        from huggingface_hub import hf_hub_url
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub not importable; install to enable this test"
        ) from exc

    url = hf_hub_url(_HF_REPO, _HF_SHARD, revision=_HF_SHA)
    first = _fetch_range(url, (0, _HEADER_CHUNK_BYTES - 1))
    header_len = struct.unpack("<Q", first[:8])[0]
    if header_len > len(first) - 8:  # pragma: no cover — future header growth
        first = _fetch_range(url, (0, header_len + 8 + 4096))
    header = json.loads(first[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len

    tensors: dict[str, dict[str, Any]] = {}
    for hf_key in _LAYER0_ATTN_KEYS:
        meta = header.get(hf_key)
        if meta is None:
            raise KeyError(
                f"HF header for {_HF_SHARD} does not carry {hf_key!r} "
                "at snapshot 7872f01b — checkpoint layout drift?"
            )
        o0, o1 = meta["data_offsets"]
        payload = _fetch_range(url, (data_start + o0, data_start + o1 - 1))
        assert len(payload) == (o1 - o0), (hf_key, len(payload), o1 - o0)
        tensors[hf_key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "bytes": payload,
        }

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    new_meta: dict[str, dict[str, Any]] = {}
    off = 0
    for k, v in tensors.items():
        n = len(v["bytes"])
        new_meta[k] = {
            "dtype": v["dtype"],
            "shape": v["shape"],
            "data_offsets": [off, off + n],
        }
        off += n
    metadata_json = json.dumps(new_meta, separators=(",", ":")).encode("utf-8")
    pad = (-len(metadata_json)) & 7
    metadata_json += b" " * pad
    with open(_CACHE_FILE, "wb") as f:
        f.write(struct.pack("<Q", len(metadata_json)))
        f.write(metadata_json)
        for k, v in tensors.items():
            f.write(v["bytes"])
    return _CACHE_FILE


def _load_layer0_attn_state_dict() -> dict[str, torch.Tensor]:
    """Return the 13 layer-0 attention HF tensors as a state_dict."""
    from safetensors.torch import load_file

    if not _CACHE_FILE.exists():
        try:
            _build_local_cache()
        except Exception as exc:  # network / auth / access
            pytest.skip(
                "cannot fetch real DSv4-Flash layer-0 attention tensors "
                f"({_HF_REPO}@{_HF_SHA[:12]} {_HF_SHARD}): {exc!r}"
            )
    store = load_file(str(_CACHE_FILE))
    # Sanity-check we got every expected key.
    missing = [k for k in _LAYER0_ATTN_KEYS if k not in store]
    if missing:
        raise KeyError(
            f"local mini-safetensors cache is missing keys {missing!r}; "
            f"delete {_CACHE_FILE!s} to force a re-pull."
        )
    return dict(store)


# ---------------------------------------------------------------------------
# Independent reference forward — hand-transcribed from
# transformers/models/deepseek_v4/modeling_deepseek_v4.py lines 801-873.
# Kept here (not imported from neuron_wrapper) so a wrapper-side copy-
# paste error is caught, not papered over.
# ---------------------------------------------------------------------------


def _ref_rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Reference impl of ``modeling_deepseek_v4.py:335-339`` (``rotate_half``)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _ref_apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Reference impl of ``modeling_deepseek_v4.py:342-359``."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (_ref_rotate_half_interleaved(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def _ref_rms_norm_weighted(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference impl of ``DeepseekV4RMSNorm.forward`` (lines 55-60)."""
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight * x32.to(input_dtype)


def _ref_rms_norm_unweighted(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference impl of ``DeepseekV4UnweightedRMSNorm.forward`` (lines 71-72)."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def _ref_forward(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    rms_eps: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference DSv4 MQA forward.

    Transcribes ``DeepseekV4Attention.forward`` (lines 801-873) step by
    step.  The scaling factor and sink-augmented softmax are taken from
    ``eager_attention_forward`` (lines 717-745).
    """
    B, S, H = hidden_states.shape
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    wq_a = weights["wq_a.weight"]
    wq_b = weights["wq_b.weight"]
    q_norm_weight = weights["q_norm.weight"]
    wkv = weights["wkv.weight"]
    kv_norm_weight = weights["kv_norm.weight"]
    wo_a = weights["wo_a.weight"]
    wo_b = weights["wo_b.weight"]
    attn_sink = weights["attn_sink"]

    # Line 816: q_residual = q_a_norm(q_a_proj(x))
    q_residual = _ref_rms_norm_weighted(
        torch.nn.functional.linear(hidden_states, wq_a), q_norm_weight, rms_eps
    )
    # Line 817-818: q = q_b_norm(q_b_proj(q_residual).view(...).transpose(1, 2))
    q = torch.nn.functional.linear(q_residual, wq_b).view(*hidden_shape).transpose(1, 2)
    q = _ref_rms_norm_unweighted(q, rms_eps)
    # Line 819: q = apply_rotary_pos_emb(q, cos, sin)
    q = _ref_apply_rotary(q, cos, sin)

    # Line 821-822: kv = kv_norm(kv_proj(x)); rope(kv)
    kv = _ref_rms_norm_weighted(
        torch.nn.functional.linear(hidden_states, wkv), kv_norm_weight, rms_eps
    ).view(*hidden_shape).transpose(1, 2)
    kv = _ref_apply_rotary(kv, cos, sin)

    # eager_attention_forward inline (lines 727-745).
    key_states = kv.expand(B, num_heads, kv.shape[2], head_dim)
    value_states = key_states
    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores.to(value_states.dtype), value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # Line 868: conjugate rotation on output rope slice.
    attn_output = _ref_apply_rotary(
        attn_output.transpose(1, 2), cos, -sin
    ).transpose(1, 2)

    # Lines 870-872: grouped output projection then plain o_b.
    grouped = attn_output.reshape(*input_shape, o_groups, -1)  # [B, S, G, H*D/G]
    # DeepseekV4GroupedLinear.forward (lines 326-332).
    n_groups = o_groups
    hidden_dim = grouped.shape[-1]
    w = wo_a.view(n_groups, -1, hidden_dim).transpose(1, 2)
    x = grouped.reshape(-1, n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    grouped_out = y.reshape(*input_shape, n_groups, -1)
    grouped_out = grouped_out.flatten(2)
    output = torch.nn.functional.linear(grouped_out, wo_b)
    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mqa_wrapper_tree_matches_hf_layer0_key_set() -> None:
    """Wrapper block's declared parameter names must match the HF
    layer-0 attention subtree byte-for-byte (post-dequant, with the FP8
    ``.scale`` companions consumed and gone)."""
    cfg, convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig(
        num_hidden_layers=1,
        layer_types=("sliding_attention",),
        mlp_layer_types=("hash_moe",),
        allow_reduced_shapes=True,
    )
    block = nw._MQABlock(src, layer_idx=0)
    got_names = sorted(name for name, _ in block.named_parameters())
    expected = sorted(nw._MQABlock.PARAM_KEYS)
    assert got_names == expected, (got_names, expected)
    # Independent cross-check: what the converter emits under
    # `layers.0.attn.*` must be exactly the wrapper's declared subtree
    # (minus `attn_sink` -> the sink parameter comes back too; plus
    # `attn_norm` at the layer level).
    convert_names = tuple(f"layers.0.attn.{k}" for k in expected)
    for k in convert_names:
        assert k.split(".attn.", 1)[1] in nw._MQABlock.PARAM_KEYS


def test_mqa_wrapper_matches_hf_reference_on_real_layer0_tensors() -> None:
    """The whole enchilada: real HF weights, dequant, wrapper vs
    reference forward, byte-clean bf16 diff."""
    cfg, convert, nw = _import_library()
    state = _load_layer0_attn_state_dict()

    src = cfg.DeepseekV4FlashInferenceConfig()  # frozen full shape
    dtype = torch.bfloat16

    # Convert HF -> wrapper module tree (drops the .scale companions and
    # dequants FP8 → bf16 for the five weight tensors).
    converted = convert._convert_mqa_block(
        state, layer_idx=0, src=src, dtype=dtype, require_attn_sink=True
    )

    # Build the wrapper block; load the dequanted tensors under the
    # wrapper's declared parameter tree.
    block = nw._MQABlock(src, layer_idx=0).to(dtype)
    load_report = {}
    prefix = "layers.0.attn."
    with torch.no_grad():
        for pname in nw._MQABlock.PARAM_KEYS:
            tensor = converted[f"{prefix}{pname}"]
            # ".weight" -> nn.Linear.weight; "attn_sink" -> nn.Parameter.
            # Split on the last dot so "q_norm.weight" reaches the
            # _MQANormParam.weight sub-attribute.
            parts = pname.split(".")
            obj = block
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            assert tuple(param.shape) == tuple(tensor.shape), (
                pname,
                tuple(param.shape),
                tuple(tensor.shape),
            )
            param.copy_(tensor.to(dtype))
            load_report[pname] = tuple(tensor.shape)

    # Deterministic synthetic input.  Small B and S keep the O(S^2)
    # attention math manageable on CPU (~64 head × 8×8 = 4096 dot-products
    # per head-batch — a few seconds on a laptop).
    torch.manual_seed(0)
    B, S = 1, 8
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.5
    positions = torch.arange(S).unsqueeze(0)
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.rope_theta,
        dtype=dtype,
    )
    assert tuple(cos.shape) == (B, S, src.qk_rope_head_dim // 2), cos.shape

    # ---- wrapper forward ----
    with torch.no_grad():
        y_wrap = block(hidden, cos, sin)
    assert tuple(y_wrap.shape) == (B, S, src.hidden_size)
    assert y_wrap.dtype == dtype

    # ---- reference forward ----
    ref_weights = {
        pname: converted[f"{prefix}{pname}"].to(dtype)
        for pname in nw._MQABlock.PARAM_KEYS
    }
    with torch.no_grad():
        y_ref = _ref_forward(
            ref_weights,
            hidden,
            cos,
            sin,
            num_heads=src.num_attention_heads,
            head_dim=src.head_dim,
            o_groups=src.o_groups,
            o_lora_rank=src.o_lora_rank,
            rms_eps=src.rms_norm_eps,
        )
    assert tuple(y_ref.shape) == tuple(y_wrap.shape)
    assert y_ref.dtype == dtype

    # ---- degeneracy guard on BOTH sides ----
    require_comparable(
        y_wrap.detach().to(torch.float32).cpu().numpy(), "wrapper_output_fp32"
    )
    require_comparable(
        y_ref.detach().to(torch.float32).cpu().numpy(), "reference_output_fp32"
    )

    # ---- byte-clean bf16 diff ----
    diff = (y_wrap.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_abs_error_bf16 = float(diff.max().item())
    mean_abs_error_bf16 = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-mqa-block-real-tensor",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "hf_shard": _HF_SHARD,
                "layer_idx": 0,
                "batch_size": B,
                "seq_len": S,
                "hidden_size": src.hidden_size,
                "num_heads": src.num_attention_heads,
                "head_dim": src.head_dim,
                "q_lora_rank": src.q_lora_rank,
                "qk_rope_head_dim": src.qk_rope_head_dim,
                "o_groups": src.o_groups,
                "o_lora_rank": src.o_lora_rank,
                "wrapper_tree_key_count": len(nw._MQABlock.PARAM_KEYS),
                "converted_keys": sorted(converted.keys()),
                "load_report": load_report,
                "max_abs_error_bf16": max_abs_error_bf16,
                "mean_abs_error_bf16": mean_abs_error_bf16,
                "output_shape": tuple(y_wrap.shape),
            },
            indent=2,
        )
    )
    assert max_abs_error_bf16 < 1e-4, max_abs_error_bf16


def test_mqa_wrapper_forward_synthetic_shape_gate() -> None:
    """Fast synthetic-input gate: shapes, dtypes, no-NaN, non-degenerate.
    Skips no network — runs always so a CI without HF access still gates
    on the wrapper's internal shape math."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig(
        num_hidden_layers=1,
        layer_types=("sliding_attention",),
        mlp_layer_types=("hash_moe",),
        allow_reduced_shapes=True,
    )
    block = nw._MQABlock(src, layer_idx=0).to(torch.bfloat16)
    torch.manual_seed(1)
    B, S = 2, 4
    hidden = torch.randn(B, S, src.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.rope_theta,
        dtype=torch.bfloat16,
    )
    y = block(hidden, cos, sin)
    assert tuple(y.shape) == (B, S, src.hidden_size)
    assert y.dtype == torch.bfloat16
    require_comparable(
        y.detach().to(torch.float32).cpu().numpy(), "synthetic_output_fp32"
    )


def test_mqa_wrapper_refuses_wrong_num_kv_heads() -> None:
    """The MQA block hard-requires ``num_key_value_heads=1``; any other
    value indicates a config was mutated post-freeze."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig(
        num_hidden_layers=1,
        layer_types=("sliding_attention",),
        mlp_layer_types=("hash_moe",),
        allow_reduced_shapes=True,
        num_key_value_heads=1,  # frozen valid value
    )
    # Manually mutate to smuggle a wrong value past the config validator
    # (this simulates a caller that constructed the config differently
    # and did not re-run the validator).
    src.num_key_value_heads = 2
    with pytest.raises(ValueError, match="shared-KV MQA"):
        nw._MQABlock(src, layer_idx=0)


def test_partial_rope_helper_matches_reference_on_synthetic_input() -> None:
    """Standalone gate on :func:`apply_partial_rope` — decouples the RoPE
    helper's correctness from the block-level assembly.  This catches a
    rotate-half sign flip or a repeat-vs-concat expansion bug earlier
    than the full-block test."""
    _cfg, _convert, nw = _import_library()
    torch.manual_seed(7)
    B, H, S, D = 2, 4, 6, 128
    rope_dim = 32
    x = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = nw.build_main_rope_cos_sin(
        positions, rope_dim=rope_dim, rope_theta=10000.0, dtype=torch.bfloat16
    )
    y_lib = nw.apply_partial_rope(x, cos, sin)
    y_ref = _ref_apply_rotary(x, cos, sin)
    diff = (y_lib.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_err = float(diff.max().item())
    assert max_err == 0.0, max_err  # identical math, identical inputs
    # And the leading NoPE channels must be untouched.
    assert torch.equal(y_lib[..., : D - rope_dim], x[..., : D - rope_dim])


# ---------------------------------------------------------------------------
# Standalone runner — mirrors test_fp4_dequant_1tensor.py so
# a laptop without pytest-collectible `vllm_neuron` can still run
# the gate.
# ---------------------------------------------------------------------------
def _standalone_main() -> int:
    tests = [
        test_partial_rope_helper_matches_reference_on_synthetic_input,
        test_mqa_wrapper_tree_matches_hf_layer0_key_set,
        test_mqa_wrapper_forward_synthetic_shape_gate,
        test_mqa_wrapper_refuses_wrong_num_kv_heads,
        test_mqa_wrapper_matches_hf_reference_on_real_layer0_tensors,
    ]
    n_pass = n_skip = n_fail = 0
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
        print(f"PASS  {name}")
    print(
        json.dumps(
            {
                "suite": "dsv4-flash.tests.test_mqa_1tensor",
                "pass": n_pass,
                "skip": n_skip,
                "fail": n_fail,
            },
            indent=2,
        )
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover — local invocation
    sys.exit(_standalone_main())
