# SPDX-License-Identifier: Apache-2.0
"""Round-6 1-block smoke: sliding-only attention forward, wrapper vs HF.

Correctness backstop for the DSv4-Flash sliding-only attention family —
the bootstrap-layer path (layers 0 and 1) and the base
:class:`_CSABlock` / :class:`_HCABlock` compose on top of.

Discipline:
  1. Load ALL 9 layer-0 attention tensors (8 MQA params + sibling
     ``attn_norm.weight``) from the real HF shard
     ``model-00002-of-00048.safetensors`` at snapshot
     ``deepseek-ai/DeepSeek-V4-Flash-0731 @
     7872f01b1d1fe23eabc4c98b48bffcef5a386062`` via HTTP-Range slice.
     Layer 0 is the first sliding-only layer of the frozen schedule.
     Shared cache with ``test_mqa_1tensor.py``.
  2. Dequant the five FP8-e4m3 UE8M0 attention weights via
     ``_convert_sliding_only_block``.
  3. Instantiate ``_SlidingOnlyAttentionBlock(config, layer_idx=0)``,
     run forward on ``[B=1, S=200, hidden]`` — S=200 > sliding_window
     (128) so the mask clips something on late queries.
  4. Independent reference forward transcribed HERE from
     ``modeling_deepseek_v4.py:801-873`` (sliding branch) +
     ``eager_attention_forward:717-745`` +
     ``masking_utils.py:{80, 92-101, 138}`` — kept out of the wrapper
     so a copy-paste error can be caught.
  5. Assert ``max_abs_error_bf16 < 1e-4`` (target 0.0).
  6. ``require_comparable`` on both sides.
  7. Assert the sliding-window mask actually applies: exhaustive
     predicate compare against HF's rule over the full S x S grid, +
     a real-forward cross-check where the wrapper's windowed output
     MUST differ from a full-causal output on late queries and MUST
     match on early queries (SUFFICIENT-CONDITION cross-check).

Skips (rather than errors) when the local cache is empty and the
network fetch cannot complete.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Degeneracy guard — same lookup convention as the sibling tests.
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
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"degeneracy_guard not importable at {_HARNESS_KERNELS!s}: {exc!r}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Library under test — importlib fallback so a dev box without the full
# ``vllm_neuron`` package (which pulls ``vllm``) still runs the test.
# ---------------------------------------------------------------------------
def _import_library():
    try:
        from vllm_neuron.model.dsv4_flash import (  # type: ignore
            checkpoint_convert as convert,
        )
        from vllm_neuron.model.dsv4_flash import (
            config as cfg,
        )
        from vllm_neuron.model.dsv4_flash import (
            neuron_wrapper as nw,
        )

        return cfg, convert, nw
    except Exception:
        pass

    import importlib.util
    import types

    dsv4_dir = Path(__file__).resolve().parent.parent
    pkg = types.ModuleType("_dsv4_flash_slidingonly_test_pkg")
    pkg.__path__ = [str(dsv4_dir)]
    sys.modules["_dsv4_flash_slidingonly_test_pkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_dsv4_flash_slidingonly_test_pkg.{name}",
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
# HF-shard slicer for the layer-0 sliding-only attention subtree.
# Shares the cache path with test_mqa_1tensor.py.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
_HF_SHARD = "model-00002-of-00048.safetensors"
_LAYER_IDX = 0

_ATTN_KEYS_FP8 = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
_ATTN_KEYS_DENSE = ("q_norm.weight", "kv_norm.weight", "attn_sink")

_LAYER0_ATTN_KEYS: tuple[str, ...] = tuple(
    [f"layers.{_LAYER_IDX}.attn.{n}.weight" for n in _ATTN_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.{n}.scale" for n in _ATTN_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.{n}" for n in _ATTN_KEYS_DENSE]
    + [f"layers.{_LAYER_IDX}.attn_norm.weight"]
)
_CACHE_DIR = Path(__file__).parent / ".hf_cache"
_CACHE_FILE = _CACHE_DIR / "dsv4_layer0_attn.safetensors"
_HEADER_CHUNK_BYTES = 1024 * 1024


def _fetch_range(url: str, byte_range: tuple[int, int]) -> bytes:
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
            "User-Agent": "vllm_neuron.dsv4_flash.tests.slidingonly/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _build_local_cache() -> Path:
    try:
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(_HF_REPO, _HF_SHARD, revision=_HF_SHA)
    except Exception:
        url = f"https://huggingface.co/{_HF_REPO}/resolve/{_HF_SHA}/{_HF_SHARD}"

    first = _fetch_range(url, (0, _HEADER_CHUNK_BYTES - 1))
    header_len = struct.unpack("<Q", first[:8])[0]
    if header_len > len(first) - 8:  # pragma: no cover
        first = _fetch_range(url, (0, header_len + 8 + 4096))
    header = json.loads(first[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len

    tensors: dict[str, dict[str, Any]] = {}
    for hf_key in _LAYER0_ATTN_KEYS:
        meta = header.get(hf_key)
        if meta is None:
            raise KeyError(
                f"HF header for {_HF_SHARD} does not carry {hf_key!r} "
                f"at snapshot {_HF_SHA[:12]} — checkpoint layout drift?"
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
    from safetensors.torch import load_file

    if not _CACHE_FILE.exists():
        try:
            _build_local_cache()
        except Exception as exc:
            pytest.skip(
                "cannot fetch real DSv4-Flash layer-0 attention tensors "
                f"({_HF_REPO}@{_HF_SHA[:12]} {_HF_SHARD}): {exc!r}"
            )
    store = load_file(str(_CACHE_FILE))
    missing = [k for k in _LAYER0_ATTN_KEYS if k not in store]
    if missing:
        raise KeyError(
            f"local mini-safetensors cache is missing keys {missing!r}; "
            f"delete {_CACHE_FILE!s} to force a re-pull."
        )
    return dict(store)


# ---------------------------------------------------------------------------
# Independent reference forward — hand-transcribed from
# modeling_deepseek_v4.py DeepseekV4Attention.forward (sliding branch) +
# eager_attention_forward + masking_utils sliding+causal predicate.
# Kept here (not imported) so a wrapper-side copy-paste error is caught.
# ---------------------------------------------------------------------------


def _ref_rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _ref_apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (
        (rope.float() * cos) + (_ref_rotate_half_interleaved(rope).float() * sin)
    ).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def _ref_rms_norm_weighted(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight * x32.to(x.dtype)


def _ref_rms_norm_unweighted(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def _ref_build_sliding_causal_mask(
    position_ids: torch.Tensor,
    *,
    sliding_window: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference sliding + causal predicate — matches HF's masking_utils
    ``sliding_window_causal_mask_function`` byte-for-byte.
    """
    q_pos = position_ids.unsqueeze(-1)
    kv_pos = position_ids.unsqueeze(-2)
    visible = (kv_pos <= q_pos) & (kv_pos > (q_pos - int(sliding_window)))
    neg_inf = torch.full((), float("-inf"), dtype=dtype)
    zero = torch.zeros((), dtype=dtype)
    mask = torch.where(visible, zero, neg_inf)
    return mask.unsqueeze(1)


def _ref_sliding_only_forward(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    rms_eps: float,
    sliding_window: int,
    position_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference sliding-only forward — transcribes HF exactly."""
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

    q_residual = _ref_rms_norm_weighted(
        torch.nn.functional.linear(hidden_states, wq_a), q_norm_weight, rms_eps
    )
    q = torch.nn.functional.linear(q_residual, wq_b).view(*hidden_shape).transpose(1, 2)
    q = _ref_rms_norm_unweighted(q, rms_eps)
    q = _ref_apply_rotary(q, cos, sin)

    kv = (
        _ref_rms_norm_weighted(
            torch.nn.functional.linear(hidden_states, wkv), kv_norm_weight, rms_eps
        )
        .view(*hidden_shape)
        .transpose(1, 2)
    )
    kv = _ref_apply_rotary(kv, cos, sin)

    sliding_mask = _ref_build_sliding_causal_mask(
        position_ids,
        sliding_window=sliding_window,
        dtype=hidden_states.dtype,
    )

    scaling = head_dim**-0.5
    key_states = kv.expand(B, num_heads, kv.shape[2], head_dim)
    value_states = key_states
    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + sliding_mask
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(
        combined_logits, dim=-1, dtype=combined_logits.dtype
    )
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores.to(value_states.dtype), value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = _ref_apply_rotary(attn_output.transpose(1, 2), cos, -sin).transpose(
        1, 2
    )

    grouped = attn_output.reshape(*input_shape, o_groups, -1)
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


def test_sliding_layer_schedule_confirms_layer_0() -> None:
    """Layer 0 must be the first sliding-only layer in the frozen schedule."""
    cfg, _convert, _nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    assert src.layer_types[0] == "sliding_attention"
    assert src.layer_types[1] == "sliding_attention"
    assert src.layer_types[2] == "compressed_sparse_attention"
    assert src.compress_ratios[0] == 0
    assert src.compress_ratios[1] == 0
    assert src.compress_ratios[2] == 4
    assert src.sliding_window == 128
    assert src.rope_theta == 10000.0
    assert src.compress_rope_theta == 160000.0


def test_sliding_only_wrapper_tree_key_set() -> None:
    """The _SlidingOnlyAttentionBlock's parameter tree must be exactly the
    8 MQA params under ``mqa.*`` — no compressor, no indexer.  The sibling
    ``attn_norm.weight`` at the decoder-layer level makes it 9 keys at the
    layer level, but is NOT owned by this block (same convention as
    _HCABlock / _CSABlock).
    """
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    block = nw._SlidingOnlyAttentionBlock(src, layer_idx=_LAYER_IDX)
    names = sorted(name for name, _ in block.named_parameters())
    expected = sorted(f"mqa.{k}" for k in nw._MQABlock.PARAM_KEYS)
    assert names == expected, (names, expected)
    assert len(nw._MQABlock.PARAM_KEYS) == 8
    assert len(names) == 8
    assert block.sliding_window == 128
    assert block.layer_idx == _LAYER_IDX


def test_sliding_only_refuses_wrong_layer_type() -> None:
    """The block hard-refuses instantiation at a non-sliding layer index."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    with pytest.raises(ValueError, match="sliding_attention"):
        nw._SlidingOnlyAttentionBlock(src, layer_idx=2)  # CSA
    with pytest.raises(ValueError, match="sliding_attention"):
        nw._SlidingOnlyAttentionBlock(src, layer_idx=3)  # HCA


def test_sliding_only_converter_refuses_wrong_layer_type() -> None:
    """The converter hard-refuses a non-sliding layer index."""
    cfg, convert, _nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    with pytest.raises(ValueError, match="sliding"):
        convert._convert_sliding_only_block({}, layer_idx=2, src=src)


def test_sliding_window_mask_helper_predicate_matches_hf() -> None:
    """Exhaustive S=200 x S=200 predicate compare — wrapper's helper
    against the HF ``sliding_window_causal_mask_function`` rule.
    """
    _cfg, _convert, nw = _import_library()
    B, S = 1, 200
    sliding_window = 128
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
    mask = nw.build_sliding_window_causal_mask(
        positions, sliding_window=sliding_window, dtype=torch.float32
    )
    assert tuple(mask.shape) == (B, 1, S, S), mask.shape
    assert not torch.isnan(mask).any()

    m = mask[0, 0]
    n_mismatch = 0
    for q_idx in range(S):
        for k_idx in range(S):
            hf_visible = (k_idx <= q_idx) and (k_idx > q_idx - sliding_window)
            got = float(m[q_idx, k_idx].item())
            exp = 0.0 if hf_visible else float("-inf")
            if got != exp:
                n_mismatch += 1
    assert n_mismatch == 0, n_mismatch

    # Spot checks
    assert m[0, 0].item() == 0.0
    assert m[0, 1].item() == float("-inf")
    assert m[100, 0].item() == 0.0
    assert m[100, 100].item() == 0.0
    assert m[100, 101].item() == float("-inf")
    assert m[199, 72].item() == 0.0
    assert m[199, 199].item() == 0.0
    assert m[199, 71].item() == float("-inf")
    assert m[199, 0].item() == float("-inf")


def test_sliding_only_synthetic_shape_gate() -> None:
    """Fast synthetic-input gate: shape math + no NaN + non-degenerate."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    block = nw._SlidingOnlyAttentionBlock(src, layer_idx=_LAYER_IDX).to(dtype)
    torch.manual_seed(3)
    B, S = 1, 200
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.rope_theta,  # MAIN rope (theta=10000)
        dtype=dtype,
    )
    y = block(hidden, cos, sin, positions)
    assert tuple(y.shape) == (B, S, src.hidden_size)
    assert y.dtype == dtype
    require_comparable(
        y.detach().to(torch.float32).cpu().numpy(),
        "sliding_only_synthetic_output_fp32",
    )


def test_sliding_only_wrapper_matches_hf_reference_on_real_layer0_tensors() -> None:
    """Real HF weights, dequant, sliding-only wrapper vs reference forward,
    byte-clean bf16 diff.  S=200 > sliding_window=128 exercises the mask
    on late queries.
    """
    cfg, convert, nw = _import_library()
    state = _load_layer0_attn_state_dict()

    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    sliding_window = 128

    converted = convert._convert_sliding_only_block(
        state, layer_idx=_LAYER_IDX, src=src, dtype=dtype, require_attn_sink=True
    )
    prefix_attn = f"layers.{_LAYER_IDX}.attn."
    expected_converted = set(
        [f"{prefix_attn}{k}" for k in nw._MQABlock.PARAM_KEYS]
        + [f"layers.{_LAYER_IDX}.attn_norm.weight"]
    )
    got_converted = set(converted.keys())
    assert got_converted == expected_converted, sorted(
        got_converted.symmetric_difference(expected_converted)
    )
    assert len(converted) == 9

    block = nw._SlidingOnlyAttentionBlock(src, layer_idx=_LAYER_IDX).to(dtype)
    load_report: dict[str, tuple[int, ...]] = {}
    with torch.no_grad():
        for pname in nw._MQABlock.PARAM_KEYS:
            tensor = converted[f"{prefix_attn}{pname}"]
            parts = pname.split(".")
            obj = block.mqa
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            assert tuple(param.shape) == tuple(tensor.shape), (
                pname,
                tuple(param.shape),
                tuple(tensor.shape),
            )
            param.copy_(tensor.to(dtype))
            load_report[f"mqa.{pname}"] = tuple(tensor.shape)

    torch.manual_seed(0)
    B, S = 1, 200
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)

    # MAIN rope (theta=10000).  A caller that fed the "compress" rope
    # (theta=160000) would silently produce wrong logits — the byte-clean
    # diff would blow through 1e-4.
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.rope_theta,
        dtype=dtype,
    )
    assert tuple(cos.shape) == (B, S, src.qk_rope_head_dim // 2)

    with torch.no_grad():
        y_wrap = block(hidden, cos, sin, positions)
    assert tuple(y_wrap.shape) == (B, S, src.hidden_size)
    assert y_wrap.dtype == dtype

    ref_weights: dict[str, torch.Tensor] = {
        pname: converted[f"{prefix_attn}{pname}"].to(dtype)
        for pname in nw._MQABlock.PARAM_KEYS
    }
    with torch.no_grad():
        y_ref = _ref_sliding_only_forward(
            hidden,
            ref_weights,
            num_heads=src.num_attention_heads,
            head_dim=src.head_dim,
            o_groups=src.o_groups,
            o_lora_rank=src.o_lora_rank,
            rms_eps=src.rms_norm_eps,
            sliding_window=sliding_window,
            position_ids=positions,
            cos=cos,
            sin=sin,
        )
    assert tuple(y_ref.shape) == tuple(y_wrap.shape)
    assert y_ref.dtype == dtype

    require_comparable(
        y_wrap.detach().to(torch.float32).cpu().numpy(),
        "sliding_only_wrapper_output_fp32",
    )
    require_comparable(
        y_ref.detach().to(torch.float32).cpu().numpy(),
        "sliding_only_reference_output_fp32",
    )

    diff = (y_wrap.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_abs_error_bf16 = float(diff.max().item())
    mean_abs_error_bf16 = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-sliding-only-block-real-tensor",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "hf_shard": _HF_SHARD,
                "layer_idx": _LAYER_IDX,
                "layer_type": src.layer_types[_LAYER_IDX],
                "compress_ratio": src.compress_ratios[_LAYER_IDX],
                "sliding_window": sliding_window,
                "batch_size": B,
                "seq_len": S,
                "rope_theta_used": src.rope_theta,
                "hidden_size": src.hidden_size,
                "num_heads": src.num_attention_heads,
                "head_dim": src.head_dim,
                "q_lora_rank": src.q_lora_rank,
                "qk_rope_head_dim": src.qk_rope_head_dim,
                "o_groups": src.o_groups,
                "o_lora_rank": src.o_lora_rank,
                "wrapper_tree_key_count": len(list(block.named_parameters())),
                "converted_key_count": len(converted),
                "load_report_key_count": len(load_report),
                "max_abs_error_bf16": max_abs_error_bf16,
                "mean_abs_error_bf16": mean_abs_error_bf16,
                "output_shape": tuple(y_wrap.shape),
            },
            indent=2,
        )
    )
    assert max_abs_error_bf16 < 1e-4, max_abs_error_bf16


def test_sliding_mask_clips_beyond_window_on_real_forward() -> None:
    """SUFFICIENT-CONDITION cross-check that the mask reaches the attention
    math: on real HF weights, the wrapper's windowed forward MUST
    * agree byte-for-byte with a full-causal reference for early queries
      (q_idx < sliding_window) — no KV clipped;
    * diverge (max_diff > 0) for late queries (q_idx >= sliding_window) —
      past KV is clipped.

    A wrapper that silently drops the sliding mask would agree everywhere.
    """
    cfg, convert, nw = _import_library()
    state = _load_layer0_attn_state_dict()

    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    sliding_window = 128

    converted = convert._convert_sliding_only_block(
        state, layer_idx=_LAYER_IDX, src=src, dtype=dtype, require_attn_sink=True
    )

    block = nw._SlidingOnlyAttentionBlock(src, layer_idx=_LAYER_IDX).to(dtype)
    prefix_attn = f"layers.{_LAYER_IDX}.attn."
    with torch.no_grad():
        for pname in nw._MQABlock.PARAM_KEYS:
            tensor = converted[f"{prefix_attn}{pname}"]
            parts = pname.split(".")
            obj = block.mqa
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            param.copy_(tensor.to(dtype))

    torch.manual_seed(0)
    B, S = 1, 200
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.rope_theta,
        dtype=dtype,
    )

    with torch.no_grad():
        y_windowed = block(hidden, cos, sin, positions)

    ref_weights: dict[str, torch.Tensor] = {
        pname: converted[f"{prefix_attn}{pname}"].to(dtype)
        for pname in nw._MQABlock.PARAM_KEYS
    }
    with torch.no_grad():
        y_fullcausal = _ref_sliding_only_forward(
            hidden,
            ref_weights,
            num_heads=src.num_attention_heads,
            head_dim=src.head_dim,
            o_groups=src.o_groups,
            o_lora_rank=src.o_lora_rank,
            rms_eps=src.rms_norm_eps,
            sliding_window=S,  # trivially-true window
            position_ids=positions,
            cos=cos,
            sin=sin,
        )

    diff = (y_windowed.to(torch.float32) - y_fullcausal.to(torch.float32)).abs()
    early_diff = float(diff[:, :sliding_window].max().item())
    late_diff = float(diff[:, sliding_window:].max().item())
    assert early_diff == 0.0, (
        f"early queries q_idx<{sliding_window} MUST agree between "
        f"windowed and full-causal forwards (nothing to clip); got "
        f"max_diff={early_diff}"
    )
    assert late_diff > 0.0, (
        f"late queries q_idx>={sliding_window} MUST diverge between "
        "windowed and full-causal forwards (the window clips past KV); "
        f"got max_diff={late_diff} — the sliding-window mask is not "
        "reaching the attention math."
    )
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-sliding-only-mask-actually-applied",
                "early_query_max_diff_bf16": early_diff,
                "late_query_max_diff_bf16": late_diff,
                "sliding_window": sliding_window,
                "seq_len": S,
            },
            indent=2,
        )
    )


def _standalone_main() -> int:
    tests = [
        test_sliding_layer_schedule_confirms_layer_0,
        test_sliding_only_wrapper_tree_key_set,
        test_sliding_only_refuses_wrong_layer_type,
        test_sliding_only_converter_refuses_wrong_layer_type,
        test_sliding_window_mask_helper_predicate_matches_hf,
        test_sliding_only_synthetic_shape_gate,
        test_sliding_only_wrapper_matches_hf_reference_on_real_layer0_tensors,
        test_sliding_mask_clips_beyond_window_on_real_forward,
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
                "suite": "dsv4-flash.tests.test_sliding_only_1layer",
                "pass": n_pass,
                "skip": n_skip,
                "fail": n_fail,
            },
            indent=2,
        )
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_standalone_main())
