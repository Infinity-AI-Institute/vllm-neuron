# SPDX-License-Identifier: Apache-2.0
"""Round-4 1-block smoke: HCA attention forward, wrapper vs HF reference.

This is the correctness backstop for the HCA composition — the smallest
attention family beyond sliding-only, exercising the ``_MQABlock``
project_q/project_kv/attend_and_project boundaries plus the new
``_HCACompressor`` pool and causal ``block_bias``.

We defend by:

  1. Loading ALL 13 layer-3 attention + compressor tensors from the real
     HF shard ``model-00005-of-00048.safetensors`` at snapshot
     ``deepseek-ai/DeepSeek-V4-Flash-0731 @
     7872f01b1d1fe23eabc4c98b48bffcef5a386062`` via HTTP-Range slice.
     Layer 3 is the first HCA layer under the frozen ``compress_ratios``
     schedule (``_COMPRESS_RATIOS_HF[3] == 128``).

  2. Dequanting the five FP8-e4m3 UE8M0 attention weights and carrying
     the compressor's four dense tensors through via
     ``_convert_hca_block``.

  3. Instantiating ``_HCABlock(config, layer_idx=3)`` and running its
     forward on a synthetic input ``[B=1, S=256, hidden]`` (S=256 gives
     exactly 2 closed HCA windows of 128 tokens).

  4. Running an INDEPENDENT reference forward that transcribes HF's HCA
     path (``DeepseekV4HCACompressor`` +
     ``DeepseekV4Attention.forward`` restricted to ``layer_type ==
     "heavily_compressed_attention"``, ``past_key_values=None``,
     ``attention_mask=None``).  The reference lives HERE, not imported
     from the wrapper, so any wrapper-side copy-paste error is caught.

  5. Asserting ``max_abs_error_bf16 < 1e-4`` (target 0.0 like MQA/MoE),
     with a degeneracy guard on both sides so all-zero / all-NaN output
     cannot vacuously pass.

  6. Asserting the compressor emits exactly ``S // compress_rate = 2``
     compressed entries.

The test skips (rather than errors) when the local cache is empty and
the network fetch cannot complete — an offline dev box is not a
correctness failure.  It never falls back to synthetic-only when the
intent is byte-clean verification; a skip is louder than a false pass.
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
# Degeneracy guard — same lookup convention as the FP4 / MQA tests.
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
# Library under test — imported via importlib fallback so a dev box
# without the full `vllm_neuron` package (which pulls `vllm`) still
# runs the test.  Mirrors test_mqa_1tensor.py::_import_library exactly.
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
    pkg = types.ModuleType("_dsv4_flash_hca_test_pkg")
    pkg.__path__ = [str(dsv4_dir)]
    sys.modules["_dsv4_flash_hca_test_pkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_dsv4_flash_hca_test_pkg.{name}",
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
# HF-shard slicer for the layer-3 attention + compressor subtree.
#
# Layer 3 lives in shard 00005-of-00048; verified against
# ``.hf_cache/dsv4_flash_index.json``.  All 13 tensors (attention subtree
# with FP8 scales + compressor dense) add up to ~89 MB — cached in
# ``.hf_cache/dsv4_layer3_hca.safetensors`` after the first pull.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
_HF_SHARD = "model-00005-of-00048.safetensors"
_LAYER_IDX = 3

_ATTN_KEYS_FP8 = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
_ATTN_KEYS_DENSE = ("q_norm.weight", "kv_norm.weight", "attn_sink")
_COMPRESSOR_KEYS_DENSE = ("wkv.weight", "wgate.weight", "ape", "norm.weight")

_LAYER3_KEYS: tuple[str, ...] = tuple(
    [f"layers.{_LAYER_IDX}.attn.{n}.weight" for n in _ATTN_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.{n}.scale" for n in _ATTN_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.{n}" for n in _ATTN_KEYS_DENSE]
    + [f"layers.{_LAYER_IDX}.attn.compressor.{n}" for n in _COMPRESSOR_KEYS_DENSE]
    + [f"layers.{_LAYER_IDX}.attn_norm.weight"]
)

_CACHE_DIR = Path(__file__).parent / ".hf_cache"
_CACHE_FILE = _CACHE_DIR / f"dsv4_layer{_LAYER_IDX}_hca.safetensors"
_HEADER_CHUNK_BYTES = 2 * 1024 * 1024  # 2 MB — shard 5's header is larger


def _fetch_range(url: str, byte_range: tuple[int, int]) -> bytes:
    """HTTP ``Range: bytes=start-end`` fetch, inclusive on both ends."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
            "User-Agent": "vllm_neuron.dsv4_flash.tests.hca/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _build_local_cache() -> Path:
    """Pull the layer-3 attn+compressor tensors and repack as mini-safetensors.

    Uses the huggingface_hub URL if available, otherwise the CDN direct URL.
    """
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
    for hf_key in _LAYER3_KEYS:
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


def _load_layer3_state_dict() -> dict[str, torch.Tensor]:
    """Return the 18 layer-3 attn + compressor HF tensors as a state_dict."""
    from safetensors.torch import load_file

    if not _CACHE_FILE.exists():
        try:
            _build_local_cache()
        except Exception as exc:  # network / auth
            pytest.skip(
                "cannot fetch real DSv4-Flash layer-3 attention+compressor "
                f"tensors ({_HF_REPO}@{_HF_SHA[:12]} {_HF_SHARD}): {exc!r}"
            )
    store = load_file(str(_CACHE_FILE))
    missing = [k for k in _LAYER3_KEYS if k not in store]
    if missing:
        raise KeyError(
            f"local mini-safetensors cache is missing keys {missing!r}; "
            f"delete {_CACHE_FILE!s} to force a re-pull."
        )
    return dict(store)


# ---------------------------------------------------------------------------
# Independent reference forward — hand-transcribed from
# transformers/models/deepseek_v4/modeling_deepseek_v4.py:
#   * DeepseekV4HCACompressor.forward (lines 362-443)
#   * DeepseekV4Attention.forward     (lines 801-873)
#     restricted to layer_type == "heavily_compressed_attention",
#     past_key_values=None, attention_mask=None
#
# Kept here (not imported from neuron_wrapper) so a wrapper-side copy-
# paste error is caught, not papered over.
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


def _ref_compressor_forward(
    hidden_states: torch.Tensor,  # [B, S, hidden]
    wkv: torch.Tensor,  # [head_dim, hidden]
    wgate: torch.Tensor,  # [head_dim, hidden]
    ape: torch.Tensor,  # [compress_rate, head_dim]
    norm_weight: torch.Tensor,  # [head_dim]
    *,
    compress_rate: int,
    head_dim: int,
    rms_eps: float,
    cos_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
    sin_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
) -> tuple[torch.Tensor, int]:
    """Reference HCA compressor: emits [B, 1, n_windows, head_dim].

    Transcribes ``DeepseekV4HCACompressor.forward`` (modeling_deepseek_v4.py
    lines 402-424) at ``past_key_values=None`` (stateless single-shot).
    """
    batch, seq, _ = hidden_states.shape
    usable = (seq // compress_rate) * compress_rate
    if usable == 0:
        return hidden_states.new_zeros((batch, 1, 0, head_dim)), 0
    chunk = hidden_states[:, :usable]
    kv = torch.nn.functional.linear(chunk, wkv)  # [B, U, D]
    gate = torch.nn.functional.linear(chunk, wgate)  # [B, U, D]
    n_windows = usable // compress_rate
    kv_r = kv.view(batch, n_windows, compress_rate, head_dim)
    gate_r = gate.view(batch, n_windows, compress_rate, head_dim) + ape
    softmax_w = gate_r.softmax(dim=2, dtype=torch.float32).to(kv_r.dtype)
    compressed = (kv_r * softmax_w).sum(dim=2)
    compressed = _ref_rms_norm_weighted(compressed, norm_weight, rms_eps)
    # Apply "compress" rope at window positions.  compressed is [B, n_windows, D];
    # unsqueeze the head axis so apply_rotary_pos_emb (unsqueeze_dim=1) broadcasts.
    compressed = _ref_apply_rotary(compressed.unsqueeze(1), cos_win, sin_win).squeeze(1)
    return compressed.unsqueeze(1), n_windows  # [B, 1, T, D]


def _ref_hca_forward(
    hidden_states: torch.Tensor,  # [B, S, hidden]
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    rms_eps: float,
    compress_rate: int,
    position_ids: torch.Tensor,  # [B, S]
    cos: torch.Tensor,  # [B, S, rope_dim/2]
    sin: torch.Tensor,  # [B, S, rope_dim/2]
    cos_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
    sin_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
) -> torch.Tensor:
    """Reference DSv4 HCA attention block forward.

    Transcribes ``DeepseekV4Attention.forward`` (lines 801-873) with
    ``layer_type == "heavily_compressed_attention"``,
    ``past_key_values=None``, ``attention_mask=None``, and inlines the
    HCA compressor from :func:`_ref_compressor_forward`.
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
    comp_wkv = weights["compressor.wkv.weight"]
    comp_wgate = weights["compressor.wgate.weight"]
    comp_ape = weights["compressor.ape"]
    comp_norm = weights["compressor.norm.weight"]

    # Q & KV projections + partial RoPE (identical to _ref_forward in the MQA
    # test but using the "compress" rope table passed in by the caller).
    q_residual = _ref_rms_norm_weighted(
        torch.nn.functional.linear(hidden_states, wq_a), q_norm_weight, rms_eps
    )
    q = torch.nn.functional.linear(q_residual, wq_b).view(*hidden_shape).transpose(1, 2)
    q = _ref_rms_norm_unweighted(q, rms_eps)
    q = _ref_apply_rotary(q, cos, sin)

    kv_main = (
        _ref_rms_norm_weighted(
            torch.nn.functional.linear(hidden_states, wkv), kv_norm_weight, rms_eps
        )
        .view(*hidden_shape)
        .transpose(1, 2)
    )
    kv_main = _ref_apply_rotary(kv_main, cos, sin)

    # HCA compressor emits [B, 1, T_c, D] compressed KV entries.
    compressed_kv, n_windows = _ref_compressor_forward(
        hidden_states,
        comp_wkv,
        comp_wgate,
        comp_ape,
        comp_norm,
        compress_rate=compress_rate,
        head_dim=head_dim,
        rms_eps=rms_eps,
        cos_win=cos_win,
        sin_win=sin_win,
    )
    t_compressed = compressed_kv.shape[2]

    # Build the block-bias over the compressed slots — HF lines 435-443.
    if t_compressed > 0 and S > 1:
        entry_indices = torch.arange(t_compressed, device=hidden_states.device)
        causal_threshold = (position_ids + 1) // compress_rate  # [B, S]
        block_bias = hidden_states.new_zeros((B, 1, S, t_compressed))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1)
            >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
    else:
        block_bias = None

    # Cat compressed KV onto main KV (HF line 832).
    kv_extended = torch.cat([kv_main, compressed_kv], dim=2)  # [B, 1, S+T_c, D]

    # Extend attention_mask.  HF's caller here passes attention_mask=None, so
    # we synthesise a zero prefix over kv_main and cat block_bias.  This
    # matches HF's behaviour at line 840 (the `if isinstance(attention_mask,
    # torch.Tensor)` branch is not taken; block_bias must still apply, and
    # our tensor-form is the additive form).
    if block_bias is not None:
        zeros_prefix = hidden_states.new_zeros((B, 1, S, kv_main.shape[2]))
        extended_mask = torch.cat([zeros_prefix, block_bias], dim=-1)
    else:
        extended_mask = None

    # Attention math with per-head sink (eager_attention_forward,
    # modeling_deepseek_v4.py:727-745).  scaling = head_dim ** -0.5.
    scaling = head_dim**-0.5
    key_states = kv_extended.expand(B, num_heads, kv_extended.shape[2], head_dim)
    value_states = key_states
    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * scaling
    if extended_mask is not None:
        attn_weights = attn_weights + extended_mask
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(
        combined_logits, dim=-1, dtype=combined_logits.dtype
    )
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores.to(value_states.dtype), value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # -sin conjugate rotation on output rope slice (HF line 868).
    attn_output = _ref_apply_rotary(attn_output.transpose(1, 2), cos, -sin).transpose(
        1, 2
    )

    # Grouped output projection.
    grouped = attn_output.reshape(*input_shape, o_groups, -1)
    n_groups = o_groups
    hidden_dim = grouped.shape[-1]
    w = wo_a.view(n_groups, -1, hidden_dim).transpose(1, 2)
    x = grouped.reshape(-1, n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    grouped_out = y.reshape(*input_shape, n_groups, -1)
    grouped_out = grouped_out.flatten(2)
    output = torch.nn.functional.linear(grouped_out, wo_b)
    return output, n_windows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hca_wrapper_tree_key_set() -> None:
    """The _HCABlock's parameter tree must be exactly:
    - 8 MQA params under `mqa.*`
    - 4 HCA compressor params under `compressor.*`
    - 12 total
    """
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    block = nw._HCABlock(src, layer_idx=_LAYER_IDX)
    names = sorted(name for name, _ in block.named_parameters())
    expected = sorted(
        [f"mqa.{k}" for k in nw._MQABlock.PARAM_KEYS]
        + [f"compressor.{k}" for k in nw._HCACompressor.PARAM_KEYS]
    )
    assert names == expected, (names, expected)
    assert len(nw._MQABlock.PARAM_KEYS) == 8
    assert len(nw._HCACompressor.PARAM_KEYS) == 4
    assert len(names) == 12


def test_hca_layer_schedule_confirms_layer_3() -> None:
    """Layer 3 must be the first HCA layer in the frozen schedule."""
    cfg, _convert, _nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    assert src.layer_types[_LAYER_IDX] == "heavily_compressed_attention"
    assert src.compress_ratios[_LAYER_IDX] == 128
    # And it is the FIRST HCA layer: layers 0, 1 sliding; layer 2 CSA;
    # layer 3 first HCA.
    assert src.layer_types[0] == "sliding_attention"
    assert src.layer_types[1] == "sliding_attention"
    assert src.layer_types[2] == "compressed_sparse_attention"
    assert src.layer_types[3] == "heavily_compressed_attention"


def test_hca_refuses_wrong_layer_type() -> None:
    """The block hard-refuses instantiation at a non-HCA layer index."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    # Layer 0 is sliding_attention.
    with pytest.raises(ValueError, match="heavily_compressed_attention"):
        nw._HCABlock(src, layer_idx=0)
    # Layer 2 is CSA.
    with pytest.raises(ValueError, match="heavily_compressed_attention"):
        nw._HCABlock(src, layer_idx=2)


def test_hca_compressor_refuses_wrong_ratio() -> None:
    """_HCACompressor hard-refuses a layer whose compress_ratios != 128."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    # Layer 2 is CSA (compress_ratios=4).
    with pytest.raises(ValueError, match="compress_ratios"):
        nw._HCACompressor(src, layer_idx=2)


def test_hca_synthetic_shape_gate() -> None:
    """Fast synthetic-input gate: exercise the wrapper without HF network
    access — shape math + compressor n_windows + no NaN + non-degenerate.
    Skips no network (uses random-init weights)."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    block = nw._HCABlock(src, layer_idx=_LAYER_IDX).to(dtype)
    torch.manual_seed(2)
    B, S = 1, 256
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    n_windows = S // block.compress_rate
    win_positions = (
        torch.arange(n_windows).unsqueeze(0).expand(B, -1) * block.compress_rate
    )
    cos_win, sin_win = nw.build_main_rope_cos_sin(
        win_positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    y = block(hidden, cos, sin, cos_win, sin_win, positions)
    assert tuple(y.shape) == (B, S, src.hidden_size)
    assert y.dtype == dtype
    require_comparable(
        y.detach().to(torch.float32).cpu().numpy(), "hca_synthetic_output_fp32"
    )
    assert n_windows == S // 128 == 2


def test_hca_wrapper_matches_hf_reference_on_real_layer3_tensors() -> None:
    """The whole enchilada: real HF weights, dequant, HCA wrapper vs
    reference forward, byte-clean bf16 diff."""
    cfg, convert, nw = _import_library()
    state = _load_layer3_state_dict()

    src = cfg.DeepseekV4FlashInferenceConfig()  # frozen full shape
    dtype = torch.bfloat16
    compress_rate = 128

    # Convert HF -> wrapper module tree.
    converted = convert._convert_hca_block(
        state, layer_idx=_LAYER_IDX, src=src, dtype=dtype, require_attn_sink=True
    )

    # Sanity: the converted dict carries exactly 12 layer.attn.* + 1 attn_norm.
    prefix_attn = f"layers.{_LAYER_IDX}.attn."
    expected_converted = set(
        [f"{prefix_attn}{k}" for k in nw._MQABlock.PARAM_KEYS]
        + [f"{prefix_attn}compressor.{k}" for k in nw._HCACompressor.PARAM_KEYS]
        + [f"layers.{_LAYER_IDX}.attn_norm.weight"]
    )
    got_converted = set(converted.keys())
    assert got_converted == expected_converted, sorted(
        got_converted.symmetric_difference(expected_converted)
    )

    # Build the wrapper block; load the dequanted tensors under the
    # wrapper's declared parameter tree.
    block = nw._HCABlock(src, layer_idx=_LAYER_IDX).to(dtype)
    load_report: dict[str, tuple[int, ...]] = {}
    with torch.no_grad():
        # MQA subtree
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
        # Compressor subtree
        for pname in nw._HCACompressor.PARAM_KEYS:
            tensor = converted[f"{prefix_attn}compressor.{pname}"]
            parts = pname.split(".")
            obj = block.compressor
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            assert tuple(param.shape) == tuple(tensor.shape), (
                pname,
                tuple(param.shape),
                tuple(tensor.shape),
            )
            param.copy_(tensor.to(dtype))
            load_report[f"compressor.{pname}"] = tuple(tensor.shape)

    # Deterministic synthetic input.  S=256 gives exactly 2 closed HCA
    # windows of 128 tokens (the minimum to see the compressor emit
    # something and to exercise the block_bias masking non-trivially —
    # early queries in window 0 see 0 compressed entries, late queries
    # in window 1 see 1 compressed entry).
    torch.manual_seed(0)
    B, S = 1, 256
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)

    # Both wrapper and reference use the same rope tables — the composability
    # smoke tests forward-math equality, not rope correctness.  Wrapper's
    # helper is source-cited byte-for-byte against HF's rope module for the
    # default (non-yarn) branch, and here we pin theta=compress_rope_theta
    # (160000) to match what an HCA layer uses in HF.
    cos, sin = nw.build_main_rope_cos_sin(
        positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    n_windows = S // compress_rate
    win_positions = torch.arange(n_windows).unsqueeze(0).expand(B, -1) * compress_rate
    cos_win, sin_win = nw.build_main_rope_cos_sin(
        win_positions,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    assert tuple(cos.shape) == (B, S, src.qk_rope_head_dim // 2)
    assert tuple(cos_win.shape) == (B, n_windows, src.qk_rope_head_dim // 2)

    # ---- wrapper forward ----
    with torch.no_grad():
        y_wrap = block(hidden, cos, sin, cos_win, sin_win, positions)
    assert tuple(y_wrap.shape) == (B, S, src.hidden_size)
    assert y_wrap.dtype == dtype

    # Directly probe the compressor to confirm it emits the expected count
    # of compressed entries — HF's spec is one per closed window.
    with torch.no_grad():
        compressed_kv_probe = block.compressor.compress(hidden, cos_win, sin_win)
    assert tuple(compressed_kv_probe.shape) == (B, 1, n_windows, src.head_dim)
    assert n_windows == 2, n_windows

    # ---- reference forward ----
    ref_weights: dict[str, torch.Tensor] = {}
    for pname in nw._MQABlock.PARAM_KEYS:
        ref_weights[pname] = converted[f"{prefix_attn}{pname}"].to(dtype)
    for pname in nw._HCACompressor.PARAM_KEYS:
        ref_weights[f"compressor.{pname}"] = converted[
            f"{prefix_attn}compressor.{pname}"
        ].to(dtype)
    with torch.no_grad():
        y_ref, ref_n_windows = _ref_hca_forward(
            hidden,
            ref_weights,
            num_heads=src.num_attention_heads,
            head_dim=src.head_dim,
            o_groups=src.o_groups,
            o_lora_rank=src.o_lora_rank,
            rms_eps=src.rms_norm_eps,
            compress_rate=compress_rate,
            position_ids=positions,
            cos=cos,
            sin=sin,
            cos_win=cos_win,
            sin_win=sin_win,
        )
    assert tuple(y_ref.shape) == tuple(y_wrap.shape)
    assert y_ref.dtype == dtype
    assert ref_n_windows == n_windows == 2

    # ---- degeneracy guard on BOTH sides ----
    require_comparable(
        y_wrap.detach().to(torch.float32).cpu().numpy(), "hca_wrapper_output_fp32"
    )
    require_comparable(
        y_ref.detach().to(torch.float32).cpu().numpy(), "hca_reference_output_fp32"
    )

    # ---- byte-clean bf16 diff ----
    diff = (y_wrap.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_abs_error_bf16 = float(diff.max().item())
    mean_abs_error_bf16 = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hca-block-real-tensor",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "hf_shard": _HF_SHARD,
                "layer_idx": _LAYER_IDX,
                "layer_type": src.layer_types[_LAYER_IDX],
                "compress_rate": compress_rate,
                "batch_size": B,
                "seq_len": S,
                "n_windows_emitted": n_windows,
                "wrapper_tree_key_count": len(list(block.named_parameters())),
                "converted_key_count": len(converted),
                "load_report_key_count": len(load_report),
                "hidden_size": src.hidden_size,
                "num_heads": src.num_attention_heads,
                "head_dim": src.head_dim,
                "compressor_head_dim": block.compressor.head_dim,
                "compress_rope_theta": src.compress_rope_theta,
                "max_abs_error_bf16": max_abs_error_bf16,
                "mean_abs_error_bf16": mean_abs_error_bf16,
                "output_shape": tuple(y_wrap.shape),
            },
            indent=2,
        )
    )
    assert max_abs_error_bf16 < 1e-4, max_abs_error_bf16


# ---------------------------------------------------------------------------
# Standalone runner — mirrors test_mqa_1tensor.py so a laptop without
# pytest-collectible `vllm_neuron` can still run the gate.
# ---------------------------------------------------------------------------
def _standalone_main() -> int:
    tests = [
        test_hca_layer_schedule_confirms_layer_3,
        test_hca_wrapper_tree_key_set,
        test_hca_refuses_wrong_layer_type,
        test_hca_compressor_refuses_wrong_ratio,
        test_hca_synthetic_shape_gate,
        test_hca_wrapper_matches_hf_reference_on_real_layer3_tensors,
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
                "suite": "dsv4-flash.tests.test_hca_1layer",
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
