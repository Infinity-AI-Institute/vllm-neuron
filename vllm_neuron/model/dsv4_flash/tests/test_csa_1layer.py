# SPDX-License-Identifier: Apache-2.0
"""Round-5 1-block smoke: CSA attention forward + overlap-state aliasing.

This is the correctness backstop for the CSA composition — the most
complex attention family in DSv4-Flash, adding two new mechanisms on
top of the HCA composition:

  1. **Overlap-state aliasing**: the outer CSA compressor's Ca/Cb window
     scheme (paper §2.3.1) needs per-window ``[B, compress_rate,
     head_dim]`` overlap KV / gate tensors persisted across forward calls.
     The Lightning Indexer's own inner compressor at ``index_head_dim=128``
     owns a second aliased pair with the same shape contract.  Getting
     the update rule wrong (writing the *current* window's Ca instead of
     the LAST window's Ca; using the wrong slice; forgetting to zero /
     -inf-init the first-call window-0 slot) silently corrupts every
     decode step — same class of bug that caught Round 5's KDA conv1d
     layout in GLM-5.3-Flash.
  2. **Lightning Indexer top-K gating**: the scorer's per-head inner
     product with ReLU + fp32 softmax scale reduces to ``[B, S,
     compressed_len]``, then top-``K = min(index_topk, compressed_len)``
     picks the compressed entries each query may attend to.  Sentinel
     ``-1`` marks entries that would land past the query's causal
     threshold.

We defend by:

  1. Loading ALL 19 layer-2 attention + CSA compressor + indexer HF
     tensors from ``model-00004-of-00048.safetensors`` at snapshot
     ``deepseek-ai/DeepSeek-V4-Flash-0731 @
     7872f01b1d1fe23eabc4c98b48bffcef5a386062`` via HTTP-Range slice.
     Layer 2 is the first CSA layer under the frozen ``compress_ratios``
     schedule (``_COMPRESS_RATIOS_HF[2] == 4``).

  2. Dequanting the six FP8-e4m3 UE8M0 attention/indexer weights and
     carrying the compressor's 8 dense tensors + indexer's dense
     ``weights_proj`` through via ``_convert_csa_block``.

  3. Instantiating ``_CSABlock(config, layer_idx=2)`` and running its
     forward on a synthetic input ``[B=1, S=16]`` (S=16 = 4 closed CSA
     windows of 4 tokens each — the smallest that gives non-trivial
     top-K selection with 4 candidates per query and exercises the
     Ca/Cb inter-window state flow).

  4. Running an INDEPENDENT reference forward that transcribes HF's CSA
     path (``DeepseekV4CSACompressor.forward`` +
     ``DeepseekV4Indexer.forward`` + ``DeepseekV4IndexerScorer.forward``
     + ``DeepseekV4Attention.forward`` restricted to
     ``layer_type == "compressed_sparse_attention"``,
     ``past_key_values=None``, ``attention_mask=None``).  The reference
     lives HERE, not imported from the wrapper, so any wrapper-side
     copy-paste error is caught.

  5. Asserting ``max_abs_error_bf16 < 1e-4`` (target 0.0 like the prior
     three block classes) on both first-step and multi-step outputs,
     with a degeneracy guard on both sides so all-zero / all-NaN output
     cannot vacuously pass.

  6. **State-evolution assertion**: run a 16-token sequence as one shot
     (no state carry) and as two 8-token chunks with overlap-state
     carry between chunks.  The compressor + indexer emissions from
     chunks 1 & 2 concatenated MUST equal the emissions of the single-
     shot run.  A wrong Ca/Cb update rule breaks this on the boundary
     window.

  7. **Lightning Indexer top-K correctness**: assert wrapper's top_k
     indices are element-for-element identical to the reference's
     (both use ``torch.topk`` on the same fp32 scores), and every
     wrapper index is either ``-1`` (sentinel) or strictly less than
     the per-query causal threshold ``(position_ids + 1) //
     compress_rate``.

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
# Degeneracy guard — same lookup convention as the FP4 / MQA / HCA tests.
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
# Library under test — importlib fallback so a dev box without the full
# `vllm_neuron` package (which pulls `vllm`) still runs the test.
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
    pkg = types.ModuleType("_dsv4_flash_csa_test_pkg")
    pkg.__path__ = [str(dsv4_dir)]
    sys.modules["_dsv4_flash_csa_test_pkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_dsv4_flash_csa_test_pkg.{name}",
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
# HF-shard slicer for the layer-2 attention + CSA compressor + indexer
# subtree.
#
# Layer 2 lives in shard 00004-of-00048; verified against
# ``.hf_cache/dsv4_flash_index.json``.  All 24 tensors (8 attn FP8 weights
# + 8 attn FP8 scales + 3 attn dense + 4 compressor dense + 4 indexer-
# compressor dense + 1 indexer weights_proj + 1 indexer wq_b + 1 indexer
# wq_b scale + 1 attn_norm sibling — with `attn_sink` counted in the "3
# attn dense") add up to <90 MB — cached in
# ``.hf_cache/dsv4_layer2_csa.safetensors`` after the first pull.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
_HF_SHARD = "model-00004-of-00048.safetensors"
_LAYER_IDX = 2

_ATTN_KEYS_FP8 = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
_ATTN_KEYS_DENSE = ("q_norm.weight", "kv_norm.weight", "attn_sink")
_CSA_COMPRESSOR_KEYS_DENSE = ("wkv.weight", "wgate.weight", "ape", "norm.weight")
_INDEXER_COMPRESSOR_KEYS_DENSE = _CSA_COMPRESSOR_KEYS_DENSE
_INDEXER_KEYS_DENSE = ("weights_proj.weight",)
_INDEXER_KEYS_FP8 = ("wq_b",)  # `.weight` and `.scale`

_LAYER2_KEYS: tuple[str, ...] = tuple(
    [f"layers.{_LAYER_IDX}.attn.{n}.weight" for n in _ATTN_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.{n}.scale" for n in _ATTN_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.{n}" for n in _ATTN_KEYS_DENSE]
    + [f"layers.{_LAYER_IDX}.attn.compressor.{n}" for n in _CSA_COMPRESSOR_KEYS_DENSE]
    + [
        f"layers.{_LAYER_IDX}.attn.indexer.compressor.{n}"
        for n in _INDEXER_COMPRESSOR_KEYS_DENSE
    ]
    + [f"layers.{_LAYER_IDX}.attn.indexer.{n}" for n in _INDEXER_KEYS_DENSE]
    + [f"layers.{_LAYER_IDX}.attn.indexer.{n}.weight" for n in _INDEXER_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn.indexer.{n}.scale" for n in _INDEXER_KEYS_FP8]
    + [f"layers.{_LAYER_IDX}.attn_norm.weight"]
)

_CACHE_DIR = Path(__file__).parent / ".hf_cache"
_CACHE_FILE = _CACHE_DIR / f"dsv4_layer{_LAYER_IDX}_csa.safetensors"
_HEADER_CHUNK_BYTES = 2 * 1024 * 1024  # 2 MB — shard 4's header fits inside


def _fetch_range(url: str, byte_range: tuple[int, int]) -> bytes:
    """HTTP ``Range: bytes=start-end`` fetch, inclusive on both ends."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
            "User-Agent": "vllm_neuron.dsv4_flash.tests.csa/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _build_local_cache() -> Path:
    """Pull the layer-2 attn+compressor+indexer tensors and repack as mini-safetensors."""
    try:
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(_HF_REPO, _HF_SHARD, revision=_HF_SHA)
    except Exception:
        url = f"https://huggingface.co/{_HF_REPO}/resolve/{_HF_SHA}/{_HF_SHARD}"

    first = _fetch_range(url, (0, _HEADER_CHUNK_BYTES - 1))
    header_len = struct.unpack("<Q", first[:8])[0]
    if header_len > len(first) - 8:  # pragma: no cover — very large header
        first = _fetch_range(url, (0, header_len + 8 + 4096))
    header = json.loads(first[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len

    tensors: dict[str, dict[str, Any]] = {}
    for hf_key in _LAYER2_KEYS:
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


def _load_layer2_state_dict() -> dict[str, torch.Tensor]:
    """Return the layer-2 attn + compressor + indexer HF tensors as a state_dict."""
    from safetensors.torch import load_file

    if not _CACHE_FILE.exists():
        try:
            _build_local_cache()
        except Exception as exc:  # network / auth
            pytest.skip(
                "cannot fetch real DSv4-Flash layer-2 attention+compressor+"
                f"indexer tensors ({_HF_REPO}@{_HF_SHA[:12]} {_HF_SHARD}): "
                f"{exc!r}"
            )
    store = load_file(str(_CACHE_FILE))
    missing = [k for k in _LAYER2_KEYS if k not in store]
    if missing:
        raise KeyError(
            f"local mini-safetensors cache is missing keys {missing!r}; "
            f"delete {_CACHE_FILE!s} to force a re-pull."
        )
    return dict(store)


# ---------------------------------------------------------------------------
# Independent reference forward — hand-transcribed from
# transformers/models/deepseek_v4/modeling_deepseek_v4.py:
#   * DeepseekV4CSACompressor.forward     (lines 623-702)
#   * DeepseekV4Indexer.forward           (lines 507-586)
#   * DeepseekV4IndexerScorer.forward     (lines 455-459)
#   * DeepseekV4Attention.forward         (lines 801-873)
#     restricted to layer_type == "compressed_sparse_attention",
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


def _ref_overlap_compressor_forward(
    hidden_states: torch.Tensor,  # [B, S, hidden]
    wkv: torch.Tensor,  # [2*head_dim, hidden]
    wgate: torch.Tensor,  # [2*head_dim, hidden]
    ape: torch.Tensor,  # [compress_rate, 2*head_dim]
    norm_weight: torch.Tensor,  # [head_dim]
    *,
    compress_rate: int,
    head_dim: int,
    rms_eps: float,
    cos_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
    sin_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
    overlap_kv_prev: torch.Tensor | None = None,  # [B, compress_rate, head_dim]
    overlap_gate_prev: torch.Tensor | None = None,  # [B, compress_rate, head_dim]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference CSA-style overlap compressor.

    Transcribes ``DeepseekV4CSACompressor.forward`` (modeling_deepseek_v4.py
    lines 638-680) at ``past_key_values`` semantically set to the overlap
    ``(overlap_kv_prev, overlap_gate_prev)`` pair.  Emits [B, 1, n_windows,
    head_dim] compressed KV + the ``[B, compress_rate, head_dim]`` overlap
    slice for the next call.
    """
    batch, seq, _ = hidden_states.shape
    usable = (seq // compress_rate) * compress_rate
    if usable == 0:
        empty_c = hidden_states.new_zeros((batch, 1, 0, head_dim))
        empty_state = hidden_states.new_zeros((batch, compress_rate, head_dim))
        return empty_c, empty_state, empty_state
    chunk = hidden_states[:, :usable]
    kv = torch.nn.functional.linear(chunk, wkv)  # [B, U, 2D]
    gate = torch.nn.functional.linear(chunk, wgate)  # [B, U, 2D]
    n_windows = usable // compress_rate
    ratio = compress_rate
    chunk_kv = kv.view(batch, n_windows, ratio, -1)
    chunk_gate = gate.view(batch, n_windows, ratio, -1) + ape

    new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, head_dim))
    new_gate = chunk_gate.new_full(
        (batch, n_windows, 2 * ratio, head_dim), float("-inf")
    )
    new_kv[:, :, ratio:] = chunk_kv[..., head_dim:]  # Cb → second half current window
    new_gate[:, :, ratio:] = chunk_gate[..., head_dim:]
    if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :head_dim]  # Ca of prior window
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :head_dim]
    if overlap_kv_prev is not None:
        new_kv[:, 0, :ratio] = overlap_kv_prev.to(new_kv.dtype)
        new_gate[:, 0, :ratio] = overlap_gate_prev.to(new_gate.dtype)

    softmax_w = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
    compressed = (new_kv * softmax_w).sum(dim=2)  # [B, n_win, D]
    compressed = _ref_rms_norm_weighted(compressed, norm_weight, rms_eps)
    compressed = _ref_apply_rotary(
        compressed.unsqueeze(1), cos_win, sin_win
    )  # [B, 1, n_win, D]
    new_overlap_kv = chunk_kv[:, -1, :, :head_dim].clone()
    new_overlap_gate = chunk_gate[:, -1, :, :head_dim].clone()
    return compressed, new_overlap_kv, new_overlap_gate


def _ref_indexer_top_k(
    hidden_states: torch.Tensor,  # [B, S, hidden]
    q_residual: torch.Tensor,  # [B, S, q_lora_rank]
    idx_wq_b: torch.Tensor,  # [n_heads*head_dim, q_lora_rank]
    idx_weights_proj: torch.Tensor,  # [n_heads, hidden]
    compressed_kv: torch.Tensor,  # [B, T, head_dim]
    *,
    cos_q: torch.Tensor,  # [B, S, rope_dim/2]
    sin_q: torch.Tensor,
    n_heads: int,
    head_dim: int,
    index_topk: int,
    compress_rate: int,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference Lightning Indexer top-K selection.

    Transcribes ``DeepseekV4Indexer.forward`` lines 563-586 + the scorer.
    """
    batch, seq, _ = hidden_states.shape
    q_flat = torch.nn.functional.linear(q_residual, idx_wq_b)  # [B, S, H*D]
    q = q_flat.view(batch, seq, n_heads, head_dim)
    # HF applies rope on [B, H, S, D]; we transpose in-out to match.
    q = _ref_apply_rotary(q.transpose(1, 2), cos_q, sin_q).transpose(1, 2)
    # Scorer.
    q_fp32 = q.float()
    k_fp32 = compressed_kv.transpose(-1, -2).float().unsqueeze(1)  # [B, 1, D, T]
    scores = torch.matmul(q_fp32, k_fp32)  # [B, S, H, T]
    scores = torch.nn.functional.relu(scores) * (head_dim**-0.5)
    weights = torch.nn.functional.linear(hidden_states, idx_weights_proj).float() * (
        n_heads**-0.5
    )  # [B, S, H]
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]

    compressed_len = compressed_kv.shape[1]
    top_k = min(index_topk, compressed_len)
    if compressed_len > 0:
        causal_threshold = (position_ids + 1) // compress_rate
        entry_indices = torch.arange(compressed_len, device=index_scores.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k_indices = index_scores.topk(top_k, dim=-1).indices
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        top_k_indices = torch.where(
            invalid, torch.full_like(top_k_indices, -1), top_k_indices
        )
    else:
        top_k_indices = torch.zeros((batch, seq, top_k), dtype=torch.int64)
    return top_k_indices


def _ref_csa_forward(
    hidden_states: torch.Tensor,  # [B, S, hidden]
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    rms_eps: float,
    compress_rate: int,
    index_head_dim: int,
    index_n_heads: int,
    index_topk: int,
    position_ids: torch.Tensor,  # [B, S]
    cos: torch.Tensor,  # [B, S, rope_dim/2]
    sin: torch.Tensor,  # [B, S, rope_dim/2]
    cos_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
    sin_win: torch.Tensor,  # [B, n_windows, rope_dim/2]
) -> torch.Tensor:
    """Reference DSv4 CSA attention block forward (stateless single-shot).

    Transcribes ``DeepseekV4Attention.forward`` lines 801-873 with
    ``layer_type == "compressed_sparse_attention"``,
    ``past_key_values=None``, ``attention_mask=None``, and inlines the CSA
    compressor + Lightning Indexer references above.
    """
    B, S, H = hidden_states.shape
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    # MQA weights.
    wq_a = weights["wq_a.weight"]
    wq_b = weights["wq_b.weight"]
    q_norm_weight = weights["q_norm.weight"]
    wkv = weights["wkv.weight"]
    kv_norm_weight = weights["kv_norm.weight"]
    wo_a = weights["wo_a.weight"]
    wo_b = weights["wo_b.weight"]
    attn_sink = weights["attn_sink"]
    # CSA compressor.
    comp_wkv = weights["compressor.wkv.weight"]
    comp_wgate = weights["compressor.wgate.weight"]
    comp_ape = weights["compressor.ape"]
    comp_norm = weights["compressor.norm.weight"]
    # Indexer.
    idx_wkv = weights["indexer.compressor.wkv.weight"]
    idx_wgate = weights["indexer.compressor.wgate.weight"]
    idx_ape = weights["indexer.compressor.ape"]
    idx_norm = weights["indexer.compressor.norm.weight"]
    idx_wq_b = weights["indexer.wq_b.weight"]
    idx_wproj = weights["indexer.weights_proj.weight"]

    # Q / KV projections + partial RoPE (CSA uses the same "compress" rope
    # frame as HCA; HF line 776-777: `rope_layer_type = "compress"` for
    # any non-`sliding_attention` layer).
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

    # CSA compressor emits [B, 1, T_c, D] + updated overlap state (unused
    # in stateless single-shot).
    compressed_kv, _, _ = _ref_overlap_compressor_forward(
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
        overlap_kv_prev=None,
        overlap_gate_prev=None,
    )
    t_compressed = compressed_kv.shape[2]

    # Indexer's own inner compressor at index_head_dim=128.
    idx_compressed_kv_bh, _, _ = _ref_overlap_compressor_forward(
        hidden_states,
        idx_wkv,
        idx_wgate,
        idx_ape,
        idx_norm,
        compress_rate=compress_rate,
        head_dim=index_head_dim,
        rms_eps=rms_eps,
        cos_win=cos_win,
        sin_win=sin_win,
        overlap_kv_prev=None,
        overlap_gate_prev=None,
    )
    idx_compressed_kv = idx_compressed_kv_bh.squeeze(1)  # [B, T, D_idx]

    # Lightning Indexer top-K.
    top_k_indices = _ref_indexer_top_k(
        hidden_states,
        q_residual,
        idx_wq_b,
        idx_wproj,
        idx_compressed_kv,
        cos_q=cos,
        sin_q=sin,
        n_heads=index_n_heads,
        head_dim=index_head_dim,
        index_topk=index_topk,
        compress_rate=compress_rate,
        position_ids=position_ids,
    )  # [B, S, K]

    # Build indexer-gated block_bias — HF lines 693-702.
    if t_compressed > 0:
        valid = top_k_indices >= 0
        safe_indices = torch.where(
            valid, top_k_indices, torch.full_like(top_k_indices, t_compressed)
        )
        block_bias = compressed_kv.new_full((B, 1, S, t_compressed + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        block_bias = block_bias[..., :t_compressed]
    else:
        block_bias = None

    # Cat compressed KV onto main KV, extend mask (attention_mask=None
    # here so we synthesise a zero prefix over the sliding KV region).
    kv_extended = torch.cat([kv_main, compressed_kv], dim=2)  # [B, 1, S+T_c, D]
    if block_bias is not None:
        zeros_prefix = hidden_states.new_zeros((B, 1, S, kv_main.shape[2]))
        extended_mask = torch.cat([zeros_prefix, block_bias], dim=-1)
    else:
        extended_mask = None

    # Attention math with per-head sink (eager_attention_forward lines
    # 727-745).
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
    grouped_out = y.reshape(*input_shape, n_groups, -1).flatten(2)
    output = torch.nn.functional.linear(grouped_out, wo_b)
    return output, t_compressed, top_k_indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_csa_layer_schedule_confirms_layer_2() -> None:
    """Layer 2 must be the first CSA layer in the frozen schedule."""
    cfg, _convert, _nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    assert src.layer_types[_LAYER_IDX] == "compressed_sparse_attention"
    assert src.compress_ratios[_LAYER_IDX] == 4
    # And it is the first CSA layer: layers 0, 1 sliding; layer 2 CSA;
    # layer 3 HCA (already covered by test_hca_1layer).
    assert src.layer_types[0] == "sliding_attention"
    assert src.layer_types[1] == "sliding_attention"
    assert src.layer_types[2] == "compressed_sparse_attention"
    assert src.layer_types[3] == "heavily_compressed_attention"


def test_csa_wrapper_tree_key_set() -> None:
    """The _CSABlock's parameter tree must be exactly:
    - 8 MQA params under `mqa.*`
    - 4 CSA compressor params under `compressor.*`
    - 4 indexer inner compressor params under `indexer.compressor.*`
    - 2 indexer projection params under `indexer.*` (wq_b, weights_proj)
    - 18 total
    """
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    block = nw._CSABlock(src, layer_idx=_LAYER_IDX)
    names = sorted(name for name, _ in block.named_parameters())
    expected = sorted(
        [f"mqa.{k}" for k in nw._MQABlock.PARAM_KEYS]
        + [f"compressor.{k}" for k in nw._CSAOverlapCompressor.PARAM_KEYS]
        + [f"indexer.compressor.{k}" for k in nw._CSAOverlapCompressor.PARAM_KEYS]
        + ["indexer.wq_b.weight", "indexer.weights_proj.weight"]
    )
    assert names == expected, (names, expected)
    assert len(nw._MQABlock.PARAM_KEYS) == 8
    assert len(nw._CSAOverlapCompressor.PARAM_KEYS) == 4
    assert len(names) == 18


def test_csa_refuses_wrong_layer_type() -> None:
    """The block hard-refuses instantiation at a non-CSA layer index."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    # Layer 0 is sliding.
    with pytest.raises(ValueError, match="compressed_sparse_attention"):
        nw._CSABlock(src, layer_idx=0)
    # Layer 3 is HCA.
    with pytest.raises(ValueError, match="compressed_sparse_attention"):
        nw._CSABlock(src, layer_idx=3)


def test_csa_state_cache_specs_shape_and_dtype() -> None:
    """state_cache_specs must declare exactly the 4 aliased pairs at the
    documented shapes: outer compressor (head_dim=512) + indexer (index_head_dim=128)
    each get an overlap_kv and overlap_gate of shape [B, compress_rate=4, D]."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    block = nw._CSABlock(src, layer_idx=_LAYER_IDX)
    specs = block.state_cache_specs(batch=3, seq_len=999)
    names = [name for name, _, _ in specs]
    assert names == [
        "compressor_overlap_kv",
        "compressor_overlap_gate",
        "indexer_overlap_kv",
        "indexer_overlap_gate",
    ]
    shape_map = {name: shape for name, shape, _ in specs}
    assert shape_map["compressor_overlap_kv"] == (
        3,
        src.compress_ratios[_LAYER_IDX],
        src.head_dim,
    )
    assert shape_map["compressor_overlap_gate"] == (
        3,
        src.compress_ratios[_LAYER_IDX],
        src.head_dim,
    )
    assert shape_map["indexer_overlap_kv"] == (
        3,
        src.compress_ratios[_LAYER_IDX],
        src.index_head_dim,
    )
    assert shape_map["indexer_overlap_gate"] == (
        3,
        src.compress_ratios[_LAYER_IDX],
        src.index_head_dim,
    )
    dtypes = {name: dt for name, _, dt in specs}
    assert dtypes == {n: src.torch_dtype for n in shape_map}


def test_csa_synthetic_shape_gate() -> None:
    """Fast synthetic-input gate: exercise wrapper without HF network."""
    cfg, _convert, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    block = nw._CSABlock(src, layer_idx=_LAYER_IDX).to(dtype)
    torch.manual_seed(3)
    B, S = 1, 16
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
    y, state = block(hidden, cos, sin, cos_win, sin_win, positions)
    assert tuple(y.shape) == (B, S, src.hidden_size)
    assert y.dtype == dtype
    for k in (
        "compressor_overlap_kv",
        "compressor_overlap_gate",
        "indexer_overlap_kv",
        "indexer_overlap_gate",
    ):
        assert k in state
    assert tuple(state["compressor_overlap_kv"].shape) == (
        B,
        block.compress_rate,
        src.head_dim,
    )
    assert tuple(state["indexer_overlap_kv"].shape) == (
        B,
        block.compress_rate,
        src.index_head_dim,
    )
    require_comparable(
        y.detach().to(torch.float32).cpu().numpy(), "csa_synthetic_output_fp32"
    )


def test_csa_wrapper_matches_hf_reference_on_real_layer2_tensors() -> None:
    """The whole enchilada: real HF weights, dequant, CSA wrapper vs
    reference forward, byte-clean bf16 diff on first-step output +
    identical top_k indices from Lightning Indexer."""
    cfg, convert, nw = _import_library()
    state = _load_layer2_state_dict()

    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    compress_rate = 4

    # Convert HF -> wrapper module tree.
    converted = convert._convert_csa_block(
        state, layer_idx=_LAYER_IDX, src=src, dtype=dtype, require_attn_sink=True
    )

    prefix_attn = f"layers.{_LAYER_IDX}.attn."
    expected_converted = set(
        [f"{prefix_attn}{k}" for k in nw._MQABlock.PARAM_KEYS]
        + [f"{prefix_attn}compressor.{k}" for k in nw._CSAOverlapCompressor.PARAM_KEYS]
        + [
            f"{prefix_attn}indexer.compressor.{k}"
            for k in nw._CSAOverlapCompressor.PARAM_KEYS
        ]
        + [
            f"{prefix_attn}indexer.wq_b.weight",
            f"{prefix_attn}indexer.weights_proj.weight",
        ]
        + [f"layers.{_LAYER_IDX}.attn_norm.weight"]
    )
    got_converted = set(converted.keys())
    assert got_converted == expected_converted, sorted(
        got_converted.symmetric_difference(expected_converted)
    )

    # Build the wrapper block; load dequanted tensors into its param tree.
    block = nw._CSABlock(src, layer_idx=_LAYER_IDX).to(dtype)
    load_report: dict[str, tuple[int, ...]] = {}
    with torch.no_grad():
        # MQA subtree.
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
        # CSA outer compressor subtree.
        for pname in nw._CSAOverlapCompressor.PARAM_KEYS:
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
        # Indexer's inner compressor subtree.
        for pname in nw._CSAOverlapCompressor.PARAM_KEYS:
            tensor = converted[f"{prefix_attn}indexer.compressor.{pname}"]
            parts = pname.split(".")
            obj = block.indexer.compressor
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            assert tuple(param.shape) == tuple(tensor.shape), (
                pname,
                tuple(param.shape),
                tuple(tensor.shape),
            )
            param.copy_(tensor.to(dtype))
            load_report[f"indexer.compressor.{pname}"] = tuple(tensor.shape)
        # Indexer projection params.
        for pname in ("wq_b.weight", "weights_proj.weight"):
            tensor = converted[f"{prefix_attn}indexer.{pname}"]
            parts = pname.split(".")
            obj = block.indexer
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            assert tuple(param.shape) == tuple(tensor.shape), (
                pname,
                tuple(param.shape),
                tuple(tensor.shape),
            )
            param.copy_(tensor.to(dtype))
            load_report[f"indexer.{pname}"] = tuple(tensor.shape)

    # Deterministic synthetic input.  S=16 gives exactly 4 CSA windows of
    # 4 tokens — the smallest non-trivial multi-window setup that
    # exercises the Ca overlap between windows 1..3.  With
    # `compressed_len=4 < index_topk=512`, `top_k = min(...) = 4`, so
    # every query picks all valid entries deterministically.  The test
    # asserts wrapper vs reference indices match element-wise.
    torch.manual_seed(0)
    B, S = 1, 16
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1
    positions = torch.arange(S).unsqueeze(0).expand(B, -1)
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

    # ---- wrapper forward (stateless first-step) ----
    with torch.no_grad():
        y_wrap, new_state = block(hidden, cos, sin, cos_win, sin_win, positions)
    assert tuple(y_wrap.shape) == (B, S, src.hidden_size)
    assert y_wrap.dtype == dtype

    # ---- reference forward ----
    ref_weights: dict[str, torch.Tensor] = {}
    for pname in nw._MQABlock.PARAM_KEYS:
        ref_weights[pname] = converted[f"{prefix_attn}{pname}"].to(dtype)
    for pname in nw._CSAOverlapCompressor.PARAM_KEYS:
        ref_weights[f"compressor.{pname}"] = converted[
            f"{prefix_attn}compressor.{pname}"
        ].to(dtype)
    for pname in nw._CSAOverlapCompressor.PARAM_KEYS:
        ref_weights[f"indexer.compressor.{pname}"] = converted[
            f"{prefix_attn}indexer.compressor.{pname}"
        ].to(dtype)
    for pname in ("wq_b.weight", "weights_proj.weight"):
        ref_weights[f"indexer.{pname}"] = converted[f"{prefix_attn}indexer.{pname}"].to(
            dtype
        )

    with torch.no_grad():
        y_ref, ref_t_compressed, ref_top_k = _ref_csa_forward(
            hidden,
            ref_weights,
            num_heads=src.num_attention_heads,
            head_dim=src.head_dim,
            o_groups=src.o_groups,
            o_lora_rank=src.o_lora_rank,
            rms_eps=src.rms_norm_eps,
            compress_rate=compress_rate,
            index_head_dim=src.index_head_dim,
            index_n_heads=src.index_n_heads,
            index_topk=src.index_topk,
            position_ids=positions,
            cos=cos,
            sin=sin,
            cos_win=cos_win,
            sin_win=sin_win,
        )
    assert ref_t_compressed == n_windows == 4

    # ---- degeneracy guard on BOTH sides ----
    require_comparable(
        y_wrap.detach().to(torch.float32).cpu().numpy(), "csa_wrapper_output_fp32"
    )
    require_comparable(
        y_ref.detach().to(torch.float32).cpu().numpy(), "csa_reference_output_fp32"
    )

    # ---- Lightning Indexer top-K correctness ----
    # Directly probe the wrapper's indexer to obtain its top_k indices.
    q_residual = block.mqa.project_q(hidden, cos, sin)[1]  # [B, S, q_lora_rank]
    with torch.no_grad():
        wrap_top_k, _, _ = block.indexer(
            hidden, q_residual, cos, sin, cos_win, sin_win, positions
        )
    assert tuple(wrap_top_k.shape) == (B, S, min(src.index_topk, n_windows))
    # top-K count = min(index_topk, compressed_len) = min(512, 4) = 4.
    assert wrap_top_k.shape[-1] == 4
    # Wrapper and reference indices must match element-wise (both use
    # torch.topk on identical fp32 scores).
    assert torch.equal(wrap_top_k, ref_top_k), (
        f"wrapper vs reference top-K mismatch: max abs diff = "
        f"{(wrap_top_k - ref_top_k).abs().max().item()}"
    )
    # Every non-sentinel index must be strictly less than the query's
    # causal threshold.
    causal_threshold = (positions + 1) // compress_rate  # [B, S]
    valid_mask = wrap_top_k >= 0
    max_causal = causal_threshold.unsqueeze(-1).expand_as(wrap_top_k)
    assert torch.all((~valid_mask) | (wrap_top_k < max_causal)), (
        "Lightning Indexer emitted a top-K index that violates causal_threshold"
    )

    # ---- byte-clean bf16 diff on first-step output ----
    diff = (y_wrap.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_abs_error_bf16 = float(diff.max().item())
    mean_abs_error_bf16 = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-csa-block-real-tensor",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "hf_shard": _HF_SHARD,
                "layer_idx": _LAYER_IDX,
                "layer_type": src.layer_types[_LAYER_IDX],
                "compress_rate": compress_rate,
                "batch_size": B,
                "seq_len": S,
                "n_windows_emitted": n_windows,
                "index_topk_effective": wrap_top_k.shape[-1],
                "wrapper_tree_key_count": len(list(block.named_parameters())),
                "converted_key_count": len(converted),
                "load_report_key_count": len(load_report),
                "hidden_size": src.hidden_size,
                "num_heads": src.num_attention_heads,
                "head_dim": src.head_dim,
                "index_head_dim": src.index_head_dim,
                "index_n_heads": src.index_n_heads,
                "compress_rope_theta": src.compress_rope_theta,
                "max_abs_error_bf16_firststep": max_abs_error_bf16,
                "mean_abs_error_bf16_firststep": mean_abs_error_bf16,
                "output_shape": tuple(y_wrap.shape),
            },
            indent=2,
        )
    )
    assert max_abs_error_bf16 < 1e-4, max_abs_error_bf16


def test_csa_state_evolution_multi_step() -> None:
    """State-aliasing verification: full-shot output must equal the
    concatenation of two half-length calls that carry overlap state.

    A wrong Ca/Cb update rule breaks this on the boundary window (the
    first window of the second call).  This is the load-bearing new
    discipline for CSA — same class of bug that caught GLM-5.3-Flash's
    KDA conv1d layout.
    """
    cfg, convert, nw = _import_library()
    state = _load_layer2_state_dict()

    src = cfg.DeepseekV4FlashInferenceConfig()
    dtype = torch.bfloat16
    compress_rate = 4

    converted = convert._convert_csa_block(
        state, layer_idx=_LAYER_IDX, src=src, dtype=dtype, require_attn_sink=True
    )
    prefix_attn = f"layers.{_LAYER_IDX}.attn."

    block = nw._CSABlock(src, layer_idx=_LAYER_IDX).to(dtype)
    with torch.no_grad():
        for pname in nw._MQABlock.PARAM_KEYS:
            parts = pname.split(".")
            obj = block.mqa
            for p in parts[:-1]:
                obj = getattr(obj, p)
            getattr(obj, parts[-1]).copy_(converted[f"{prefix_attn}{pname}"].to(dtype))
        for pname in nw._CSAOverlapCompressor.PARAM_KEYS:
            parts = pname.split(".")
            obj = block.compressor
            for p in parts[:-1]:
                obj = getattr(obj, p)
            getattr(obj, parts[-1]).copy_(
                converted[f"{prefix_attn}compressor.{pname}"].to(dtype)
            )
        for pname in nw._CSAOverlapCompressor.PARAM_KEYS:
            parts = pname.split(".")
            obj = block.indexer.compressor
            for p in parts[:-1]:
                obj = getattr(obj, p)
            getattr(obj, parts[-1]).copy_(
                converted[f"{prefix_attn}indexer.compressor.{pname}"].to(dtype)
            )
        for pname in ("wq_b.weight", "weights_proj.weight"):
            parts = pname.split(".")
            obj = block.indexer
            for p in parts[:-1]:
                obj = getattr(obj, p)
            getattr(obj, parts[-1]).copy_(
                converted[f"{prefix_attn}indexer.{pname}"].to(dtype)
            )

    torch.manual_seed(7)
    B, S = 1, 16
    hidden = torch.randn(B, S, src.hidden_size, dtype=dtype) * 0.1

    # ---- Path A: full-shot compressor.compress on the entire sequence ----
    positions_full = torch.arange(S).unsqueeze(0).expand(B, -1)
    n_win_full = S // compress_rate
    wp_full = torch.arange(n_win_full).unsqueeze(0).expand(B, -1) * compress_rate
    cw_full, sw_full = nw.build_main_rope_cos_sin(
        wp_full,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    with torch.no_grad():
        comp_full, _, _ = block.compressor.compress(
            hidden,
            cw_full,
            sw_full,
            overlap_kv_prev=None,
            overlap_gate_prev=None,
        )
        idx_full, _, _ = block.indexer.compressor.compress(
            hidden,
            cw_full,
            sw_full,
            overlap_kv_prev=None,
            overlap_gate_prev=None,
        )
    assert comp_full.shape == (B, 1, n_win_full, src.head_dim)
    assert idx_full.shape == (B, 1, n_win_full, src.index_head_dim)

    # ---- Path B: split into two chunks, carry overlap state.
    # Chunk 1: hidden[:, :8], window positions [0, 4].
    # Chunk 2: hidden[:, 8:], window positions [8, 12].
    # For Path B to match Path A, the compressor MUST see window positions
    # in the SAME absolute frame — chunk 2's local windows are absolute
    # windows 2 and 3 → positions [8, 12], not [0, 4].  This is what HF's
    # `first_window_position = entry_count * compress_rate` accomplishes.
    S_chunk = 8
    n_win_chunk = S_chunk // compress_rate
    # Chunk 1 window positions [0, 4].
    wp1 = torch.arange(n_win_chunk).unsqueeze(0).expand(B, -1) * compress_rate  # [0, 4]
    cw1, sw1 = nw.build_main_rope_cos_sin(
        wp1,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    # Chunk 2 window positions [8, 12] (absolute).
    wp2 = wp1 + S_chunk  # [8, 12]
    cw2, sw2 = nw.build_main_rope_cos_sin(
        wp2,
        rope_dim=src.qk_rope_head_dim,
        rope_theta=src.compress_rope_theta,
        dtype=dtype,
    )
    with torch.no_grad():
        # ---- outer CSA compressor split ----
        comp_c1, new_comp_kv, new_comp_gate = block.compressor.compress(
            hidden[:, :S_chunk],
            cw1,
            sw1,
            overlap_kv_prev=None,
            overlap_gate_prev=None,
        )
        comp_c2, _, _ = block.compressor.compress(
            hidden[:, S_chunk:],
            cw2,
            sw2,
            overlap_kv_prev=new_comp_kv,
            overlap_gate_prev=new_comp_gate,
        )
        comp_concat = torch.cat([comp_c1, comp_c2], dim=2)
        # ---- indexer inner compressor split ----
        idx_c1, new_idx_kv, new_idx_gate = block.indexer.compressor.compress(
            hidden[:, :S_chunk],
            cw1,
            sw1,
            overlap_kv_prev=None,
            overlap_gate_prev=None,
        )
        idx_c2, _, _ = block.indexer.compressor.compress(
            hidden[:, S_chunk:],
            cw2,
            sw2,
            overlap_kv_prev=new_idx_kv,
            overlap_gate_prev=new_idx_gate,
        )
        idx_concat = torch.cat([idx_c1, idx_c2], dim=2)

    require_comparable(
        comp_full.detach().to(torch.float32).cpu().numpy(),
        "csa_comp_full_fp32",
    )
    require_comparable(
        comp_concat.detach().to(torch.float32).cpu().numpy(),
        "csa_comp_concat_fp32",
    )
    require_comparable(
        idx_full.detach().to(torch.float32).cpu().numpy(),
        "csa_idx_full_fp32",
    )
    require_comparable(
        idx_concat.detach().to(torch.float32).cpu().numpy(),
        "csa_idx_concat_fp32",
    )

    comp_diff = (comp_full.to(torch.float32) - comp_concat.to(torch.float32)).abs()
    idx_diff = (idx_full.to(torch.float32) - idx_concat.to(torch.float32)).abs()
    comp_max = float(comp_diff.max().item())
    idx_max = float(idx_diff.max().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-csa-state-evolution",
                "compressor_max_abs_error_bf16_multistep": comp_max,
                "indexer_compressor_max_abs_error_bf16_multistep": idx_max,
                "seq_len": S,
                "chunk_size": S_chunk,
                "n_windows_full": n_win_full,
                "compress_rate": compress_rate,
            },
            indent=2,
        )
    )
    assert comp_max < 1e-4, comp_max
    assert idx_max < 1e-4, idx_max


# ---------------------------------------------------------------------------
# Standalone runner — mirrors test_hca_1layer.py so a laptop without
# pytest-collectible `vllm_neuron` can still run the gate.
# ---------------------------------------------------------------------------


def _standalone_main() -> int:
    tests = [
        test_csa_layer_schedule_confirms_layer_2,
        test_csa_wrapper_tree_key_set,
        test_csa_refuses_wrong_layer_type,
        test_csa_state_cache_specs_shape_and_dtype,
        test_csa_synthetic_shape_gate,
        test_csa_wrapper_matches_hf_reference_on_real_layer2_tensors,
        test_csa_state_evolution_multi_step,
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
                "suite": "dsv4-flash.tests.test_csa_1layer",
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
