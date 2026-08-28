# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash streaming per-rank checkpoint sharder (Round 6).

Round 5 rewrote `_convert_glm53_checkpoint` to emit the Round-4 module
tree correctly, but the eager path materialises a ~611 GiB BF16 dict —
larger than the compile host's RAM.  Round 6 replaces that eager path
with a per-rank streaming writer.

For each rank ``r`` in ``[0, tp_degree)``:

  1. Open every relevant HF FP8 safetensors shard via ``safe_open(mmap=True)``
     (kernel-cached, no bulk read).
  2. For each target parameter of the Round-4 wrapper, fetch the HF
     tensor(s) needed for that param, dequantize any FP8 blocks to bf16
     via ``dequantize_block_fp8`` (byte-identical to Round-5's smoke —
     ``max_abs_error_bf16 == 0.0``), and **slice the rank-r partition
     immediately**.  Full tensors are freed as soon as the slice is
     computed.
  3. Save the rank's dict to
     ``{compiled_model_path}/weights/tp{r}_sharded_checkpoint.safetensors``
     (matches NxDI's load-time contract at
     ``application_base.load_weights`` line ~396).
  4. Free the rank dict; ``gc.collect()``; next rank.

Peak resident BF16 tensor is one full expert weight
(``[intermediate, hidden] = [2048, 4096]`` bf16 = ~16 MiB) per moment.
Peak per-rank output buffer is the rank's ~19 GiB slice of the 611 GiB
BF16 checkpoint; that is what gets flushed to disk each iteration.
Total working set stays well under 40 GiB, far below the 100 GiB budget
the coordinator set.

Sharding rules (read off the Round-4 wrapper's declared modules):

  * ``embed_tokens.weight``: ``ParallelEmbedding(shard_across_embedding=True,
    pad=True)``.  Sharded on **dim=0** (vocab axis).  ``pad=True`` rounds
    up to a TP-multiple; NxDI handles the pad at load-time via
    ``_shard_and_pad_dim0``, so we emit the full unsharded vocab tensor
    for that key and let NxDI's loader handle the pad.  (Simpler than
    reproducing NxDI's pad-and-shard math here.)
  * ``final_norm_weight``: replicated (each rank keeps the full tensor).
  * ``lm_head.weight``: ``ColumnParallelLinear(pad=True)``.  Same story
    — emit the unsharded tensor.
  * Per-layer BF16 (``input_norm_weight``, ``post_attention_norm_weight``,
    ``hc_attn.{base,fn,scale}``, ``hc_mlp.{base,fn,scale}``): replicated.
  * KDA q_proj / k_proj / v_proj / b_proj / f_a_proj / f_b_proj /
    g_a_proj / g_b_proj: ``ColumnParallelLinear`` on dim=0 (output axis).
    Rank r takes ``weight[r*out_per_r:(r+1)*out_per_r, :]``.
  * KDA o_proj: ``RowParallelLinear`` on dim=1 (input axis).  Rank r
    takes ``weight[:, r*in_per_r:(r+1)*in_per_r]``.
  * KDA ``A_log``, ``dt_bias``, ``o_norm_weight``: replicated (the
    KDA forward code takes a rank-local slice at runtime via
    ``self._local_heads()`` — the parameter is the full-width tensor).
  * KDA conv1d.weight: fused per-head across streams.  The Round-5
    converter concatenated the three per-stream conv weights on dim=0
    (channel axis), which produced ``[q_all_channels, k_all_channels,
    v_all_channels]``.  The wrapper's KDA forward expects the ordering
    ``[for each local head h: q_h, k_h, v_h]`` — split of the conv
    output by ``head_dim`` on the trailing axis gives ``(q_c, k_c, v_c)``
    per head.  This module produces that per-head-interleaved layout,
    then slices to rank-local heads.  (Fixes a Round-5 KDA layout bug
    the smoke could not catch — the wrapper's forward
    ``convolved.view(batch, length, heads_per_rank, 3 * head_dim)``
    only makes sense under per-head-interleaved channels.)
  * MLA q_a/q_b/kv_a/o: ColumnParallel (q_a/q_b/kv_a) or RowParallel
    (o) with the appropriate axis; sizes vary per layer.
  * MLA kv_b_proj: BF16 in HF (no scale) — replicated to
    ``kv_b_proj.weight``.  Wrapper reshapes at forward.
  * MLA q_a_layernorm / kv_a_layernorm: replicated (small RMS gain).
  * Indexer k_proj: ``ColumnParallelLinear(gather_output=True)`` on
    dim=0.  Each rank gets a slice but the forward gathers — so we
    still need per-rank slice.
  * Indexer q_proj: rank-3 param ``[pooled_index_heads, heads_per_rank
    * qk_head_dim, index_head_dim]``.  The middle axis is rank-local
    (heads_per_rank × qk_head_dim per rank).
  * Indexer pool_weights: replicated (small ``[index_kpool]`` param).
  * Dense MLP gate/up/down (layers 0-2): ColumnParallel / RowParallel
    on the intermediate axis (``intermediate_size = 12288``).  Rank r
    takes 12288/TP columns.
  * Shared expert gate/up/down: same as dense MLP but on
    ``moe_intermediate_size = 2048``.  Rank r takes 2048/TP.
  * Router weight (``mlp.router.weight``): shape ``[n_routed_experts,
    hidden]`` = ``[288, 4096]``.  The wrapper's ``self.router`` is a
    plain ``nn.Linear`` — no TP.  Replicated.
  * ``e_score_correction_bias``: ``[n_routed_experts]``, replicated
    (fp32 — the wrapper's parameter dtype).
  * MoE ``expert_mlps.mlp_op.gate_up_proj.weight`` — the dominant
    class.  Full shape ``[288, hidden, 2*moe_intermediate] = [288,
    4096, 4096]`` bf16 = 9.66 GiB per layer.  Full 42 MoE layers ×
    9.66 = 405 GiB.  Rank shard: NxDI ``stride=2`` semantics.
    Per rank r at TP=32:
      * Divide the 2×intermediate=4096-column fused axis into
        ``2*TP = 64`` chunks of ``intermediate/TP = 64`` cols each.
      * Rank r gets chunks ``[r, TP+r]`` — that is, gate cols
        ``[r*I_TP : (r+1)*I_TP]`` concatenated with up cols
        ``[r*I_TP : (r+1)*I_TP]``.
      * Rank-r tensor: ``[288, 4096, 2*I_TP] = [288, 4096, 128]``
        (bf16 = 302 MiB per layer per rank).
    We produce this without ever holding the full 9.66 GiB slab: for
    each expert e, we dequant gate + up in-place, slice rank-r cols,
    concat, store into the rank's stacked accumulator.
  * MoE ``down_proj.weight``: full ``[288, moe_intermediate, hidden]
    = [288, 2048, 4096]``, RowParallelLinear on dim=1 (moe_intermediate
    axis).  Rank r gets ``[288, I_TP, hidden] = [288, 64, 4096]`` bf16
    = 151 MiB per layer per rank.

All slice offsets keyed off the wrapper's declared constants so a
future TP change (16 or 8) reroutes without touching this file.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoint_convert import (
    HC_PARAM_SUFFIXES,
    dequantize_block_fp8,
)
from .config import Glm53FlashInferenceConfig


def _load_hf_index(model_path: str) -> dict[str, str]:
    """Return ``{hf_key: shard_filename}`` from the HF safetensors index."""
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as fh:
        return json.load(fh)["weight_map"]


def _open_shards(model_path: str, shards: set[str]) -> dict[str, Any]:
    """Open a set of shard safe_open handles.  Caller closes via ``_close_shards``."""
    handles: dict[str, Any] = {}
    for shard in shards:
        p = os.path.join(model_path, shard)
        handles[shard] = safe_open(p, framework="pt", device="cpu")
    return handles


def _close_shards(handles: dict[str, Any]) -> None:
    for h in handles.values():
        try:
            h.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


def _fetch_and_dequant(
    hf_key: str,
    weight_map: dict[str, str],
    handles: dict[str, Any],
    block_size: tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Fetch an HF tensor by name; dequant FP8 -> ``out_dtype`` if a scale exists."""
    shard = weight_map.get(hf_key)
    if shard is None:
        return None
    h = handles[shard]
    weight = h.get_tensor(hf_key)
    scale_key = f"{hf_key}_scale_inv"
    if scale_key in weight_map:
        scale_shard = weight_map[scale_key]
        scale = handles[scale_shard].get_tensor(scale_key)
        return dequantize_block_fp8(weight, scale, block_size, out_dtype)
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            f"{hf_key} is FP8 but has no paired {scale_key}; refusing to load"
        )
    return weight.to(out_dtype)


def _row_shard(
    tensor: torch.Tensor, rank: int, tp_degree: int, dim: int
) -> torch.Tensor:
    """Return rank's contiguous shard along ``dim``.  No stride handling."""
    n = tensor.shape[dim]
    if n % tp_degree != 0:
        raise ValueError(
            f"cannot shard dim {dim} of size {n} across TP={tp_degree}"
        )
    step = n // tp_degree
    slicer = [slice(None)] * tensor.ndim
    slicer[dim] = slice(rank * step, (rank + 1) * step)
    return tensor[tuple(slicer)].contiguous()


def _write_rank_dict(
    rank_dict: dict[str, torch.Tensor], serialize_path: str, rank: int
) -> tuple[str, int]:
    out_path = os.path.join(
        serialize_path, f"tp{rank}_sharded_checkpoint.safetensors"
    )
    # save_file requires contiguous tensors; enforce here rather than at
    # every producer.
    contiguous = {k: (v.contiguous() if not v.is_contiguous() else v) for k, v in rank_dict.items()}
    save_file(contiguous, out_path)
    return out_path, os.path.getsize(out_path)


def _kda_conv1d_per_head_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Assemble the wrapper's expected per-head-interleaved conv1d channel layout.

    HF stores per-stream conv weights as ``[num_heads*head_dim, 1, K]`` with
    channels indexed ``h*head_dim + d`` (head-major).  The wrapper's KDA
    forward reads the conv output as
    ``convolved.view(batch, length, heads_per_rank, 3 * head_dim)`` and
    then ``torch.split(..., head_dim, dim=-1)`` — which only produces
    ``(q_c, k_c, v_c)`` correctly when the conv channels are laid out as
    ``[for h in heads: q_h[0..D-1], k_h[0..D-1], v_h[0..D-1]]``.

    Round 5's converter used ``torch.cat([q, k, v], dim=0)`` which yields
    the WRONG layout (stream-major, not head-major-interleaved).  Correct
    it here:

      * View each stream as ``[num_heads, head_dim, 1, K]``.
      * Stack along a new stream axis at dim=1: ``[num_heads, 3, head_dim,
        1, K]``.
      * Reshape to ``[num_heads * 3 * head_dim, 1, K]``.

    Result channel ``h*3*head_dim + s*head_dim + d`` matches the wrapper's
    forward view.
    """
    K = q.shape[-1]
    q = q.view(num_heads, head_dim, 1, K)
    k = k.view(num_heads, head_dim, 1, K)
    v = v.view(num_heads, head_dim, 1, K)
    combined = torch.stack([q, k, v], dim=1)  # [num_heads, 3, head_dim, 1, K]
    return combined.reshape(num_heads * 3 * head_dim, 1, K)


def stream_shard_glm53(
    model_path: str,
    serialize_path: str,
    src: Glm53FlashInferenceConfig,
    tp_degree: int,
    *,
    logger: Any = None,
    ranks: list[int] | None = None,
) -> dict[str, Any]:
    """Stream-write per-rank sharded safetensors from a GLM-5.3-Flash HF snapshot.

    Args:
      model_path: HF snapshot directory (contains ``model.safetensors.index.json``
        and ``model-*-of-*.safetensors``).
      serialize_path: output dir; per-rank files land at
        ``tp{rank}_sharded_checkpoint.safetensors``.
      src: frozen GLM-5.3-Flash source config (declares hidden, intermediate,
        num_heads, etc.).
      tp_degree: TP degree — controls sharding factor across dim-0/dim-1.
      logger: optional logger; if provided, ``.info()`` is called per phase.
      ranks: which ranks to write (default: all ``0..tp_degree-1``).  Useful
        when running on multiple hosts.

    Returns:
      A conversion report dict with per-rank paths, sizes, timings, and a
      ``blocker`` list of unmapped tensors.
    """
    os.makedirs(serialize_path, exist_ok=True)
    log = (lambda m: logger.info(m)) if logger is not None else print
    dtype = src.torch_dtype
    block = tuple(src.quantization_config.weight_block_size)
    H = src.hidden_size
    Vsz = src.vocab_size
    D_h = src.linear_attn_config.head_dim
    num_heads = src.linear_attn_config.num_heads
    kernel = src.linear_attn_config.short_conv_kernel_size
    qkv_dim = num_heads * D_h  # 64 * 128 = 8192
    I_dense = src.intermediate_size  # 12288
    I_moe = src.moe_intermediate_size  # 2048
    n_experts = src.n_routed_experts  # 288
    num_layers = src.num_hidden_layers  # 45
    # MLA sizes.
    n_attn_heads = src.num_attention_heads  # 64
    qk_head_dim = src.qk_head_dim  # 256
    qk_nope = src.qk_nope_head_dim  # 256
    v_head_dim = src.v_head_dim  # 256
    q_lora = src.q_lora_rank  # 1536
    kv_lora = src.kv_lora_rank  # 512
    # DSA indexer sizes.
    idx_n_heads = src.index_n_heads  # 32
    idx_head_dim = src.index_head_dim  # 128
    idx_kpool = src.index_kpool  # 4

    if tp_degree <= 0:
        raise ValueError(f"tp_degree must be positive; got {tp_degree}")
    if num_heads % tp_degree != 0:
        raise ValueError(
            f"KDA num_heads {num_heads} not divisible by TP={tp_degree}"
        )
    if n_attn_heads % tp_degree != 0:
        raise ValueError(
            f"MLA num_attention_heads {n_attn_heads} not divisible by TP={tp_degree}"
        )
    if I_dense % tp_degree != 0:
        raise ValueError(
            f"Dense intermediate_size {I_dense} not divisible by TP={tp_degree}"
        )
    if I_moe % tp_degree != 0:
        raise ValueError(
            f"MoE intermediate_size {I_moe} not divisible by TP={tp_degree}"
        )

    heads_per_rank_kda = num_heads // tp_degree
    heads_per_rank_mla = n_attn_heads // tp_degree
    I_dense_per_tp = I_dense // tp_degree
    I_moe_per_tp = I_moe // tp_degree
    qkv_per_rank = heads_per_rank_kda * D_h  # KDA output per rank

    ranks_iter = list(range(tp_degree)) if ranks is None else list(ranks)

    report: dict[str, Any] = {
        "model_path": model_path,
        "serialize_path": serialize_path,
        "tp_degree": tp_degree,
        "ranks_written": [],
        "unmapped": [],
        "rank_bytes": {},
        "rank_wall_s": {},
        "start_unix": int(time.time()),
    }

    weight_map = _load_hf_index(model_path)
    # Prefetch shards actually touched (avoid opening files we do not need).
    text_prefix = "model.language_model."
    lm_head_key = "lm_head.weight"

    def _hf(k: str) -> str:
        """Return the HF key with the model.language_model. prefix."""
        return f"{text_prefix}{k}"

    # Build a set of the shards we'll need for text tensors + lm_head.
    needed_shards: set[str] = set()
    for hk, shard in weight_map.items():
        if hk.startswith(text_prefix):
            # Drop layer 45 (MTP) and vision (already excluded by prefix).
            if ".layers." in hk:
                # cheap layer index parse
                after = hk.split(".layers.", 1)[1]
                try:
                    li = int(after.split(".", 1)[0])
                except ValueError:
                    li = -1
                if li == 45:
                    continue
            needed_shards.add(shard)
        elif hk == lm_head_key:
            needed_shards.add(shard)
    log(f"[stream_shard] opening {len(needed_shards)} FP8 shards mmap-lazy")
    handles = _open_shards(model_path, needed_shards)

    def fetch(k: str) -> torch.Tensor | None:
        return _fetch_and_dequant(k, weight_map, handles, block, dtype)

    def fetch_fp32(k: str) -> torch.Tensor | None:
        t = _fetch_and_dequant(k, weight_map, handles, block, torch.float32)
        return t

    try:
        for rank in ranks_iter:
            t0 = time.time()
            rank_dict: dict[str, torch.Tensor] = {}
            _emit_rank(
                rank=rank,
                tp_degree=tp_degree,
                src=src,
                fetch=fetch,
                fetch_fp32=fetch_fp32,
                weight_map=weight_map,
                rank_dict=rank_dict,
                report=report,
                dtype=dtype,
            )
            path, nbytes = _write_rank_dict(rank_dict, serialize_path, rank)
            wall = time.time() - t0
            report["ranks_written"].append(rank)
            report["rank_bytes"][str(rank)] = nbytes
            report["rank_wall_s"][str(rank)] = round(wall, 2)
            log(
                f"[stream_shard] rank {rank}: wrote {nbytes/1e9:.2f} GB in {wall:.1f}s "
                f"({path})"
            )
            del rank_dict
            gc.collect()
    finally:
        _close_shards(handles)

    report["end_unix"] = int(time.time())
    report["total_wall_s"] = report["end_unix"] - report["start_unix"]
    return report


def _emit_rank(
    *,
    rank: int,
    tp_degree: int,
    src: Glm53FlashInferenceConfig,
    fetch,
    fetch_fp32,
    weight_map: dict[str, str],
    rank_dict: dict[str, torch.Tensor],
    report: dict[str, Any],
    dtype: torch.dtype,
) -> None:
    """Populate ``rank_dict`` with the wrapper's declared param subset for ``rank``."""
    H = src.hidden_size
    D_h = src.linear_attn_config.head_dim
    num_heads = src.linear_attn_config.num_heads
    kernel = src.linear_attn_config.short_conv_kernel_size
    qkv_dim = num_heads * D_h
    heads_per_rank_kda = num_heads // tp_degree
    qkv_per_rank = heads_per_rank_kda * D_h
    I_dense = src.intermediate_size
    I_moe = src.moe_intermediate_size
    n_experts = src.n_routed_experts
    num_layers = src.num_hidden_layers
    n_attn_heads = src.num_attention_heads
    qk_head_dim = src.qk_head_dim
    qk_nope = src.qk_nope_head_dim
    v_head_dim = src.v_head_dim
    q_lora = src.q_lora_rank
    kv_lora = src.kv_lora_rank
    idx_n_heads = src.index_n_heads
    idx_head_dim = src.index_head_dim

    # --- top-level ---
    embed = fetch("model.language_model.embed_tokens.weight")
    if embed is None:
        raise RuntimeError("missing embed_tokens in HF index")
    # ParallelEmbedding(shard_across_embedding=True): shard the vocab axis
    # (dim=0).  vocab_size=154880 is TP-divisible for TP in {8,16,32} so no
    # padding needed.
    if embed.shape[0] % tp_degree != 0:
        raise ValueError(
            f"embed_tokens vocab {embed.shape[0]} not divisible by TP={tp_degree}; "
            "pre-shard pad not implemented — bump vocab or add pad here"
        )
    rank_dict["embed_tokens.weight"] = _row_shard(embed, rank, tp_degree, dim=0)
    fn = fetch("model.language_model.norm.weight")
    if fn is None:
        raise RuntimeError("missing final norm")
    rank_dict["final_norm_weight"] = fn  # replicated
    lm = fetch("lm_head.weight")
    if lm is None:
        if getattr(src, "tie_word_embeddings", False):
            lm = embed
        else:
            raise RuntimeError("missing lm_head.weight and tie_word_embeddings is False")
    # ColumnParallelLinear(pad=True) on lm_head: shard vocab (dim=0).
    if lm.shape[0] % tp_degree != 0:
        raise ValueError(
            f"lm_head vocab {lm.shape[0]} not divisible by TP={tp_degree}"
        )
    rank_dict["lm_head.weight"] = _row_shard(lm, rank, tp_degree, dim=0)

    # --- per layer ---
    for L in range(num_layers):
        base = f"model.language_model.layers.{L}."
        out = f"layers.{L}."

        # Norms (replicated bf16).
        for hk, tk in (
            ("input_layernorm.weight", "input_norm_weight"),
            ("post_attention_layernorm.weight", "post_attention_norm_weight"),
        ):
            t = fetch(f"{base}{hk}")
            if t is not None:
                rank_dict[f"{out}{tk}"] = t

        # mHC params (replicated bf16).
        for suf in HC_PARAM_SUFFIXES:
            t = fetch(f"{base}{suf}")
            if t is None:
                continue
            target = "hc_attn" if "attn" in suf else "hc_mlp"
            leaf = suf.split("_", 2)[-1]
            rank_dict[f"{out}{target}.{leaf}"] = t

        # Attention branch.
        attn_kind = src.layer_types[L]
        if attn_kind == "linear_attention":
            _emit_kda_rank(
                base=base,
                out=out,
                rank=rank,
                tp_degree=tp_degree,
                fetch=fetch,
                rank_dict=rank_dict,
                num_heads=num_heads,
                head_dim=D_h,
                qkv_dim=qkv_dim,
                qkv_per_rank=qkv_per_rank,
                heads_per_rank=heads_per_rank_kda,
                kernel=kernel,
                dtype=dtype,
            )
        elif attn_kind == "deepseek_sparse_attention":
            _emit_dsa_rank(
                base=base,
                out=out,
                rank=rank,
                tp_degree=tp_degree,
                fetch=fetch,
                rank_dict=rank_dict,
                n_attn_heads=n_attn_heads,
                qk_head_dim=qk_head_dim,
                qk_nope=qk_nope,
                v_head_dim=v_head_dim,
                q_lora=q_lora,
                kv_lora=kv_lora,
                idx_n_heads=idx_n_heads,
                idx_head_dim=idx_head_dim,
                report=report,
            )
        else:
            raise ValueError(f"unknown layer_type {attn_kind!r} at layer {L}")

        # MLP branch.
        mlp_kind = src.mlp_layer_types[L]
        if mlp_kind == "dense":
            _emit_dense_mlp_rank(
                base=base,
                out=out,
                rank=rank,
                tp_degree=tp_degree,
                fetch=fetch,
                rank_dict=rank_dict,
                hidden=H,
                intermediate=I_dense,
            )
        elif mlp_kind == "sparse":
            _emit_moe_rank(
                base=base,
                out=out,
                rank=rank,
                tp_degree=tp_degree,
                fetch=fetch,
                fetch_fp32=fetch_fp32,
                rank_dict=rank_dict,
                hidden=H,
                moe_intermediate=I_moe,
                n_experts=n_experts,
            )
        else:
            raise ValueError(f"unknown mlp_type {mlp_kind!r} at layer {L}")


def _emit_kda_rank(
    *,
    base: str,
    out: str,
    rank: int,
    tp_degree: int,
    fetch,
    rank_dict: dict[str, torch.Tensor],
    num_heads: int,
    head_dim: int,
    qkv_dim: int,
    qkv_per_rank: int,
    heads_per_rank: int,
    kernel: int,
    dtype: torch.dtype,
) -> None:
    attn = f"{base}self_attn."
    target = f"{out}self_attn."

    # Column-parallel projections (sharded on dim=0 output axis).
    for hk_suffix, tk_suffix in (
        ("q_proj.weight", "q_proj.weight"),
        ("k_proj.weight", "k_proj.weight"),
        ("v_proj.weight", "v_proj.weight"),
        ("b_proj.weight", "b_proj.weight"),
        ("f_b_proj.weight", "f_b_proj.weight"),
        ("g_b_proj.weight", "g_b_proj.weight"),
    ):
        t = fetch(f"{attn}{hk_suffix}")
        if t is None:
            continue
        rank_dict[f"{target}{tk_suffix}"] = _row_shard(t, rank, tp_degree, dim=0)

    # f_a_proj / g_a_proj: ColumnParallel with gather_output=True.  The
    # forward gathers across ranks, so each rank still stores its own
    # dim=0 shard.
    for hk_suffix, tk_suffix in (
        ("f_a_proj.weight", "f_a_proj.weight"),
        ("g_a_proj.weight", "g_a_proj.weight"),
    ):
        t = fetch(f"{attn}{hk_suffix}")
        if t is None:
            continue
        rank_dict[f"{target}{tk_suffix}"] = _row_shard(t, rank, tp_degree, dim=0)

    # o_proj: RowParallelLinear (input axis sharded).
    t = fetch(f"{attn}o_proj.weight")
    if t is not None:
        rank_dict[f"{target}o_proj.weight"] = _row_shard(t, rank, tp_degree, dim=1)

    # Replicated small params.
    for hk_suffix, tk_suffix in (
        ("A_log", "A_log"),
        ("dt_bias", "dt_bias"),
        ("o_norm.weight", "o_norm_weight"),
    ):
        t = fetch(f"{attn}{hk_suffix}")
        if t is not None:
            rank_dict[f"{target}{tk_suffix}"] = t

    # Fused KDA conv1d — per-head-interleaved layout, sliced to rank heads.
    parts = []
    for stream in ("q", "k", "v"):
        t = fetch(f"{attn}{stream}_conv1d.weight")
        if t is None:
            parts = []
            break
        parts.append(t)
    if parts:
        q, k, v = parts
        full = _kda_conv1d_per_head_layout(q, k, v, num_heads, head_dim)
        # Full-width shape: [num_heads * 3 * head_dim, 1, K]. Rank slice:
        # heads [r*hpr : (r+1)*hpr], each contributing 3*head_dim channels.
        full_view = full.view(num_heads, 3 * head_dim, 1, kernel)
        rank_slice = full_view[rank * heads_per_rank : (rank + 1) * heads_per_rank]
        rank_dict[f"{target}conv1d.weight"] = rank_slice.reshape(
            heads_per_rank * 3 * head_dim, 1, kernel
        ).contiguous()


def _emit_dsa_rank(
    *,
    base: str,
    out: str,
    rank: int,
    tp_degree: int,
    fetch,
    rank_dict: dict[str, torch.Tensor],
    n_attn_heads: int,
    qk_head_dim: int,
    qk_nope: int,
    v_head_dim: int,
    q_lora: int,
    kv_lora: int,
    idx_n_heads: int,
    idx_head_dim: int,
    report: dict[str, Any],
) -> None:
    attn = f"{base}self_attn."
    target = f"{out}self_attn."

    # q_a_proj / q_b_proj / kv_a_proj / o_proj (FP8 in HF; dequant to bf16).
    for hk_suffix, tk_suffix, tp_dim in (
        ("q_a_proj.weight", "mla.q_a_proj.weight", 0),   # ColumnParallel dim=0
        ("q_b_proj.weight", "mla.q_b_proj.weight", 0),
        ("kv_a_proj_with_mqa.weight", "mla.kv_a_proj.weight", 0),
        ("o_proj.weight", "mla.o_proj.weight", 1),        # RowParallel dim=1
    ):
        t = fetch(f"{attn}{hk_suffix}")
        if t is None:
            continue
        rank_dict[f"{target}{tk_suffix}"] = _row_shard(t, rank, tp_degree, dim=tp_dim)

    # kv_b_proj is BF16 in HF (no scale).  Wrapper's kv_b_proj is
    # ColumnParallelLinear so still shard on dim=0.
    t = fetch(f"{attn}kv_b_proj.weight")
    if t is not None:
        rank_dict[f"{target}mla.kv_b_proj.weight"] = _row_shard(t, rank, tp_degree, dim=0)

    # MLA latent norms — replicated small vectors.
    for hk_suffix, tk_suffix in (
        ("q_a_layernorm.weight", "mla.q_a_norm"),
        ("kv_a_layernorm.weight", "mla.kv_a_norm"),
    ):
        t = fetch(f"{attn}{hk_suffix}")
        if t is not None:
            rank_dict[f"{target}{tk_suffix}"] = t

    # Round-7 DSA indexer sharding.  Every projection is
    # ``ColumnParallelLinear(gather_output=True)`` (see ``_DSAIndexerBlock``
    # in ``neuron_wrapper.py``); each rank stores a per-rank dim=0 slice and
    # the forward all-gathers on the output.  ``k_norm`` and both
    # ``index_kpool_compress_*`` params are replicated.
    #
    # ``pool_weights`` and the rank-3 ``q_proj`` scaffolds are gone in Round
    # 7: they mapped provably-lossily onto HF's ``weights_proj`` (per-token,
    # per-head) and ``wq_b`` (low-rank off q_lora).  See
    # ``DSA-INDEXER-MAPPING-FIX-2026-08-28.md`` for the derivation that
    # rules out Option B (converter-side reformulation).
    for hf_suffix, tp_dim in (
        ("indexer.wq_b.weight", 0),
        ("indexer.wk.weight", 0),
        ("indexer.weights_proj.weight", 0),
    ):
        t = fetch(f"{attn}{hf_suffix}")
        if t is not None:
            rank_dict[f"{target}{hf_suffix}"] = _row_shard(
                t, rank, tp_degree, dim=tp_dim
            )

    for hf_suffix in (
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.index_kpool_compress_ape",
        "indexer.index_kpool_compress_gate",
    ):
        t = fetch(f"{attn}{hf_suffix}")
        if t is not None:
            rank_dict[f"{target}{hf_suffix}"] = t


def _emit_dense_mlp_rank(
    *,
    base: str,
    out: str,
    rank: int,
    tp_degree: int,
    fetch,
    rank_dict: dict[str, torch.Tensor],
    hidden: int,
    intermediate: int,
) -> None:
    mlp = f"{base}mlp."
    target = f"{out}mlp."
    # gate_proj: ColumnParallel dim=0. up_proj: same.  down_proj: RowParallel dim=1.
    for hk_suffix, tk_suffix, tp_dim in (
        ("gate_proj.weight", "gate_proj.weight", 0),
        ("up_proj.weight", "up_proj.weight", 0),
        ("down_proj.weight", "down_proj.weight", 1),
    ):
        t = fetch(f"{mlp}{hk_suffix}")
        if t is None:
            continue
        rank_dict[f"{target}{tk_suffix}"] = _row_shard(t, rank, tp_degree, dim=tp_dim)


def _emit_moe_rank(
    *,
    base: str,
    out: str,
    rank: int,
    tp_degree: int,
    fetch,
    fetch_fp32,
    rank_dict: dict[str, torch.Tensor],
    hidden: int,
    moe_intermediate: int,
    n_experts: int,
) -> None:
    mlp = f"{base}mlp."
    target = f"{out}mlp."

    # Router (replicated fp32 — matches nn.Linear with dtype=float32 in the wrapper).
    router = fetch_fp32(f"{mlp}gate.weight")
    if router is not None:
        rank_dict[f"{target}router.weight"] = router

    # e_score_correction_bias (replicated fp32).
    corr = fetch_fp32(f"{mlp}gate.e_score_correction_bias")
    if corr is not None:
        rank_dict[f"{target}e_score_correction_bias"] = corr

    # Shared expert: ColumnParallel/RowParallel on moe_intermediate axis.
    for hk_suffix, tk_suffix, tp_dim in (
        ("shared_experts.gate_proj.weight", "shared_expert.gate_proj.weight", 0),
        ("shared_experts.up_proj.weight", "shared_expert.up_proj.weight", 0),
        ("shared_experts.down_proj.weight", "shared_expert.down_proj.weight", 1),
    ):
        t = fetch(f"{mlp}{hk_suffix}")
        if t is None:
            continue
        rank_dict[f"{target}{tk_suffix}"] = _row_shard(t, rank, tp_degree, dim=tp_dim)

    # Routed experts — this is the dominant memory class.  Process one
    # expert at a time to keep peak resident tensor small.
    I_tp = moe_intermediate // tp_degree
    # Accumulate rank-r slices for all 288 experts.
    gate_up_slices: list[torch.Tensor] = []
    down_slices: list[torch.Tensor] = []
    for e in range(n_experts):
        gate = fetch(f"{mlp}experts.{e}.gate_proj.weight")   # [I, H]
        up = fetch(f"{mlp}experts.{e}.up_proj.weight")       # [I, H]
        down = fetch(f"{mlp}experts.{e}.down_proj.weight")   # [H, I]
        if gate is None or up is None or down is None:
            raise RuntimeError(
                f"expert {e} incomplete in {mlp}: gate={gate is not None} "
                f"up={up is not None} down={down is not None}"
            )
        # For gate_up_proj: transpose to [H, I], slice rank cols, fuse [gate | up].
        gate_t = gate.transpose(0, 1)  # [H, I]
        up_t = up.transpose(0, 1)      # [H, I]
        gate_rank = gate_t[:, rank * I_tp : (rank + 1) * I_tp]  # [H, I_tp]
        up_rank = up_t[:, rank * I_tp : (rank + 1) * I_tp]      # [H, I_tp]
        fused = torch.cat([gate_rank, up_rank], dim=1).contiguous()  # [H, 2*I_tp]
        gate_up_slices.append(fused)
        # For down_proj: HF stores [H, I]; wrapper wants [E, I, H].  Transpose,
        # slice on I axis (RowParallel dim=1 of [E, I, H]).
        down_t = down.transpose(0, 1)  # [I, H]
        down_rank = down_t[rank * I_tp : (rank + 1) * I_tp, :].contiguous()  # [I_tp, H]
        down_slices.append(down_rank)
        # Free the fetched tensors as we go.
        del gate, up, down, gate_t, up_t, gate_rank, up_rank, down_t, down_rank
    # Stack across experts along a new leading axis.
    gate_up_stacked = torch.stack(gate_up_slices, dim=0)  # [E, H, 2*I_tp]
    down_stacked = torch.stack(down_slices, dim=0)        # [E, I_tp, H]
    rank_dict[f"{target}expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_stacked
    rank_dict[f"{target}expert_mlps.mlp_op.down_proj.weight"] = down_stacked
    del gate_up_slices, down_slices


__all__ = [
    "stream_shard_glm53",
]
