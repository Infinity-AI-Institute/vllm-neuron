# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash streaming per-rank checkpoint sharder (Round 1 skeleton).

Ports the Round-6 GLM-5.3-Flash streaming-writer scaffold at
``vllm_neuron/model/glm53_flash/stream_shard.py`` verbatim for the
file-handle machinery: ``safe_open(mmap=True)`` handle pool, per-rank
slice-then-flush loop, ``gc.collect()`` between ranks, output layout at
``{compiled_model_path}/weights/tp{r}_sharded_checkpoint.safetensors``.

What changes for DeepSeek-V4-Flash:

* Total in-file footprint is ~167 GB (FP4 experts + FP8 non-experts), NOT
  the ~610 GB BF16 hydrated size GLM-5.3-Flash Round-6 dealt with.  Peak
  per-rank output buffer at TP=32 is roughly ``167/32 ≈ 5.2 GiB`` if we
  were to keep the checkpoint at its native quant format, but we
  currently dequant to bf16 at converter time — so the per-rank bf16
  output slice is closer to ``167 * 2 / 32 ≈ 10.4 GiB`` (FP4 experts
  4x expand to bf16; FP8 non-experts 2x).  Well under the 100 GiB
  budget.

* Sharding rules per parameter differ from GLM-5.3-Flash's KDA/DSA/MoE
  layout — see the enablement-draft §2 for the block-by-block delta.
  The core rules:

    * ``embed_tokens.weight``: ``ParallelEmbedding(shard_across_embedding
      = True, pad = True)``.  Sharded on dim=0 (vocab axis 129280).
      Emit unsharded; NxDI's loader handles the pad.
    * ``final_norm_weight``: replicated.
    * ``lm_head.weight``: ``ColumnParallelLinear(pad=True)``.  Emit unsharded.
    * MQA per-layer weights:
        - ``wq_a`` [q_lora_rank=1024, hidden=4096]: ColumnParallel on dim=0.
          Small; each rank gets 32 rows at TP=32.
        - ``wq_b`` [num_heads=64 * head_dim=512, q_lora_rank=1024] =
          [32768, 1024]: ColumnParallel on dim=0 (head-count axis).  Rank
          r gets ``[64/32=2] * 512 = 1024`` rows.
        - ``kv_proj`` (shared K=V) [1 * head_dim=512, hidden=4096]:
          replicated across TP (KV heads=1 < TP=32, per user memory
          gqa-hbm-heuristic-correction-20260828).
        - ``wo_a`` grouped output projection [num_heads * head_dim,
          o_groups=8 * o_lora_rank=1024] = [32768, 8192]: ColumnParallel
          on dim=0.  Layout decision (o_groups × head slices) is a
          Round-2 correctness gate.
        - ``wo_b`` [o_groups * o_lora_rank = 8192, hidden=4096]:
          RowParallel on dim=1.
        - ``attn_sink`` [num_heads=64]: ColumnParallel on dim=0
          (rank-local heads only).
        - ``attn_norm``, ``ffn_norm``, ``q_norm``, ``kv_norm``: replicated.
    * mHC (``hc_attn_*``, ``hc_ffn_*``): replicated.
    * CSA/HCA compressor weights (per-layer, layer-type-dependent) and
      Lightning Indexer weights: Round-2 layout decision — see enablement
      draft §2 for the mapping deferrals.
    * Hash-MoE bootstrap: ``mlp.hash_table`` [vocab=129280, top_k=6]
      int32 lookup table, REPLICATED across TP (small; ~3 MiB).
    * MoE routed experts:
        - ``experts.w1.weight`` [n_experts=256, moe_intermediate=2048,
          hidden=4096] (FP4-packed): expand to bf16 during shard, then
          same ExpertMLPs stride=2 gate|up fused layout as GLM-5.3-Flash.
          Rank-r shard at TP=32 = 64 intermediate cols/rank.
        - ``experts.w2.weight`` [n_experts, moe_intermediate,
          hidden] = down_proj: RowParallel on the moe_intermediate axis.
        - ``experts.w3.weight`` = second half of fused gate|up (present in
          some safetensors layouts; concat into gate|up on load).
        - ``experts.<gate>.weight`` (router linear): replicated
          (small).
        - ``e_score_correction_bias``: replicated fp32.
    * Shared expert (``shared_experts.*``): same rules as GLM-5.3-Flash
      shared-expert branch.

All slice offsets keyed off the source config's constants so a future
TP=16 or TP=8 lane reroutes without touching this file.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .config import DeepseekV4FlashInferenceConfig


def stream_shard_dsv4_checkpoint(
    hf_model_path: str,
    compiled_model_path: str,
    src: DeepseekV4FlashInferenceConfig,
    *,
    tp_degree: int,
    ep_degree: int | None = None,
) -> dict[str, Any]:
    """Per-rank streaming shard writer — Round 1 skeleton.

    Returns a receipt dict shaped like GLM-5.3-Flash's Round-6 output
    (peak-memory bytes, output-file sizes, ranks written, dequant errors
    accumulated during streaming).

    NOT YET IMPLEMENTED — per-parameter sharding rules from the docstring
    above land in Round 2 after the FP4-UE8M0 dequant math has closed on
    a 1-tensor smoke against a real routed-expert shard.
    """
    if tp_degree <= 0:
        raise ValueError(f"tp_degree must be positive; got {tp_degree}")
    if ep_degree is None:
        ep_degree = tp_degree
    if not os.path.isdir(hf_model_path):
        raise FileNotFoundError(f"HF model path {hf_model_path!r} is not a directory")
    raise NotImplementedError(
        "stream_shard_dsv4_checkpoint is Round 2.  Depends on the FP4-UE8M0 "
        "dequant closure (see checkpoint_convert.dequantize_block_fp4_ue8m0) and "
        "the Round-2 per-parameter sharding rules in this module's docstring."
    )


__all__ = [
    "stream_shard_dsv4_checkpoint",
]
