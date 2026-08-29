# SPDX-License-Identifier: Apache-2.0
"""Round-7 DSA indexer HF-parity mini-golden.

Purpose
-------
Proves that ``_DSAIndexerBlock`` (post-Round-7 rewrite in
``neuron_wrapper.py``) is numerically equivalent to HF's
``Glm5NextTextIndexer.forward`` on identical synthetic input, at BF16
tolerance.  This is the acceptance gate for the wrapper-side mapping fix
called out in the Round-6 status doc (``NXDI-WRAPPER-ROUND6-STATUS-
2026-08-28.md``, §3): ``indexer.wq_b``, ``indexer.weights_proj``,
``indexer.index_kpool_compress_{ape,gate}`` and the ``k_norm`` params could
not land under the Round-4 rank-3 ``q_proj`` / scalar ``pool_weights``
scaffolds.  Round 7 rewires the wrapper so all seven HF tensors carry
through 1:1 and the forward math matches HF bit-for-bit at fp32 (BF16 rel
tolerance downstream).

Golden source
-------------
HF's ``Glm5NextTextIndexer`` reference is not universally installed on the
campaign hosts (transformers ``glm5_next`` shipped in the uv cache but not
in the standard Python 3.12 site-packages).  We would not want to depend
on that lookup at gate time anyway — a mini-golden should not silently
skip on a missing import.  This module carries an **inline port** of
``Glm5NextTextIndexer.forward`` (transformers 5.14.x, ``glm5_next`` /
``glm_moe_dsa`` families, snapshot 2026-08-28) that produces the same
tensors on the same inputs.  If the operator later wants a
"golden-of-goldens" cross-check, ``_maybe_import_hf_reference`` will pick
up the ``Glm5NextTextIndexer`` class when available; when it is not, this
module's own inline port stands alone.

Discipline
----------
* Every comparison runs through ``require_comparable`` (degeneracy guard
  from ``harness-v2/staging/reference-sweep-20260826T2150Z/kernels/
  degeneracy_guard.py``) so a NaN- or all-zero output cannot vacuously
  pass a max-abs-error check.
* Assertions are on the pre-top-k ``index_scores`` tensor (fp32 output of
  the pool-collapse pipeline) rather than on the ``topk_indices`` tensor
  (int32 argmax result), because the mini-golden is testing the
  arithmetic mapping, not top-k tie-breaking.  A parallel exact-equality
  check on ``topk_indices`` follows as a downstream witness.
* BF16 tolerance target: ``max_abs_error < 1e-4`` on ``index_scores`` —
  well inside the wrapper's own ``bf16_round`` boundary.  The inline HF
  reference runs in bf16 for the projections and up-casts to fp32 for the
  softmax + score compose, mirroring HF exactly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# ``degeneracy_guard`` lives in the reference-sweep bundle.  The path lookup
# mirrors ``_reference_kernels._kernel_dir`` — the campaign's canonical
# location — so this test does not carry a private copy of the guard.
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
        "degeneracy_guard not importable — the Round-7 DSA indexer gate "
        f"cannot run without it: {exc!r}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Inline port of HF ``Glm5NextTextIndexer.forward``.
# ---------------------------------------------------------------------------


class _HFReferenceIndexer(nn.Module):
    """Bit-exact port of HF ``Glm5NextTextIndexer`` (transformers 5.14.x).

    Independent of ``vllm_neuron`` and of ``transformers``: takes the same
    synthetic weight tensors the wrapper's block takes, produces the same
    ``index_scores`` and ``topk_indices`` HF emits on the same inputs.
    Kept small on purpose — this is a golden, not a scaffold.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        q_lora_rank: int,
        index_kpool: int,
        index_topk: int,
        always_select_tail: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.index_kpool = index_kpool
        self.index_topk = index_topk
        self.always_select_tail = always_select_tail
        self.softmax_scale = head_dim**-0.5
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False, dtype=dtype)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6, dtype=dtype)
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False, dtype=dtype)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(index_kpool, head_dim, dtype=dtype)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(head_dim, hidden_size, dtype=dtype)
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(index_scores[B, S, P] fp32, topk_indices[B, S, W] int32)``."""
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)

        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]

        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        current_length = seq_len
        keys_c, gate_c, valid_c = torch.split(
            packed_states, [self.head_dim, self.head_dim, 1], dim=-1
        )
        valid_keys = valid_c.bool().squeeze(-1)
        # Fresh-sequence path only (no cache): first_key = 0.
        device = keys_c.device
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        pool_offsets = torch.arange(
            number_of_pools * self.index_kpool, device=device, dtype=torch.long
        )
        pool_indices = pool_offsets.view(1, number_of_pools, self.index_kpool).expand(
            batch_size, -1, -1
        )
        safe_indices = pool_indices.clamp(0, seq_len - 1)
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        grouped_keys = keys_c[batch_idx, safe_indices]
        grouped_gate = gate_c[batch_idx, safe_indices]
        grouped_valid = valid_keys[batch_idx, safe_indices]
        grouped_valid = grouped_valid & (pool_indices < seq_len)
        pool_valid = grouped_valid.all(-1)
        pool_indices_masked = pool_indices.masked_fill(~grouped_valid, -1)
        logits = (
            grouped_gate.float() + self.index_kpool_compress_ape.float()[None, None]
        )
        logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)

        # Causality / visibility.
        kv_positions = torch.arange(seq_len, device=device)
        q_positions = current_length - seq_len + torch.arange(seq_len, device=device)
        causal = kv_positions[None, None, :] <= q_positions[None, :, None]
        visible_tokens = causal & valid_keys[:, None, :]

        scores = torch.matmul(
            q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1)
        )
        scores = F.relu(scores * self.softmax_scale)
        weights = self.weights_proj(
            hidden_states.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        pool_end = pool_indices_masked[..., -1].clamp(0, seq_len - 1)
        pool_visible = visible_tokens.gather(
            dim=-1,
            index=pool_end[:, None, :].expand(batch_size, seq_len, -1),
        )
        valid_candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(
            ~valid_candidates,
            torch.finfo(index_scores.dtype).min,
        )

        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices_masked[batch_idx, selected]
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
            -1,
        )
        output_width = self.index_topk
        if self.always_select_tail:
            output_width += self.index_kpool - 1
        topk_indices = F.pad(
            topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1
        )
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return index_scores, topk_indices.to(torch.int32)


def _maybe_import_hf_reference():
    """Return HF's own ``Glm5NextTextIndexer`` class when installed; else None."""
    try:
        from transformers.models.glm5_next.modeling_glm5_next import (  # type: ignore
            Glm5NextTextIndexer,
        )

        return Glm5NextTextIndexer
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Wrapper harness.  Instantiates ``_DSAIndexerBlock`` via the NxDI-shaped
# config, then copies the golden's weights into it so the two blocks are
# arithmetically identical modulo the compute path.
# ---------------------------------------------------------------------------


def _build_wrapper_block(
    *,
    hidden_size,
    n_heads,
    head_dim,
    q_lora_rank,
    index_kpool,
    index_topk,
    always_select_tail,
    num_attention_heads,
    qk_head_dim,
    dtype,
):
    """Instantiate ``_DSAIndexerBlock`` under a stubbed NxDI-availability guard.

    The block class lives inside ``if _NXDI_AVAILABLE:`` at module import
    time; the CPU-only test env does not have NxDI installed, so we import
    the block through a small trampoline that patches ``_NXDI_AVAILABLE`` +
    ``_NxdColumnParallelLinear`` = ``nn.Linear`` before the module is
    re-executed.  This trades a bit of import gymnastics for a self-
    contained test that runs anywhere torch runs — no Neuron toolchain
    required.
    """
    import importlib
    import types

    # ---- Stub the NxDI parallel primitives with plain nn.Linear ----
    fake_nxd_dist = types.ModuleType("neuronx_distributed")
    fake_parallel_layers = types.ModuleType("neuronx_distributed.parallel_layers")
    fake_layers = types.ModuleType("neuronx_distributed.parallel_layers.layers")

    class _StubCPL(nn.Linear):
        """ColumnParallelLinear stub — replicated Linear with matching kwargs."""

        def __init__(
            self,
            in_features,
            out_features,
            *,
            bias=False,
            gather_output=True,
            dtype=torch.float32,
            **kwargs,
        ):
            super().__init__(in_features, out_features, bias=bias, dtype=dtype)

    class _StubRPL(nn.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            *,
            bias=False,
            input_is_parallel=True,
            dtype=torch.float32,
            **kwargs,
        ):
            super().__init__(in_features, out_features, bias=bias, dtype=dtype)

    class _StubEmbed(nn.Embedding):
        def __init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx,
            *,
            dtype=torch.float32,
            shard_across_embedding=False,
            pad=False,
            sequence_parallel_enabled=False,
            **kwargs,
        ):
            super().__init__(
                num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=dtype
            )

    fake_layers.ColumnParallelLinear = _StubCPL
    fake_layers.RowParallelLinear = _StubRPL
    fake_layers.ParallelEmbedding = _StubEmbed
    fake_parallel_layers.layers = fake_layers
    fake_nxd_dist.parallel_layers = fake_parallel_layers

    fake_parallel_state = types.ModuleType(
        "neuronx_distributed.parallel_layers.parallel_state"
    )
    fake_parallel_state.get_tensor_model_parallel_rank = lambda: 0
    fake_parallel_state.get_tensor_model_parallel_size = lambda: 1
    fake_parallel_layers.parallel_state = fake_parallel_state

    fake_mappings = types.ModuleType("neuronx_distributed.parallel_layers.mappings")
    fake_mappings.reduce_from_tensor_model_parallel_region = lambda x: x
    fake_parallel_layers.mappings = fake_mappings

    fake_moe = types.ModuleType("neuronx_distributed.modules.moe")
    fake_expert_mlps = types.ModuleType("neuronx_distributed.modules.moe.expert_mlps")

    class _StubExpertMLPs:  # pragma: no cover - never instantiated by this test
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "MoE expert path is not exercised by the DSA indexer gate"
            )

    fake_expert_mlps.ExpertMLPs = _StubExpertMLPs
    fake_model_utils = types.ModuleType("neuronx_distributed.modules.moe.model_utils")
    fake_model_utils.GLUType = object
    fake_moe.expert_mlps = fake_expert_mlps
    fake_moe.model_utils = fake_model_utils

    fake_nxdi = types.ModuleType("neuronx_distributed_inference")
    fake_nxdi_models = types.ModuleType("neuronx_distributed_inference.models")
    fake_model_base = types.ModuleType(
        "neuronx_distributed_inference.models.model_base"
    )

    class _StubBase(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    fake_model_base.NeuronBaseForCausalLM = _StubBase
    fake_model_base.NeuronBaseModel = _StubBase
    fake_config_mod = types.ModuleType("neuronx_distributed_inference.models.config")

    class _StubInferenceConfig:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_config_mod.InferenceConfig = _StubInferenceConfig
    fake_config_mod.MoENeuronConfig = _StubInferenceConfig
    fake_config_mod.NeuronConfig = _StubInferenceConfig
    fake_nxdi_models.model_base = fake_model_base
    fake_nxdi_models.config = fake_config_mod

    for name, mod in (
        ("neuronx_distributed", fake_nxd_dist),
        ("neuronx_distributed.parallel_layers", fake_parallel_layers),
        ("neuronx_distributed.parallel_layers.layers", fake_layers),
        ("neuronx_distributed.parallel_layers.parallel_state", fake_parallel_state),
        ("neuronx_distributed.parallel_layers.mappings", fake_mappings),
        (
            "neuronx_distributed.modules",
            types.ModuleType("neuronx_distributed.modules"),
        ),
        ("neuronx_distributed.modules.moe", fake_moe),
        ("neuronx_distributed.modules.moe.expert_mlps", fake_expert_mlps),
        ("neuronx_distributed.modules.moe.model_utils", fake_model_utils),
        ("neuronx_distributed_inference", fake_nxdi),
        ("neuronx_distributed_inference.models", fake_nxdi_models),
        ("neuronx_distributed_inference.models.model_base", fake_model_base),
        ("neuronx_distributed_inference.models.config", fake_config_mod),
    ):
        sys.modules.setdefault(name, mod)

    if "vllm_neuron.model.glm53_flash.neuron_wrapper" in sys.modules:
        del sys.modules["vllm_neuron.model.glm53_flash.neuron_wrapper"]
    neuron_wrapper = importlib.import_module(
        "vllm_neuron.model.glm53_flash.neuron_wrapper"
    )

    # ``_DSAIndexerBlock`` reads the frozen ``Glm53FlashInferenceConfig`` via
    # ``_require_source_config``; a duck-typed stand-in fails the isinstance
    # check.  Build a real one at the tiny test dims — ``allow_reduced_shapes
    # =True`` relaxes the layer-stack invariants that would otherwise reject
    # ``num_attention_heads=8``, etc.
    from vllm_neuron.model.glm53_flash.config import (
        Glm53FlashInferenceConfig,
        Glm53LinearAttentionConfig,
    )

    src = Glm53FlashInferenceConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        intermediate_size=hidden_size * 2,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=q_lora_rank,
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_head_dim,
        v_head_dim=qk_head_dim,
        index_n_heads=n_heads,
        index_head_dim=head_dim,
        index_topk=index_topk,
        index_kpool=index_kpool,
        index_kpool_always_select_tail=always_select_tail,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=hidden_size,
        torch_dtype=dtype,
        static_fp8=False,
        allow_reduced_shapes=True,
        linear_attn_config=Glm53LinearAttentionConfig(
            num_heads=num_attention_heads, head_dim=head_dim
        ),
    )

    class _FakeNeuronConfig:
        pass

    nc = _FakeNeuronConfig()
    nc.torch_dtype = dtype
    nc.tp_degree = 1

    class _FakeConfig:
        pass

    cfg = _FakeConfig()
    cfg.source_config = src
    cfg.neuron_config = nc

    # ``_require_source_config`` reads ``config.source_config`` — the stubbed
    # attribute above satisfies it.
    block = neuron_wrapper._DSAIndexerBlock(cfg, layer_idx=3)
    return block, neuron_wrapper


def _copy_hf_weights_to_wrapper(*, hf_block, wrapper_block):
    """Point-by-point weight copy — HF is source-of-truth."""
    with torch.no_grad():
        wrapper_block.wq_b.weight.copy_(hf_block.wq_b.weight)
        wrapper_block.wk.weight.copy_(hf_block.wk.weight)
        wrapper_block.k_norm.weight.copy_(hf_block.k_norm.weight)
        wrapper_block.k_norm.bias.copy_(hf_block.k_norm.bias)
        wrapper_block.weights_proj.weight.copy_(hf_block.weights_proj.weight)
        wrapper_block.index_kpool_compress_ape.copy_(hf_block.index_kpool_compress_ape)
        wrapper_block.index_kpool_compress_gate.copy_(
            hf_block.index_kpool_compress_gate
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 17])
def test_wrapper_index_scores_match_hf_within_bf16_tolerance(seed: int) -> None:
    """``compute_index_scores`` output agrees with HF at ``<1e-4`` in BF16.

    Tiny shapes on the indexer side (``n_heads=8``, ``head_dim=32``,
    ``hidden_size=128``, ``q_lora_rank=64``) keep the golden fast; the
    MLA-side ``qk_head_dim`` stays pinned at the GLM-5.3-Flash frozen 256
    because ``Glm53FlashInferenceConfig._validate_architecture`` gates it
    (the DSA indexer never reads ``qk_head_dim`` post-Round-7 anyway — that
    axis was only ever used by the rank-3 ``q_proj`` scaffold this fix
    replaces).
    """
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    batch, seq_len = 2, 12
    hidden_size = 128
    n_heads = 8
    head_dim = 32
    q_lora_rank = 64
    index_kpool = 4
    index_topk = 8
    always_select_tail = True

    hf = _HFReferenceIndexer(
        hidden_size=hidden_size,
        n_heads=n_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        index_kpool=index_kpool,
        index_topk=index_topk,
        always_select_tail=always_select_tail,
        dtype=dtype,
    ).eval()
    # Randomize weights so scores carry variation — a zero-initialised
    # block would render the max-abs-error check vacuous (degeneracy_guard
    # would then reject the input).
    for p in hf.parameters():
        if p.dtype in (torch.float32, torch.bfloat16, torch.float16):
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.02)

    wrapper, wrapper_mod = _build_wrapper_block(
        hidden_size=hidden_size,
        n_heads=n_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        index_kpool=index_kpool,
        index_topk=index_topk,
        always_select_tail=always_select_tail,
        num_attention_heads=8,
        qk_head_dim=256,  # pinned by Glm53FlashInferenceConfig; unused by the indexer
        dtype=dtype,
    )
    _copy_hf_weights_to_wrapper(hf_block=hf, wrapper_block=wrapper)

    hidden_states = torch.randn(batch, seq_len, hidden_size, dtype=dtype)
    q_resid = torch.randn(batch, seq_len, q_lora_rank, dtype=dtype)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)

    hf_scores, hf_topk = hf(hidden_states, q_resid, attention_mask)

    # Wrapper caches K + gate for the current window; ``key_lengths`` is
    # ``seq_len`` (fresh sequence, no left padding).
    k_cache = wrapper.project_index_k(hidden_states)
    gate_cache = wrapper.project_index_gate(hidden_states)
    position_ids = torch.arange(seq_len, dtype=torch.int64)[None, :].expand(batch, -1)
    key_lengths = torch.full((batch,), seq_len, dtype=torch.int64)

    wrapper_scores, pool_indices, pool_valid, valid_cand = wrapper.compute_index_scores(
        hidden_states,
        q_resid,
        k_cache,
        gate_cache,
        position_ids,
        key_lengths,
    )

    # Both sides must be comparable — reject NaN / all-zero degenerates
    # BEFORE any max-abs-error math (which could silently pass over a
    # degenerate row otherwise).
    require_comparable(hf_scores.detach().cpu().numpy(), "hf_index_scores")
    require_comparable(wrapper_scores.detach().cpu().numpy(), "wrapper_index_scores")

    # Compare where the mask is finite on both sides — the ``-inf``
    # sentinel entries are identical by construction (both mask on the
    # same rule); we do not want the max-abs-error to inherit the NaN
    # from ``-inf - -inf``.
    finite = torch.isfinite(hf_scores) & torch.isfinite(wrapper_scores)
    err = (hf_scores - wrapper_scores).masked_fill(~finite, 0.0).abs().max().item()
    print(f"[seed={seed}] wrapper vs HF index_scores max_abs_err = {err:.3e}")
    assert math.isfinite(err), f"non-finite error slipped through: {err!r}"
    assert err < 1e-4, (
        f"wrapper index_scores diverged from HF reference: max_abs_err={err!r}. "
        "This gate rejects any BF16-tolerance drift — the two forwards should "
        "be arithmetically identical modulo a matmul reassociation."
    )

    # Downstream witness: the wrapper's top-k picks match HF's picks
    # exactly (top-k is deterministic + tie-breaking-stable at this scale).
    wrapper_topk = wrapper.select_topk(wrapper_scores, pool_indices, valid_cand)
    assert torch.equal(hf_topk, wrapper_topk), (
        "topk_indices disagreement after index_scores parity — the top-k "
        "reduction is deterministic, so a disagreement here means the "
        "score-composition disagreement is bigger than the ties."
    )


def test_wrapper_matches_hf_when_installed() -> None:
    """Belt-and-braces: if HF's ``Glm5NextTextIndexer`` imports, compare directly.

    Skipped when transformers doesn't ship ``glm5_next`` in this
    environment.  The inline golden above is the primary gate; this second
    check protects against a drift in the inline port at some future
    transformers version.
    """
    Glm5NextTextIndexer = _maybe_import_hf_reference()
    if Glm5NextTextIndexer is None:
        pytest.skip(
            "transformers.models.glm5_next.Glm5NextTextIndexer not "
            "installed in this env — the inline golden is the gate"
        )

    torch.manual_seed(99)
    dtype = torch.bfloat16
    batch, seq_len = 1, 16
    hidden_size = 96
    n_heads = 4
    head_dim = 24
    q_lora_rank = 32
    index_kpool = 4
    index_topk = 8

    # Build a stub config namespace HF's Glm5NextTextIndexer reads.
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.hidden_size = hidden_size
    cfg.index_n_heads = n_heads
    cfg.index_head_dim = head_dim
    cfg.qk_rope_head_dim = 0
    cfg.index_topk = index_topk
    cfg.q_lora_rank = q_lora_rank
    cfg.index_kpool = index_kpool
    cfg.index_kpool_always_select_tail = True

    upstream = Glm5NextTextIndexer(cfg, layer_idx=0).eval()

    inline = _HFReferenceIndexer(
        hidden_size=hidden_size,
        n_heads=n_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        index_kpool=index_kpool,
        index_topk=index_topk,
        always_select_tail=True,
        dtype=dtype,
    ).eval()
    # Copy upstream weights into the inline golden so the two share
    # parameters.
    with torch.no_grad():
        inline.wq_b.weight.copy_(upstream.wq_b.weight.to(dtype))
        inline.wk.weight.copy_(upstream.wk.weight.to(dtype))
        inline.k_norm.weight.copy_(upstream.k_norm.weight.to(dtype))
        inline.k_norm.bias.copy_(upstream.k_norm.bias.to(dtype))
        inline.weights_proj.weight.copy_(upstream.weights_proj.weight.to(dtype))
        inline.index_kpool_compress_ape.copy_(
            upstream.index_kpool_compress_ape.to(dtype)
        )
        inline.index_kpool_compress_gate.copy_(
            upstream.index_kpool_compress_gate.to(dtype)
        )

    hidden_states = torch.randn(batch, seq_len, hidden_size, dtype=dtype)
    q_resid = torch.randn(batch, seq_len, q_lora_rank, dtype=dtype)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    inline_scores, inline_topk = inline(hidden_states, q_resid, attention_mask)
    # HF's own forward emits int32 topk_indices only; the inline scores
    # are exercised in the primary test above.  Here we only cross-check
    # that HF's own top-k picks agree with the inline port's — sanity of
    # the port itself.
    upstream_topk = upstream(
        hidden_states, q_resid, attention_mask, past_key_values=None
    )
    require_comparable(
        upstream_topk.detach().cpu().numpy().astype("int64").tolist(),
        "upstream_topk_indices",
        require_variation=False,  # int arrays — variation is optional
    )
    assert torch.equal(upstream_topk, inline_topk), (
        "inline HF port diverged from upstream Glm5NextTextIndexer top-k — "
        "the port needs to be re-derived against the current transformers "
        "revision"
    )
