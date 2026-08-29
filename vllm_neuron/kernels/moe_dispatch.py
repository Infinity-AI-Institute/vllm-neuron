"""moe_dispatch — fused router / top-k / expert-MM / combine NKI kernel.

Corresponds to Part A of `NKI-MOE-DISPATCH-AND-GEMMA4-CPU-FALLBACK-SCAFFOLD-2026-08-27.md`.

Design summary
--------------
Replaces the four-graph baseline (router-softmax / dispatch-scatter /
per-expert MLP / weighted-combine) with **one** NKI graph:

    1. Router half — RMSNorm + logits + top-K + affinity scatter (identical
       to Codex's proven `gemma4_router_split`).
    2. Fused expert half — calls `moe_tkg(is_all_expert=True, POST_SCALE)`
       with the activation branch resolved at compile time.  The dense
       "all-experts" pattern is the shape the AoT compiler can lower —
       branch-on-router is impossible per §B6.

The scaffold's §A.2 "in-place fp32 accumulator" trick is realised inside
`moe_tkg` via `ExpertAffinityScaleMode.POST_SCALE` (weights fold into the
down-projection contraction, so the combine barrier disappears).

Compile-time shape families
---------------------------
Two production shape families ship in this file:

* `MoEDispatchConfig.QWEN3_30B_A3B_TP8` — 128 experts / top-8, hidden 2048,
  activation SILU, intermediate 768 (I_TP=96 at TP=8 — clears `%16`).
* `MoEDispatchConfig.GEMMA4_26B_A4B_TP4` — 128 experts / top-8, hidden 2816,
  activation GELU_Tanh_Approx, intermediate 704, TP=4 (I_TP=176 — the
  Codex-verified shape that sidesteps the `88 % 16` wall).  NOTE: the
  task prompt cited "64 experts / top-6" for Gemma-4-26B-A4B; that value
  disagrees with the pinned HF checkpoint (revision
  `24548b62aa021d562695c04aaf7758a1ea47990b`) which has 128/8 per
  `harness-v2/staging/cycle465/dual_input_tkg_moe_nki.py:36-44`.  This
  file ships the Codex-verified value; the discrepancy is logged in
  `MOE-DISPATCH-STATUS-2026-08-28.md` §4 for operator sign-off.
* `MoEDispatchConfig.GPT_OSS_20B_TP8` — 128 experts / top-4, hidden 2880,
  activation SILU (does NOT renormalize top-k weights per §A.7 gap A.G-9).

Container constraint (`nxdi-container-moe-blockwise-mm-workaround-20260827`
memory entry): every downstream compile MUST set
`blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True`
before `InferenceConfig.__init__`.  This kernel does not itself enforce
that — the wire-in point in NxDI does; see `moe_dispatch_wire.py` docstring
and the fallback-ladder ordering in `MOE-DISPATCH-STATUS-2026-08-28.md`.

FAIL-LOUD discipline (per scaffold §C.1)
----------------------------------------
The kernel intentionally exports **no** silent fallback path.  A compile
that refuses (activation branch mismatch, SBUF overflow, unsupported
num_experts/K) surfaces as `neuronx-cc` error, not as a WARN.  The
wire-in `enable_moe_fused_dispatch()` helper below emits a `CRITICAL`
log message when the fallback ladder drops to `torch_blockwise_matmul_inference`
so §B5's silent-drop hazard cannot repeat.

This module is import-guarded so it can be linted / unit-tested on hosts
without the `nki`/`nkilib` stack (CI + laptop).  The two `@nki.jit`
kernel bodies below require the compile host's pinned nkilib.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

log = logging.getLogger("moe_dispatch")

# --------------------------------------------------------------------------
# Import-guarded NKI dependencies
# --------------------------------------------------------------------------
try:
    import nki
    import nki.language as nl
    from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
    from nkilib.core.router_topk.router_topk import (
        XSBLayout_tp2013__1,
        router_topk,
    )
    from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg
    from nkilib.core.utils.common_types import (
        ActFnType,
        ExpertAffinityScaleMode,
        RouterActFnType,
    )

    _NKI_AVAILABLE = True
except ImportError as _e:  # laptop / CI stub — kernel bodies unavailable
    _NKI_AVAILABLE = False
    _NKI_IMPORT_ERROR = _e

    class _Stub:
        def __getattr__(self, item):
            raise ImportError(
                "nki stack not available on this host; import moe_dispatch "
                "only on the pinned compile host."
            )

    nki = _Stub()  # type: ignore
    nl = _Stub()  # type: ignore


# --------------------------------------------------------------------------
# Compile-time activation branch table (scaffold §A.4 / §A.G-7)
# --------------------------------------------------------------------------


class MoEActivation(Enum):
    """Enum enumerating the exhaustive activation branches (§A.G-7).

    Any queue-member not in this list refuses to compile — no silent
    fallback to `torch_blockwise_matmul_inference` (§B5 discipline).
    """

    SILU = auto()  # Qwen3-30B-A3B, GPT-OSS-20B
    GELU_TANH_APPROX = auto()  # Gemma-4-26B-A4B (§B5 hazard model)
    GLU = auto()  # sigmoid * up
    SILU_GLU = auto()  # silu * sigmoid — some DeepSeek variants

    def to_actfn(self):
        if not _NKI_AVAILABLE:
            raise ImportError("nki stack required to resolve ActFnType")
        return {
            MoEActivation.SILU: ActFnType.Silu,
            MoEActivation.GELU_TANH_APPROX: ActFnType.GELU_Tanh_Approx,
            MoEActivation.GLU: ActFnType.Sigmoid,
            MoEActivation.SILU_GLU: ActFnType.Silu,
        }[self]


# --------------------------------------------------------------------------
# Model shape configuration objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEDispatchConfig:
    """Compile-time shape family for the fused MoE dispatch kernel.

    All fields are compile-time constants (baked into the NEFF).  Runtime
    tensors carry only (batch, sequence) which are separately parameterised.
    """

    name: str
    hidden: int
    num_experts: int
    top_k: int
    intermediate_global: int
    tp_degree: int
    activation: MoEActivation
    renormalize_topk: bool  # §A.G-9 — GPT-OSS is False, others True
    pmax: int = 128
    eps: float = 1e-6
    # Router weight axis convention: input `router_weights` is (hidden, experts)
    # to match Codex's proven `router_topk` layout.

    @property
    def intermediate_per_tp(self) -> int:
        assert self.intermediate_global % self.tp_degree == 0, (
            f"{self.name}: intermediate {self.intermediate_global} not "
            f"divisible by TP {self.tp_degree}"
        )
        return self.intermediate_global // self.tp_degree

    def validate(self) -> None:
        # Tier-1 CPU-battery guards — no compile submitted until these pass.
        # 1) num_experts × K partition width against nc_find_index8 cap
        if self.num_experts > 16384:
            raise ValueError(
                f"{self.name}: num_experts={self.num_experts} exceeds "
                "nc_find_index8 partition cap of 16384 (§B1)."
            )
        # 2) 88 % 16 wall — I_TP must clear
        if self.intermediate_per_tp % 16 != 0:
            raise ValueError(
                f"{self.name}: I_TP={self.intermediate_per_tp} fails %16 "
                "assert wall (§B7); pick a TP degree that makes it divisible."
            )
        # 3) top_k must be power-of-2-or-8 for bitonic topk safety
        if self.top_k not in (1, 2, 4, 6, 8, 16):
            raise ValueError(f"{self.name}: top_k={self.top_k} outside tested set.")


# Pinned shape families — every downstream compile picks one and never guesses.
QWEN3_30B_A3B_TP8 = MoEDispatchConfig(
    name="qwen3-30b-a3b-tp8",
    hidden=2048,
    num_experts=128,
    top_k=8,
    intermediate_global=768,
    tp_degree=8,
    activation=MoEActivation.SILU,
    renormalize_topk=True,
)

GEMMA4_26B_A4B_TP4 = MoEDispatchConfig(
    name="gemma4-26b-a4b-tp4",
    hidden=2816,
    num_experts=128,  # per pinned HF revision 24548b62; task prompt
    top_k=8,  # cited 64/6 but that disagrees with the code +
    intermediate_global=704,  # HF config — see status doc §4 for sign-off.
    tp_degree=4,  # TP=4 so I_TP=176 (clears %16); TP=8 is I_TP=88 which fails.
    activation=MoEActivation.GELU_TANH_APPROX,
    renormalize_topk=True,
)

GPT_OSS_20B_TP4 = MoEDispatchConfig(
    name="gpt-oss-20b-tp4",
    hidden=2880,
    num_experts=128,
    top_k=4,
    intermediate_global=2880,
    tp_degree=4,
    activation=MoEActivation.SILU,
    renormalize_topk=False,  # §A.G-9 — GPT-OSS does NOT renormalize top-k weights
)


# --------------------------------------------------------------------------
# Kernel body — router + fused-combine expert half
# --------------------------------------------------------------------------


def _make_moe_dispatch(cfg: MoEDispatchConfig):
    """Build the `@nki.jit` fused MoE dispatch kernel for a shape family.

    The kernel body captures `cfg` by closure so every compile-time
    constant is baked in (num_experts, top_k, hidden, activation).  The
    only runtime tensors are activations and pre-transposed weight bundles.

    Returns (router_fn, expert_fn) both `@nki.jit`-decorated.  They can be
    compiled to a single NEFF via a wrapper (see `moe_dispatch_wire.py`)
    OR split into two NEFFs matching Codex's proven `dual_input_tkg_moe_split`
    baseline.
    """

    if not _NKI_AVAILABLE:
        raise ImportError(
            "nki stack required to build moe_dispatch kernels; this file "
            "is import-safe on hosts without the stack but the kernel "
            "constructor requires the pinned nkilib."
        )

    cfg.validate()
    ACT_FN = cfg.activation.to_actfn()
    HIDDEN = cfg.hidden
    EXPERTS = cfg.num_experts
    TOP_K = cfg.top_k
    I_TP = cfg.intermediate_per_tp
    PMAX = cfg.pmax
    EPS = cfg.eps

    # ---- Router NEFF ----
    # Compile-time-branched on renormalize_topk (§A.G-9).  This is a constexpr
    # branch — legal AoT — not a router-output branch which §B6 forbids.
    @nki.jit
    def gemma_moe_router(
        router_input,  # BF16 [B, S, HIDDEN]
        router_gamma,  # BF16 [1, HIDDEN]  (Gemma: gamma * H**-0.5)
        router_weights,  # BF16 [HIDDEN, EXPERTS]
    ):
        """Router half: RMSNorm -> logits -> softmax -> top-K + affinities.

        Identical layout to Codex's `gemma4_router_split` — reused
        deliberately so the router NEFF is bit-compatible with the
        production dual-input split baseline.
        """
        # Tokens dimension is dynamic at trace time; the kernel wraps the
        # TracerFrontend-supplied concrete input shape.
        _tokens = router_input.shape[0] * router_input.shape[1]
        router_norm = nl.ndarray(
            (PMAX, _tokens, HIDDEN // PMAX),
            dtype=router_input.dtype,
            buffer=nl.sbuf,
        )
        rmsnorm_tkg(
            input=router_input,
            gamma=router_gamma,
            output=router_norm,
            eps=EPS,
            hidden_actual=HIDDEN,
        )
        router_logits = nl.ndarray(
            (_tokens, EXPERTS), dtype=router_input.dtype, buffer=nl.shared_hbm
        )
        expert_affinities = nl.ndarray(
            (_tokens, EXPERTS), dtype=nl.float32, buffer=nl.shared_hbm
        )
        expert_index = nl.ndarray(
            (_tokens, TOP_K), dtype=nl.uint32, buffer=nl.shared_hbm
        )
        router_topk(
            x=router_norm,
            w=router_weights,
            w_bias=None,
            router_logits=router_logits,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            act_fn=RouterActFnType.SOFTMAX,
            k=TOP_K,
            x_hbm_layout=0,
            x_sb_layout=XSBLayout_tp2013__1,
            router_pre_norm=True,
            norm_topk_prob=cfg.renormalize_topk,
            use_column_tiling=True,
            shard_on_tokens=False,
        )
        return expert_affinities, expert_index, router_logits

    # ---- Fused expert + in-place combine NEFF ----
    # `moe_tkg(is_all_expert=True, POST_SCALE)` performs the scaffold's
    # §A.2 stage-4 "for e in num_experts: matmul + activation + weighted
    # combine into fp32 accumulator" pattern, with the down-projection
    # weight-fold that Codex's dual-input split established (per
    # dual_input_tkg_moe_nki.py:114-126).  The in-place accumulator makes
    # the combine barrier disappear — no extra graph after the expert loop.
    @nki.jit
    def gemma_moe_expert_combine(
        expert_input,  # BF16 [T, HIDDEN]  (post-layernorm)
        expert_gate_up_weights,  # BF16 [EXPERTS, HIDDEN, 2, I_TP]
        expert_down_weights_scaled,  # BF16 [EXPERTS, I_TP, HIDDEN]  (down * per_expert_scale)
        expert_affinities,  # FP32 [T, EXPERTS]  from router
        expert_index,  # U32  [T, TOP_K]    from router
        rank_id,  # I32  [1, 1]        EP rank (0 for EP1)
    ):
        """Fused expert-MM + weighted in-place combine (Part A stages 3-5).

        Runs one dense pass over all EXPERTS regardless of top-K contents
        (per §B6 AoT constraint); zero-affinity experts contribute zero
        to the combine.  POST_SCALE folds the router affinity into the
        down-projection contraction so no separate combine graph fires —
        the fp32 accumulator lives in SBUF throughout.
        """
        return moe_tkg(
            hidden_input=expert_input,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights_scaled,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            is_all_expert=True,
            rank_id=rank_id,
            mask_unselected_experts=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ACT_FN,
            output_dtype=expert_input.dtype,
        )

    return gemma_moe_router, gemma_moe_expert_combine


# --------------------------------------------------------------------------
# Public factories per shape family (compile-driver entry points)
# --------------------------------------------------------------------------


def make_qwen3_30b_a3b_tp8_kernel():
    """Fused MoE dispatch for Qwen3-30B-A3B TP=8 — first-strike test bed.

    Qwen3-30B-A3B is the higher-signal validation target per scope §4.4
    because it's already MFU 12.05% on the shipped stack (measurable
    baseline vs Gemma-4's 0.06% CPU-fallback state).  Compile shape:
    HIDDEN=2048, EXPERTS=128, TOP_K=8, I_TP=96 (clears %16), SILU.
    """
    return _make_moe_dispatch(QWEN3_30B_A3B_TP8)


def make_gemma4_26b_a4b_tp4_kernel():
    """Fused MoE dispatch for Gemma-4-26B-A4B TP=4 dual-input split.

    Matches Codex's proven `harness-v2/staging/cycle465` compile shape:
    HIDDEN=2816, EXPERTS=128, TOP_K=8, I_TP=176 (clears %16), GELU_Tanh_Approx.

    IMPORTANT: For Gemma-4 whole-window fire, this MoE kernel is
    necessary but NOT sufficient — the attention half still hits
    `_MAX_D_HEAD=128` and CPU-fallbacks until AWS PR #172 (validated
    `nki_flash_attn_d256_swa` + `nki_flash_attn_large_d`) is wired in.
    See `gemma4_no_fallback_mitigations.import_pr172_flash_attention`.
    """
    return _make_moe_dispatch(GEMMA4_26B_A4B_TP4)


def make_gpt_oss_20b_tp4_kernel():
    """Fused MoE dispatch for GPT-OSS-20B TP=4 — cross-model validation.

    GPT-OSS-20B does NOT renormalize top-K weights (§A.G-9) — the
    kernel emits the different convention automatically via the compile-time
    `renormalize_topk=False` branch in `router_topk`.
    """
    return _make_moe_dispatch(GPT_OSS_20B_TP4)


# --------------------------------------------------------------------------
# Fallback ladder (§A.5) — wire-in helper with FAIL-LOUD discipline
# --------------------------------------------------------------------------


class MoEFallbackRung(Enum):
    FUSED_DISPATCH = 0  # this kernel
    BLOCKWISE_MM_SHARD_INTERMEDIATE_HYBRID = 1  # current Qwen3-30B-A3B production path
    TORCH_BLOCKWISE_MATMUL_INFERENCE = 2  # §B5 CPU-fallback — FAIL LOUD


def enable_moe_fused_dispatch(neuron_config, cfg: MoEDispatchConfig) -> None:
    """Wire the fused MoE dispatch kernel into an NxDI NeuronConfig.

    This helper MUST be called before `InferenceConfig.__init__` and MUST
    also set `blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True`
    per the container `sha256:011d49c7…` workaround memory entry.

    Fallback discipline (§B5 / §C.1): every rung except the target rung
    emits a CRITICAL log when reached.  No silent drop to torch_blockwise.
    """
    cfg.validate()

    # 1) The mandatory container workaround (top-level rule).
    if not getattr(neuron_config, "blockwise_matmul_config", None):
        raise RuntimeError(
            "NeuronConfig.blockwise_matmul_config is None; refuse to wire "
            "moe_fused_dispatch without the container 011d49c7 workaround."
        )
    neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

    # 2) Opt-in flag on NeuronConfig (per scaffold §A.5).
    neuron_config.moe_fused_dispatch = True
    neuron_config.moe_fused_dispatch_debug = getattr(
        neuron_config, "moe_fused_dispatch_debug", False
    )

    # 3) Attach the shape config so downstream compile driver can pick it up.
    neuron_config._moe_fused_dispatch_shape = cfg

    # 4) Register the FAIL-LOUD hook.  NxDI's model_loader will emit an
    #    identifiable print — the Tier-1 CPU battery greps runtime output
    #    for `MoE fused dispatch = enabled` and refuses to proceed if the
    #    label is missing (per §A.6 Tier-1 gate).
    log.critical(
        "MoE fused dispatch = enabled | shape=%s | activation=%s | "
        "renormalize_topk=%s | I_TP=%d",
        cfg.name,
        cfg.activation.name,
        cfg.renormalize_topk,
        cfg.intermediate_per_tp,
    )


def log_moe_fallback(reached: MoEFallbackRung, cfg_name: str, reason: str) -> None:
    """Emit CRITICAL-level fallback marker (§C.1 discipline).

    The integration test greps for this exact string; a silent drop to
    rung 2 without this marker is a FAIL per the Tier-1 CPU battery.
    """
    log.critical(
        "MoE fallback ladder rung=%d name=%s cfg=%s reason=%s",
        reached.value,
        reached.name,
        cfg_name,
        reason,
    )
    if reached is MoEFallbackRung.TORCH_BLOCKWISE_MATMUL_INFERENCE:
        # This is the "silent CPU-fallback" hazard — never let it be silent.
        log.critical(
            "MoE dispatch has fallen to torch_blockwise_matmul_inference "
            "(host-side execution). §B5 discipline: fail loud; do NOT ship "
            "a customer receipt from this rung."
        )
