# SPDX-License-Identifier: Apache-2.0
"""Round-6 per-layer smoke: DSv4-Flash hash-MoE bootstrap block on layer 0
(the first hash-MoE layer per the frozen ``mlp_layer_types`` schedule).

Gates the LAYER-level composition of :class:`_HashMoEBlock` (the 6th and
final DSv4-Flash block class):

  1. **Router-with-side-channel forward** (pure PyTorch, no NxDI).
     :func:`dsv4_hash_route_affinities` under an all-in-range synthetic
     input must:
       * produce ``[T, n_routed_experts]`` fp32 affinities that are
         finite and strictly positive;
       * produce ``[T, num_experts_per_tok]`` int64 expert indices where
         every value is a direct lookup from ``tid2eid[input_ids]`` (NOT
         from a topk on the router scores);
       * refuse a wrong ``scoring_func`` loudly (silent-quality failure
         if a caller ever mixed up the routed-MoE and hash-MoE scoring
         convention);
       * refuse out-of-range ``input_ids`` loudly (an off-by-one on the
         token ids would silently route every affected token to
         whatever tid2eid[0] holds).

  2. **HF-key catalog for layer 0**.  Fetch the safetensors index and
     confirm every hash-MoE-required key exists under ``layers.0.ffn.*``:
     router weight + tid2eid + shared expert + 256 routed experts × 3
     tensors × 2 fields.  ``ffn.gate.bias`` MUST be absent (HF sets
     ``Gate.bias = None`` in hash mode).  Metadata-only pull under 10 MiB.

  3. **256-expert converter shape gate** (synthetic FP4 + FP8).  Invoke
     :func:`_convert_hash_moe_block` with a synthetic state_dict carrying
     all 256 routed experts and assert:
       * exactly **7** wrapper-tree keys land under ``layers.0.mlp.*``
         (task deliverable);
       * ``tid2eid`` has shape ``(vocab_size, num_experts_per_tok)`` and
         dtype ``int32``;
       * fused ``gate_up_proj.weight`` shape ``[E, hidden, 2*I]``;
       * ``down_proj.weight`` shape ``[E, I, hidden]``;
       * every one of the 256 experts' contributions is non-degenerate.
       * the converter refuses a wrong layer type (routed-MoE layer 3
         through the hash-MoE converter must fail loud).
       * the converter refuses a checkpoint that carries
         ``ffn.gate.bias`` at a hash layer (schedule drift).

  4. **Real-HF wrapper vs reference forward on real layer-0 tensors**.
     Vendor the real ``layers.0.ffn.gate.tid2eid`` + ``ffn.gate.weight``
     via HTTP-Range slice from shard ``model-00002-of-00048`` of
     ``deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe...`` (skips on
     no network).  Combined with the routed-MoE tests' expert pattern
     (synthetic FP4 experts + real router + real tid2eid), assert
     ``max_abs_error_bf16 == 0.0`` between the wrapper's forward and
     :func:`dsv4_reference_hash_moe_forward` — bit-clean composability
     against a hand-transcribed HF ``MoE.forward`` @ hash-MoE reference
     (``inference/model.py::MoE.forward`` lines 634-649).

  5. **Input-ids side-channel verification** (bit-exact + differentiated).
     Route the same hidden_states with two different input_ids sequences;
     assert the wrapper's expert indices differ on the tokens whose ids
     changed and match on the tokens whose ids stayed the same.  This is
     the "input_ids side channel is REALLY the routing knob" gate —
     numerical byte-cleanness alone doesn't prove the wrapper is using
     input_ids rather than silently defaulting to a topk on scores.

Design constraints honoured:

  * NO 3.6 GiB shard download — the real tensor pull is
    router.weight (~4 MiB after dequant) + tid2eid (~1.5 MiB int32).
  * NO device compile.  Wrapper + converter path only, gated on
    :func:`degeneracy_guard.require_comparable`.
  * NO speculative-decode surface.
  * Full absolute paths in receipts.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import struct
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Degeneracy guard — same discovery convention as the other DSv4 tests.
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
# Library discovery — same fallback pattern as test_hca_1layer.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
_HF_SHARD = "model-00002-of-00048.safetensors"
_LAYER_IDX = 0
_CACHE_DIR = Path(__file__).parent / ".hf_cache"
_CACHE_FILE = _CACHE_DIR / f"dsv4_layer{_LAYER_IDX}_hash_moe_router.safetensors"
_INDEX_CACHE = _CACHE_DIR / "dsv4_flash_index.json"
_HEADER_CHUNK_BYTES = 2 * 1024 * 1024  # 2 MB — shard headers can be ~1.4 MB

_LAYER0_HASH_ROUTER_KEYS: tuple[str, ...] = (
    f"layers.{_LAYER_IDX}.ffn.gate.weight",
    f"layers.{_LAYER_IDX}.ffn.gate.tid2eid",
)


def _import_library():
    """Import ``config`` + ``checkpoint_convert`` + ``neuron_wrapper``.

    Prefer the natural top-level import; fall back to explicit importlib
    on CPU-only laptops without the ``vllm`` package.
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
    pkg_name = "_dsv4_flash_test_pkg_hash_moe"
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
    try:
        wrap_mod = _load("neuron_wrapper")
    except Exception as exc:
        pytest.skip(f"neuron_wrapper unimportable even on CPU-only path: {exc!r}")
    return cfg_mod, conv_mod, wrap_mod


# ---------------------------------------------------------------------------
# HF index fetcher (metadata only).
# ---------------------------------------------------------------------------


def _fetch_index() -> dict[str, Any] | None:
    """Return the parsed ``model.safetensors.index.json`` or ``None`` if
    the network is unreachable.  Cached under ``.hf_cache/`` to avoid
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
            headers={"User-Agent": "vllm_neuron.dsv4_flash.tests.hash_moe/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        _INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_CACHE.write_bytes(data)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HF-shard slicer for the layer-0 hash-MoE router + tid2eid tensors.
#
# Layer 0 lives in shard 00002-of-00048; verified against
# ``.hf_cache/dsv4_flash_index.json``.  Router weight + tid2eid together
# are small (~5 MiB after dequant; the packed FP8 router is ~4 MiB with
# its scale, tid2eid is 129280*6*4 = 3.1 MiB int32).  Cached under
# ``.hf_cache/dsv4_layer0_hash_moe_router.safetensors`` after first pull.
# ---------------------------------------------------------------------------


def _fetch_range(url: str, byte_range: tuple[int, int]) -> bytes:
    """HTTP ``Range: bytes=start-end`` fetch, inclusive on both ends."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
            "User-Agent": "vllm_neuron.dsv4_flash.tests.hash_moe/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _build_local_cache_router_and_tid2eid() -> Path:
    """Pull layer-0 router.weight + router.scale + tid2eid; repack as
    mini-safetensors.  ~5 MiB after dequant, no shard-full download.
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

    # Router weight (FP8), router scale, and tid2eid.  Router scale is
    # named `<...>.gate.weight.scale`? or `<...>.gate.scale`? — verify via
    # the header.  DSv4 convention (from checkpoint_convert.py) is
    # `<...>.gate.scale`, and the index confirmed layer 0 does NOT have
    # a `layers.0.ffn.gate.scale` — the router is stored dense (fp32) at
    # inference-time; the safetensors index list confirmed there's no
    # `.scale` sibling for `ffn.gate.weight`.  So we pull just gate.weight
    # (dense) + tid2eid.
    keys_to_pull = list(_LAYER0_HASH_ROUTER_KEYS)
    tensors: dict[str, dict[str, Any]] = {}
    for hf_key in keys_to_pull:
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


def _load_layer0_router_state_dict() -> dict[str, torch.Tensor]:
    """Return the layer-0 router.weight + tid2eid as a state_dict."""
    from safetensors.torch import load_file

    if not _CACHE_FILE.exists():
        try:
            _build_local_cache_router_and_tid2eid()
        except Exception as exc:  # network / auth
            pytest.skip(
                "cannot fetch real DSv4-Flash layer-0 hash-MoE router "
                f"tensors ({_HF_REPO}@{_HF_SHA[:12]} {_HF_SHARD}): {exc!r}"
            )
    store = load_file(str(_CACHE_FILE))
    missing = [k for k in _LAYER0_HASH_ROUTER_KEYS if k not in store]
    if missing:
        raise KeyError(
            f"local mini-safetensors cache is missing keys {missing!r}; "
            f"delete {_CACHE_FILE!s} to force a re-pull."
        )
    return dict(store)


# ---------------------------------------------------------------------------
# Synthetic layer state dict — real router + real tid2eid + synthetic
# experts + synthetic shared expert.  Keeps the test fast (no need to
# dequant 256 * 3 real expert tensors) while exercising the full converter
# path.  Same synthesis convention as test_routed_moe_1layer.
# ---------------------------------------------------------------------------


def _synth_hash_moe_state_dict(
    layer_idx: int,
    src,
    *,
    real_router: torch.Tensor | None = None,
    real_tid2eid: torch.Tensor | None = None,
    seed: int = 20260828,
) -> dict[str, torch.Tensor]:
    """Build a state dict carrying every HF key the hash-MoE converter
    reads for one layer.  If ``real_router`` / ``real_tid2eid`` are
    passed, they replace the synthetic router + lookup — enabling the
    real-tensor byte-clean gate to route through the same converter path
    while keeping expert dequant fast.
    """
    hidden = src.hidden_size
    inter = src.moe_intermediate_size
    n_experts = src.n_routed_experts
    top_k = src.num_experts_per_tok
    vocab = src.vocab_size

    torch.manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    base = f"layers.{layer_idx}."

    if real_router is not None:
        assert real_router.shape == (n_experts, hidden), real_router.shape
        sd[f"{base}ffn.gate.weight"] = real_router
    else:
        sd[f"{base}ffn.gate.weight"] = (
            torch.randn(n_experts, hidden, dtype=torch.float32) * 0.02
        )
    if real_tid2eid is not None:
        assert real_tid2eid.shape == (vocab, top_k), real_tid2eid.shape
        assert real_tid2eid.dtype == torch.int32, real_tid2eid.dtype
        sd[f"{base}ffn.gate.tid2eid"] = real_tid2eid
    else:
        sd[f"{base}ffn.gate.tid2eid"] = torch.randint(
            0,
            n_experts,
            (vocab, top_k),
            dtype=torch.int32,
        )
    # NO ffn.gate.bias in hash mode — HF sets Gate.bias = None.

    # Shared expert: 3 x FP8 e4m3 tensors with UE8M0 block-scale (128, 128).
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

    # Routed experts: FP4 packed + UE8M0 block-scale (1, 32) on K axis —
    # same synthesis convention as test_routed_moe_1layer.
    for e in range(n_experts):
        base_e = f"{base}ffn.experts.{e}."
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
            nblk = in_logical // 32
            return torch.randint(
                122, 132, (out_dim, nblk), dtype=torch.uint8, generator=g
            ).contiguous()

        sd[f"{base_e}w1.weight"] = _packed(inter, hidden // 2)
        sd[f"{base_e}w1.scale"] = _scale(inter, hidden)
        sd[f"{base_e}w3.weight"] = _packed(inter, hidden // 2)
        sd[f"{base_e}w3.scale"] = _scale(inter, hidden)
        sd[f"{base_e}w2.weight"] = _packed(hidden, inter // 2)
        sd[f"{base_e}w2.scale"] = _scale(hidden, inter)

    return sd


# ===========================================================================
# Tests
# ===========================================================================


def test_hash_route_affinities_shape_and_range() -> None:
    """Router smoke: dsv4_hash_route_affinities returns finite affinities
    and correct-shape indices for a synthetic input."""
    cfg, _, wrap = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    hidden = src.hidden_size
    n_experts = src.n_routed_experts
    top_k = src.num_experts_per_tok
    vocab = src.vocab_size

    torch.manual_seed(1)
    hidden_states = torch.randn(1, 4, hidden, dtype=torch.float32)
    router_weight = torch.randn(n_experts, hidden, dtype=torch.float32) * 0.02
    tid2eid = torch.randint(0, n_experts, (vocab, top_k), dtype=torch.int32)
    input_ids = torch.randint(0, vocab, (1, 4), dtype=torch.int64)

    affinities, indices = wrap.dsv4_hash_route_affinities(
        hidden_states,
        router_weight,
        tid2eid,
        input_ids,
        scoring_func=src.scoring_func,
    )
    assert affinities.shape == (4, n_experts), affinities.shape
    assert affinities.dtype == torch.float32
    assert indices.shape == (4, top_k), indices.shape
    assert indices.dtype == torch.int64
    assert torch.isfinite(affinities).all(), affinities
    assert (affinities > 0).all(), affinities.min()
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) < n_experts

    # Indices came from tid2eid[input_ids], NOT from topk on scores.
    ref_indices = tid2eid.to(torch.long)[input_ids.reshape(-1)]
    assert torch.equal(indices, ref_indices), (indices, ref_indices)

    require_comparable(affinities.detach().cpu().numpy(), "hash_route_affinities")


def test_hash_route_refuses_wrong_scoring_func() -> None:
    """A wrong scoring_func must be refused — silent-quality failure if a
    caller ever mixed up the routed-MoE and hash-MoE scoring conventions.
    """
    cfg, _, wrap = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    hidden = src.hidden_size
    router_weight = torch.randn(src.n_routed_experts, hidden)
    tid2eid = torch.randint(
        0,
        src.n_routed_experts,
        (src.vocab_size, src.num_experts_per_tok),
        dtype=torch.int32,
    )
    with pytest.raises(NotImplementedError, match="sqrtsoftplus"):
        wrap.dsv4_hash_route_affinities(
            torch.randn(1, 1, hidden),
            router_weight,
            tid2eid,
            torch.tensor([[0]], dtype=torch.int32),
            scoring_func="sigmoid",
        )


def test_hash_route_refuses_out_of_range_input_ids() -> None:
    """Out-of-range input_ids must be refused loudly — an off-by-one on
    the token ids would silently route every affected token."""
    cfg, _, wrap = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    hidden = src.hidden_size
    router_weight = torch.randn(src.n_routed_experts, hidden)
    tid2eid = torch.randint(
        0,
        src.n_routed_experts,
        (src.vocab_size, src.num_experts_per_tok),
        dtype=torch.int32,
    )
    hidden_states = torch.randn(1, 1, hidden)
    # Just above vocab
    bad_ids = torch.tensor([[src.vocab_size]], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of vocab range"):
        wrap.dsv4_hash_route_affinities(
            hidden_states,
            router_weight,
            tid2eid,
            bad_ids,
            scoring_func=src.scoring_func,
        )
    # Negative
    bad_ids2 = torch.tensor([[-1]], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of vocab range"):
        wrap.dsv4_hash_route_affinities(
            hidden_states,
            router_weight,
            tid2eid,
            bad_ids2,
            scoring_func=src.scoring_func,
        )


def test_hash_moe_layer_schedule_confirms_layers_012() -> None:
    """Layers 0, 1, 2 must be hash-MoE per the frozen schedule."""
    cfg, _, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    assert src.num_hash_layers == 3
    assert src.mlp_layer_types[0] == "hash_moe"
    assert src.mlp_layer_types[1] == "hash_moe"
    assert src.mlp_layer_types[2] == "hash_moe"
    # Layer 3+ is routed MoE (noaux_tc with correction bias)
    assert src.mlp_layer_types[3] == "moe"


def test_hash_moe_wrapper_tree_key_set() -> None:
    """The _HashMoEBlock's parameter tree must be exactly the 7 keys the
    converter emits."""
    cfg, _, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    block = nw._HashMoEBlock(src, layer_idx=0)
    names = sorted(name for name, _ in block.named_parameters())
    expected = sorted(nw._HashMoEBlock.PARAM_KEYS)
    assert names == expected, (names, expected)
    assert len(nw._HashMoEBlock.PARAM_KEYS) == 7
    # tid2eid dtype and shape sanity.
    assert block.tid2eid.dtype == torch.int32
    assert tuple(block.tid2eid.shape) == (src.vocab_size, src.num_experts_per_tok)
    # Router lives in fp32 (numerical rationale documented on the class).
    assert block.router.weight.dtype == torch.float32
    # No e_score_correction_bias in hash mode.
    assert not any("e_score_correction_bias" in n for n, _ in block.named_parameters())


def test_hash_moe_block_refuses_wrong_layer_type() -> None:
    """The block hard-refuses instantiation at a non-hash-MoE layer index."""
    cfg, _, nw = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    # Layer 3 is routed MoE.
    with pytest.raises(ValueError, match="hash_moe"):
        nw._HashMoEBlock(src, layer_idx=3)


def test_hf_index_catalog_layer0_hash_moe() -> None:
    """Every HF-side hash-MoE key for layer 0 exists in the safetensors
    index; ``ffn.gate.bias`` MUST be absent (HF sets Gate.bias=None in
    hash mode).  Metadata-only pull.
    """
    cfg, _, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    idx = _fetch_index()
    if idx is None:
        pytest.skip(
            "safetensors index unreachable; layer-0 HF-key catalog cannot "
            "be verified offline"
        )
    weight_map = idx["weight_map"]
    keys = set(weight_map.keys())
    L = 0
    required = [
        f"layers.{L}.ffn.gate.weight",
        f"layers.{L}.ffn.gate.tid2eid",
        f"layers.{L}.ffn.shared_experts.w1.weight",
        f"layers.{L}.ffn.shared_experts.w1.scale",
        f"layers.{L}.ffn.shared_experts.w3.weight",
        f"layers.{L}.ffn.shared_experts.w3.scale",
        f"layers.{L}.ffn.shared_experts.w2.weight",
        f"layers.{L}.ffn.shared_experts.w2.scale",
    ]
    for e in range(src.n_routed_experts):
        for w in ("w1", "w2", "w3"):
            required.append(f"layers.{L}.ffn.experts.{e}.{w}.weight")
            required.append(f"layers.{L}.ffn.experts.{e}.{w}.scale")
    missing = [k for k in required if k not in keys]
    assert not missing, missing[:10]
    # Refuse a bias key for a hash layer.
    forbidden = f"layers.{L}.ffn.gate.bias"
    assert forbidden not in keys, (
        f"layer {L} is hash-MoE (Gate.bias should be None per HF "
        f"model.py:565) but the checkpoint carries {forbidden!r}"
    )
    total = 2 + 6 + src.n_routed_experts * 3 * 2  # router+tid2eid + shared + 256*3*2
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hf-index-layer0-hash-moe",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "layer_idx": L,
                "expected_hf_key_count": total,
                "verified_present": total - len(missing),
                "missing_count": len(missing),
                "gate_bias_absent": True,
                "index_cache_path": str(_INDEX_CACHE),
            },
            indent=2,
        )
    )


def test_synth_layer0_convert_shape_and_keys() -> None:
    """Round-6 gate: run _convert_hash_moe_block with synthetic tensors
    for all 256 experts, assert 7 wrapper-tree keys land."""
    cfg, conv, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    env_e = os.environ.get("DSV4_HASH_MOE_SMOKE_N_EXPERTS")
    n_experts = int(env_e) if env_e is not None else src.n_routed_experts
    if n_experts != src.n_routed_experts:
        # Rebuild config with reduced n_routed_experts for CPU memory tests.
        src = cfg.DeepseekV4FlashInferenceConfig(
            allow_reduced_shapes=True,
            num_hidden_layers=src.num_hidden_layers,
            layer_types=src.layer_types,
            mlp_layer_types=src.mlp_layer_types,
            n_routed_experts=n_experts,
        )
    sd = _synth_hash_moe_state_dict(_LAYER_IDX, src)
    converted: dict[str, Any] = {}
    report = conv._convert_hash_moe_block(
        sd, converted, layer_idx=_LAYER_IDX, src=src, dtype=torch.bfloat16
    )
    target = f"layers.{_LAYER_IDX}.mlp."
    expected_keys = {
        f"{target}router.weight",
        f"{target}tid2eid",
        f"{target}shared_expert.gate_proj.weight",
        f"{target}shared_expert.up_proj.weight",
        f"{target}shared_expert.down_proj.weight",
        f"{target}expert_mlps.mlp_op.gate_up_proj.weight",
        f"{target}expert_mlps.mlp_op.down_proj.weight",
    }
    got_keys = {k for k in converted if not k.startswith("_")}
    assert got_keys == expected_keys, sorted(
        got_keys.symmetric_difference(expected_keys)
    )
    assert len(expected_keys) == 7, len(expected_keys)

    gu = converted[f"{target}expert_mlps.mlp_op.gate_up_proj.weight"]
    dn = converted[f"{target}expert_mlps.mlp_op.down_proj.weight"]
    tid = converted[f"{target}tid2eid"]
    assert tuple(gu.shape) == (
        n_experts,
        src.hidden_size,
        2 * src.moe_intermediate_size,
    ), gu.shape
    assert tuple(dn.shape) == (n_experts, src.moe_intermediate_size, src.hidden_size), (
        dn.shape
    )
    assert tuple(tid.shape) == (src.vocab_size, src.num_experts_per_tok), tid.shape
    assert tid.dtype == torch.int32, tid.dtype

    # Every expert's contribution to both stacked tensors is non-degenerate.
    for e in range(n_experts):
        require_comparable(
            gu[e].detach().to(torch.float32).cpu().numpy(),
            f"hash_moe_expert_{e}_gate_up",
        )
        require_comparable(
            dn[e].detach().to(torch.float32).cpu().numpy(),
            f"hash_moe_expert_{e}_down",
        )
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hash-moe-converter-synth-layer0",
                "layer_idx": _LAYER_IDX,
                "wrapper_tree_key_count": len(got_keys),
                "n_experts_stacked": n_experts,
                "gate_up_shape": tuple(gu.shape),
                "down_shape": tuple(dn.shape),
                "tid2eid_shape": tuple(tid.shape),
                "tid2eid_dtype": str(tid.dtype),
                "report": report,
            },
            indent=2,
        )
    )


def test_converter_refuses_wrong_layer_type() -> None:
    """_convert_hash_moe_block must refuse a layer that is not hash_moe."""
    cfg, conv, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig()
    sd: dict[str, Any] = {}
    with pytest.raises(ValueError, match="hash_moe"):
        conv._convert_hash_moe_block(sd, {}, layer_idx=3, src=src, dtype=torch.bfloat16)


def test_converter_refuses_router_bias_at_hash_layer() -> None:
    """A checkpoint carrying ``ffn.gate.bias`` at a hash-MoE layer means
    the layer schedule drifted; converter must fail loud."""
    cfg, conv, _ = _import_library()
    src = cfg.DeepseekV4FlashInferenceConfig(
        allow_reduced_shapes=True,
        num_hidden_layers=3,
        layer_types=tuple(["sliding_attention"] * 3),
        mlp_layer_types=tuple(["hash_moe"] * 3),
        n_routed_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        moe_intermediate_size=64,
        hidden_size=128,
        compress_ratios=tuple([0] * 3),
    )
    sd = _synth_hash_moe_state_dict(0, src)
    # Inject the forbidden bias key.
    sd["layers.0.ffn.gate.bias"] = torch.zeros(
        src.n_routed_experts, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="Gate.bias=None"):
        conv._convert_hash_moe_block(sd, {}, layer_idx=0, src=src, dtype=torch.bfloat16)


def _reduced_config(cfg_mod):
    """Reduced config used by the wrapper-forward tests.  Keeps every
    invariant (head_dim=512, o_groups=8, ...) so DeepseekV4FlashInferenceConfig
    stops complaining, and reduces the moving parts (n_experts, vocab,
    moe_intermediate, hidden) to something a laptop can dequant + expert-loop
    in seconds instead of minutes.
    """
    return cfg_mod.DeepseekV4FlashInferenceConfig(
        allow_reduced_shapes=True,
        num_hidden_layers=3,
        layer_types=tuple(["sliding_attention"] * 3),
        mlp_layer_types=tuple(["hash_moe"] * 3),
        n_routed_experts=8,
        num_experts_per_tok=3,
        vocab_size=64,
        moe_intermediate_size=64,
        hidden_size=128,
        compress_ratios=tuple([0] * 3),
    )


def test_hash_moe_wrapper_matches_reference_synthetic() -> None:
    """Wrapper's forward path is bit-exact against
    :func:`dsv4_reference_hash_moe_forward` (which is a hand transcription
    of HF's ``MoE.forward`` @ hash-MoE) on synthetic weights + a random
    input.  Uses a reduced config so the per-expert loop terminates
    quickly on CPU.
    """
    cfg, _, nw = _import_library()
    src = _reduced_config(cfg)
    block = nw._HashMoEBlock(src, layer_idx=0)
    torch.manual_seed(0)
    with torch.no_grad():
        block.router.weight.copy_(torch.randn_like(block.router.weight) * 0.02)
        block.tid2eid.copy_(
            torch.randint(
                0,
                src.n_routed_experts,
                block.tid2eid.shape,
                dtype=torch.int32,
            )
        )
        block.expert_mlps.mlp_op.gate_up_proj.weight.copy_(
            torch.randn_like(block.expert_mlps.mlp_op.gate_up_proj.weight) * 0.02
        )
        block.expert_mlps.mlp_op.down_proj.weight.copy_(
            torch.randn_like(block.expert_mlps.mlp_op.down_proj.weight) * 0.02
        )
        for lin in (
            block.shared_expert.gate_proj,
            block.shared_expert.up_proj,
            block.shared_expert.down_proj,
        ):
            lin.weight.copy_(torch.randn_like(lin.weight) * 0.02)

    hidden_states = torch.randn(2, 5, src.hidden_size, dtype=torch.bfloat16) * 0.1
    input_ids = torch.randint(0, src.vocab_size, (2, 5), dtype=torch.int32)
    with torch.no_grad():
        y_wrap = block(hidden_states, input_ids)
        y_ref = nw.dsv4_reference_hash_moe_forward(
            hidden_states,
            input_ids,
            block.router.weight,
            block.tid2eid,
            shared_gate=block.shared_expert.gate_proj.weight,
            shared_up=block.shared_expert.up_proj.weight,
            shared_down=block.shared_expert.down_proj.weight,
            expert_gate_up_stack=block.expert_mlps.mlp_op.gate_up_proj.weight,
            expert_down_stack=block.expert_mlps.mlp_op.down_proj.weight,
            swiglu_limit=src.swiglu_limit,
            routed_scaling_factor=src.routed_scaling_factor,
        )
    assert y_wrap.shape == y_ref.shape
    assert y_wrap.dtype == y_ref.dtype
    require_comparable(
        y_wrap.detach().to(torch.float32).cpu().numpy(),
        "hash_moe_wrapper_output_synth",
    )
    require_comparable(
        y_ref.detach().to(torch.float32).cpu().numpy(),
        "hash_moe_reference_output_synth",
    )
    diff = (y_wrap.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hash-moe-wrapper-vs-ref-synth",
                "layer_idx": 0,
                "max_abs_error_bf16": max_abs,
                "mean_abs_error_bf16": mean_abs,
                "n_experts": src.n_routed_experts,
                "vocab_size": src.vocab_size,
                "top_k": src.num_experts_per_tok,
                "output_shape": tuple(y_wrap.shape),
            },
            indent=2,
        )
    )
    # BYTE-CLEAN target — task deliverable.
    assert max_abs == 0.0, max_abs


def test_input_ids_side_channel_actually_routes() -> None:
    """Prove the wrapper is using input_ids for routing, not silently
    defaulting to a topk on router scores.  Flip a single token's id;
    the resulting expert indices for that token MUST change AND the
    output for that token MUST change; the output for the other tokens
    MUST NOT change.
    """
    cfg, _, nw = _import_library()
    src = _reduced_config(cfg)
    block = nw._HashMoEBlock(src, layer_idx=0)
    torch.manual_seed(5)
    with torch.no_grad():
        block.router.weight.copy_(torch.randn_like(block.router.weight) * 0.02)
        # Deterministic tid2eid so we can predict what a flip does.
        block.tid2eid.copy_(
            torch.randint(
                0,
                src.n_routed_experts,
                block.tid2eid.shape,
                dtype=torch.int32,
            )
        )
        block.expert_mlps.mlp_op.gate_up_proj.weight.copy_(
            torch.randn_like(block.expert_mlps.mlp_op.gate_up_proj.weight) * 0.02
        )
        block.expert_mlps.mlp_op.down_proj.weight.copy_(
            torch.randn_like(block.expert_mlps.mlp_op.down_proj.weight) * 0.02
        )
        for lin in (
            block.shared_expert.gate_proj,
            block.shared_expert.up_proj,
            block.shared_expert.down_proj,
        ):
            lin.weight.copy_(torch.randn_like(lin.weight) * 0.02)

    hidden_states = torch.randn(1, 4, src.hidden_size, dtype=torch.bfloat16) * 0.1
    ids_a = torch.tensor([[3, 7, 11, 19]], dtype=torch.int32)
    # Find a flip that actually changes tid2eid picks for token 0.
    changed_id = 3
    for cand in range(src.vocab_size):
        if cand == 3:
            continue
        if not torch.equal(block.tid2eid[3], block.tid2eid[cand]):
            changed_id = cand
            break
    assert changed_id != 3, "vocab too degenerate; every row of tid2eid identical"
    ids_b = ids_a.clone()
    ids_b[0, 0] = changed_id

    with torch.no_grad():
        # Direct router probe: indices from tid2eid must differ for token 0.
        _, idx_a = nw.dsv4_hash_route_affinities(
            hidden_states,
            block.router.weight,
            block.tid2eid,
            ids_a,
            scoring_func=src.scoring_func,
        )
        _, idx_b = nw.dsv4_hash_route_affinities(
            hidden_states,
            block.router.weight,
            block.tid2eid,
            ids_b,
            scoring_func=src.scoring_func,
        )
        # Indices for token 0 differ (side channel active).
        assert not torch.equal(idx_a[0], idx_b[0]), (idx_a[0], idx_b[0])
        # Indices for the unchanged tokens are identical.
        for t in (1, 2, 3):
            assert torch.equal(idx_a[t], idx_b[t]), (t, idx_a[t], idx_b[t])

        # Full-block forward: output for token 0 differs, others identical.
        y_a = block(hidden_states, ids_a)
        y_b = block(hidden_states, ids_b)

    delta = (y_b.to(torch.float32) - y_a.to(torch.float32)).abs()
    max_at_0 = float(delta[:, 0].max().item())
    max_elsewhere = float(delta[:, 1:].max().item())
    assert max_at_0 > 0.0, max_at_0
    # Elsewhere the output MUST be byte-identical — the input_ids for tokens
    # 1..3 didn't change, so their routed experts and weights didn't change,
    # so their outputs must be bit-identical.
    assert max_elsewhere == 0.0, max_elsewhere
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hash-moe-input-ids-side-channel",
                "changed_input_id_from": 3,
                "changed_input_id_to": changed_id,
                "delta_max_at_changed_token": max_at_0,
                "delta_max_at_unchanged_tokens": max_elsewhere,
                "side_channel_verified": True,
            },
            indent=2,
        )
    )


def test_hash_moe_wrapper_matches_hf_reference_on_real_layer0_router() -> None:
    """The whole enchilada: REAL layer-0 router.weight + tid2eid pulled
    via HTTP-Range, synthetic experts, wrapper forward vs
    :func:`dsv4_reference_hash_moe_forward`, byte-clean bf16 diff.

    Skips on no network.  The reason we use synthetic experts (rather
    than real FP4-dequant) is a memory + wall-time trade: dequanting 256
    real experts at full DSv4-Flash sizes is ~6 GB peak RSS and 60-120 s
    on a laptop — the FP4 dequant path itself is byte-exact against a
    real HF tensor already (``test_fp4_dequant_1tensor``), so the value
    add here is the *hash-MoE-specific* wiring: real router + real
    tid2eid + gather + normalise + scale + shared expert composition.
    """
    cfg, conv, nw = _import_library()

    # Load the real layer-0 router + tid2eid.  On CPU we cannot afford the
    # full 256-expert dequant; the routed-MoE tests already cover FP4
    # dequant byte-cleanness end-to-end.
    real = _load_layer0_router_state_dict()
    real_router = real[f"layers.{_LAYER_IDX}.ffn.gate.weight"]
    real_tid2eid = real[f"layers.{_LAYER_IDX}.ffn.gate.tid2eid"]

    # Sanity: shapes must match the frozen config.
    src_full = cfg.DeepseekV4FlashInferenceConfig()
    assert tuple(real_router.shape) == (
        src_full.n_routed_experts,
        src_full.hidden_size,
    ), real_router.shape
    assert tuple(real_tid2eid.shape) == (
        src_full.vocab_size,
        src_full.num_experts_per_tok,
    ), real_tid2eid.shape
    assert real_tid2eid.dtype == torch.int32, real_tid2eid.dtype

    # The real router weight is dense-stored on HF (index confirmed no
    # `.scale` sibling).  Its dtype is bf16 in the HF snapshot; cast to
    # fp32 for the wrapper's router.
    real_router_fp32 = real_router.to(torch.float32)

    # Run the reduced-size wrapper with the FULL-vocab tid2eid + FULL-size
    # router.  We shrink only n_experts (via env override on the wrapper)
    # for the CPU expert loop's sake?  No — the wrapper is defined at
    # full n_experts.  Better: rebuild a synthetic-experts state_dict that
    # carries the FULL config shape, using the real router + tid2eid, and
    # let the converter emit the full-size stacked tensors.  256 stacks
    # at bf16 hidden=4096 inter=2048 = ~6 GB — too much on a 16 GB laptop.
    #
    # Instead: subsample n_experts down to a small k (say 8) — since we
    # use the real tid2eid, we must first REMAP its expert indices to fall
    # inside [0, k).  This gives us a hash-MoE forward that uses the real
    # router structure (its per-token scores + gather math) with the real
    # tid2eid selection PATTERN (each token still picks its own top_k
    # experts based on token id), just projected onto a reduced expert
    # bank.
    n_experts_reduced = int(os.environ.get("DSV4_HASH_MOE_REAL_N_EXPERTS", "8"))
    top_k = src_full.num_experts_per_tok
    # Remap tid2eid mod n_experts_reduced, but stay INJECTIVE per row
    # (top_k distinct experts per row) — HF's tid2eid rows are already
    # distinct, we just need the remap to preserve distinctness with
    # high probability.  Since n_experts_reduced >= top_k, take
    # `real_tid2eid % n_experts_reduced` and fix collisions by shifting.
    tid_reduced = real_tid2eid.to(torch.int64) % n_experts_reduced
    for row in range(tid_reduced.shape[0]):
        # Fix duplicates by incrementing until distinct.
        seen: set[int] = set()
        for k in range(top_k):
            v = int(tid_reduced[row, k].item())
            while v in seen:
                v = (v + 1) % n_experts_reduced
            tid_reduced[row, k] = v
            seen.add(v)
    tid_reduced = tid_reduced.to(torch.int32)

    src = cfg.DeepseekV4FlashInferenceConfig(
        allow_reduced_shapes=True,
        num_hidden_layers=3,
        layer_types=tuple(["sliding_attention"] * 3),
        mlp_layer_types=tuple(["hash_moe"] * 3),
        n_routed_experts=n_experts_reduced,
        num_experts_per_tok=top_k,
        vocab_size=src_full.vocab_size,
        moe_intermediate_size=64,  # reduced
        hidden_size=src_full.hidden_size,
        compress_ratios=tuple([0] * 3),
    )

    # Reduced router that lives on the reduced expert bank.  We build it
    # by pulling the first n_experts_reduced rows of the real router (bf16
    # -> fp32).  Same-distribution, no wall-time cost.
    reduced_router = real_router_fp32[:n_experts_reduced].contiguous()

    # Build a state_dict with the reduced router + reduced tid2eid +
    # synthetic experts, and run the converter to get the wrapper-tree
    # state.
    sd = _synth_hash_moe_state_dict(
        _LAYER_IDX,
        src,
        real_router=reduced_router,
        real_tid2eid=tid_reduced,
    )
    converted: dict[str, Any] = {}
    conv._convert_hash_moe_block(
        sd, converted, layer_idx=_LAYER_IDX, src=src, dtype=torch.bfloat16
    )

    # Build the wrapper block; load the converted tensors.
    block = nw._HashMoEBlock(src, layer_idx=_LAYER_IDX)
    target = f"layers.{_LAYER_IDX}.mlp."
    with torch.no_grad():
        block.router.weight.copy_(converted[f"{target}router.weight"])
        block.tid2eid.copy_(converted[f"{target}tid2eid"])
        block.expert_mlps.mlp_op.gate_up_proj.weight.copy_(
            converted[f"{target}expert_mlps.mlp_op.gate_up_proj.weight"]
        )
        block.expert_mlps.mlp_op.down_proj.weight.copy_(
            converted[f"{target}expert_mlps.mlp_op.down_proj.weight"]
        )
        block.shared_expert.gate_proj.weight.copy_(
            converted[f"{target}shared_expert.gate_proj.weight"]
        )
        block.shared_expert.up_proj.weight.copy_(
            converted[f"{target}shared_expert.up_proj.weight"]
        )
        block.shared_expert.down_proj.weight.copy_(
            converted[f"{target}shared_expert.down_proj.weight"]
        )

    # Deterministic synthetic input; real vocab-range input_ids so we
    # exercise the tid2eid gather at true-vocab scale.
    torch.manual_seed(11)
    B, S = 1, 8
    hidden = torch.randn(B, S, src.hidden_size, dtype=torch.bfloat16) * 0.1
    input_ids = torch.randint(0, src.vocab_size, (B, S), dtype=torch.int32)

    with torch.no_grad():
        y_wrap = block(hidden, input_ids)
        y_ref = nw.dsv4_reference_hash_moe_forward(
            hidden,
            input_ids,
            block.router.weight,
            block.tid2eid,
            shared_gate=block.shared_expert.gate_proj.weight,
            shared_up=block.shared_expert.up_proj.weight,
            shared_down=block.shared_expert.down_proj.weight,
            expert_gate_up_stack=block.expert_mlps.mlp_op.gate_up_proj.weight,
            expert_down_stack=block.expert_mlps.mlp_op.down_proj.weight,
            swiglu_limit=src.swiglu_limit,
            routed_scaling_factor=src.routed_scaling_factor,
        )

    require_comparable(
        y_wrap.detach().to(torch.float32).cpu().numpy(),
        "hash_moe_wrapper_output_real_router",
    )
    require_comparable(
        y_ref.detach().to(torch.float32).cpu().numpy(),
        "hash_moe_reference_output_real_router",
    )
    diff = (y_wrap.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-hash-moe-real-router-tid2eid",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "hf_shard": _HF_SHARD,
                "layer_idx": _LAYER_IDX,
                "mlp_layer_type": src.mlp_layer_types[_LAYER_IDX],
                "reduced_n_experts": src.n_routed_experts,
                "full_vocab_size": src.vocab_size,
                "top_k": src.num_experts_per_tok,
                "batch_size": B,
                "seq_len": S,
                "wrapper_tree_key_count": len(list(block.named_parameters())),
                "max_abs_error_bf16": max_abs,
                "mean_abs_error_bf16": mean_abs,
                "output_shape": tuple(y_wrap.shape),
                "real_router_source_path": str(_CACHE_FILE),
            },
            indent=2,
        )
    )
    assert max_abs == 0.0, max_abs


# ---------------------------------------------------------------------------
# Standalone runner — mirrors test_hca_1layer.py so a laptop without
# pytest-collectible `vllm_neuron` can still run the gate.
# ---------------------------------------------------------------------------


def _standalone_main() -> int:
    tests = [
        test_hash_route_affinities_shape_and_range,
        test_hash_route_refuses_wrong_scoring_func,
        test_hash_route_refuses_out_of_range_input_ids,
        test_hash_moe_layer_schedule_confirms_layers_012,
        test_hash_moe_wrapper_tree_key_set,
        test_hash_moe_block_refuses_wrong_layer_type,
        test_hf_index_catalog_layer0_hash_moe,
        test_synth_layer0_convert_shape_and_keys,
        test_converter_refuses_wrong_layer_type,
        test_converter_refuses_router_bias_at_hash_layer,
        test_hash_moe_wrapper_matches_reference_synthetic,
        test_input_ids_side_channel_actually_routes,
        test_hash_moe_wrapper_matches_hf_reference_on_real_layer0_router,
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
                "suite": "dsv4-flash.tests.test_hash_moe_1layer",
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
