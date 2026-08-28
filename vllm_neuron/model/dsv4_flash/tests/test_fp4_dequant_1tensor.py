# SPDX-License-Identifier: Apache-2.0
"""Round-1 1-tensor gate: byte-exact FP4-UE8M0 dequant.

This test is the correctness backstop for every routed-expert weight in
DeepSeek-V4-Flash.  A silent bug in the FP4 packing convention or the
UE8M0 bias would render every routed-MoE compile plausible-looking and
quietly wrong.  We defend that by:

  1. Loading one **real** routed-expert w2 shard fragment
     (``layers.3.ffn.experts.0.w2.weight`` + ``.scale``) sliced out of
     HF snapshot ``deepseek-ai/DeepSeek-V4-Flash-0731`` @
     ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``.  The fetch is HTTP-
     range-based (no full 3.6 GB shard download) — 4.3 MB of tensor
     bytes total.  On first run the mini-safetensors file is cached
     under ``.hf_cache/`` next to this test; subsequent runs are offline.
  2. Running our :func:`dequantize_block_fp4_ue8m0` against that tensor.
  3. Comparing against an INDEPENDENT reference — a byte-for-byte
     transcription of the DSv4-Flash HF inference reference's dequant
     path (``inference/convert.py::cast_e2m1fn_to_e4m3fn`` @ the same
     SHA, lines 30-33 for nibble unpack + line 11-14 for the FP4-E2M1
     codebook).  The reference is written from scratch in this file so
     the test does not import back into the code it is supposed to
     verify.
  4. Asserting ``max_abs_error_bf16 == 0.0`` — byte-exact.  The FP4-E2M1
     codebook is a 16-value table of small integers and half-integers;
     any single UE8M0 multiplier ``2**(X-127)`` is exact in fp32; the
     product is a bf16-representable value; therefore any nonzero error
     indicates a real discrepancy, not rounding slack.
  5. Running the output through the campaign's degeneracy guard
     (``require_comparable``) on both sides so an all-zero or all-NaN
     result cannot vacuously pass.
  6. Covering synthetic edge cases (all-zero packing → all-zero output;
     sentinel scale ``2**0`` → identity; max positive int4 × max sane
     UE8M0 code → no bf16 overflow at typical block scales).

The test skips (rather than errors) if the local HF cache miss cannot
be filled (no network).  It never falls back to synthetic-only when
the intent is the byte-exact gate — a skip is louder than a false pass.
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
# Degeneracy guard: same lookup convention as the GLM-5.3-Flash tests so
# we don't carry a private copy.
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
# Independent reference: FP4-E2M1 codebook + nibble unpack.
#
# Transcribed byte-for-byte from ``deepseek-ai/DeepSeek-V4-Flash-0731`` @
# HF SHA ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``,
# ``inference/convert.py::FP4_TABLE`` (lines 11-14) and the nibble-
# unpack block (lines 30-33).  This constant intentionally lives here,
# NOT imported from ``checkpoint_convert``, so a copy-paste error in the
# library tensor would be caught rather than papered over.
_REF_FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _ref_fp4_ue8m0_dequant(
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Independent FP4-UE8M0 -> bf16 dequant reference.

    Matches ``inference/convert.py::cast_e2m1fn_to_e4m3fn`` in every
    numerical step, but expresses the E8M0 → fp32 conversion via
    PyTorch's own ``float8_e8m0fnu`` cast so the bias-127 semantics are
    tested against the storage dtype directly.  For an integer scale
    tensor (e.g. a synthetic case where the caller stored raw bytes as
    uint8) we redo the bias-127 arithmetic explicitly.
    """
    assert weight_int8.ndim == 2, weight_int8.shape
    assert weight_int8.dtype in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2)
    out_dim, in_bytes = weight_int8.shape
    in_dim = in_bytes * 2
    block_out, block_in = block_size
    assert scale.shape == (
        math.ceil(out_dim / block_out),
        math.ceil(in_dim / block_in),
    ), (scale.shape, (out_dim, in_dim), block_size)

    # 1. Nibble unpack (spec-cited).
    x = weight_int8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    table = _REF_FP4_TABLE.to(x.device)
    fp4_fp32 = torch.stack(
        [table[low.long()], table[high.long()]], dim=-1
    ).flatten(-2)

    # 2. UE8M0 → fp32 multiplier.
    if scale.dtype == torch.float8_e8m0fnu:
        scale_fp32 = scale.to(torch.float32)
    elif scale.dtype.is_floating_point:
        raise AssertionError(
            f"reference refuses scale dtype {scale.dtype} — E8M0 or integer only"
        )
    else:
        exp = scale.to(torch.int32) - 127
        scale_fp32 = torch.ldexp(
            torch.ones_like(exp, dtype=torch.float32), exp
        )
    assert torch.isfinite(scale_fp32).all(), "reference scale carries NaN/inf"

    # 3. Block broadcast + product.
    scale_bcast = scale_fp32.repeat_interleave(block_out, dim=-2).repeat_interleave(
        block_in, dim=-1
    )[..., :out_dim, :in_dim]
    return (fp4_fp32 * scale_bcast).to(out_dtype)


# ---------------------------------------------------------------------------
# HF-shard slicer: HTTP-Range fetch of one routed-expert tensor pair.
# ---------------------------------------------------------------------------

_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
_HF_SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
_HF_SHARD = "model-00005-of-00048.safetensors"
_HF_KEYS = (
    ("layers.3.ffn.experts.0.w2.weight", "w2_weight"),
    ("layers.3.ffn.experts.0.w2.scale",  "w2_scale"),
)
_CACHE_DIR = Path(__file__).parent / ".hf_cache"
_CACHE_FILE = _CACHE_DIR / "dsv4_expert0_w2.safetensors"
# 512 KB is enough to cover the safetensors header (172 KB observed
# 2026-08-28); we grow the pull if a future re-upload adds header
# metadata.
_HEADER_CHUNK_BYTES = 1024 * 1024


def _fetch_range(url: str, byte_range: tuple[int, int]) -> bytes:
    """HTTP ``Range: bytes=start-end`` fetch, inclusive on both ends."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
            "User-Agent": "vllm_neuron.dsv4_flash.tests/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _build_local_cache() -> Path:
    """Pull the two tensors from the HF shard via HTTP-Range and repack.

    Returns the local mini-safetensors path.  Raises the network error
    on failure — the caller decides whether to skip the test.
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
    for hf_key, out_key in _HF_KEYS:
        meta = header[hf_key]
        o0, o1 = meta["data_offsets"]
        payload = _fetch_range(url, (data_start + o0, data_start + o1 - 1))
        assert len(payload) == (o1 - o0), (out_key, len(payload), o1 - o0)
        tensors[out_key] = {
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


def _load_real_expert_tensor() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(w2_weight[int8, 4096, 1024], w2_scale[e8m0, 4096, 64])``.

    Skips (not fails) the test if the cache is empty AND the network
    fetch cannot complete — a network-less local dev box is not a
    correctness failure.
    """
    from safetensors.torch import load_file

    if not _CACHE_FILE.exists():
        try:
            _build_local_cache()
        except Exception as exc:  # network / auth / access
            pytest.skip(
                "cannot fetch real DSv4-Flash routed-expert tensor "
                f"({_HF_REPO}@{_HF_SHA[:12]} {_HF_SHARD} "
                f"layers.3.ffn.experts.0.w2): {exc!r}"
            )
    store = load_file(str(_CACHE_FILE))
    return store["w2_weight"], store["w2_scale"]


# ---------------------------------------------------------------------------
# Library under test — imported here rather than at module-scope so a
# checkpoint_convert import error surfaces as a test failure with a
# clear trace instead of a module-collection failure.
# ---------------------------------------------------------------------------


def _import_library():
    """Import the checkpoint_convert module.

    Prefer the natural top-level import so a compile-host CI hits the
    same code path as production.  Fall back to importlib-only loading
    when ``vllm_neuron/__init__.py`` cannot execute (e.g. a dev box
    without the ``vllm`` package installed) — that keeps this test
    runnable locally so the FP4 gate does not silently rot behind an
    environment gate.
    """
    try:
        from vllm_neuron.model.dsv4_flash.checkpoint_convert import (  # type: ignore
            _FP4_E2M1_TABLE,
            dequantize_block_fp4_ue8m0,
        )
        from vllm_neuron.model.dsv4_flash.config import (  # type: ignore
            validate_ue8m0_scale,
        )
        return _FP4_E2M1_TABLE, dequantize_block_fp4_ue8m0, validate_ue8m0_scale
    except Exception:
        pass

    import importlib.util
    import types

    dsv4_dir = Path(__file__).resolve().parent.parent
    pkg = types.ModuleType("_dsv4_flash_test_pkg")
    pkg.__path__ = [str(dsv4_dir)]
    sys.modules["_dsv4_flash_test_pkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_dsv4_flash_test_pkg.{name}",
            str(dsv4_dir / f"{name}.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    config = _load("config")
    convert = _load("checkpoint_convert")
    return (
        convert._FP4_E2M1_TABLE,
        convert.dequantize_block_fp4_ue8m0,
        config.validate_ue8m0_scale,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fp4_e2m1_codebook_matches_hf_reference() -> None:
    """The library's ``_FP4_E2M1_TABLE`` must equal the HF reference
    codebook byte-for-byte.  Any drift is a silent-corruption vector."""
    lib_table, _, _ = _import_library()
    assert tuple(lib_table) == tuple(_REF_FP4_TABLE.tolist()), (
        lib_table,
        _REF_FP4_TABLE.tolist(),
    )
    # Sanity: DSv4 codebook is the standard OCP MXFP4 codebook.
    assert lib_table[0] == 0.0 and lib_table[8] == 0.0  # +0 and -0 codes
    assert lib_table[7] == 6.0 and lib_table[15] == -6.0  # max magnitude


def test_fp4_dequant_real_hf_tensor_byte_exact() -> None:
    """Round-1 byte-exact gate: real DSv4-Flash routed-expert w2."""
    _, dequantize_block_fp4_ue8m0, validate_ue8m0_scale = _import_library()
    weight_int8, scale_e8m0 = _load_real_expert_tensor()

    # Preflight the loaded tensor: shape + dtype + validator.
    assert weight_int8.dtype == torch.int8, weight_int8.dtype
    assert weight_int8.shape == (4096, 1024), weight_int8.shape
    assert scale_e8m0.dtype == torch.float8_e8m0fnu, scale_e8m0.dtype
    assert scale_e8m0.shape == (4096, 64), scale_e8m0.shape
    # No NaN codes (byte 255) — the validator would refuse the run.
    validate_ue8m0_scale(scale_e8m0, "w2_scale")

    block_size = (1, 32)  # DSv4-Flash FP4 spec: per-row × per-32
    lib_bf16 = dequantize_block_fp4_ue8m0(
        weight_int8, scale_e8m0, block_size, torch.bfloat16
    )
    ref_bf16 = _ref_fp4_ue8m0_dequant(
        weight_int8, scale_e8m0, block_size, torch.bfloat16
    )
    assert lib_bf16.shape == (4096, 2048), lib_bf16.shape
    assert lib_bf16.dtype == torch.bfloat16, lib_bf16.dtype
    assert ref_bf16.shape == lib_bf16.shape

    # Degeneracy guard — both sides must be non-degenerate before we
    # gate on max_abs_error.  For bf16 tensors we compare fp32-cast arrays.
    require_comparable(
        lib_bf16.detach().to(torch.float32).cpu().numpy(), "lib_output_fp32"
    )
    require_comparable(
        ref_bf16.detach().to(torch.float32).cpu().numpy(), "ref_output_fp32"
    )

    # Byte-exact match: max_abs_error_bf16 == 0.0.
    diff = (
        lib_bf16.to(torch.float32) - ref_bf16.to(torch.float32)
    ).abs()
    max_abs_error_bf16 = float(diff.max().item())
    mean_abs_error_bf16 = float(diff.mean().item())
    print(
        json.dumps(
            {
                "smoke": "dsv4-flash-fp4-ue8m0-real-tensor",
                "hf_repo": _HF_REPO,
                "hf_sha": _HF_SHA,
                "hf_shard": _HF_SHARD,
                "hf_key": "layers.3.ffn.experts.0.w2",
                "packed_shape": tuple(weight_int8.shape),
                "logical_shape": tuple(lib_bf16.shape),
                "scale_shape": tuple(scale_e8m0.shape),
                "scale_dtype": str(scale_e8m0.dtype),
                "block_size": block_size,
                "max_abs_error_bf16": max_abs_error_bf16,
                "mean_abs_error_bf16": mean_abs_error_bf16,
            },
            indent=2,
        )
    )
    assert max_abs_error_bf16 == 0.0, max_abs_error_bf16


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------


def test_fp4_dequant_all_zero_packing() -> None:
    """A packing of all-zero bytes must produce an all-zero output for
    any valid UE8M0 scale (nibble 0 → codebook value 0.0)."""
    _, dequantize_block_fp4_ue8m0, _ = _import_library()
    weight = torch.zeros((4, 32), dtype=torch.int8)  # 4 x 64 logical FP4
    # Scale with mixed non-degenerate codes — ensure the zero survives.
    scale = torch.tensor([[100, 127, 130, 140]] * 4, dtype=torch.uint8)
    out = dequantize_block_fp4_ue8m0(weight, scale, (1, 16), torch.bfloat16)
    assert out.shape == (4, 64), out.shape
    assert torch.all(out == 0), (out.min(), out.max())


def test_fp4_dequant_identity_scale() -> None:
    """UE8M0 raw code 127 → multiplier 2**0 = 1.0.  The dequant output
    must equal the FP4 codebook lookup directly."""
    lib_table, dequantize_block_fp4_ue8m0, _ = _import_library()
    # Weight bytes 0x00..0xFF cover every FP4 codepoint pair; take 16
    # bytes = 32 FP4 values which is exactly one block.
    row = torch.arange(0, 32, dtype=torch.uint8).view(torch.int8)  # 32 bytes
    weight = row.reshape(1, 32)  # 1 x 32 packed -> 1 x 64 logical
    scale = torch.full((1, 2), 127, dtype=torch.uint8)  # 2**0 = 1.0
    out = dequantize_block_fp4_ue8m0(weight, scale, (1, 32), torch.bfloat16)
    assert out.shape == (1, 64), out.shape
    # Reference: unpack manually and look up.
    x = row.view(torch.uint8)
    expected = torch.stack(
        [
            torch.tensor(lib_table)[(x & 0x0F).long()],
            torch.tensor(lib_table)[((x >> 4) & 0x0F).long()],
        ],
        dim=-1,
    ).flatten().to(torch.bfloat16)
    assert torch.equal(out[0], expected), (
        out[0].tolist()[:16],
        expected.tolist()[:16],
    )


def test_fp4_dequant_max_positive_no_overflow() -> None:
    """FP4 max magnitude (6.0, code 0x07) × largest UE8M0 code that keeps
    the product in bf16 range must not overflow to inf.  bf16's max is
    ``~3.39e38 = 2**128``, so ``6.0 * 2**(x-127) < 2**128`` means
    ``x - 127 + log2(6) < 128`` → ``x < 128 + 127 - log2(6) ≈ 252.4``.
    Pick raw code 250 — well inside range."""
    _, dequantize_block_fp4_ue8m0, _ = _import_library()
    # Byte 0x77 packs two 0x07 nibbles = two +6.0 FP4 values.
    packed = torch.full((1, 16), 0x77, dtype=torch.uint8).view(torch.int8)  # 32 fp4 vals
    scale = torch.full((1, 1), 250, dtype=torch.uint8)  # 2**123 ≈ 8.5e36
    out = dequantize_block_fp4_ue8m0(packed, scale, (1, 32), torch.bfloat16)
    assert out.shape == (1, 32), out.shape
    fp32 = out.to(torch.float32)
    assert torch.isfinite(fp32).all(), fp32
    # Value should be 6.0 * 2**123 = 3 * 2**124 -> exactly representable in bf16
    expected = 6.0 * (2 ** 123)
    assert float(fp32.max().item()) == expected, (
        float(fp32.max().item()), expected,
    )


def test_fp4_dequant_rejects_nan_scale() -> None:
    """A ``float8_e8m0fnu`` scale carrying raw byte 255 (NaN) must be
    refused by the validator — silently broadcasting NaN across a whole
    block would poison every accumulated activation."""
    _, dequantize_block_fp4_ue8m0, _ = _import_library()
    weight = torch.zeros((1, 32), dtype=torch.int8)
    bad = torch.tensor([[0, 255]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="NaN"):
        dequantize_block_fp4_ue8m0(weight, bad, (1, 32), torch.bfloat16)


def test_fp4_dequant_rejects_scale_shape_mismatch() -> None:
    """A scale with the wrong trailing shape must be refused rather
    than silently broadcast-guessed."""
    _, dequantize_block_fp4_ue8m0, _ = _import_library()
    weight = torch.zeros((2, 32), dtype=torch.int8)  # logical 2 x 64
    # Correct scale shape for block (1, 32) would be (2, 2); pass (2, 1).
    wrong = torch.full((2, 1), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="block-scale shape"):
        dequantize_block_fp4_ue8m0(weight, wrong, (1, 32), torch.bfloat16)


def test_fp4_dequant_rejects_wrong_weight_dtype() -> None:
    """A bf16 weight tensor must be refused — otherwise a caller who
    forgot to view() would silently reinterpret every second bf16 byte
    as a nibble packet."""
    _, dequantize_block_fp4_ue8m0, _ = _import_library()
    bad_weight = torch.zeros((1, 32), dtype=torch.bfloat16)
    scale = torch.full((1, 2), 127, dtype=torch.uint8)
    with pytest.raises(TypeError):
        dequantize_block_fp4_ue8m0(bad_weight, scale, (1, 32), torch.bfloat16)


def _standalone_main() -> int:
    """Direct-call runner for environments where pytest collection can't
    import ``vllm_neuron`` (e.g. a laptop without the ``vllm`` package).

    Runs every test function in module order and prints a compact receipt
    per test.  Returns 0 iff every test passes.  A pytest ``skip`` becomes
    a SKIP receipt, not a failure.  Byte-exactness is still checked via
    ``assert`` — a mismatch aborts with a non-zero exit code.
    """
    tests = [
        test_fp4_e2m1_codebook_matches_hf_reference,
        test_fp4_dequant_real_hf_tensor_byte_exact,
        test_fp4_dequant_all_zero_packing,
        test_fp4_dequant_identity_scale,
        test_fp4_dequant_max_positive_no_overflow,
        test_fp4_dequant_rejects_nan_scale,
        test_fp4_dequant_rejects_scale_shape_mismatch,
        test_fp4_dequant_rejects_wrong_weight_dtype,
    ]
    n_pass = 0
    n_skip = 0
    n_fail = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except pytest.skip.Exception as skip_exc:  # noqa: F401
            n_skip += 1
            print(f"SKIP  {name}: {skip_exc}")
            continue
        except Exception as exc:  # noqa: BLE001
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
                "suite": "dsv4-flash.tests.test_fp4_dequant_1tensor",
                "pass": n_pass,
                "skip": n_skip,
                "fail": n_fail,
            },
            indent=2,
        )
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover — for local invocation
    sys.exit(_standalone_main())
