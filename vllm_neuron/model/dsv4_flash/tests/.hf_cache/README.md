# DSv4-Flash HF tensor test fixtures

This directory holds slices of the DeepSeek-V4-Flash HF checkpoint used
by the local unit tests, so the tests run byte-exact offline against
real weights without pulling the full 166 GB checkpoint.

## `dsv4_expert0_w2.safetensors` (4.4 MB)

A two-tensor mini-safetensors file carved out of
`model-00005-of-00048.safetensors` from repo
`deepseek-ai/DeepSeek-V4-Flash-0731` @ HF SHA
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` (MIT-licensed).

Contents:

| tensor      | source HF key                              | dtype              | shape        |
|-------------|--------------------------------------------|--------------------|--------------|
| `w2_weight` | `layers.3.ffn.experts.0.w2.weight`         | `int8`             | `[4096, 1024]` |
| `w2_scale`  | `layers.3.ffn.experts.0.w2.scale`          | `float8_e8m0fnu`   | `[4096, 64]`   |

Fetched via HTTP-Range slicing from the source safetensors — the full
3.6 GB shard is never downloaded.  See
`tests/test_fp4_dequant_1tensor.py::_build_local_cache` for the fetcher;
delete this file to force a re-fetch on next test run.

Consumed by `test_fp4_dequant_1tensor.py`
(`test_fp4_dequant_real_hf_tensor_byte_exact`).

## `dsv4_layer0_attn.safetensors` (~107 MB) — GITIGNORED

The full layer-0 attention subtree (all 13 tensors under
`layers.0.attn.*` plus the sibling `layers.0.attn_norm.weight`) needed
by `tests/test_mqa_1tensor.py` for the `_MQABlock` byte-clean gate.

Contents (from `model-00002-of-00048.safetensors`, same repo/SHA):

| tensor                     | dtype            | shape          |
|----------------------------|------------------|----------------|
| `layers.0.attn.attn_sink`  | `bf16`           | `[64]`         |
| `layers.0.attn.wq_a.weight`| `float8_e4m3fn`  | `[1024, 4096]` |
| `layers.0.attn.wq_a.scale` | `float8_e8m0fnu` | `[8, 32]`      |
| `layers.0.attn.wq_b.weight`| `float8_e4m3fn`  | `[32768, 1024]`|
| `layers.0.attn.wq_b.scale` | `float8_e8m0fnu` | `[256, 8]`     |
| `layers.0.attn.q_norm.weight` | `bf16`        | `[1024]`       |
| `layers.0.attn.wkv.weight` | `float8_e4m3fn`  | `[512, 4096]`  |
| `layers.0.attn.wkv.scale`  | `float8_e8m0fnu` | `[4, 32]`      |
| `layers.0.attn.kv_norm.weight`| `bf16`        | `[512]`        |
| `layers.0.attn.wo_a.weight`| `float8_e4m3fn`  | `[8192, 4096]` |
| `layers.0.attn.wo_a.scale` | `float8_e8m0fnu` | `[64, 32]`     |
| `layers.0.attn.wo_b.weight`| `float8_e4m3fn`  | `[4096, 8192]` |
| `layers.0.attn.wo_b.scale` | `float8_e8m0fnu` | `[32, 64]`     |
| `layers.0.attn_norm.weight`| `bf16`           | `[4096]`       |

At 107 MB this fixture is too large for the git repo (per `.gitignore`
in this directory), but small enough that a first-run HTTP-Range fetch
completes in ~30 s.  Delete the file to force a re-pull.  When missing
on an offline dev box, `test_mqa_1tensor.py::
test_mqa_wrapper_matches_hf_reference_on_real_layer0_tensors` skips
rather than fails — the synthetic gate `test_mqa_wrapper_forward_
synthetic_shape_gate` still runs, and the standalone RoPE helper gate
`test_partial_rope_helper_matches_reference_on_synthetic_input` still
runs.
