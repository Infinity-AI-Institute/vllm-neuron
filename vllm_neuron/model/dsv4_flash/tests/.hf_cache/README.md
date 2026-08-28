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
