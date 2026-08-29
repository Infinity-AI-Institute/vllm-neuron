# GLM-5.3-Flash converter preflight receipt

This patch isolates the host-only conversion contract for
`zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`.

The existing `glm52_moe_dsa/checkpoint_converter.py` cannot be reused:

- GLM-5.2 is a text-only `Glm52MoeDsa` path; GLM-5.3 is the conditional,
  multimodal `Glm5NextForConditionalGeneration` architecture with a nested
  `glm5_next_text` backbone.
- GLM-5.3 has 45 inference text layers (34 KDA and 11 DSA), plus a separate
  layer-45 MTP subtree and a visual tower. MTP and vision are intentionally
  excluded from this no-spec-decode text lane.
- GLM-5.3 stores native FP8 weights with reciprocal
  `weight_scale_inv` tensors on 128x128 blocks. The conversion operation is
  `weight * scale_inv`; the older per-tensor assumptions are invalid.
- KDA short-convolution weights must be interleaved per head as Q/K/V. A
  stream-major `cat(q, k, v)` loads and compiles but feeds the wrong channels
  to `view(..., heads, 3 * head_dim)`.
- `hc_attn_scale` and `hc_ffn_scale` are BF16 mHC parameters, not FP8 scales.

Immutable metadata evidence:

- `config.json` SHA-256:
  `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`
- `model.safetensors.index.json` SHA-256:
  `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`
- 76,108 indexed tensors; 37,338 reciprocal scales; 347 visual tensors;
  1,760 MTP tensors; 84 DSA-indexer tensors.

The smallest falsifiable next unblock is a streaming per-rank writer that
consumes only checkpoints which pass `preflight_checkpoint_metadata`, uses
the two tested transforms here, and proves one complete rank manifest without
materializing the approximately 611 GiB BF16 model. This patch does not claim
that writer, model enablement, correctness, performance, or tokenomics.

For a real checkpoint directory, callers must use `preflight_checkpoint_dir`.
It resolves the HF snapshot path, requires the directory name to equal the
full pinned revision, and verifies both immutable metadata SHA-256 values
before parsing the schema. The in-memory metadata helper exists for tests and
does not by itself establish checkpoint provenance.
