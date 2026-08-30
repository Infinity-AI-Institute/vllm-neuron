# GLM-5.3-Flash readiness roadmap

This is a host-readiness record only.  It does not authorize a compile, card
load, correctness claim, performance claim, or tokenomics calculation.

## Current retained evidence

- Source PR: #28, head `9f800ee5453e0f1620b06ffb7f241596cd376690`, tree
  `2b764184b16c7d34bb331180332683a079fe8c2c`.
- Runtime contract: TP32/LNC2/B1/S128, BF16, no runtime quantization, no
  speculative decoding.
- TKG config SHA:
  `38c9d7992e40b0c050589f2efb102daa9672c9fd1a20562cd73add16567aeba8`;
  NEFF SHA `b6f12210459ce13deb4b0be24d6f79d896df8e8c6425135723c6538bb4bdb41d`;
  HLO SHA `cf1d196cc892aae712217fb945437a2cb5979cca2ad08976c57eebda41fa2fc5`.
- CTE uses the same config SHA; NEFF SHA
  `d4885422f31a0b14e23ed12f7162f60d246baac99911740517fceb72b947826b`;
  HLO SHA `069b5bdae35c0239fce707f6ecbcc82e63a14445fe1fe5e8ddd4318d0c2b76e2`.
- Rank producer `/glm53-ranks1-31-2669054` is active on Trn CPUs 96-111 with
  a 128 GiB limit.  The last read-only snapshot had 15/32 rank files and
  `317761017760` bytes in the partial directory; do not duplicate or restart.

## Source/runtime gates

`phase_runtime.py` requires TKG to initialize through
`LayoutTransformation.forward(checkpoint, False)` and CTE through
`torch.ops.neuron._parallel_load(checkpoint)`.  Their KV state key schemas,
count, shape, and dtype must agree.  CTE output must be logits plus state;
bare logits fail closed.  The two phase weight layouts remain phase-local;
weights are not copied between phases.  The paired handoff uses the existing
wrapper `_copy_past_key_values` hook and keeps runtime permission false.

After all ranks finish, run `tools/glm53_phase_handoff.py` against the staged
`tkg` and `cte` roots.  Then load the resident phase models, perform the tested
CTE-to-TKG state handoff, and capture every planned slot/prompt/position as a
full 154880-wide raw row.

## Reference and native correctness gates

`reference_target.py` accepts only a manifest bound to checkpoint revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`, the pinned config/index hashes,
full vocabulary, explicit semantics, and per-row hashes.  It reads one row at
a time and never manufactures a bank.  No native/original canonical bank is
currently bound, so strict correctness remains unproven.  Q4 is diagnostic
only.  The native checkpoint is dynamic block-FP8 converted to BF16; FP32
storage is not evidence of FP32 execution.

## Throughput and tokenomics sequence

Only after one canonical target is selected and strict full-vocabulary 40/40
correctness passes may the 8-card candidate be loaded.  TP32 across eight
cards is four ranks/card, about 73.98 GiB payload/card.  Reserve 16 host CPUs,
128 GiB RAM, and at least 1 TiB free scratch.

Every throughput run must retain both unprofiled timing and a matched Neuron
Explorer trace for the identical artifact, topology, workload, prompts,
sampling, and token counts.  Trace overhead is reported separately.  Only
then compute measured tok/s/card, output tok/s, and cost per million tokens.
