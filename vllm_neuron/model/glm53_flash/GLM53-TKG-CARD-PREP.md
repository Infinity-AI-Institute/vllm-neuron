# GLM-5.3 TKG card-preparation boundary

`RETAINED-TKG-ARTIFACT-PACKET.json` binds the retained compiler evidence: the
effective-shape receipt, emitted `neuron_config.json`, serialized `model.pt`,
compile result, HLO, NEFF, source/tree identities, and pinned
NxDI/checkpoint identities.  The HLO, NEFF, and model archive are evidence
only; this change does not recompile or authorize a card launch.

The artifact is TKG-only: it contains exactly one
`token_generation_model`, no `context_encoding_model`, and no CTE bucket.  A
fresh prompt therefore remains fail-closed until a separately compiled CTE
artifact is supplied.  The retained TKG NEFF is reusable unchanged for the
token-generation side of that future paired runtime.

Run the host-only preparation probe inside the pinned image with the HF cache
mounted as a whole (the snapshot contains relative blob symlinks):

```bash
docker run --rm --entrypoint python --network none --cpuset-cpus=32-47 \
  -v /mnt/compile/runroot/glm53-pr22-tkg-tp32-b1-s128-20260830T004316Z:/artifact:ro \
  -v /mnt/compile/hf-cache/models--zai-org--GLM-5.3-Flash:/hf:ro \
  -v /path/to/vllm-neuron:/src:ro \
  -w /src \
  public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1 \
  tools/glm53_tkg_card_prep.py \
  --artifact-root /artifact \
  --checkpoint-dir /hf/snapshots/04c4e9e95c5da8862dced7e5056455116f83a7e0
```

The probe audits SafeTensors headers only (`payload_bytes_loaded_during_audit`
must remain zero) and computes the existing TP32 streaming transform's exact
per-rank payload.  The required later host transform is one transactional
`stream_glm53_rank_checkpoint` output plus manifest for each rank `0..31`; it
must preserve BF16 target weights, the 64 MiB chunk bound, and the pinned
checkpoint/index identities.  The probe does not perform that multi-hundred-GB
write.

The retained-artifact gate is enabled with
`GLM53_RETAINED_ARTIFACT_ROOT=/artifact`.  It verifies the actual HLO/NEFF
bytes and compiler result, rejects `%sort.` and `aten__topk`, and checks the
TP32/LNC2/B1/S128/BF16/no-quant/no-spec emitted contract.  Without that
environment variable it skips rather than claiming compiler evidence.

When the retained packet and compiled files are supplied from an immutable
external artifact mount, pass the packet explicitly; the packet's relative
paths are resolved against `--artifact-root` and every referenced byte is
rehashed:

```bash
tools/glm53_tkg_card_prep.py \
  --artifact-root /artifact \
  --retained-packet /evidence/RETAINED-TKG-ARTIFACT-PACKET.json \
  --checkpoint-dir /hf/snapshots/04c4e9e95c5da8862dced7e5056455116f83a7e0
```

Omitting `--retained-packet` preserves the checked-in packet default.  The
option only selects the evidence manifest; it never authorizes compile,
runtime, card load, correctness, or performance claims.

Before any paired runtime load, run the separate host-only phase handoff
probe.  It reads the serialized `model.pt` archives without importing torch:

```bash
python tools/glm53_phase_handoff.py \
  --tkg-artifact-root /mnt/compile/runroot/glm53-pr22-tkg-tp32-b1-s128-20260830T004316Z \
  --cte-artifact-root /mnt/compile/runroot/glm53-pr22-cte-tp32-b1-s128-20260830T0140Z \
  --compose-receipt /mnt/compile/runroot/glm53-pr22-cte-tp32-b1-s128-20260830T0140Z/artifacts/tkg-cte-compose-receipt.json \
  --output /mnt/compile/runroot/glm53-pr22-phase-handoff/phase-handoff-receipt.json
```

The probe requires TKG's serialized `LayoutTransformation` and CTE's
serialized `torch.ops.neuron._parallel_load` to remain phase-local, binds the
`model.pt` bytes for both phases, and requires identical serialized
`past_key_values.*` state-key sets, emitted-config hash, source identity, and
compose-receipt hashes.  CTE may omit `artifacts/source-identity.json` only
when the immutable compose/launch receipt supplies the same identity.  Its
receipt is evidence only and keeps runtime/card/correctness claims false.

Readiness remains false unless the optional inputs are independently bound:
the CTE companion must carry the exact TP32/LNC2/B1/S128 context-encoding
contract, matching source/config/checkpoint identities, and its own BF16,
no-sort compiler evidence; the rank directory must contain all 32 non-empty
transactional outputs whose manifests match the pinned checkpoint, rank
inventory/plan hashes, resource bound, tensor dtypes, and exact byte totals.
Shape-only CTE metadata and empty or placeholder rank files never publish
fresh-prompt or continuation readiness.  Even a passing receipt keeps
`card_launch_authorized=false`; hardware correctness, performance, and
tokenomics require separate evidence.
