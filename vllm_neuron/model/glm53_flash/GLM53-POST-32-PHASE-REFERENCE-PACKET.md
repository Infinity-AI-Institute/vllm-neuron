# GLM-5.3 post-32/32 phase and reference packet

This packet is prepared while `/glm53-ranks1-31-2669054` is still writing.  Do
not inspect, hash, restart, or duplicate the partial directory.  Execute the
checks below only after the existing producer's own receipt says all ranks
`0..31` are atomically published.

All checks are host-only and read-only except for their receipt output.  They
reuse the retained NEFF/HLO and do not authorize a compile, card load, runtime,
correctness, performance, or tokenomics claim.

## Immutable bindings

- Checkpoint revision:
  `04c4e9e95c5da8862dced7e5056455116f83a7e0`
- Checkpoint config SHA:
  `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`
- Checkpoint index SHA:
  `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`
- TKG emitted config SHA:
  `38c9d7992e40b0c050589f2efb102daa9672c9fd1a20562cd73add16567aeba8`
- TKG NEFF/HLO:
  `b6f12210459ce13deb4b0be24d6f79d896df8e8c6425135723c6538bb4bdb41d` /
  `cf1d196cc892aae712217fb945437a2cb5979cca2ad08976c57eebda41fa2fc5`
- CTE NEFF/HLO:
  `d4885422f31a0b14e23ed12f7162f60d246baac99911740517fceb72b947826b` /
  `069b5bdae35c0239fce707f6ecbcc82e63a14445fe1fe5e8ddd4318d0c2b76e2`
- Required runtime: TP32/LNC2/B1/S128, BF16, no quantization, greedy, no
  speculation.

## Observed post-32 terminal

The existing producer completed naturally with 32/32 rank files and 32/32
manifests in
`/mnt/instance-scratch/glm53-rank-bundle-2669054.partial`.  Every rank file
was verified against its manifest at exactly `19,859,842,320` bytes and its
declared SHA-256.  The producer's directory was not renamed or modified.

The first phase-handoff attempt against the retained staged artifacts exposed
one exact metadata blocker after the staged-cache lookup was corrected in
memory: CTE `artifacts/launch-receipt.json` contains null `source_commit` and
`source_tree`, while the compose/TKG identity is
`source_commit=93c3d8773d268612bc93307eb7a68ca70b8b9b23` and
`source_tree=61d5b9ad0b42672762839ae1b4e388a115cb8aa2`.  The phase receipt is
therefore not emitted.  Repair or replace that immutable CTE launch
provenance before rerunning Gate 2; do not weaken the verifier or infer source
identity from the checkpoint.

The authorized metadata-only repair has now added those two fields to the
existing CTE launch receipt.  The repaired receipt SHA is
`f8be30850c8a15e0e388cb563ff1e8bc66394e5ef245e44ee953dd08ef4c14be`; no
compiled or rank artifact was changed.  The nested-cache verifier then passed
and emitted
`/mnt/instance-scratch/glm53-phase-handoff-9f800ee/phase-handoff-receipt.json`
with SHA
`b07694ddc28af0be8ea3dcbc80b24432da19377da25d31420c4f2ba96eee0478`.
Its receipt records shared state schema/config and both model bindings true,
phase-local loader difference true, and card/runtime/correctness claims false.

## Gate 1: completed-rank verification

Use the exact checkpoint directory already bound in the producer's immutable
launch receipt; do not guess or substitute a path.  Set only these shell
variables from that receipt and the producer's final output directory:

```bash
export GLM53_CHECKPOINT_DIR=/exact/pinned/04c4e9e95c5da8862dced7e5056455116f83a7e0
export GLM53_RANK_DIR=/mnt/instance-scratch/glm53-rank-bundle-2669054.partial
export GLM53_REQUESTED_CONFIG=/exact/emitted/requested-runtime-config.json
export GLM53_EMITTED_CONFIG=/exact/emitted/neuron_config.json
```

The command below must print a bundle with exactly 32 verified ranks and all
claims false.  Any missing/extra rank, manifest drift, source identity drift,
plan/inventory drift, wrong dtype, byte/hash mismatch, or chunk bound failure
is terminal for this packet:

```bash
python - <<'PY'
import os
import sys
import types
from pathlib import Path

source_root = Path.cwd()
package_root = source_root / "vllm_neuron"
for name, path in (
    ("vllm_neuron", package_root),
    ("vllm_neuron.model", package_root / "model"),
    ("vllm_neuron.model.glm53_flash", package_root / "model" / "glm53_flash"),
):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

from vllm_neuron.model.glm53_flash.runtime_factory import Glm53RuntimeFactory

policy = {
    "ownership_path": "/mnt/compile/OWNERSHIP.md",
    "active_compile_cap": 2,
    "systemd_unit": "glm53-compile-post-rank-20260830",
    "systemd_nice": 15,
    "systemd_scope": False,
    "network_mode": "none",
    "atomic_staging_suffix": ".partial-<run-id>",
    "compile_permitted": False,
}
bundle = Glm53RuntimeFactory.from_paths(
    checkpoint_dir=os.environ["GLM53_CHECKPOINT_DIR"],
    rank_dir=os.environ["GLM53_RANK_DIR"],
    requested_config=os.environ["GLM53_REQUESTED_CONFIG"],
    emitted_config=os.environ["GLM53_EMITTED_CONFIG"],
    compile_policy=policy,
)
receipt = bundle.to_mapping()
assert len(receipt["ranks"]) == 32
assert receipt["claims"] == {
    "rank_files_verified": True,
    "compile_permitted": False,
    "runtime_permitted": False,
    "correctness_40_of_40": False,
    "performance": False,
    "tokenomics": False,
}
print(bundle.sha256())
PY
```

## Gate 2: retained TKG/CTE phase handoff

Run the existing serialized-artifact verifier.  It must read the already
staged roots and emit a receipt with `shared_state_schema=true`,
`shared_emitted_config=true`, `model_pt_bound_for_both_phases=true`, and
`card_launch_authorized=false`:

```bash
python tools/glm53_phase_handoff.py \
  --tkg-artifact-root /mnt/instance-scratch/glm53-paired-stage-2669054/tkg \
  --cte-artifact-root /mnt/instance-scratch/glm53-paired-stage-2669054/cte \
  --compose-receipt /mnt/instance-scratch/glm53-paired-stage-2669054/cte/artifacts/tkg-cte-compose-receipt.json \
  --output /mnt/instance-scratch/glm53-phase-handoff-9f800ee/phase-handoff-receipt.json
```

The serialized loader markers must remain phase-local: TKG
`LayoutTransformation`, CTE `_parallel_load`.  The verifier must reject any
loader swap, state-key mismatch, config/hash mismatch, stale NEFF/HLO, or
non-BF16/no-quant/no-spec compose contract.

## Gate 3: original/native 4x10 target producer

The producer contract is
`Glm53OriginalTargetProducerSpec`: one exact selected target, explicit loader
versions, native block-FP8 or explicitly declared converted-BF16 semantics,
vocabulary `154880`, prompts
`feedback-0..feedback-3`, and positions `0..9`.  The injected loader must be a
real original-target implementation bound to the checkpoint above; the
injected runner must return ten full-vocabulary rows per prompt.

Expected output is 40 row files and a `reference.json` accepted by
`Glm53ReferenceTarget.from_manifest`.  Every row must pass dtype, shape,
finite-value, and SHA checks.  The manifest must include exact loader-version
identities.  A missing full-checkpoint CPU loader/runner is a capability gap;
do not substitute Q4, a generic FP32 bank, or a confidence-only reference.
The producer's successful receipt still does not authorize correctness.

The concrete provider is now
`vllm_neuron/model/glm53_flash/original_target_provider.py`.  It binds
`transformers==5.16.1`'s `Glm5NextProcessor` and exact
`Glm5NextForConditionalGeneration` class, loads only from the pinned snapshot,
and passes the processor's official chat-template IDs into an own-greedy
runner.  The runner calls the model ten times with `use_cache=False`, takes
the final-position logits, and emits FP32 comparison rows only after the
  native forward.  It does not enable MTP/speculation or replace the
  checkpoint with Q4.  Native execution is admitted only when CUDA/XPU is
  present because the upstream FP8 quantizer rejects CPU-native execution.  A
  CPU reference may explicitly pass
  `native-block-fp8-dequantized-bfloat16`; the provider then supplies
  `FineGrainedFP8Config(dequantize=True, weight_block_size=(128, 128))`.  It
  rejects undeclared conversion and original-CPU-FP32 labels.

The generic example
`examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron.py` remains only a
near match: its `AutoModelForCausalLM.from_pretrained` and
`AutoTokenizer.from_pretrained` calls do not pin this revision, record native
block-FP8 semantics, or implement the required token-bound ten-row runner.

The provider's metadata-only dry run was executed against the exact revision
with the seven non-weight files `config.json`,
`model.safetensors.index.json`, `processor_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`, and `generation_config.json`.
It returned four non-empty integer ID sequences, including the expected
15-token feedback prompts, while reporting `weights_loaded=false` and
`device_used=false`.

The bounded CPU FP8 audit under `torch==2.9.1+cpu` is not a native full-model
execution authorization: `torch._scaled_mm` accepts tiny FP8 operands and
returns FP32 when explicitly requested, while ordinary `F.linear` rejects
mixed FP32/FP8 operands.  More importantly, the real Transformers
`FineGrainedFP8HfQuantizer` changes a pre-quantized CPU load to
`dequantize=True`; the provider therefore refuses native-block-FP8 loading on
CPU rather than inheriting that silent conversion.  A CPU bank must use the
explicit converted-BF16 semantics and its declared 128x128 conversion path.

The exact fail-fast entry point is
`tools/glm53_reference_target_producer.py`.  Its new `--configure
MODULE:CALLABLE` seam is called before tokenization and receives
`(checkpoint_dir, semantics)`.  The actual metadata-only binding command is:

```bash
python tools/glm53_reference_target_producer.py \
  --checkpoint-dir /exact/pinned/04c4e9e95c5da8862dced7e5056455116f83a7e0 \
  --output-dir /not-created-on-dry-run \
  --reference-id glm53-original-native-20260830 \
  --semantics native-block-fp8 \
  --configure vllm_neuron.model.glm53_flash.original_target_provider:configure \
  --loader vllm_neuron.model.glm53_flash.original_target_provider:load \
  --runner vllm_neuron.model.glm53_flash.original_target_provider:run \
  --tokenizer vllm_neuron.model.glm53_flash.original_target_provider:tokenize \
  --loader-version transformers=5.16.1 \
  --loader-version torch=2.9.1+cpu \
  --tokenizer-version processor=Glm5NextProcessor@5.16.1 \
  --dry-run
```

The dry run validates pinned checkpoint metadata and the tokenizer's four
non-empty integer `input_ids` sequences, then emits the 4x10 contract without
loading weights.  Remove `--dry-run` only when the exact full checkpoint is
resident and capacity gates pass; the provider then loads with `dtype="auto"`
to preserve serialized precision and writes 40 rows transactionally.  The
producer materializes at most 11 runner values per prompt (the required 10
plus one overflow probe), so a generator cannot publish short or extra
coverage and cannot cause unbounded host-side materialization.

Admission is fail-closed at >=1.1 TiB available RAM and >=1.1 TiB free scratch,
with 32-64 physical CPU cores isolated from active lanes.  Prefer one SMT
thread per physical core for the memory-bound load; use sibling SMT threads
only after measured conversion utilization justifies it.  The source plus
BF16 converted working set, not the ~23.6 MiB 40-row bank, determines capacity.

## Gate 4: device handoff and correctness

Only after Gates 1–3 pass may the parent schedule the existing paired TKG/CTE
NEFFs on cards.  The resident runtime must call TKG's
`LayoutTransformation.forward(checkpoint, False)` and CTE's
`torch.ops.neuron._parallel_load(checkpoint)`, then use the tested
`_copy_past_key_values` CTE-to-TKG handoff.  Capture full 154880-wide raw rows
for every planned slot, prompt, and position, bind exactly one canonical
reference before scoring, and require strict full-vocabulary 40/40.

No throughput or tokenomics result is admissible before correctness.  Every
later throughput run must retain both unprofiled timing and a matched Neuron
Explorer trace for the same artifact, topology, workload, prompts, sampling,
and token counts; trace overhead is reported separately.
