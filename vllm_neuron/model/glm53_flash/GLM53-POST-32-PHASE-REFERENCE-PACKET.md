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

## Gate 1: completed-rank verification

Use the exact checkpoint directory already bound in the producer's immutable
launch receipt; do not guess or substitute a path.  Set only these shell
variables from that receipt and the producer's final output directory:

```bash
export GLM53_CHECKPOINT_DIR=/exact/pinned/04c4e9e95c5da8862dced7e5056455116f83a7e0
export GLM53_RANK_DIR=/mnt/instance-scratch/glm53-rank-bundle-2669054
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
