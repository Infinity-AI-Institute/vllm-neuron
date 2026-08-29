# exec()-to-file-import patch — Fleet A NKI kernel dispatch shims

**Date:** 2026-08-27 (patch applied), doc-slug 2026-08-28 per lane convention
**Callsign:** exec-to-file-import-patch
**Trigger:** Kimi K3 Route B fire surfaced the universal blocker across all
three Fleet A NKI source-string kernels.

Absolute local paths (per `[[always-give-full-local-paths]]`):
- Patch doc: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md`
- Body directory: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\_kernel_bodies\`

---

## 1. Bug diagnosis

The Fleet A NKI kernel dispatch shims followed this pattern:

```python
_NKI_KERNEL_SOURCE = """
@nki.jit
def kernel_body(...):
    ...
"""
_ns = {}
exec(_NKI_KERNEL_SOURCE, _ns)
_kernel = _ns['kernel_body']
```

When `_kernel(...)` runs, `@nki.jit`'s `KernelRewriter.reparse_function`
(in `neuronxcc/nki/compile/kernel_rewriter.py:410`) calls
`inspect.getsource(kernel_body)` to re-parse the AST. But `exec()`'d
functions don't have a physical file to walk — `inspect.getsource` raises

```
OSError: could not get source code
```

Every NKI compile then fails at this OSError. The bug is universal across
every kernel that uses the source-string + `exec()` dispatch pattern.

Root cause: `exec(src, ns)` assigns the function's `__module__` to
`"<string>"` and leaves no `__file__`; Python's `inspect.getsourcefile`
returns None and `inspect.getsource` raises. NKI's compile pass needs the
actual source text of the decorated function (re-parsed and rewritten)
and cannot recover it from bytecode alone.

Surfaced-by / receipt: Kimi K3 Route B fire, 2026-08-27.

---

## 2. Fix pattern (file-based import)

Replace `exec(source_str, ns)` with `importlib.util.spec_from_file_location`
+ `spec.loader.exec_module`. The body file is a physical Python file the
`spec.loader` reads through the file system, so the decorated function gets
a real `__file__` and `inspect.getsource` can walk it.

```python
from pathlib import Path
import importlib.util

_KERNEL_MODULE_SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "_kernel_bodies"
    / "<name>_body.py"
)

def _load_nki_kernel_module():
    spec = importlib.util.spec_from_file_location(
        "<name>_body",
        _KERNEL_MODULE_SOURCE_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_KERNEL_MODULE = _load_nki_kernel_module()
_kernel = _KERNEL_MODULE.kernel_body
```

`inspect.getsource(_kernel)` now walks the physical body file and NKI's
compile pass succeeds.

---

## 3. Kernels patched

### 3.1 DSA Lightning Indexer NKI v1
- Shim: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer_nki_v1.py`
- Body: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\_kernel_bodies\dsa_lightning_indexer_nki_v1_body.py`
- Slug preserved: `dsa_sparse_attention.nki_v1`
- Constant preserved: `LSE_BASE_CONVENTION = "natural"`
- Cache key preserved: `build_v1_cache_key(...)` unchanged
- Dispatch: `_compile_nki_kernel_if_available()` now file-imports the body
  module via `importlib.util.spec_from_file_location` instead of
  `exec(_NKI_KERNEL_SOURCE, exec_globals)`. The source-string constant
  `_NKI_KERNEL_SOURCE` is still exposed for consumers that grep the source
  (read from the body file via `pathlib.Path.read_text`).
- `_NKI_KERNEL_NAMESPACE` cache retained (idempotent load).

### 3.2 KDA state NKI v2
- Shim: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\kda_state_nki_v2.py`
- Body: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\_kernel_bodies\kda_state_nki_v2_body.py`
- Slug preserved: `kda_state.decode.kda_gate.rank1_delta.bf16_state.nki_v1`
- Four FLA-parity fixes preserved (all present in body file):
  - (a) KDA per-channel gate: `Diag(alpha_h)` applied before delta step
  - (b) In-kernel L2-norm on Q and K (`eps=1e-6`)
  - (c) Query scale `q *= 1/sqrt(D_qk)` applied AFTER L2-norm
  - (d) bf16 state layout `[num_slots, HV, V, K]` + bf16 store on state + y
- The KDA v2 shim did NOT previously `exec()` the source (the source was
  passed as a text string to the compile driver). This patch still lands
  the file-based body so any downstream call that would `exec` the source
  can be migrated to file-import, and adds two public helpers used by the
  compile driver: `load_nki_kernel_module()` and
  `get_nki_kernel_source_path()`.
- `_kda_state_nki_v2_source(lower_bound, l2norm_eps, tiling)` retained for
  the non-default constant path; whenever that path is next exercised, the
  compile driver should materialize a per-constant-set body file rather
  than fall back to `exec()`.

### 3.3 DMA coalescing NKI v1
- Shim: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dma_coalescing_nki_v1.py`
- Body: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\_kernel_bodies\dma_coalescing_nki_v1_body.py`
- Slug preserved: `dma_coalesced_gather.nki_v1`
- Callsign preserved: `dma-coalescing-nki-v1-agent`
- First-fire lane manifest preserved: `FIRST_FIRE_LANE` dict unchanged
- Dispatch: the module-level `exec(DMA_COALESCED_GATHER_NKI_V1_SOURCE, _NS)`
  branch is replaced with an `importlib.util.spec_from_file_location`
  file-import when `_NKI_AVAILABLE`. The stub fallback for non-Trn2 hosts
  is unchanged (still raises `NotImplementedError` with the same message
  pointing callers to v0's Path B/C planners).
- `DMA_COALESCED_GATHER_NKI_V1_SOURCE` still exposed as the body file's
  TEXT (read via `pathlib.Path.read_text`) for source-hygiene tests.
- Two new diagnostics: `_KERNEL_BODY_MODULE` (the imported module or None)
  and `_KERNEL_LOAD_ERROR` (repr of any load exception).

---

## 4. Test suite results

- Test root: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\`
- Baseline (pre-patch): **300 passed, 7 skipped** in 80.90s
- Post-patch: **300 passed, 7 skipped** in 70.89s

No regressions. The 7 skips are the NKI-runtime-gated tests that only run
on Trn2 (unchanged skip conditions: `neuronxcc` unimportable, or torch
unavailable in a torch-gated fixture).

Every source-hygiene test that greps `_NKI_KERNEL_SOURCE` /
`KDA_STATE_NKI_V2_SOURCE` / `DMA_COALESCED_GATHER_NKI_V1_SOURCE` still
passes because those constants are now the body-file text — same
substrings, same content, different origin.

Command:
```
py -3 -m pytest -q \
  C:/Users/apumu/research/InfinityAI/gemma4-trn2-handoff/harness-v2/staging/reference-sweep-20260826T2150Z/kernels/tests/
```

---

## 5. Container dry-import check (compile host)

- Compile host: `ec2-user@13.222.20.119` (SSH key never printed)
- Container: `hopeful_hofstadter` (image
  `public.ecr.aws/neuron/pytorch-inference-neuronx`)
- Status at check-in: `docker ps` responsive; `docker exec` / `docker inspect`
  sluggish (both timed out at 45-120s during this patch tick).
- Action: **deferred per task instruction** ("If docker daemon is stuck /
  unresponsive / no container running, DEFER dry-import to next tick;
  don't force docker recovery"). No rsync-and-fire attempted this tick.

Next tick fire plan (record for the follow-up agent):
```
rsync -avz -e "ssh -i ~/.ssh/apuroop-trial-key.pem" \
  C:/Users/apumu/research/InfinityAI/gemma4-trn2-handoff/harness-v2/staging/reference-sweep-20260826T2150Z/kernels/ \
  ec2-user@13.222.20.119:/tmp/fleet-a-kernels/

ssh -i ~/.ssh/apuroop-trial-key.pem ec2-user@13.222.20.119 \
  'docker cp /tmp/fleet-a-kernels hopeful_hofstadter:/tmp/fleet-a-kernels && \
   docker exec hopeful_hofstadter python3 -c "
import sys; sys.path.insert(0, \"/tmp/fleet-a-kernels\")
import kda_state_nki_v2 as m
mod = m.load_nki_kernel_module()
print(\"KDA body module:\", mod is not None)
print(\"body path:\", m.get_nki_kernel_source_path())
if mod is not None:
    import inspect
    src = inspect.getsource(mod.kda_state_decode_forward_nki_v2_body)
    print(\"inspect.getsource OK, len=\", len(src))
"'
```

The KEY assertion: `inspect.getsource(<decorated_fn>)` returns non-empty
text on the file-imported path. On the pre-patch `exec()` path this
call raised `OSError: could not get source code`. The same check applies
to the DSA and DMA body modules.

---

## 6. Recommendation for future NKI source-string kernels

Any FUTURE Fleet A NKI kernel authored under a source-string dispatch
pattern MUST use file-based import from day 1:

1. Author the `@nki.jit`-decorated function in a physical file under
   `_kernel_bodies/<name>_body.py`. Do NOT keep the body only as a string
   inside the shim.
2. In the sibling shim, expose the source text (if callers grep it) via
   `pathlib.Path(<body>).read_text()`, never via a hand-copied constant
   that could drift.
3. Dispatch the compiled callable via `importlib.util.spec_from_file_location`
   + `spec.loader.exec_module` and pull the attribute off the returned
   module. Do NOT use `exec(source, ns)`.
4. If constants are baked in at source-generation time (KDA v2 pattern),
   the generator still returns a string for text callers, but the
   file-import path materializes a per-constant-set body file the first
   time a non-default constant set is compiled. `exec()` is never the
   fallback.
5. Cache the imported module so `inspect.getsource` walks the same file
   every time the compiled callable is re-parsed.

Failure mode if this rule is violated: every `@nki.jit` compile raises
`OSError: could not get source code` inside `KernelRewriter.reparse_function`,
which is silently swallowed by the campaign's fallback discipline and
surfaces only as "NKI backend cold" — a false negative that costs a full
Route-B fire before diagnosis.
