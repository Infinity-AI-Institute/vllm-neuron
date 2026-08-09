# NKI compile-cache identity

The NKI cache is an optimization, never an authority for kernel equivalence.
Key generation therefore fails closed: if a compile-time value cannot be
serialized deterministically, `create_nki_cache_key()` returns `None` and the
kernel compiles without either persistent or process-local reuse.

## Schema 2 identity

Schema 2 keys bind all of the following:

- the kernel module, qualified name, normalized Python bytecode, constants,
  defaults, keyword defaults, function attributes, closure values, referenced
  globals, and recursively referenced Python helper functions;
- a SHA-256 seal of the kernel's defining source file when one exists;
- each argument name and order;
- tensor shape, dtype, stride, layout, storage offset, and device role (including
  a FakeTensor's target device when exposed as `fake_device`);
- deterministic scalar/container/dataclass compile-time arguments;
- grid/LNC, target platform, Python, NKI, neuronx-cc, and torch-neuronx versions;
- device-dump mode; and
- optional source-overlay/revision identity from
  `VLLM_NEURON_NKI_SOURCE_IDENTITY`.

Set `VLLM_NEURON_NKI_SOURCE_IDENTITY` to an immutable source revision or overlay
manifest digest in source-overlay deployments. It supplements rather than
replaces the semantic code and source-file digests.

The key uses SHA-256 truncated to the existing 32-character filename width.
Cache records also carry schema version 2. Schema 1 records are deliberately
rejected because they bind only top-level source, tensor shape/dtype, and a
subset of the runtime identity; equivalence cannot be proven safely.

## Process and persistent behavior

The process cache is namespaced by normalized local and remote cache roots plus
the complete semantic key. Every hit revalidates a materialized NKI binary.
Deleting that binary turns the next lookup into a miss.

After POSIX `fork`, an at-fork handler gives the child fresh process-cache and
source-digest locks, then drops inherited process entries and counters. The
immutable, stat-keyed source digests remain reusable. Replacing the locks is
necessary because another parent thread may have held an inherited copy at
fork; a PID check inside that lock would deadlock. The child may repopulate from
the locked persistent cache, but its updates remain child-local and cannot
masquerade as parent state. Cache-mode transitions and explicit clears likewise
advance the process generation so an in-flight lookup cannot resurrect stale
entries.

## Qualification boundary

These checks establish deterministic cache invalidation, not kernel numerical
correctness or NEFF portability across rank classes. Cache reuse still requires
the caller's model, shape, topology, compiler flags, source overlay, and SDK
contract to be compatible. Unknown custom objects, recursive containers,
incomplete tensor metadata, cyclic wrapped callables, uninspectable Python
callables, or classes without inspectable source disable caching for that
invocation.
