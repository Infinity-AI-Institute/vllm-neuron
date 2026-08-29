# GLM-5.3 emitted runtime-config contract

This host-only contract closes one blocker from trainium-autoresearcher PR #95:
requested compile/runtime fields can now be serialized canonically, reloaded
with duplicate-key rejection, and compared against the configuration actually
emitted by a future runtime adapter. No unknown field has a default, and
missing, extra, silently dropped, or changed values fail closed.

The contract pins the checkpoint architecture/revision and TP32 rank-plan
boundary. It also requires explicit source commit/tree, compiler image ID and
registry digest, compiler version and flags, package identities, LNC, batch,
sequence/bucket profile, weight/cache dtypes, runtime quantization, greedy
sampling, and no speculative decode.

This change does **not** provide a runtime model, factory, or registry entry.
It does not select any of the currently unknown compile-profile values.
Therefore compilation and runtime remain unauthorized until a later reviewed
adapter and authorization packet bind real emitted values through this API.
