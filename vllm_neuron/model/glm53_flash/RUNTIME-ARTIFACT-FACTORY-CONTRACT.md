# GLM-5.3-Flash runtime-artifact factory contract

This host-only factory joins the exact checkpoint identity, all 32 transactional
TP-rank artifacts, and the requested-versus-emitted runtime configuration. It
does not instantiate or register a model and cannot authorize compilation or
runtime.

The factory fails closed unless every rank file and manifest exists, hashes and
sizes match, the source and TP32/LNC2 rank-plan identities are exact, observed writer
chunks are no larger than 64 MiB, and requested and emitted configuration bytes
are canonical and equal. It also requires an explicit future launch policy:
`/mnt/compile/OWNERSHIP.md`, cap 2, a named `systemd-run --unit` operation,
nice 15, no `--scope`, network disabled, and atomic `.partial-<run-id>` staging.
The policy must keep `compile_permitted=false`.

This closes only the missing artifact-factory seam after the emitted-config
roundtrip. The exact compiler image ID/digest, package versions, compiler flags,
actual emitted configuration, hardware correctness, performance, and tokenomics
remain unclaimed until independently observed and validated.
