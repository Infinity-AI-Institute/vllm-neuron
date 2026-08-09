"""Fail-closed policy for opt-in rank-invariant graph extraction."""


def should_extract_graphs(*, rank: int, leader_only: bool) -> bool:
    """Return whether this distributed worker must run graph extraction.

    ``leader_only`` is deliberately an explicit opt-in. It is safe only after
    the exact model/shape stack has produced a semantic rank-HLO census showing
    that every rank maps to the same graph hashes and normalized HLO digests.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a non-negative integer, got {rank!r}")
    return not leader_only or rank == 0
