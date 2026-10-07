"""Validity checks used by the tests and by every experiment checkpoint."""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple


def conflicting_edges(adj: Sequence[Set[int]], col: Sequence[int]) -> List[Tuple[int, int]]:
    """All edges {u, v} with col[u] == col[v] (empty for a proper colouring)."""
    return [(u, v) for u in range(len(adj)) for v in adj[u] if u < v and col[u] == col[v]]


def is_proper(adj: Sequence[Set[int]], col: Sequence[int]) -> bool:
    if len(col) != len(adj):
        return False
    for u in range(len(adj)):
        cu = col[u]
        if cu < 0:
            return False
        for v in adj[u]:
            if col[v] == cu:
                return False
    return True


def number_of_colours(col: Sequence[int]) -> int:
    return len(set(col))


def recoloured(before: Sequence[int], after: Sequence[int]) -> int:
    """Hamming distance between two colourings (vertices whose colour changed)."""
    return sum(1 for a, b in zip(before, after) if a != b)


def assert_proper(adj: Sequence[Set[int]], col: Sequence[int], context: str = "") -> None:
    bad = conflicting_edges(adj, col)
    if bad:
        raise AssertionError(f"improper colouring {context}: {len(bad)} conflicting edges, e.g. {bad[:3]}")


def assert_budget(before: Sequence[int], after: Sequence[int], budget: int, context: str = "") -> None:
    r = recoloured(before, after)
    if r > budget:
        raise AssertionError(f"budget violated {context}: {r} recoloured vertices > R = {budget}")
