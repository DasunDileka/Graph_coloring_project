"""Undirected simple graph that supports edge insertions and deletions.

The graph is stored as a list of adjacency *sets* (one hash set per vertex).
This gives expected O(1) edge lookup, insertion and deletion, O(deg(v))
neighbourhood iteration and O(n + m) memory, which is what the dynamic
colouring algorithms need (see Chapter 6 of the report).
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Set, Tuple

Edge = Tuple[int, int]


class GraphError(ValueError):
    """Raised for invalid graph operations (self-loops, bad vertex ids)."""


class DynamicGraph:
    """A simple undirected graph G = (V, E) with V = {0, ..., n-1}.

    The vertex set is fixed when the object is created; the edge set can
    change over time through :meth:`add_edge` and :meth:`remove_edge`.
    """

    __slots__ = ("n", "adj", "m")

    def __init__(self, n: int, edges: Iterable[Edge] = ()) -> None:
        if not isinstance(n, int) or n < 0:
            raise GraphError(f"number of vertices must be a non-negative int, got {n!r}")
        self.n: int = n
        self.adj: List[Set[int]] = [set() for _ in range(n)]
        self.m: int = 0
        for u, v in edges:
            self.add_edge(u, v)

    # ------------------------------------------------------------------ checks
    def _check_vertex(self, v: int) -> None:
        if not (0 <= v < self.n):
            raise GraphError(f"vertex {v} is outside the range 0..{self.n - 1}")

    # --------------------------------------------------------------- updates
    def add_edge(self, u: int, v: int) -> bool:
        """Insert edge {u, v}. Returns False if it was already present."""
        self._check_vertex(u)
        self._check_vertex(v)
        if u == v:
            raise GraphError(f"self-loop {{{u}, {v}}} is not allowed in a simple graph")
        if v in self.adj[u]:
            return False
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.m += 1
        return True

    def remove_edge(self, u: int, v: int) -> bool:
        """Delete edge {u, v}. Returns False if it was not present."""
        self._check_vertex(u)
        self._check_vertex(v)
        if v not in self.adj[u]:
            return False
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        self.m -= 1
        return True

    # --------------------------------------------------------------- queries
    def has_edge(self, u: int, v: int) -> bool:
        return 0 <= u < self.n and v in self.adj[u]

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def max_degree(self) -> int:
        return max((len(a) for a in self.adj), default=0)

    def edges(self) -> Iterator[Edge]:
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:
                    yield (u, v)

    def copy(self) -> "DynamicGraph":
        g = DynamicGraph(self.n)
        g.adj = [set(a) for a in self.adj]
        g.m = self.m
        return g

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"DynamicGraph(n={self.n}, m={self.m})"
