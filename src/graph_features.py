"""Structural features used to describe benchmark graphs and dynamic workloads."""

from __future__ import annotations

from typing import Dict, Sequence, Set

import numpy as np

from static_coloring import degeneracy, greedy_clique_lower_bound


def connected_components(adj: Sequence[Set[int]]) -> int:
    n = len(adj)
    seen = [False] * n
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
    return comps


def graph_features(adj: Sequence[Set[int]], clique_starts: int = 64) -> Dict[str, float]:
    n = len(adj)
    deg = np.array([len(a) for a in adj], dtype=float)
    m = int(deg.sum() // 2)
    density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    return {
        "n": n,
        "m": m,
        "density": density,
        "min_degree": float(deg.min()) if n else 0.0,
        "mean_degree": float(deg.mean()) if n else 0.0,
        "max_degree": float(deg.max()) if n else 0.0,
        "degree_cv": float(deg.std() / deg.mean()) if n and deg.mean() > 0 else 0.0,
        "degeneracy": degeneracy(adj),
        "clique_lb": greedy_clique_lower_bound(adj, clique_starts),
        "components": connected_components(adj),
        "isolated": int((deg == 0).sum()),
    }
