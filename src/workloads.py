"""Update sequences (workloads) for the dynamic colouring experiments.

A workload is an initial edge list for G_0 plus a list of single-edge
operations ``(op, u, v)`` with op in {"insert", "delete"}.

* :func:`random_churn` -- random edge deletions and re-insertions on a fixed
  benchmark graph.  A fraction ``rho`` of the benchmark edges starts in a reserve pool;
  each update deletes a random present edge or re-inserts a random pool edge
  with equal probability, so the graph stays close to its original structure.
* :func:`sliding_window` -- converts a real time-stamped interaction stream
  into a dynamic graph: edge {u, v} is present at time t if u and v interacted
  during (t - W, t].  A new interaction of an absent pair is an insertion; the
  expiry of the last interaction of a pair is a deletion (Holme and
  Saramaki, 2012, discuss such time-window aggregations).
"""

from __future__ import annotations

import heapq
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from proposed_algorithm import DELETE, INSERT

Edge = Tuple[int, int]
Op = Tuple[str, int, int]


class _IndexedSet:
    """List-backed set with O(1) add, remove and uniform random choice."""

    def __init__(self, items: Iterable[Edge] = ()) -> None:
        self.items: List[Edge] = []
        self.pos: Dict[Edge, int] = {}
        for e in items:
            self.add(e)

    def add(self, e: Edge) -> None:
        if e in self.pos:
            return
        self.pos[e] = len(self.items)
        self.items.append(e)

    def remove(self, e: Edge) -> None:
        i = self.pos.pop(e)
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def choice(self, rng: random.Random) -> Edge:
        return self.items[rng.randrange(len(self.items))]

    def __len__(self) -> int:
        return len(self.items)


def random_churn(edges: Sequence[Edge], rho: float = 0.2, updates: int = 5000,
                 seed: int = 0, p_insert: float = 0.5) -> Tuple[List[Edge], List[Op]]:
    """Random deletion/re-insertion workload on a fixed benchmark graph."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    rng = random.Random(seed)
    canon = sorted({(min(u, v), max(u, v)) for u, v in edges if u != v})
    rng.shuffle(canon)
    k = int(round(rho * len(canon)))
    pool = _IndexedSet(canon[:k])
    present = _IndexedSet(canon[k:])
    initial = list(present.items)
    ops: List[Op] = []
    for _ in range(updates):
        do_insert = (rng.random() < p_insert and len(pool) > 0) or len(present) == 0
        if do_insert:
            e = pool.choice(rng)
            pool.remove(e)
            present.add(e)
            ops.append((INSERT, e[0], e[1]))
        else:
            e = present.choice(rng)
            present.remove(e)
            pool.add(e)
            ops.append((DELETE, e[0], e[1]))
    return initial, ops


def sliding_window(events: Sequence[Tuple[int, int, float]], window: float) -> List[Tuple[str, int, int, float]]:
    """Turn time-stamped interactions (u, v, t), sorted by t, into edge updates.

    Returns a list of (op, u, v, time).  Self-interactions are ignored.  At
    equal times, expiries are processed before new interactions.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    last: Dict[Edge, float] = {}
    expiry: List[Tuple[float, Edge]] = []
    ops: List[Tuple[str, int, int, float]] = []
    prev_t = float("-inf")
    for u, v, t in events:
        if t < prev_t:
            raise ValueError("events must be sorted by time")
        prev_t = t
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        while expiry and expiry[0][0] <= t:
            te, old = heapq.heappop(expiry)
            seen = last.get(old)
            if seen is not None and seen + window == te:
                del last[old]
                ops.append((DELETE, old[0], old[1], te))
        if e not in last:
            ops.append((INSERT, e[0], e[1], t))
        last[e] = t
        heapq.heappush(expiry, (t + window, e))
    return ops


def relabel_vertices(n: int, initial: Sequence[Edge], ops: Sequence[Sequence], seed: int):
    """Apply a random vertex permutation (used to vary tie-breaking between seeds)."""
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    new_initial = [(perm[u], perm[v]) for u, v in initial]
    new_ops = [(op[0], perm[op[1]], perm[op[2]]) + tuple(op[3:]) for op in ops]
    return new_initial, new_ops, perm
