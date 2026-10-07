"""Dynamic colouring baselines that KERB is compared against.

* :class:`FirstFitRepair` -- the classical greedy local repair: when an
  inserted edge creates a conflict, one endpoint takes the smallest colour not
  used by its neighbours (a new colour if there is none).  At most one vertex
  is recoloured per update.  Deletions are ignored.  It applies the first-fit
  rule studied for online colouring (Gyarfas and Lehel, 1988) as a local
  repair.
* :class:`TabuWarmStart` -- quality-oriented repair that follows the
  warm-start idea studied by Hardy, Lewis and Thompson (2016, 2018): keep the
  previous colouring, let TabuCol remove the clash with the current number of
  colours, and periodically try to remove one colour.  The number of
  recoloured vertices is not limited.
* :class:`DsaturRecompute` -- the recompute-from-scratch strategy: DSATUR is
  run on every new graph and its classes are matched to the previous labels
  with the Hungarian algorithm so that as many vertices as possible keep their
  colour.  Because DSATUR is deterministic, its per-update cost can be
  measured exactly at sampled updates (see :func:`dsatur_recompute_cost`).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from coloring_state import ColouringState
from dynamic_graph import DynamicGraph
from proposed_algorithm import DELETE, INSERT, UpdateResult
from static_coloring import dsatur, tabucol


class _DynamicBase:
    name = "base"

    def __init__(self, graph: DynamicGraph, initial_colouring: Optional[Sequence[int]] = None) -> None:
        self.g = graph
        colours = list(initial_colouring) if initial_colouring is not None else dsatur(graph.adj)
        self.state = ColouringState(colours)
        self.t = 0
        self.recolour_count = [0] * graph.n

    @property
    def colouring(self) -> List[int]:
        return self.state.col

    @property
    def num_colours(self) -> int:
        return self.state.num_colours

    def apply(self, op: str, u: int, v: int) -> UpdateResult:
        if op == INSERT:
            return self.insert_edge(u, v)
        if op == DELETE:
            return self.delete_edge(u, v)
        raise ValueError(f"unknown operation {op!r}")

    def _finish(self, op: str, u: int, v: int, applied: bool, conflict: bool,
                repair: str, eliminated: int = 0) -> UpdateResult:
        st = self.state
        for x in st.changed_vertices():
            self.recolour_count[x] += 1
        self.t += 1
        return UpdateResult(op, u, v, applied, conflict, repair, st.changed,
                            st.num_colours, eliminated)


class FirstFitRepair(_DynamicBase):
    name = "FF-Repair"

    def insert_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        applied = self.g.add_edge(u, v)
        conflict = applied and st.col[u] == st.col[v]
        repair = "none"
        if conflict:
            adj = self.g.adj
            # try the lower-degree endpoint first, then the other one
            for w in sorted((u, v), key=lambda x: (len(adj[x]), x)):
                used = {st.col[x] for x in adj[w]}
                free = [b for b in st.classes if b not in used]
                if free:
                    st.set_colour(w, min(free))
                    repair = "direct"
                    break
            else:
                w = min((u, v), key=lambda x: (len(adj[x]), x))
                st.set_colour(w, st.new_label())
                repair = "new_colour"
        return self._finish(INSERT, u, v, applied, conflict, repair)

    def delete_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        applied = self.g.remove_edge(u, v)
        return self._finish(DELETE, u, v, applied, False, "none")


class TabuWarmStart(_DynamicBase):
    name = "Tabu-WarmStart"

    def __init__(self, graph: DynamicGraph, initial_colouring: Optional[Sequence[int]] = None,
                 repair_iters: int = 500, reduce_every: int = 50, reduce_iters: int = 2000,
                 seed: int = 0) -> None:
        super().__init__(graph, initial_colouring)
        self.repair_iters = repair_iters
        self.reduce_every = reduce_every
        self.reduce_iters = reduce_iters
        self.rng = random.Random(seed)
        self.tabu_iterations = 0

    def _commit(self, new_col: List[int], moved) -> None:
        """Apply the TabuCol result; only vertices TabuCol moved can differ."""
        st = self.state
        for x in sorted(moved):
            c = new_col[x]
            if st.col[x] != c:
                st.set_colour(x, c)

    def insert_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        applied = self.g.add_edge(u, v)
        conflict = applied and st.col[u] == st.col[v]
        repair = "none"
        if conflict:
            labels = sorted(st.classes)
            trial = list(st.col)
            moved = set()
            result, conflicts, its = tabucol(self.g.adj, trial, labels, self.repair_iters,
                                             self.rng, initial_conflicting=(u, v), moved=moved)
            self.tabu_iterations += its
            if conflicts == 0:
                self._commit(result, moved)
                repair = "tabu"
            else:
                adj = self.g.adj
                w = min((u, v), key=lambda x: (len(adj[x]), x))
                st.set_colour(w, st.new_label())
                repair = "new_colour"
        eliminated = self._maybe_reduce()
        return self._finish(INSERT, u, v, applied, conflict, repair, eliminated)

    def delete_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        applied = self.g.remove_edge(u, v)
        eliminated = self._maybe_reduce()
        return self._finish(DELETE, u, v, applied, False, "none", eliminated)

    def _maybe_reduce(self) -> int:
        """Every ``reduce_every`` updates, try to remove the smallest class."""
        st = self.state
        if self.reduce_every <= 0 or (self.t + 1) % self.reduce_every or st.num_colours <= 1:
            return 0
        adj = self.g.adj
        sizes = {c: len(m) for c, m in st.classes.items()}
        victim = min(sizes, key=lambda c: (sizes[c], -c))
        labels = [c for c in sorted(st.classes) if c != victim]
        trial = list(st.col)
        moved = list(st.classes[victim])
        for x in moved:
            counts: Dict[int, int] = {}
            for y in adj[x]:
                counts[trial[y]] = counts.get(trial[y], 0) + 1
            trial[x] = min(labels, key=lambda c: (counts.get(c, 0), c))
        touched = set(moved)
        for x in moved:
            touched |= adj[x]
        changed = set(moved)
        result, conflicts, its = tabucol(adj, trial, labels, self.reduce_iters, self.rng,
                                         initial_conflicting=touched, moved=changed)
        self.tabu_iterations += its
        if conflicts == 0:
            self._commit(result, changed)
            return 1
        return 0


# ------------------------------------------------------------ DSATUR recompute
def match_labels(previous: Sequence[int], new: Sequence[int]) -> List[int]:
    """Relabel ``new`` to agree with ``previous`` on as many vertices as possible.

    Solves a maximum-weight bipartite matching between new classes and old
    labels (Hungarian algorithm, scipy.optimize.linear_sum_assignment).
    Unmatched new classes receive the smallest unused labels.
    """
    old_labels = sorted(set(previous))
    new_labels = sorted(set(new))
    o_index = {c: i for i, c in enumerate(old_labels)}
    n_index = {c: i for i, c in enumerate(new_labels)}
    weight = np.zeros((len(new_labels), len(old_labels)), dtype=np.int64)
    for p, q in zip(previous, new):
        weight[n_index[q], o_index[p]] += 1
    rows, cols = linear_sum_assignment(weight, maximize=True)
    mapping: Dict[int, int] = {}
    used = set()
    for r, c in zip(rows, cols):
        if weight[r, c] > 0:
            mapping[new_labels[r]] = old_labels[c]
            used.add(old_labels[c])
    nxt = 0
    for q in new_labels:
        if q not in mapping:
            while nxt in used:
                nxt += 1
            mapping[q] = nxt
            used.add(nxt)
    return [mapping[q] for q in new]


def hamming(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


class DsaturRecompute(_DynamicBase):
    """Recompute DSATUR after every update and relabel to the previous labels."""

    name = "DSATUR-Recompute"

    def _recompute(self) -> int:
        st = self.state
        new = match_labels(st.col, dsatur(self.g.adj))
        for x, c in enumerate(new):
            if st.col[x] != c:
                st.set_colour(x, c)
        return 0

    def insert_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        applied = self.g.add_edge(u, v)
        conflict = applied and st.col[u] == st.col[v]
        self._recompute()
        return self._finish(INSERT, u, v, applied, conflict, "recompute")

    def delete_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        applied = self.g.remove_edge(u, v)
        self._recompute()
        return self._finish(DELETE, u, v, applied, False, "recompute")


def dsatur_recompute_cost(adj_before, adj_after) -> Tuple[int, int]:
    """Exact (recoloured vertices, colours) of DSATUR-Recompute for one update.

    The minimum Hamming distance between two colourings over all relabellings
    does not depend on which labels the earlier colouring used, so the cost of
    the recompute strategy at update t only needs DSATUR(G_{t-1}) and
    DSATUR(G_t).
    """
    before = dsatur(adj_before)
    after = dsatur(adj_after)
    relabelled = match_labels(before, after)
    return hamming(before, relabelled), len(set(after))
