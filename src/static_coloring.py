"""Static graph colouring algorithms used as initialisers and baselines.

All functions take an adjacency structure ``adj`` (a sequence of sets, one per
vertex 0..n-1) and return a list ``col`` with col[v] >= 0.  Colour labels are
0..k-1.  The implementations follow the cited original descriptions:

* greedy first-fit in a given order            (Welsh and Powell, 1967 family)
* Welsh-Powell / largest-first ordering        (Welsh and Powell, 1967)
* smallest-last (degeneracy) ordering          (Matula and Beck, 1983)
* DSATUR                                        (Brelaz, 1979)
* Recursive Largest First (RLF)                 (Leighton, 1979)
* TabuCol local search for a fixed k            (Hertz and de Werra, 1987;
                                                 tenure rule of Galinier and Hao, 1999)
"""

from __future__ import annotations

import heapq
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Adjacency = Sequence[Set[int]]


# --------------------------------------------------------------------- helpers
def _first_fit_colour(neigh_colours: Set[int]) -> int:
    c = 0
    while c in neigh_colours:
        c += 1
    return c


def greedy_colouring(adj: Adjacency, order: Iterable[int]) -> List[int]:
    """First-fit greedy colouring of the vertices in the given order."""
    n = len(adj)
    col = [-1] * n
    for v in order:
        used = {col[u] for u in adj[v] if col[u] >= 0}
        col[v] = _first_fit_colour(used)
    # vertices missing from ``order`` (should not happen) get colour 0
    return [c if c >= 0 else 0 for c in col]


def natural_order_greedy(adj: Adjacency) -> List[int]:
    return greedy_colouring(adj, range(len(adj)))


def random_order_greedy(adj: Adjacency, seed: int = 0) -> List[int]:
    order = list(range(len(adj)))
    random.Random(seed).shuffle(order)
    return greedy_colouring(adj, order)


def welsh_powell(adj: Adjacency) -> List[int]:
    """Largest-first ordering (non-increasing degree), ties by vertex id."""
    order = sorted(range(len(adj)), key=lambda v: (-len(adj[v]), v))
    return greedy_colouring(adj, order)


def smallest_last_order(adj: Adjacency) -> List[int]:
    """Degeneracy ordering with a bucket queue; O(n^2 + m) with the min tie-break."""
    n = len(adj)
    deg = [len(a) for a in adj]
    max_deg = max(deg, default=0)
    buckets: List[Set[int]] = [set() for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[deg[v]].add(v)
    removed = [False] * n
    removal_order: List[int] = []
    d = 0
    for _ in range(n):
        d = max(d - 1, 0)
        while not buckets[d]:
            d += 1
        v = min(buckets[d])  # deterministic tie-break
        buckets[d].discard(v)
        removed[v] = True
        removal_order.append(v)
        for u in adj[v]:
            if not removed[u]:
                du = deg[u]
                buckets[du].discard(u)
                deg[u] = du - 1
                buckets[du - 1].add(u)
    removal_order.reverse()  # colour the last-removed vertex first
    return removal_order


def smallest_last(adj: Adjacency) -> List[int]:
    return greedy_colouring(adj, smallest_last_order(adj))


def degeneracy(adj: Adjacency) -> int:
    """Degeneracy of the graph (max over the smallest-last removal degrees)."""
    n = len(adj)
    deg = [len(a) for a in adj]
    max_deg = max(deg, default=0)
    buckets: List[Set[int]] = [set() for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[deg[v]].add(v)
    removed = [False] * n
    best = 0
    d = 0
    for _ in range(n):
        d = max(d - 1, 0)
        while not buckets[d]:
            d += 1
        best = max(best, d)
        v = buckets[d].pop()
        removed[v] = True
        for u in adj[v]:
            if not removed[u]:
                du = deg[u]
                buckets[du].discard(u)
                deg[u] = du - 1
                buckets[du - 1].add(u)
    return best


def dsatur(adj: Adjacency) -> List[int]:
    """DSATUR (Brelaz, 1979) with a binary heap and lazy deletion.

    Priority: largest saturation degree, then largest degree in the subgraph
    induced by the uncoloured vertices, then smallest vertex id.
    Time O((n + m) log(n + m)), space O(n + m).
    """
    n = len(adj)
    col = [-1] * n
    sat: List[Set[int]] = [set() for _ in range(n)]
    udeg = [len(a) for a in adj]
    heap: List[Tuple[int, int, int]] = [(0, -udeg[v], v) for v in range(n)]
    heapq.heapify(heap)
    while heap:
        neg_s, neg_d, v = heapq.heappop(heap)
        if col[v] != -1 or -neg_s != len(sat[v]) or -neg_d != udeg[v]:
            continue  # stale heap entry
        c = _first_fit_colour(sat[v])
        col[v] = c
        for u in adj[v]:
            if col[u] == -1:
                udeg[u] -= 1
                sat[u].add(c)
                heapq.heappush(heap, (-len(sat[u]), -udeg[u], u))
    return col


def rlf(adj: Adjacency) -> List[int]:
    """Recursive Largest First (Leighton, 1979).

    Builds one colour class at a time.  The first vertex of a class is the
    uncoloured vertex with most uncoloured neighbours; each further vertex is
    the candidate with most neighbours among the vertices already excluded
    from the class (ties: fewest neighbours among the remaining candidates).
    Worst-case time O(n^3); adequate for the benchmark sizes used here.
    """
    n = len(adj)
    col = [-1] * n
    uncoloured: Set[int] = set(range(n))
    colour = 0
    while uncoloured:
        candidates = set(uncoloured)  # U: may still join this class
        excluded: Set[int] = set()    # W: uncoloured but adjacent to the class
        v = max(candidates, key=lambda x: (len(adj[x] & uncoloured), -x))
        while True:
            col[v] = colour
            uncoloured.discard(v)
            candidates.discard(v)
            newly = adj[v] & candidates
            candidates -= newly
            excluded |= newly
            if not candidates:
                break
            v = max(candidates,
                    key=lambda x: (len(adj[x] & excluded), -len(adj[x] & candidates), -x))
        colour += 1
    return col


# --------------------------------------------------------------------- TabuCol
def count_conflicts(adj: Adjacency, col: Sequence[int]) -> int:
    return sum(1 for v in range(len(adj)) for u in adj[v] if u > v and col[u] == col[v])


def tabucol(adj: Adjacency, col: List[int], labels: Sequence[int],
            max_iters: int, rng: random.Random,
            initial_conflicting: Optional[Iterable[int]] = None,
            alpha: float = 0.6, random_tenure: int = 10,
            origin: Optional[Callable[[int], int]] = None, changed: int = 0,
            max_changed: Optional[int] = None,
            moved: Optional[Set[int]] = None) -> Tuple[List[int], int, int]:
    """TabuCol: minimise the number of conflicting edges using colours ``labels``.

    ``col`` is modified in place and also returned.  Neighbour colour counts
    (the gamma table of Galinier and Hao, 1999) are computed lazily, so a run
    that starts from an almost-proper colouring only touches the vertices near
    the conflicts.  Tabu tenure = alpha * |conflicting vertices| + U[0, 9].

    Budgeted variant (used by KERB): if ``origin`` and ``max_changed`` are
    given, ``origin(v)`` is the colour v had before the current update and
    ``changed`` the current number of vertices whose colour differs from it.
    Moves that would make that number exceed ``max_changed`` are never made,
    so every colouring visited stays within the recolouring budget.  Every
    moved vertex is added to ``moved`` if a set is supplied.

    Returns (colouring, remaining conflicting edges, iterations used).
    """
    label_set = list(labels)
    gamma: Dict[int, Dict[int, int]] = {}

    def g(v: int) -> Dict[int, int]:
        d = gamma.get(v)
        if d is None:
            d = {}
            for u in adj[v]:
                cu = col[u]
                d[cu] = d.get(cu, 0) + 1
            gamma[v] = d
        return d

    if initial_conflicting is None:
        initial_conflicting = range(len(adj))
    conflicting: Set[int] = set()
    conflicts2 = 0  # twice the number of conflicting edges
    for v in initial_conflicting:
        k = g(v).get(col[v], 0)
        if k:
            conflicting.add(v)
            conflicts2 += k
    conflicts = conflicts2 // 2
    tabu: Dict[Tuple[int, int], int] = {}
    best_conflicts = conflicts
    best_col = list(col) if conflicts else col
    capped = origin is not None and max_changed is not None
    it = 0
    while conflicts > 0 and it < max_iters:
        best_delta = None
        moves: List[Tuple[int, int]] = []
        for v in conflicting:
            gv = g(v)
            cv = col[v]
            base = gv.get(cv, 0)
            if capped:
                ov = origin(v)
                room = max_changed - changed + (cv != ov)  # 1: any colour, 0: only ov
            for c in label_set:
                if c == cv:
                    continue
                if capped and (c != ov) > room:
                    continue  # would exceed the recolouring budget
                delta = gv.get(c, 0) - base
                if tabu.get((v, c), -1) > it and conflicts + delta >= best_conflicts:
                    continue  # tabu and no aspiration
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    moves = [(v, c)]
                elif delta == best_delta:
                    moves.append((v, c))
        if not moves:  # every move tabu: take a random allowed move to keep going
            allowed = [(x, c) for x in sorted(conflicting) for c in label_set
                       if c != col[x] and (not capped or changed - (col[x] != origin(x))
                                           + (c != origin(x)) <= max_changed)]
            if not allowed:
                break
            v, c = allowed[rng.randrange(len(allowed))]
            gv = g(v)
            best_delta = gv.get(c, 0) - gv.get(col[v], 0)
        else:
            v, c = moves[rng.randrange(len(moves))]
        old = col[v]
        if capped:
            ov = origin(v)
            changed += (c != ov) - (old != ov)
        if moved is not None:
            moved.add(v)
        # update neighbour counts of already-computed neighbours
        for u in adj[v]:
            du = gamma.get(u)
            if du is not None:
                du[old] -= 1
                if du[old] == 0:
                    del du[old]
                du[c] = du.get(c, 0) + 1
        col[v] = c
        conflicts += best_delta
        # refresh conflicting membership of v and its neighbours
        gv = g(v)
        if gv.get(c, 0):
            conflicting.add(v)
        else:
            conflicting.discard(v)
        for u in adj[v]:
            cu = col[u]
            if cu == c:
                conflicting.add(u)
            elif cu == old:
                if g(u).get(cu, 0):
                    conflicting.add(u)
                else:
                    conflicting.discard(u)
        tabu[(v, old)] = it + int(alpha * len(conflicting)) + rng.randint(0, random_tenure - 1)
        it += 1
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            if conflicts > 0:
                best_col = list(col)
            else:
                best_col = col
    if best_conflicts == 0:
        return col if conflicts == 0 else best_col, 0, it
    return best_col, best_conflicts, it


def remove_class_and_reassign(adj: Adjacency, col: List[int], victim: int,
                              labels: Sequence[int]) -> List[int]:
    """Move every vertex of colour ``victim`` to its least-conflicting label."""
    members = [v for v in range(len(adj)) if col[v] == victim]
    for v in members:
        counts: Dict[int, int] = {}
        for u in adj[v]:
            counts[col[u]] = counts.get(col[u], 0) + 1
        col[v] = min(labels, key=lambda c: (counts.get(c, 0), c))
    return members


def tabucol_minimise(adj: Adjacency, initial: Sequence[int], total_iters: int,
                     seed: int = 0) -> Tuple[List[int], int]:
    """Decrease k from a proper initial colouring with repeated TabuCol runs.

    Each round deletes the smallest colour class, reassigns its vertices
    greedily and runs TabuCol with the remaining k-1 labels.  Stops when a
    round fails or the iteration budget is spent.  Returns the best proper
    colouring found (relabelled 0..k-1) and the iterations used.
    """
    rng = random.Random(seed)
    best = list(initial)
    used = 0
    while used < total_iters:
        labels = sorted(set(best))
        if len(labels) <= 1:
            break
        sizes = {c: 0 for c in labels}
        for c in best:
            sizes[c] += 1
        victim = min(labels, key=lambda c: (sizes[c], -c))
        remaining = [c for c in labels if c != victim]
        trial = list(best)
        moved = remove_class_and_reassign(adj, trial, victim, remaining)
        touched = set(moved)
        for v in moved:
            touched |= adj[v]
        trial, conflicts, its = tabucol(adj, trial, remaining, total_iters - used, rng,
                                        initial_conflicting=touched)
        used += its
        if conflicts == 0:
            best = trial
        else:
            break
    return relabel_compact(best), used


def relabel_compact(col: Sequence[int]) -> List[int]:
    """Relabel colours to 0..k-1 in order of first appearance."""
    mapping: Dict[int, int] = {}
    out = []
    for c in col:
        if c not in mapping:
            mapping[c] = len(mapping)
        out.append(mapping[c])
    return out


def greedy_clique_lower_bound(adj: Adjacency, starts: int = 64) -> int:
    """Size of a clique found greedily (a valid lower bound on chi(G)).

    Starts from the ``starts`` highest-degree vertices; each clique is grown by
    repeatedly adding the candidate with most neighbours among the candidates.
    """
    n = len(adj)
    if n == 0:
        return 0
    order = sorted(range(n), key=lambda v: -len(adj[v]))[:max(1, starts)]
    best = 1
    for s in order:
        clique_size = 1
        cand = set(adj[s])
        while cand:
            x = max(cand, key=lambda y: (len(adj[y] & cand), -y))
            clique_size += 1
            cand &= adj[x]
        best = max(best, clique_size)
    return best


STATIC_ALGORITHMS = {
    "Greedy (natural order)": natural_order_greedy,
    "Welsh-Powell": welsh_powell,
    "Smallest-last": smallest_last,
    "DSATUR": dsatur,
    "RLF": rlf,
}
