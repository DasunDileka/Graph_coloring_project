"""KERB: Kempe/ejection-chain Repair with class Elimination under a recolouring Budget.

KERB maintains a proper vertex colouring of a graph that changes by single
edge insertions and deletions.  After every update the colouring must be
proper and at most ``budget`` (R) vertices may have a different colour than
before the update.  Within that hard limit KERB tries to keep the number of
colours small.  It separates *computation* (limited only by a search effort
parameter) from *disruption* (limited to R recoloured vertices per update).

Components (Chapter 9 of the report):

C1  Initial colouring of G_0: DSATUR (Brelaz, 1979), or any proper colouring
    supplied by the caller (the experiments supply DSATUR improved offline by
    TabuCol, the same start for every method).
C2  Budgeted minimum-cost repair.  When an inserted edge joins two vertices
    of the same colour, KERB tries, in this order: a direct move of one
    endpoint into an existing colour; an ejection chain found by depth-first
    search with iterative deepening on its exact cost (Kempe-chain
    interchanges are a special case); optionally a TabuCol search restricted
    to colourings within the budget (off by default); a direct move into the
    class that C4 is removing; and finally a new colour for the endpoint of
    smaller degree (cost 1), so every R >= 1 is feasible.
C3  Quick class elimination.  Budget left over in an update is used to empty
    one of the two smallest colour classes when all of its vertices can be
    moved out within the remaining budget.  The attempt is transactional (it
    is rolled back if it fails) and a failed class is retried with
    exponential back-off.
C4  Planned migration.  Periodically KERB runs TabuCol (Hertz and de Werra,
    1987) on a copy of the colouring to find a colouring with one class fewer
    that is close to the current one.  The difference is a migration plan that
    is executed a few vertices per update with the budget that C2 and C3 left,
    keeping the colouring proper after every move.  The removed class
    disappears when the plan is complete; stalled or stale plans are dropped.

Invariants (proved in Chapter 9): the colouring is proper after every
update; the number of recoloured vertices r_t never exceeds R; and the number
of colours never exceeds max(k_0, Delta_max + 1), because only the last step
of C2 opens a colour and it is reached only when every existing colour occurs
next to both endpoints.

The recolouring cost is the exact Hamming distance between the colourings
before and after an update (see :mod:`coloring_state`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from coloring_state import ColouringState
from dynamic_graph import DynamicGraph
from static_coloring import dsatur, tabucol

INSERT = "insert"
DELETE = "delete"
REPAIR_MODES = ("ejection", "kempe", "direct")


@dataclass(frozen=True)
class KerbConfig:
    """Parameters of KERB.  The values used in the evaluation are fixed in
    experiments/config.py before the evaluation runs."""

    budget: int = 3                    # R: max recoloured vertices per update (R >= 1)
    repair_mode: str = "ejection"      # C2: 'ejection' (default), 'kempe' or 'direct'
    search_nodes: int = 1000           # C2-C4: search nodes allowed per update (time control)
    tabu_iters: int = 0                # C2/C3: budgeted TabuCol fallback (0 = off; pilot choice)
    use_elimination: bool = True       # C3: quick transactional class elimination
    use_migration: bool = True         # C4: planned migration
    plan_every: int = 50               # updates between planning attempts
    plan_iters: int = 1000             # TabuCol iterations per planning attempt
    max_stall: int = 100               # abandon a plan after this many updates without progress
    elimination_candidates: int = 2    # smallest classes tried by C3 per update
    prefer_large_classes: bool = True  # tie-break: move vertices into larger classes
    max_backoff: int = 32              # C3/C4: longest wait (updates) before retrying a failed attempt
    seed: int = 0                      # random seed for TabuCol in C4

    def validate(self) -> None:
        if not isinstance(self.budget, int) or self.budget < 1:
            raise ValueError("budget R must be an integer >= 1 (R = 0 cannot repair a conflict)")
        if self.repair_mode not in REPAIR_MODES:
            raise ValueError(f"repair_mode must be one of {REPAIR_MODES}")
        if self.search_nodes < 1 or self.plan_every < 1 or self.max_stall < 1:
            raise ValueError("search_nodes, plan_every and max_stall must be >= 1")
        if self.plan_iters < 0 or self.tabu_iters < 0:
            raise ValueError("plan_iters and tabu_iters must be >= 0")
        if self.elimination_candidates < 1 or self.max_backoff < 1:
            raise ValueError("elimination_candidates and max_backoff must be >= 1")


@dataclass
class UpdateResult:
    """What happened during one update (used by the metrics module)."""

    op: str
    u: int
    v: int
    applied: bool          # False if the edge was already present / absent
    conflict: bool         # inserted edge joined two equal colours
    repair: str            # 'none', 'direct', 'chain', 'kempe', 'new_colour', ...
    recoloured: int        # r_t: Hamming distance to the previous colouring
    colours: int           # k_t after the update
    eliminated: int        # colour classes that disappeared during this update


class KERB:
    """Dynamic graph colouring with a hard per-update recolouring budget."""

    name = "KERB"

    def __init__(self, graph: DynamicGraph, config: KerbConfig = KerbConfig(),
                 initial_colouring: Optional[Sequence[int]] = None) -> None:
        config.validate()
        self.g = graph
        self.cfg = config
        colours = list(initial_colouring) if initial_colouring is not None else dsatur(graph.adj)
        if len(colours) != graph.n:
            raise ValueError("initial colouring has the wrong length")
        self.state = ColouringState(colours)
        self.rng = random.Random(config.seed)
        self.t = 0                                  # number of processed updates
        self.recolour_count: List[int] = [0] * graph.n
        self._nodes = 0                             # nodes used by the current search
        # planned-migration state (C4)
        self.target: Dict[int, int] = {}            # pending vertex -> planned colour
        self.victim: Optional[int] = None           # class the plan removes
        self.next_plan_at = config.plan_every
        self.last_progress = 0
        # retry control: a failed attempt is not repeated for 2^f updates
        # (f = consecutive failures, capped at max_backoff)
        self._class_retry: Dict[int, Tuple[int, int, int]] = {}  # class -> (size, f, retry_at)
        self._move_retry: Dict[int, Tuple[int, int]] = {}        # vertex -> (f, retry_at)
        self._plan_failures = 0                                  # consecutive plans not found
        self._plan_k = -1                                        # colours when planning last failed
        self.stats = {"plans": 0, "plans_found": 0, "migrations_done": 0,
                      "migrations_abandoned": 0, "plans_stale": 0, "plans_skipped": 0, "planned_moves": 0, "tabu_iterations": 0,
                      "search_nodes": 0}

    # ------------------------------------------------------------ public API
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

    def insert_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        self._nodes = 0
        before = st.num_colours
        applied = self.g.add_edge(u, v)
        conflict = applied and st.col[u] == st.col[v]
        repair = self._repair(u, v) if conflict else "none"
        if applied and self.target:
            self._patch_plan(u, v)
        self._use_residual_budget()
        return self._finish(INSERT, u, v, applied, conflict, repair, before)

    def delete_edge(self, u: int, v: int) -> UpdateResult:
        st = self.state
        st.begin_update()
        self._nodes = 0
        before = st.num_colours
        applied = self.g.remove_edge(u, v)
        self._use_residual_budget()
        return self._finish(DELETE, u, v, applied, False, "none", before)

    # -------------------------------------------------------------- internals
    def _finish(self, op: str, u: int, v: int, applied: bool, conflict: bool,
                repair: str, colours_before: int) -> UpdateResult:
        st = self.state
        for x in st.changed_vertices():
            self.recolour_count[x] += 1
        self.stats["search_nodes"] += self._nodes
        self.t += 1
        after = st.num_colours
        return UpdateResult(op, u, v, applied, conflict, repair, st.changed,
                            after, max(0, colours_before - after))

    def _remaining(self) -> int:
        return self.cfg.budget - self.state.changed

    def _backoff(self, failures: int) -> int:
        return min(2 ** failures, self.cfg.max_backoff)

    def _backoff_factor(self, failures: int) -> int:
        """Planning interval multiplier: 1 after a success, then 2, 4, 8, 16."""
        return min(2 ** (failures - 1), 16)

    def _class_key(self, b: int) -> Tuple[int, int]:
        size = len(self.state.classes.get(b, ()))
        return ((-size if self.cfg.prefer_large_classes else 0), b)

    def _forbidden(self, *extra: int) -> FrozenSet[int]:
        """Colours no vertex may move into (the class being removed, if any)."""
        f = set(extra)
        if self.victim is not None:
            f.add(self.victim)
        return frozenset(f)

    # ------------------------------------------- ejection-chain search (C2-C4)
    def _must_move(self, f: int, forbidden: FrozenSet[int], must: Set[int], locked: Set[int]) -> bool:
        col = self.state.col
        cf = col[f]
        if f in must and f not in locked:
            return True
        if cf in forbidden:
            return True
        return any(col[y] == cf for y in self.g.adj[f])

    def _search(self, pending: List[int], locked: Set[int], budget_left: int,
                forbidden: FrozenSet[int], must: Set[int]) -> bool:
        """Depth-first search for recolourings that satisfy every pending vertex
        at an extra cost of at most ``budget_left``.

        A pending vertex must move because it shares its colour with a vertex
        that has already been moved in this search (and may not move again),
        because its colour is forbidden, or because it is the vertex the search
        was started for.  Each vertex moves at most once (``locked``), so every
        pending vertex that is still unchanged costs at least one recolouring;
        their number is a valid lower bound used for pruning.  Moves are
        applied to the state and rolled back on failure, so on success the
        solution is already in place.
        """
        st = self.state
        col = st.col
        adj = self.g.adj
        pending = [x for x in dict.fromkeys(pending) if self._must_move(x, forbidden, must, locked)]
        if not pending:
            return True
        need = sum(1 for x in pending if st.is_unchanged(x))
        if need > budget_left:
            return False
        f = pending[-1]
        if f in locked:
            return False
        rest = pending[:-1]
        rest_set = set(rest)
        slack = budget_left - (need - (1 if st.is_unchanged(f) else 0))
        cnt = st.neighbour_colour_counts(adj[f])
        cf = col[f]
        sign = -1 if self.cfg.prefer_large_classes else 0
        options = sorted((cnt.get(c, 0), sign * len(members), c)
                         for c, members in st.classes.items() if c != cf and c not in forbidden)
        changed_now = st.changed
        for disp, _, c in options:
            if disp - changed_now - len(rest) - 1 > slack:
                break  # too many neighbours of colour c would have to move as well
            if self._nodes >= self.cfg.search_nodes:
                return False
            self._nodes += 1
            displaced = [y for y in adj[f] if col[y] == c] if disp else []
            if any(y in locked for y in displaced):
                continue
            cost = st.cost_if_set(f, c)
            extra = sum(1 for y in displaced if y not in rest_set and st.is_unchanged(y))
            if cost + extra > slack:
                continue
            sp = st.savepoint()
            st.set_colour(f, c)
            locked.add(f)
            if self._search(rest + displaced, locked, budget_left - cost, forbidden, must):
                return True
            st.rollback(sp)
            locked.discard(f)
        return False

    def _move_out(self, w: int, forbidden: FrozenSet[int], limit: int) -> bool:
        """Recolour w (and possibly others) at minimum cost <= limit so that the
        colouring becomes proper and w leaves its colour; iterative deepening."""
        for budget in range(1, limit + 1):
            if self._search([w], set(), budget, forbidden, {w}):
                return True
            if self._nodes >= self.cfg.search_nodes:
                return False
        return False

    def _move_to(self, x: int, b: int, forbidden: FrozenSet[int], limit: int) -> bool:
        """Give x colour b and recolour the displaced neighbours (cost <= limit)."""
        st = self.state
        if b not in st.classes:
            return False  # never re-open a colour that is no longer in use
        cost = st.cost_if_set(x, b)
        if cost > limit:
            return False
        sp = st.savepoint()
        displaced = [y for y in self.g.adj[x] if st.col[y] == b]
        st.set_colour(x, b)
        for budget in range(len([y for y in displaced if st.is_unchanged(y)]), limit - cost + 1):
            if self._search(list(displaced), {x}, budget, forbidden, set()):
                return True
            if self._nodes >= self.cfg.search_nodes:
                break
        st.rollback(sp)
        return False

    # ------------------------------------------- budgeted TabuCol fallback (C2, C3)
    def _tabu_resolve(self, trial: List[int], conflicted: Iterable[int], labels: List[int],
                      changed: int) -> bool:
        """Remove every conflict of the tentative colouring ``trial`` with TabuCol
        restricted to ``labels`` and to colourings within the recolouring budget
        (Hamming distance to the colouring at the start of the update <= R).
        ``changed`` is that distance for ``trial``.  Commits on success."""
        st = self.state
        if self.cfg.tabu_iters <= 0 or changed > self.cfg.budget:
            return False
        conflicted = list(conflicted)
        moved: Set[int] = {x for x in conflicted if trial[x] != st.col[x]}
        result, conflicts, its = tabucol(self.g.adj, trial, labels, self.cfg.tabu_iters, self.rng,
                                         initial_conflicting=conflicted, origin=st.original_colour,
                                         changed=changed, max_changed=self.cfg.budget, moved=moved)
        self.stats["tabu_iterations"] += its
        if conflicts:
            return False
        for x in moved:
            if st.col[x] != result[x]:
                st.set_colour(x, result[x])
        return True

    def _tabu_repair(self, u: int, v: int, forbidden: FrozenSet[int]) -> bool:
        st = self.state
        labels = [c for c in st.classes if c not in forbidden]
        return self._tabu_resolve(list(st.col), (u, v), labels, st.changed)

    def _tabu_empty_class(self, h: int) -> bool:
        """Empty class h at once: give its vertices their least-conflicting other
        colour, then let the budgeted TabuCol remove the resulting conflicts."""
        st = self.state
        adj = self.g.adj
        forbidden = self._forbidden(h)
        labels = [c for c in st.classes if c not in forbidden]
        if not labels:
            return False
        trial = list(st.col)
        changed = st.changed
        members = list(st.classes[h])
        for x in members:
            counts: Dict[int, int] = {}
            for y in adj[x]:
                counts[trial[y]] = counts.get(trial[y], 0) + 1
            ox = st.original_colour(x)
            b = min(labels, key=lambda c: (counts.get(c, 0), c != ox, c))
            changed += (b != ox) - (trial[x] != ox)
            trial[x] = b
        touched = set(members)
        for x in members:
            touched |= adj[x]
        return self._tabu_resolve(trial, touched, labels, changed)

    # ------------------------------------------------------ Kempe chains (C2)
    def _kempe_chain(self, w: int, a: int, b: int, limit: int,
                     forbidden: int = -1) -> Optional[List[int]]:
        """Kempe chain of w for colours {a, b}: the vertices reachable from w along
        edges joining colour a to colour b; None if it contains ``forbidden`` or its
        exact cost would exceed ``limit``."""
        st = self.state
        col = st.col
        adj = self.g.adj
        cost = 1 if st.is_unchanged(w) else 0
        if cost > limit:
            return None
        chain = [w]
        seen = {w}
        i = 0
        while i < len(chain):
            x = chain[i]
            i += 1
            other = b if col[x] == a else a
            for y in adj[x]:
                if col[y] == other and y not in seen:
                    if y == forbidden:
                        return None
                    seen.add(y)
                    chain.append(y)
                    if st.is_unchanged(y):
                        cost += 1
                        if cost > limit:
                            return None
        return chain

    def _kempe_repair(self, u: int, v: int, limit: int) -> bool:
        st = self.state
        col = st.col
        a = col[u]
        forbidden = self._forbidden()
        best = None
        best_cost = limit + 1
        for w in (u, v):
            other = v if w == u else u
            cnt = st.neighbour_colour_counts(self.g.adj[w])
            for c_b, _, b in sorted((cnt.get(b, 0), self._class_key(b), b)
                                    for b in st.classes if b != a and b not in forbidden):
                if 1 + c_b >= best_cost:
                    break
                chain = self._kempe_chain(w, a, b, best_cost - 1, forbidden=other)
                if chain is not None and len(chain) < best_cost:
                    best, best_cost = (chain, b), len(chain)
        if best is None:
            return False
        chain, b = best
        for x in chain:
            st.set_colour(x, b if col[x] == a else a)
        return True

    # ----------------------------------------------------- C2: conflict repair
    def _best_direct(self, u: int, v: int, forbidden: FrozenSet[int],
                     limit: int) -> Optional[Tuple[Tuple[int, int, int], int, int]]:
        """Cheapest single-vertex move that resolves the conflict on {u, v}:
        an endpoint w and an existing colour b that no neighbour of w has."""
        st = self.state
        adj = self.g.adj
        best = None
        for w in (u, v):
            cnt = st.neighbour_colour_counts(adj[w])
            cw = st.col[w]
            for b in st.classes:
                if b == cw or b in forbidden or cnt.get(b, 0):
                    continue
                key = (st.cost_if_set(w, b),) + self._class_key(b)
                if key[0] <= limit and (best is None or key < best[0]):
                    best = (key, w, b)
        return best

    def _repair(self, u: int, v: int) -> str:
        """Resolve the conflict on the new edge {u, v} (col[u] == col[v]).

        Order of preference: (1) a direct move into an existing class,
        (2) an ejection chain / Kempe interchange of total cost <= the budget,
        (3) a direct move into the class that C4 is currently removing,
        (4) a new colour for the endpoint of smaller degree.  Only step (4)
        can increase the number of colours, and it is reached only when every
        existing colour occurs in the neighbourhood of both endpoints.
        """
        st = self.state
        adj = self.g.adj
        limit = self._remaining()
        forbidden = self._forbidden()
        mode = self.cfg.repair_mode
        best = self._best_direct(u, v, forbidden, limit)
        if best is not None:
            st.set_colour(best[1], best[2])
            return "direct"
        if limit >= 2 and mode == "ejection":
            # iterative deepening on the cost: the first chain found is a cheapest one
            for budget in range(2, limit + 1):
                for w in (u, v):
                    if self._search([w], set(), budget, forbidden, {w}):
                        return "chain"
                if self._nodes >= self.cfg.search_nodes:
                    break
            # the exact search is complete unless it ran out of nodes; only then
            # can the (heuristic) budgeted TabuCol still find a repair
            if self._nodes >= self.cfg.search_nodes and self._tabu_repair(u, v, forbidden):
                return "tabu"
        elif limit >= 2 and mode == "kempe":
            if self._kempe_repair(u, v, limit):
                return "kempe"
        if self.victim is not None:
            best = self._best_direct(u, v, frozenset(), limit)
            if best is not None:
                st.set_colour(best[1], best[2])
                return "direct_victim"
        w = u if (len(adj[u]), u) <= (len(adj[v]), v) else v
        st.set_colour(w, st.new_label())
        return "new_colour"

    # ----------------------------------------------- residual budget (C3, C4)
    def _use_residual_budget(self) -> None:
        if self.cfg.use_elimination:
            self._eliminate_small_class()
        if self.cfg.use_migration:
            if self.victim is not None:
                self._migrate()
            elif self.t >= self.next_plan_at:
                if self._plan():
                    self._migrate()

    # ------------------------------------------------- C3: quick elimination
    def _eliminate_small_class(self) -> bool:
        st = self.state
        remaining = self._remaining()
        if remaining <= 0 or st.num_colours <= 1:
            return False
        ranked = sorted(st.classes.items(), key=lambda kv: (len(kv[1]), -kv[0]))
        for h, members in ranked[: self.cfg.elimination_candidates]:
            # every unchanged member costs one recolouring and at most st.changed
            # members can already be changed, so a class this large cannot be
            # emptied (checked in O(1) before the exact count)
            if len(members) - st.changed > remaining:
                break  # later classes are at least as large
            need = sum(1 for x in members if st.is_unchanged(x))
            if need > remaining:
                break
            size = len(members)
            retry = self._class_retry.get(h)
            if retry is not None and retry[0] == size and self.t < retry[2]:
                continue  # failed recently and unchanged in size: wait
            if self._empty_class(h):
                self._class_retry.pop(h, None)
                if h == self.victim:
                    self._end_plan(success=True)
                return True
            f = retry[1] + 1 if retry is not None and retry[0] == size else 1
            self._class_retry[h] = (size, f, self.t + self._backoff(f))
        return False

    def _empty_class(self, h: int) -> bool:
        """Transactionally move every vertex out of class h within the budget."""
        st = self.state
        adj = self.g.adj
        sp = st.savepoint()
        forbidden = self._forbidden(h)
        for x in sorted(st.classes[h], key=lambda y: (-len(adj[y]), y)):  # hardest first
            if st.col[x] != h:
                continue  # already moved out by an earlier chain
            ok = (self._move_out(x, forbidden, self._remaining())
                  if self.cfg.repair_mode == "ejection" else self._direct_out(x, forbidden))
            if not ok:
                st.rollback(sp)
                # one vertex at a time failed; try to move the class out jointly
                return (self.cfg.repair_mode == "ejection" and self._remaining() >= 2
                        and self._tabu_empty_class(h))
        return h not in st.classes

    def _direct_out(self, x: int, forbidden: FrozenSet[int]) -> bool:
        st = self.state
        cnt = st.neighbour_colour_counts(self.g.adj[x])
        options = [b for b in st.classes if b != st.col[x] and b not in forbidden and not cnt.get(b, 0)]
        if not options:
            return False
        b = min(options, key=self._class_key)
        if st.cost_if_set(x, b) > self._remaining():
            return False
        st.set_colour(x, b)
        return True

    # -------------------------------------------------- C4: planned migration
    def _plan(self) -> bool:
        """Search (without recolouring anything) for a proper colouring with
        one class fewer that stays close to the current colouring."""
        st = self.state
        cfg = self.cfg
        if st.num_colours != self._plan_k:
            self._plan_failures = 0      # the colouring changed: forget earlier failures
        # after f consecutive failures the next attempt waits plan_every * 2^f updates
        self.next_plan_at = self.t + cfg.plan_every * self._backoff_factor(self._plan_failures + 1)
        if st.num_colours <= 1 or cfg.plan_iters <= 0:
            return False
        adj = self.g.adj
        col = st.col
        h = min(st.classes, key=lambda c: (len(st.classes[c]), -c))
        if len(st.classes[h]) > cfg.budget * cfg.max_stall:
            # emptying the class would take more than max_stall updates even at
            # full budget, and planning it is costly on large graphs: a cost cap
            # (deviation D1), not a consequence of the stall rule
            self.stats["plans_skipped"] += 1
            self._plan_failures += 1
            self._plan_k = st.num_colours
            return False
        self.stats["plans"] += 1
        labels = [c for c in st.classes if c != h]
        trial = list(col)
        members = list(st.classes[h])
        for x in members:
            counts: Dict[int, int] = {}
            for y in adj[x]:
                counts[trial[y]] = counts.get(trial[y], 0) + 1
            trial[x] = min(labels, key=lambda c: (counts.get(c, 0), c))
        touched = set(members)
        for x in members:
            touched |= adj[x]
        moved: Set[int] = set(members)
        trial, conflicts, its = tabucol(adj, trial, labels, cfg.plan_iters, self.rng,
                                        initial_conflicting=touched, moved=moved)
        self.stats["tabu_iterations"] += its
        if conflicts:
            self._plan_failures += 1
            self._plan_k = st.num_colours
            return False
        self._plan_failures = 0
        # revert pass: keep the current colour wherever the plan allows it
        # (only moved vertices can differ, so this costs O(sum of their degrees))
        for x in sorted(moved):
            c0 = col[x]
            if trial[x] != c0 and c0 != h and all(trial[y] != c0 for y in adj[x]):
                trial[x] = c0
        self.target = {x: trial[x] for x in sorted(moved) if trial[x] != col[x]}
        self.victim = h
        self.last_progress = self.t
        self.stats["plans_found"] += 1
        return True

    def _end_plan(self, success: bool, stale: bool = False) -> None:
        self.target.clear()
        self._move_retry.clear()
        self.victim = None
        if success:
            self.stats["migrations_done"] += 1
            self.next_plan_at = self.t + 1        # try to remove another class soon
        elif stale:
            self.stats["plans_stale"] += 1
            self.next_plan_at = self.t + 1        # the colouring improved meanwhile: re-plan
        else:
            self.stats["migrations_abandoned"] += 1
            self.next_plan_at = self.t + self.cfg.plan_every

    def _patch_plan(self, u: int, v: int) -> None:
        """Keep the plan consistent with a newly inserted edge {u, v}.

        The plan is the colouring ``target.get(x, col[x])``.  If the new edge
        joins two vertices with the same planned colour, the planned move of
        an endpoint outside the victim class is dropped (it keeps its current
        colour); if that does not help, the plan is abandoned.  The plan is a
        guide only: every executed move is checked against the current graph.
        """
        col = self.state.col
        target = self.target
        for _ in range(2):
            if target.get(u, col[u]) != target.get(v, col[v]):
                return
            for w in (u, v):
                if w in target and col[w] != self.victim:
                    del target[w]
                    break
            else:
                break
        self._end_plan(success=False)

    def _migrate(self) -> None:
        """Execute the plan with the budget that is left in this update."""
        st = self.state
        col = st.col
        adj = self.g.adj
        target = self.target
        h = self.victim
        if h is None:
            return
        if h not in st.classes:
            self._end_plan(success=True)
            return
        for x in [x for x, b in target.items() if col[x] == b]:
            del target[x]
        if any(b not in st.classes for b in target.values()):
            # a planned destination class has disappeared since planning (it
            # was eliminated by C3 or emptied by a chain).  Moving into it would
            # re-open a colour, so the plan is stale and is computed again.
            self._end_plan(success=False, stale=True)
            return
        progress = False
        forbidden = self._forbidden()
        order = sorted(target, key=lambda y: (col[y] != h, y))  # victim members first
        # (1) vertices whose planned colour is free in their neighbourhood
        for x in order:
            if self._remaining() <= 0:
                break
            b = target[x]
            if b not in st.classes:
                continue  # emptied during this update; the plan is re-made next update
            if st.cost_if_set(x, b) <= self._remaining() and all(col[y] != b for y in adj[x]):
                st.set_colour(x, b)
                del target[x]
                progress = True
                self.stats["planned_moves"] += 1
        # (2) blocked vertices: move x to its planned colour with an ejection chain
        if target and self._remaining() >= 2 and self.cfg.repair_mode == "ejection":
            retry = self._move_retry
            for x in [y for y in order if y in target]:
                if self._remaining() < 2:
                    break
                b = target[x]
                if col[x] == b:
                    del target[x]
                    continue
                info = retry.get(x)
                if info is not None and self.t < info[1]:
                    continue  # this move failed recently: wait
                if self._move_to(x, b, forbidden, self._remaining()):
                    del target[x]
                    retry.pop(x, None)
                    progress = True
                    self.stats["planned_moves"] += 1
                else:
                    f = info[0] + 1 if info is not None else 1
                    retry[x] = (f, self.t + self._backoff(f))
        if h not in st.classes:
            self._end_plan(success=True)
            return
        if progress:
            self.last_progress = self.t
        if not any(col[x] == h for x in target):
            # no planned move leaves class h any more but h is not empty:
            # vertices entered h after planning; try to clear it directly
            if self._remaining() > 0 and self._empty_class(h):
                self._end_plan(success=True)
                return
            if not target:
                self._end_plan(success=False)
                return
        if self.t - self.last_progress > self.cfg.max_stall:
            self._end_plan(success=False)
