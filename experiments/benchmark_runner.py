"""Runs one dynamic colouring algorithm on one update sequence and measures it.

The harness checks every update independently of the algorithm's own
bookkeeping:

* properness -- if the colouring was proper before an update, a new conflict
  can only involve the inserted edge or a vertex whose colour changed, so the
  neighbourhoods of those vertices are checked after every update ("local"
  mode).  The whole graph is checked at every checkpoint and at the end;
* recolouring -- the Hamming distance between the colourings before and after
  the update is recomputed from a copy of the colouring and compared with the
  value the algorithm reports, and with the budget R when one is set;
* colours -- the reported k_t is compared with the number of distinct colours
  at every checkpoint.

In "sample" mode (used for graphs with up to 10^6 vertices) the full Hamming
check is done on every 100th update and the local check on the vertices the
algorithm reports as changed; everything else is as above.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from baseline_algorithms import (DsaturRecompute, FirstFitRepair, TabuWarmStart, hamming,
                                 match_labels)
from dynamic_graph import DynamicGraph
from metrics import RunRecorder
from proposed_algorithm import INSERT, KERB, KerbConfig
from static_coloring import dsatur, greedy_clique_lower_bound
from validation import conflicting_edges

REPAIR_CODES = {"none": 0, "direct": 1, "chain": 2, "kempe": 3, "tabu": 4, "direct_victim": 5,
                "new_colour": 6, "recompute": 7}


class ValidationError(AssertionError):
    pass


def make_algorithm(name: str, graph: DynamicGraph, initial: Optional[Sequence[int]],
                   seed: int, budget: Optional[int] = None, kerb_params: Optional[Dict] = None,
                   tws_params: Optional[Dict] = None):
    if name.startswith("KERB"):
        params = dict(kerb_params or {})
        return KERB(graph, KerbConfig(budget=budget, seed=seed, **params), initial_colouring=initial)
    if name == "FF-Repair":
        return FirstFitRepair(graph, initial_colouring=initial)
    if name == "Tabu-WarmStart":
        return TabuWarmStart(graph, initial_colouring=initial, seed=seed, **(tws_params or {}))
    if name == "DSATUR-Recompute":
        return DsaturRecompute(graph, initial_colouring=initial)
    raise ValueError(f"unknown algorithm {name}")


def _check_local(adj, col, vertices, context):
    for x in vertices:
        cx = col[x]
        for y in adj[x]:
            if col[y] == cx:
                raise ValidationError(f"{context}: conflict on edge ({x}, {y})")


def run_stream(alg, graph: DynamicGraph, ops: Sequence[Tuple], budget: Optional[int] = None,
               validate: str = "local", checkpoint_every: int = 0,
               checkpoint_fn: Optional[Callable[[int, DynamicGraph, Sequence[int]], Dict]] = None
               ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], List[Dict]]:
    """Apply ``ops`` with ``alg`` and return (summary, per-update traces, checkpoints)."""
    adj = graph.adj
    n = graph.n
    if conflicting_edges(adj, alg.colouring):
        raise ValidationError("initial colouring is not proper")
    rec = RunRecorder()
    checkpoints: List[Dict] = []
    full_every = 1 if validate == "local" else 100
    for t, op in enumerate(ops, 1):
        _, u, v = op[0], op[1], op[2]
        full = validate != "none" and t % full_every == 0
        before = list(alg.colouring) if full else None
        res = rec.run_update(alg, op[0], u, v)
        col = alg.colouring
        if validate != "none":
            if full:
                diff = [x for x in range(n) if before[x] != col[x]]
                if len(diff) != res.recoloured:
                    raise ValidationError(f"update {t}: reported r={res.recoloured}, measured {len(diff)}")
            else:
                diff = alg.state.changed_vertices()
            if budget is not None and len(diff) > budget:
                raise ValidationError(f"update {t}: {len(diff)} recoloured > R={budget}")
            _check_local(adj, col, list(diff) + ([u, v] if op[0] == INSERT else []), f"update {t}")
        if checkpoint_every and t % checkpoint_every == 0:
            if validate != "none" and conflicting_edges(adj, col):
                raise ValidationError(f"checkpoint {t}: colouring not proper")
            k = len(set(col))
            if k != res.colours:
                raise ValidationError(f"checkpoint {t}: reported k={res.colours}, measured {k}")
            row = {"t": t, "colours": k, "edges": graph.m, "max_degree": graph.max_degree()}
            if checkpoint_fn is not None:
                row.update(checkpoint_fn(t, graph, col))
            checkpoints.append(row)
    if validate != "none" and conflicting_edges(adj, alg.colouring):
        raise ValidationError("final colouring is not proper")
    summary = rec.summary(alg.recolour_count)
    traces = {
        "colours": np.asarray(rec.colours, dtype=np.int32),
        "recoloured": np.asarray(rec.recoloured, dtype=np.int32),
        "time_us": (np.asarray(rec.time_ns, dtype=np.float64) / 1000.0).astype(np.float32),
        "repair": np.asarray([REPAIR_CODES.get(x, 99) for x in rec.repair], dtype=np.int8),
    }
    return summary, traces, checkpoints


def dsatur_checkpoint(t: int, graph: DynamicGraph, col: Sequence[int]) -> Dict:
    """Context at a checkpoint: DSATUR colours and a clique lower bound of G_t."""
    t0 = time.perf_counter()
    k_dsatur = len(set(dsatur(graph.adj)))
    dt = time.perf_counter() - t0
    return {"dsatur_colours": k_dsatur, "dsatur_time_s": dt,
            "clique_lb": greedy_clique_lower_bound(graph.adj, 64)}


def run_dsatur_recompute_sampled(n: int, initial_edges, ops, sample_every: int,
                                 max_samples: Optional[int] = None) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Exact per-update cost of DSATUR-Recompute at every ``sample_every``-th update.

    The graph is replayed; at a sampled update t the colourings DSATUR(G_{t-1})
    and DSATUR(G_t) are computed, the second is relabelled to agree with the
    first as much as possible (Hungarian matching) and the Hamming distance is
    recorded.  The recompute time is the time for DSATUR(G_t) plus matching.
    """
    g = DynamicGraph(n, initial_edges)
    rec_r, rec_k, rec_t, ts = [], [], [], []
    for t, op in enumerate(ops, 1):
        sample = t % sample_every == 0 and (max_samples is None or len(ts) < max_samples)
        before = [set(a) for a in g.adj] if sample else None
        if op[0] == INSERT:
            g.add_edge(op[1], op[2])
        else:
            g.remove_edge(op[1], op[2])
        if sample:
            previous = dsatur(before)                     # the colouring DSR held before update t
            t0 = time.perf_counter()
            new = match_labels(previous, dsatur(g.adj))   # what DSR computes at update t
            dt = time.perf_counter() - t0
            rec_r.append(hamming(previous, new))
            rec_k.append(len(set(new)))
            rec_t.append(dt * 1e6)
            ts.append(t)
    r = np.asarray(rec_r, dtype=float)
    k = np.asarray(rec_k, dtype=float)
    tt = np.asarray(rec_t, dtype=float)
    summary = {
        "updates": len(ops), "samples": len(ts),
        "mean_colours": float(k.mean()) if k.size else float("nan"),
        "max_colours": float(k.max()) if k.size else float("nan"),
        "final_colours": float(k[-1]) if k.size else float("nan"),
        "mean_recoloured": float(r.mean()) if r.size else float("nan"),
        "max_recoloured": float(r.max()) if r.size else float("nan"),
        "total_recoloured": float("nan"),
        "share_updates_recolouring": float((r > 0).mean()) if r.size else float("nan"),
        "mean_time_us": float(tt.mean()) if tt.size else float("nan"),
        "median_time_us": float(np.median(tt)) if tt.size else float("nan"),
        "p95_time_us": float(np.percentile(tt, 95)) if tt.size else float("nan"),
        "max_time_us": float(tt.max()) if tt.size else float("nan"),
    }
    traces = {"t": np.asarray(ts, dtype=np.int32), "colours": k.astype(np.int32),
              "recoloured": r.astype(np.int32), "time_us": tt.astype(np.float32)}
    return summary, traces
