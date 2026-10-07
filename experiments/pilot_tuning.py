"""Pilot study that fixes KERB's effort parameters BEFORE the evaluation runs.

Only generated G(n, p) tuning graphs are used here; none of them is part of
the evaluation (DIMACS and real temporal networks), so the evaluation data
cannot influence the chosen parameter values.

Grid:     search_nodes in {1000, 4000} x plan_iters in {1000, 4000} x tabu_iters in {0, 300}
Fixed:    plan_every = 50, max_stall = 100, elimination_candidates = 2, max_backoff = 32
Budgets:  R in {3, 10}
Graphs:   G(120, 0.10), G(250, 0.05), G(200, 0.30), G(250, 0.50); graph seeds 1000 + s, s = 0, 1, 2
Start:    DSATUR improved by TabuCol (20,000 iterations, seed s), as in the evaluation
Workload: random churn, rho = 0.2, 2000 updates, workload seed s

(An earlier pilot of a previous KERB version, without the budgeted TabuCol
fallback and started from plain DSATUR, is kept in results/pilot/v1.)

Selection rule (written before the pilot was run): average the mean number
of colours over all pilot runs of a configuration; among the configurations
whose average is within 1% of the best average, choose the one with the
lowest mean update time.

Usage:  python experiments/pilot_tuning.py   (writes results/pilot/*.csv)
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from dynamic_graph import DynamicGraph  # noqa: E402
from metrics import RunRecorder  # noqa: E402
from proposed_algorithm import KERB, KerbConfig  # noqa: E402
from static_coloring import dsatur, tabucol_minimise  # noqa: E402
from workloads import random_churn  # noqa: E402

GRID = list(itertools.product([1000, 4000], [1000, 4000], [0, 300]))
BUDGETS = [3, 10]
GRAPHS = [(120, 0.10), (250, 0.05), (200, 0.30), (250, 0.50)]
SEEDS = [0, 1, 2]
UPDATES = 2000


_START = {}


def start_colouring(n, p, s, init):
    key = (n, p, s)
    if key not in _START:
        adj = DynamicGraph(n, init).adj
        _START[key] = tabucol_minimise(adj, dsatur(adj), 20000, seed=s)[0]
    return _START[key]


def run(task):
    (nodes, iters, tabu), R, (n, p), s = task
    G = nx.gnp_random_graph(n, p, seed=1000 + s)
    init, ops = random_churn(list(G.edges()), rho=0.2, updates=UPDATES, seed=s)
    col0 = start_colouring(n, p, s, init)
    g = DynamicGraph(n, init)
    alg = KERB(g, KerbConfig(budget=R, search_nodes=nodes, plan_iters=iters, tabu_iters=tabu, seed=s),
               initial_colouring=col0)
    rec = RunRecorder()
    t0 = time.perf_counter()
    for op, u, v in ops:
        res = rec.run_update(alg, op, u, v)
        assert res.recoloured <= R
    out = rec.summary(alg.recolour_count)
    out.update(search_nodes=nodes, plan_iters=iters, tabu_iters=tabu, R=R, graph=f"G({n},{p})",
               seed=s, k0=len(set(col0)), wall_s=time.perf_counter() - t0)
    return out


def main() -> None:
    # tasks of the same graph and seed run in sequence so the start colouring is computed once per worker
    tasks = [(cfg, R, gp, s) for gp in GRAPHS for s in SEEDS for cfg in GRID for R in BUDGETS]
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        rows = pool.map(run, tasks, chunksize=8)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pilot_raw.csv", index=False)
    summ = (df.groupby(["search_nodes", "plan_iters", "tabu_iters"])
              .agg(mean_colours=("mean_colours", "mean"), mean_recoloured=("mean_recoloured", "mean"),
                   mean_time_us=("mean_time_us", "mean"), p95_time_us=("p95_time_us", "mean"))
              .reset_index())
    best = summ["mean_colours"].min()
    summ["within_1pct"] = summ["mean_colours"] <= best * 1.01
    chosen = summ[summ["within_1pct"]].sort_values("mean_time_us").iloc[0]
    summ["chosen"] = ((summ["search_nodes"] == chosen["search_nodes"]) & (summ["plan_iters"] == chosen["plan_iters"])
                      & (summ["tabu_iters"] == chosen["tabu_iters"]))
    summ.to_csv(out / "pilot_summary.csv", index=False)
    print(summ.to_string(index=False))
    print(f"chosen: search_nodes={int(chosen['search_nodes'])}, plan_iters={int(chosen['plan_iters'])}, "
          f"tabu_iters={int(chosen['tabu_iters'])}")


if __name__ == "__main__":
    main()
