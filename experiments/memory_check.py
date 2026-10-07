"""Supplementary measurement (not pre-registered): memory of the data structures.

The E5 runs record the resident memory (RSS) of the worker process, which also
contains the interpreter, temporary edge lists and memory kept from earlier
tasks in the same process.  This script measures the Python allocations of the
structures themselves with ``tracemalloc``, on the same random geometric graphs
as E5 (mean degree 8, same seed and churn split):

* the dynamic graph (adjacency sets), per stored neighbour entry (2m entries);
* the extra memory of KERB and of FF-Repair after they are constructed on that
  graph from the same initial DSATUR colouring, per vertex;
* KERB and FF-Repair again after 2,000 churn updates, per vertex, and the peak
  during those updates (which includes the temporary copies made by planning).

Writes results/summary/memory_check.json.  Run: python3 experiments/memory_check.py
"""

from __future__ import annotations

import gc
import json
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import config as C  # noqa: E402
from benchmark_runner import make_algorithm  # noqa: E402
from dynamic_graph import DynamicGraph  # noqa: E402
from run_experiments import rgg_edges  # noqa: E402
from static_coloring import dsatur  # noqa: E402
from workloads import random_churn  # noqa: E402

UPDATES = 2000


def traced() -> int:
    gc.collect()
    return tracemalloc.get_traced_memory()[0]


def measure(n: int) -> dict:
    edges = rgg_edges(n, C.E5_RGG_MEAN_DEGREE, C.E5_SEED)
    initial_edges, ops = random_churn(edges, rho=C.E2_RHO, updates=UPDATES, seed=C.E5_SEED)
    del edges
    m = len(initial_edges)
    tracemalloc.start()
    base = traced()
    g = DynamicGraph(n, initial_edges)
    graph_bytes = traced() - base
    initial = dsatur(g.adj)
    out = {"n": n, "m": m, "graph_bytes": graph_bytes, "graph_bytes_per_entry": graph_bytes / (2 * m)}
    for label in ("KERB-R3", "FF-Repair"):
        g = DynamicGraph(n, initial_edges)
        before = traced()
        budget = C.MAIN_BUDGET if label.startswith("KERB") else None
        alg = make_algorithm(label, g, list(initial), C.E5_SEED, budget=budget, kerb_params=C.KERB_PARAMS)
        built = traced() - before
        tracemalloc.reset_peak()
        for op in ops:
            alg.apply(op[0], op[1], op[2])
        peak = tracemalloc.get_traced_memory()[1] - before
        after = traced() - before
        key = "kerb" if label.startswith("KERB") else "ff"
        out[f"{key}_bytes_per_vertex_start"] = built / n
        out[f"{key}_bytes_per_vertex_after_updates"] = after / n
        out[f"{key}_peak_bytes_per_vertex_during_updates"] = peak / n
        del alg, g
    tracemalloc.stop()
    return out


def main() -> None:
    res = {"python": sys.version.split()[0], "updates": UPDATES,
           "runs": [measure(n) for n in (10_000, 100_000)]}
    path = ROOT / "results" / "summary" / "memory_check.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
