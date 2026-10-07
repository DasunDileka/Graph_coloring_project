# KERB: dynamic graph colouring with a recolouring budget

Code, data and results for the CMP 7002 PRAC 1 project "Dynamic graph colouring under a per-update recolouring budget" (Dasun Perera, Cardiff Metropolitan University / ICBT).

**Problem.** A graph changes one edge at a time. After every change the vertex colouring must be proper again, and at most R vertices may change colour (each recolouring is a real cost, for example a radio channel switch). Within that limit, keep the number of colours as small as possible.

**KERB** (Kempe/ejection-chain Repair with class Elimination under a recolouring Budget) combines a cheapest-first ejection-chain repair, transactional removal of small colour classes and a migration plan computed by TabuCol and executed a few vertices per update. It never exceeds R recolourings in an update, and its colour count never exceeds max(k_0, Δ_max + 1).

Repository: [GitHub repository URL to be inserted after upload]

## Layout

```
src/
  dynamic_graph.py        adjacency-set graph with edge insertion and deletion
  coloring_state.py       colouring with colour classes, exact recolouring count, savepoints
  proposed_algorithm.py   KERB (components C1-C4)
  baseline_algorithms.py  FF-Repair, Tabu-WarmStart, DSATUR-Recompute
  static_coloring.py      greedy, Welsh-Powell, smallest-last, DSATUR, RLF, TabuCol (+ budget cap)
  workloads.py            random churn and sliding-window update streams
  graph_loader.py         DIMACS and temporal edge-list readers with validation
  graph_features.py       structural features of a graph
  validation.py           properness and budget checks
  metrics.py              per-update recording and run summaries
experiments/
  config.py               every pre-registered parameter
  pilot_tuning.py         parameter pilot on generated tuning graphs only
  benchmark_runner.py     validated run of one algorithm on one stream
  run_experiments.py      E1 static, E2/E4/E7 churn, E3 temporal, E5 scalability
  test_case_table.py      E6 executed test-case table
  analyse_results.py      summary tables, statistics, coverage score
  plot_results.py         figures (called by analyse_results.py)
  check_deviation_equivalence.py  re-runs E2 tasks to check deviation D1
  memory_check.py         supplementary memory measurement (deviation D3)
  stream_stats.py         descriptive statistics of the real streams (deviation D4)
  fallback_overhead.py    diagnostic of the wasted fallback work (deviation D5)
tests/                    pytest suites (unit, edge cases, invalid input)
datasets/                 DIMACS instances and temporal streams with provenance and checksums
results/                  pilot/, raw/ (one JSON + trace per run), summary/ (CSV tables)
figures/                  result figures, in the report or referred to from it
docs/                     pre-registration, deviations and the source check log
```

## Reproducing the results

```bash
python -m pip install -r requirements.txt
python datasets/fetch_large_streams.py          # only needed for E5 (two large files)
python -m pytest -q                              # 79 tests
python experiments/pilot_tuning.py               # parameter pilot (tuning graphs only)
python experiments/run_experiments.py e1 --workers 1
python experiments/run_experiments.py e2 --workers 3   # E2, E4 and E7
python experiments/run_experiments.py e3 --workers 3
python experiments/run_experiments.py e5 --workers 1
python experiments/test_case_table.py
python experiments/analyse_results.py            # tables in results/summary, figures in figures/
python experiments/memory_check.py               # supplementary measurements added after the
python experiments/stream_stats.py               # evaluation (deviations D3 to D5 in
python experiments/fallback_overhead.py          # docs/deviations.md)
```

Runs are deterministic for a given seed. Update times depend on the machine; the reported times were measured on a 4-core Intel Xeon virtual machine at 2.8 GHz with Python 3.11.15, with up to four runs in parallel (E5 ran alone).

## Using KERB

```python
from dynamic_graph import DynamicGraph
from proposed_algorithm import KERB, KerbConfig

g = DynamicGraph(5, [(0, 1), (1, 2)])
kerb = KERB(g, KerbConfig(budget=3))
result = kerb.insert_edge(0, 2)      # also: kerb.delete_edge(u, v)
print(result.recoloured, result.colours, kerb.colouring)
```

## Data

See `datasets/PROVENANCE.md`. DIMACS instances (Johnson and Trick, 1996) and temporal streams from SocioPatterns and SNAP, copied from pinned GitHub mirrors and checked against the published vertex and edge counts.

## Licence

Code: to be chosen by the author before the repository is made public (a permissive licence such as MIT is common for coursework). Data: the licences of the original publishers apply (see `datasets/PROVENANCE.md`).
