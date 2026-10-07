# Pre-registration of the evaluation

Written on 3 October 2026, before any evaluation run (E1 to E7) was started. The machine-readable version of every value below is `experiments/config.py`; the SHA-256 hashes of both files are stored in `results/raw/preregistration.json` when the runs start, so later edits would be visible.

## 1. What was decided before this point

KERB's design and its effort parameters were developed on generated G(n, p) tuning graphs only (`experiments/pilot_tuning.py`, results in `results/pilot/`). The pilot's selection rule was written into the pilot script before it ran: among configurations whose mean colour count is within 1% of the best, choose the fastest. It chose `search_nodes = 1000`, `plan_iters = 1000`, `tabu_iters = 0` (the budgeted TabuCol fallback is therefore off by default; E7 measures what it would add). No DIMACS instance or temporal network influenced these values.

## 2. Research questions and hypotheses

* **RQ1** For a fixed budget R = 3, how many colours does KERB use compared with first-fit repair, warm-started TabuCol and DSATUR recomputation?
  * H1: KERB uses fewer colours than FF-Repair (one-sided Wilcoxon signed-rank test on instance means, alpha = 0.05, Holm correction over the primary tests).
  * H2: KERB recolours fewer vertices per update on average than Tabu-WarmStart and than DSATUR-Recompute (one-sided).
  * H3: KERB uses fewer colours than DSATUR-Recompute (one-sided).
  * Secondary, reported without a directional claim: colours of KERB versus Tabu-WarmStart (two-sided), and the maximum recolouring per update versus Tabu-WarmStart.
* **RQ2** How does the colour/recolouring trade-off change with R? Descriptive: mean colours and recolourings for R in {1, 2, 3, 5, 10, 25} (DIMACS churn) and R in {1, 2, 3, 5, 10} (temporal streams); Spearman correlation between R and mean colours per instance.
* **RQ3** Does measured update time follow the per-update complexity derived in the report? Descriptive: update time against degree and against n (E5), log-log slope.

## 3. Data

* **E1 static:** 38 DIMACS instances listed in `config.E1_INSTANCES` (all instance families; the largest dense graphs DSJC500.9, DSJC1000.5/.9 and flat1000 are excluded for run time).
* **E2/E4/E7 churn:** 14 DIMACS instances (`config.E2_INSTANCES`) covering random, geometric, flat, Leighton, queen, Mycielski, timetabling, register-allocation and book graphs. Workload: random churn with rho = 0.2 (20% of the benchmark edges start in a reserve pool), 5,000 single-edge updates, insert/delete probability 0.5, seeds 0 to 4. Every method starts from the same colouring: DSATUR improved offline by TabuCol-minimise (20,000 iterations, seed = run seed).
* **E3 real temporal streams:** primary-school, high-school-2013, SFHH-conf (window W = 1 hour) and CollegeMsg, email-Eu-core-temporal (W = 1 week), converted with a sliding window (an edge exists while its pair interacted within the last W seconds). Seeds 0 to 2 permute the vertex ids. The graph starts empty.
* **E5 scalability:** sx-superuser and digg-friends (W = 1 week, full streams for KERB and FF-Repair, first 200,000 updates for Tabu-WarmStart, 20 timed recomputations for DSATUR-Recompute) and random geometric graphs with mean degree 8 and n in {10^3, 10^4, 10^5, 10^6} (synthetic, used only to control n; 20,000 churn updates).

## 4. Methods compared

KERB with R = 3 (main), R sweep, and ablations at R = 3: Kempe-chain repair, direct-move-only repair, no class elimination (C3), no migration (C4), budgeted TabuCol on. Baselines: FF-Repair (one-vertex first-fit repair), Tabu-WarmStart (TabuCol repair with 500 iterations plus a k-1 attempt with 2,000 iterations every 50 updates, unlimited recolouring; modelled on Hardy, Lewis and Thompson, 2018), DSATUR-Recompute (DSATUR on every graph, labels matched to the previous colouring by the Hungarian method; its exact cost measured at every 10th update on churn and every 50th on temporal streams).

## 5. Measurements and validity checks

Per update: colours k_t, recoloured vertices r_t (Hamming distance), update time (perf_counter_ns). Per run: mean, maximum and final colours; mean, maximum and total recolourings; share of updates that recolour; conflict and new-colour rates; mean, median, 95th percentile and maximum update time; Gini coefficient and maximum of recolourings per vertex; peak resident memory (E5). Every update is checked independently of the algorithm (properness around changed vertices, exact Hamming distance, budget), with full-graph checks at checkpoints. A single violation fails criterion C1 or C2.

## 6. Analysis plan

Instance-level means over seeds are compared with paired Wilcoxon signed-rank tests across the 14 E2 instances (and, separately, the 5 E3 streams, where n = 5 limits power; those tests are reported as descriptive). Effect sizes: median paired difference and matched-pairs rank-biserial correlation. Holm correction over the four primary tests. Superiority is claimed only when the corrected p-value is below 0.05.

## 7. Coverage score

The twelve problem-derived criteria C1 to C12, their weights (total 21) and their pass rules are listed in `config.COVERAGE_CRITERIA`. Coverage = sum(weight x score) / 21 with scores 1, 0.5 or 0. The brief's target of 95% is a target, not a promise; the computed value is reported whatever it is.

## 8. Deviations

Any change after this point is listed in `docs/deviations.md` with its reason.
