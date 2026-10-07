"""Pre-registered experimental configuration.

Everything an evaluation run depends on is fixed here BEFORE the evaluation
runs (docs/preregistration.md explains each choice).  KERB's effort
parameters were chosen by the pilot study in experiments/pilot_tuning.py on
generated tuning graphs only, so no evaluation instance influenced them.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RESULTS = ROOT / "results"

# --------------------------------------------------------------- algorithms
# KERB defaults (src/proposed_algorithm.py) chosen by the pilot study
KERB_PARAMS = dict(search_nodes=1000, plan_iters=1000, tabu_iters=0, plan_every=50,
                   max_stall=100, elimination_candidates=2, max_backoff=32,
                   prefer_large_classes=True)
MAIN_BUDGET = 3                                  # R used in the main comparison (E2, E3)
BUDGET_SWEEP = [1, 2, 3, 5, 10, 25]              # E4 trade-off on DIMACS churn
BUDGET_SWEEP_TEMPORAL = [1, 2, 3, 5, 10]         # E4 trade-off on temporal streams
ABLATIONS = {                                    # E7, all at R = MAIN_BUDGET
    "KERB-kempe": dict(repair_mode="kempe"),
    "KERB-direct": dict(repair_mode="direct"),
    "KERB-noElim": dict(use_elimination=False),
    "KERB-noMig": dict(use_migration=False),
    "KERB-tabu": dict(tabu_iters=300),           # optional fallback that the pilot did not select
}
TABU_WARMSTART = dict(repair_iters=500, reduce_every=50, reduce_iters=2000)
DSR_SAMPLE_EVERY = {"churn": 10, "temporal": 50}  # DSATUR-Recompute cost sampled every s-th update

# ----------------------------------------------------------- E1: static
E1_INSTANCES = [
    "anna", "david", "homer", "huck", "jean", "games120", "miles250", "miles750", "miles1500",
    "myciel5", "myciel6", "myciel7", "queen8_8", "queen11_11", "queen13_13",
    "fpsol2.i.1", "inithx.i.1", "mulsol.i.1", "zeroin.i.1", "school1", "school1_nsh",
    "le450_5a", "le450_15a", "le450_15c", "le450_25a", "le450_25c",
    "DSJC125.1", "DSJC125.5", "DSJC125.9", "DSJC250.1", "DSJC250.5", "DSJC250.9",
    "DSJC500.1", "DSJC500.5", "DSJR500.1", "DSJR500.5", "flat300_28_0", "DSJC1000.1",
]
E1_RANDOM_GREEDY_SEEDS = list(range(10))
E1_TABUCOL_ITERS = 100_000          # TabuCol-minimise budget per instance (started from DSATUR)

# ------------------------------------------------- E2/E4/E7: DIMACS churn
E2_INSTANCES = [
    "DSJC125.5", "DSJC250.5", "DSJC500.1", "DSJR500.1", "flat300_28_0", "le450_15c",
    "le450_25c", "queen11_11", "myciel7", "school1", "inithx.i.1", "fpsol2.i.1",
    "miles750", "homer",
]
E2_SEEDS = [0, 1, 2, 3, 4]
E2_UPDATES = 5000
E2_RHO = 0.2                        # share of benchmark edges that starts in the reserve pool
E2_P_INSERT = 0.5
E2_INITIAL_TABU_ITERS = 20_000      # common start: DSATUR improved by TabuCol, shared by all methods
E2_CHECKPOINT_EVERY = 1000          # DSATUR colours and clique lower bound of G_t

# ------------------------------------------------ E3: real temporal streams
E3_STREAMS = {
    # name: (relative path under datasets/temporal, window W in seconds)
    "primary-school": ("primary-school.txt", 3600),
    "high-school-2013": ("high-school-2013.txt", 3600),
    "SFHH-conf": ("SFHH-conf.txt", 3600),
    "CollegeMsg": ("CollegeMsg.txt", 604800),
    "email-Eu-core-temporal": ("email-Eu-core-temporal.txt", 604800),
}
E3_SEEDS = [0, 1, 2]                # seed = vertex relabelling and algorithm seed

# ------------------------------------------------------- E5: scalability
E5_REAL = {
    "sx-superuser": ("sx-superuser.txt", 604800),
    "digg-friends": ("digg-friends.txt", 604800),
}
E5_TWS_PREFIX = 200_000             # Tabu-WarmStart is run on this prefix of each large stream
E5_DSR_SAMPLES = 20                 # DSATUR recomputations timed per large stream
E5_RGG_SIZES = [1_000, 10_000, 100_000, 1_000_000]
E5_RGG_MEAN_DEGREE = 8.0            # random geometric graph in the unit square (synthetic, scaling only)
E5_RGG_UPDATES = 20_000
E5_SEED = 0

# ------------------------------------------------------- statistics plan
ALPHA = 0.05                        # Wilcoxon signed-rank on instance means, Holm correction
PRIMARY_COMPARISONS = [             # (metric, KERB vs, alternative hypothesis for KERB - other)
    ("mean_colours", "FF-Repair", "less"),
    ("mean_recoloured", "Tabu-WarmStart", "less"),
    ("mean_recoloured", "DSATUR-Recompute", "less"),
    ("mean_colours", "DSATUR-Recompute", "less"),
]
SECONDARY_COMPARISONS = [
    ("mean_colours", "Tabu-WarmStart", "two-sided"),
    ("max_recoloured", "Tabu-WarmStart", "less"),
]

# --------------------------------------------------- coverage score (C1-C12)
# weights and pass rules are fixed here, before any evaluation result is seen
COVERAGE_CRITERIA = [
    # id, weight, short name, rule for 1 / 0.5 / 0
    ("C1", 3, "Valid colouring after every update",
     "1: no invalid colouring in any evaluated update; 0 otherwise"),
    ("C2", 3, "Hard per-update recolouring cap",
     "1: r_t <= R in every evaluated update; 0 otherwise"),
    ("C3", 3, "Few colours under the cap",
     "1: mean colours < FF-Repair in >= 80% of E2 runs AND mean over E2 instances within 10% of "
     "Tabu-WarmStart; 0.5: only the first part; 0: neither"),
    ("C4", 2, "Low disruption",
     "1: mean recolourings per update < Tabu-WarmStart and < DSATUR-Recompute in >= 80% of E2 runs; "
     "0.5: against one of the two; 0: neither"),
    ("C5", 2, "Fully dynamic (insertions and deletions)",
     "1: both handled incrementally and tested; 0.5: one of them; 0: neither"),
    ("C6", 1, "Fast updates",
     "1: median update time <= 1 ms and p95 <= 10 ms over E2 runs (R = 3); 0.5: median <= 10 ms; 0: otherwise"),
    ("C7", 2, "Scales to large graphs",
     "1: E5 RGG median update time grows < 3x while n grows 1000x and the 10^6 run completes; "
     "0.5: the 10^6 run completes; 0: it does not"),
    ("C8", 1, "Reproducible",
     "1: identical results for a fixed seed (unit test) and all scripts/configs published; 0.5: one; 0: neither"),
    ("C9", 1, "Feasible for every budget R >= 1",
     "1: R = 1 runs complete with no violation; 0 otherwise"),
    ("C10", 1, "Robust to its parameters",
     "1: every pilot configuration within 5% of the best pilot mean colours; 0.5: within 10%; 0: otherwise"),
    ("C11", 1, "Evidence on real dynamic data",
     "1: evaluated on >= 3 real temporal networks; 0.5: 1-2; 0: none"),
    ("C12", 1, "Verified implementation",
     "1: full test suite passes; 0.5: >= 90% of tests pass; 0: otherwise"),
]
COVERAGE_TARGET = 0.95              # the project's own target (not part of the assessment brief)
