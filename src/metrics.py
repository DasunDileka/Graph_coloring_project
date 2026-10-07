"""Per-update measurement and summary statistics for one experimental run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from proposed_algorithm import UpdateResult


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of non-negative values (0 = perfectly even)."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) / (n * x.sum())) - (n + 1.0) / n)


@dataclass
class RunRecorder:
    """Collects r_t, k_t and the update time for every update of a run."""

    recoloured: List[int] = field(default_factory=list)
    colours: List[int] = field(default_factory=list)
    time_ns: List[int] = field(default_factory=list)
    conflict: List[bool] = field(default_factory=list)
    repair: List[str] = field(default_factory=list)
    eliminated: List[int] = field(default_factory=list)

    def run_update(self, algorithm, op: str, u: int, v: int) -> UpdateResult:
        start = time.perf_counter_ns()
        res = algorithm.apply(op, u, v)
        self.time_ns.append(time.perf_counter_ns() - start)
        self.recoloured.append(res.recoloured)
        self.colours.append(res.colours)
        self.conflict.append(res.conflict)
        self.repair.append(res.repair)
        self.eliminated.append(res.eliminated)
        return res

    def summary(self, recolour_count_per_vertex: Sequence[int] | None = None) -> Dict[str, float]:
        r = np.asarray(self.recoloured, dtype=float)
        k = np.asarray(self.colours, dtype=float)
        t = np.asarray(self.time_ns, dtype=float) / 1000.0  # microseconds
        out: Dict[str, float] = {
            "updates": int(r.size),
            "mean_colours": float(k.mean()) if k.size else float("nan"),
            "max_colours": float(k.max()) if k.size else float("nan"),
            "final_colours": float(k[-1]) if k.size else float("nan"),
            "mean_recoloured": float(r.mean()) if r.size else float("nan"),
            "max_recoloured": float(r.max()) if r.size else float("nan"),
            "total_recoloured": float(r.sum()),
            "share_updates_recolouring": float((r > 0).mean()) if r.size else float("nan"),
            "conflict_rate": float(np.mean(self.conflict)) if self.conflict else float("nan"),
            "new_colour_rate": float(np.mean([x == "new_colour" for x in self.repair])) if self.repair else float("nan"),
            "eliminations": int(np.sum(self.eliminated)),
            "mean_time_us": float(t.mean()) if t.size else float("nan"),
            "median_time_us": float(np.median(t)) if t.size else float("nan"),
            "p95_time_us": float(np.percentile(t, 95)) if t.size else float("nan"),
            "max_time_us": float(t.max()) if t.size else float("nan"),
        }
        if recolour_count_per_vertex is not None:
            counts = np.asarray(recolour_count_per_vertex, dtype=float)
            out["max_recolourings_per_vertex"] = float(counts.max()) if counts.size else 0.0
            out["gini_recolourings"] = gini(counts)
        return out
