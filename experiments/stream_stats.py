"""Descriptive statistics of the real temporal networks (not an experiment).

For each interaction file: number of interactions, time span from the first to the last
time stamp, and the number of edge updates per day of the derived sliding-window stream
(the update counts come from the E3 and E5 summaries).  Used in the problem chapter of the
report to show how often real conflict graphs change.

Writes results/summary/stream_stats.json.  Run: python3 experiments/stream_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_loader import read_temporal  # noqa: E402

STREAMS = ["primary-school", "high-school-2013", "SFHH-conf", "CollegeMsg", "email-Eu-core-temporal",
           "sx-superuser", "digg-friends"]


def main() -> None:
    e3 = pd.read_csv(ROOT / "results" / "summary" / "e3_by_stream.csv").drop_duplicates("instance").set_index("instance")
    e5 = pd.read_csv(ROOT / "results" / "summary" / "e5_real.csv")
    out = {}
    for name in STREAMS:
        inst = read_temporal(ROOT / "datasets" / "temporal" / f"{name}.txt", name)
        ts = [e[2] for e in inst.events]
        span_days = (max(ts) - min(ts)) / 86400.0
        if name in e3.index:
            updates = int(e3.loc[name].updates)
        else:
            updates = int(e5[e5.workload.str.startswith(name)].iloc[0].updates_total)
        out[name] = {"n": inst.n, "interactions": len(inst.events), "span_days": span_days,
                     "updates": updates, "updates_per_day": updates / span_days if span_days > 0 else None}
    path = ROOT / "results" / "summary" / "stream_stats.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
