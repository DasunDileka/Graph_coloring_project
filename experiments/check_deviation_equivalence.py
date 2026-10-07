"""Checks that the post-registration changes to KERB (docs/deviations.md, D1)
leave E2 results unchanged: re-runs E2 tasks with the current code and
compares the per-update traces with the stored ones."""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

STORED = ROOT / "results" / "raw" / "e2" / "traces"
TASKS = [a.split(":") for a in sys.argv[1:]]          # instance:label:seed


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    C.RESULTS = tmp
    import run_experiments as X
    for sub in ("e2/initial", "e2/runs", "e2/traces", "errors"):
        (X.RAW / sub).mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "results" / "raw" / "e2" / "initial").glob("*.json"):
        (X.RAW / "e2" / "initial" / f.name).write_bytes(f.read_bytes())
    variants = {label: (budget, params) for label, budget, params in X.e2_variants()}
    for name, label, seed in TASKS:
        budget, params = variants[label]
        X.e2_task((name, int(seed), label, budget, params))
        a = np.load(STORED / f"{name}__{label}__s{seed}.npz")
        b = np.load(X.RAW / "e2" / "traces" / f"{name}__{label}__s{seed}.npz")
        same = all(np.array_equal(a[k], b[k]) for k in ("colours", "recoloured", "repair"))
        print(f"{name} {label} s{seed}: identical traces = {same}")


if __name__ == "__main__":
    main()
