"""Input readers with validation for the two dataset formats used.

* DIMACS ``.col`` files (Johnson and Trick, 1996): ``p edge n m`` header and
  ``e u v`` lines with 1-based vertex ids.  Duplicate edges (several files
  list each edge twice) are merged; self-loops are reported and dropped.
* Temporal edge lists ``u v t`` (one interaction per line, SNAP and
  SocioPatterns style).  Raw ids are mapped to 0..n-1 in order of first
  appearance; the events are returned sorted by time (stable).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

Edge = Tuple[int, int]


class InputFormatError(ValueError):
    pass


@dataclass
class StaticInstance:
    name: str
    n: int
    edges: List[Edge]
    header_edges: int
    self_loops: int
    duplicate_lines: int
    sha256: str


@dataclass
class TemporalInstance:
    name: str
    n: int
    events: List[Tuple[int, int, float]]
    raw_ids: List[str] = field(default_factory=list)
    self_loops: int = 0
    sha256: str = ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dimacs(path: str | Path) -> StaticInstance:
    path = Path(path)
    n = None
    header_m = None
    edges = set()
    loops = 0
    lines = 0
    with open(path, "r", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line[0] == "c":
                continue
            parts = line.split()
            if parts[0] == "p":
                if len(parts) < 4:
                    raise InputFormatError(f"{path.name}:{lineno}: malformed problem line")
                n, header_m = int(parts[2]), int(parts[3])
            elif parts[0] == "e":
                if n is None:
                    raise InputFormatError(f"{path.name}:{lineno}: edge before 'p' line")
                if len(parts) < 3:
                    raise InputFormatError(f"{path.name}:{lineno}: malformed edge line")
                u, v = int(parts[1]) - 1, int(parts[2]) - 1
                if not (0 <= u < n and 0 <= v < n):
                    raise InputFormatError(f"{path.name}:{lineno}: vertex id out of range")
                lines += 1
                if u == v:
                    loops += 1
                    continue
                edges.add((u, v) if u < v else (v, u))
    if n is None:
        raise InputFormatError(f"{path.name}: no 'p edge n m' line found")
    return StaticInstance(path.name.rsplit(".col", 1)[0], n, sorted(edges), header_m or 0,
                          loops, lines - loops - len(edges), sha256_of(path))


def read_temporal(path: str | Path, name: str | None = None) -> TemporalInstance:
    path = Path(path)
    ids: Dict[str, int] = {}
    raw_ids: List[str] = []
    events: List[Tuple[int, int, float]] = []
    loops = 0
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            parts = raw.split()
            if not parts or parts[0].startswith(("#", "%")):
                continue
            if len(parts) < 3:
                raise InputFormatError(f"{path.name}:{lineno}: expected 'u v t'")
            a, b = parts[0], parts[1]
            try:
                t = float(parts[2])
            except ValueError as exc:
                raise InputFormatError(f"{path.name}:{lineno}: bad timestamp") from exc
            for x in (a, b):
                if x not in ids:
                    ids[x] = len(raw_ids)
                    raw_ids.append(x)
            if a == b:
                loops += 1
            events.append((ids[a], ids[b], t))
    events.sort(key=lambda e: e[2])  # Python's sort is stable
    return TemporalInstance(name or path.stem, len(raw_ids), events, raw_ids, loops, sha256_of(path))
