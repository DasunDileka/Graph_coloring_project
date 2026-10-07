"""Colouring state with colour classes and exact recolouring accounting.

A colouring is stored in two synchronised structures:

* ``col[v]``      -- the colour (an integer label) of vertex v, O(1) lookup;
* ``classes[c]``  -- the set of vertices that currently have colour c.

Colour labels are *identities* (for example radio channels), so emptying a
class never relabels the remaining vertices.  The number of colours in use
is ``len(classes)``.

During one dynamic update the state also records the colour every touched
vertex had at the start of the update.  ``changed`` is therefore always the
exact Hamming distance between the current colouring and the colouring at
the start of the update, i.e. the number of *recoloured* vertices r_t in the
formal model.  A vertex that is moved twice counts once, and a vertex that
is moved back to its original colour counts zero.  A journal of assignments
supports savepoints and rollback, so tentative moves cost nothing if they
are undone.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple


class ColouringState:
    __slots__ = ("col", "classes", "_orig", "changed", "_journal")

    def __init__(self, colours: Sequence[int]) -> None:
        self.col: List[int] = list(colours)
        self.classes: Dict[int, Set[int]] = {}
        for v, c in enumerate(self.col):
            if c < 0:
                raise ValueError(f"vertex {v} has invalid colour {c}")
            self.classes.setdefault(c, set()).add(v)
        self._orig: Dict[int, int] = {}
        self.changed: int = 0
        self._journal: List[Tuple[int, int]] = []

    # ------------------------------------------------------------ properties
    @property
    def num_colours(self) -> int:
        return len(self.classes)

    def new_label(self) -> int:
        """Smallest non-negative colour label that is not currently in use."""
        label = 0
        classes = self.classes
        while label in classes:
            label += 1
        return label

    # ------------------------------------------------------- update tracking
    def begin_update(self) -> None:
        """Start a new update: the current colouring becomes the reference."""
        self._orig.clear()
        self.changed = 0
        self._journal.clear()

    def original_colour(self, v: int) -> int:
        return self._orig.get(v, self.col[v])

    def is_unchanged(self, v: int) -> bool:
        """True if v still has the colour it had at the start of the update."""
        return self._orig.get(v, self.col[v]) == self.col[v]

    def cost_if_set(self, v: int, c: int) -> int:
        """Change in the recolouring count if v were given colour c (-1, 0 or +1)."""
        orig = self._orig.get(v, self.col[v])
        return (c != orig) - (self.col[v] != orig)

    def changed_vertices(self) -> List[int]:
        col = self.col
        return [v for v, o in self._orig.items() if col[v] != o]

    # --------------------------------------------------------------- changes
    def _move(self, v: int, old: int, new: int) -> None:
        cls = self.classes[old]
        cls.discard(v)
        if not cls:
            del self.classes[old]
        target = self.classes.get(new)
        if target is None:
            self.classes[new] = {v}
        else:
            target.add(v)
        self.col[v] = new

    def set_colour(self, v: int, c: int) -> None:
        """Give vertex v colour c, updating classes, journal and cost."""
        old = self.col[v]
        if old == c:
            return
        orig = self._orig.get(v)
        if orig is None:
            self._orig[v] = old
            orig = old
        self.changed += (c != orig) - (old != orig)
        self._journal.append((v, old))
        self._move(v, old, c)

    def savepoint(self) -> int:
        return len(self._journal)

    def rollback(self, savepoint: int) -> None:
        """Undo every assignment made after ``savepoint`` (no recolouring cost)."""
        journal = self._journal
        while len(journal) > savepoint:
            v, previous = journal.pop()
            current = self.col[v]
            orig = self._orig[v]
            self.changed += (previous != orig) - (current != orig)
            self._move(v, current, previous)

    # ---------------------------------------------------------------- helpers
    def neighbour_colour_counts(self, neighbours: Set[int]) -> Dict[int, int]:
        """Return {colour: number of neighbours with that colour}."""
        col = self.col
        counts: Dict[int, int] = {}
        for u in neighbours:
            c = col[u]
            counts[c] = counts.get(c, 0) + 1
        return counts
