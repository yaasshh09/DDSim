"""The assembled form every discretization in this package returns.

One residual vector and one Jacobian in COO triplet form, nothing else. The
type is shared rather than duplicated per equation so that boundary conditions
are written once: an ohmic contact pins psi in the Poisson system and pins n
and p in the two continuity systems, and that is the same row replacement three
times over.

COO because assembly naturally emits one triplet per contribution and expects
duplicates. solve/linear.py sums them on the conversion to CSC.

This satisfies the structural Assembly protocol in solve/newton.py without
either module importing the other, which is what keeps solve/ free of
semiconductor knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SparseAssembly:
    """A residual vector and a Jacobian in COO form, ready for solve/linear.py."""

    residual: npt.NDArray[np.float64]
    """F(x) [1], one entry per unknown."""

    rows: npt.NDArray[np.int64]
    """Jacobian row indices."""

    cols: npt.NDArray[np.int64]
    """Jacobian column indices."""

    values: npt.NDArray[np.float64]
    """Jacobian values, same length as rows and cols."""

    shape: tuple[int, int]
    """Jacobian shape, (n_unknowns, n_unknowns)."""
