"""Sparse direct solver, a thin wrapper over SuperLU.

Nothing semiconductor specific belongs in this package, ever. solve/ takes a
matrix and a right hand side and returns a solution. That is what lets
continuation.py be lifted into the SPICE layer without modification.

Assembly convention: build in COO, because assembly naturally emits one triplet
per contribution and duplicates are expected, then convert to CSC once and
factorize. The conversion sums duplicates for us.

On reusing the factorization across Newton steps
------------------------------------------------
docs/02-numerics.md says to reuse the symbolic factorization between Newton
iterations, and calls it a significant speedup for free. Neither half of that
holds with scipy, and it is worth writing down exactly why so that nobody
tries it again.

scipy.sparse.linalg.splu takes permc_spec as a string and returns perm_c. There
is no way to hand a previously computed symbolic factorization back in, and no
way to pass a precomputed permutation. So a true symbolic and numeric split is
simply not available.

The obvious workaround is to reuse the fill reducing ordering: keep perm_c from
the first factorization, then permute the columns yourself and ask for NATURAL
ordering. That was implemented and benchmarked, and it is much worse:

    1D tridiagonal    n = 3000    fresh   1.41 ms   reused    1.35 ms
    1D coupled        n = 9000    fresh   4.11 ms   reused    3.34 ms
    2D 5-point    100 x 100       fresh  23.24 ms   reused  352.40 ms
    2D 5-point    200 x 200       fresh 134.14 ms   reused 6146.03 ms

The column gather itself costs 0.6 ms, so it is not the permutation. The
factorization is what blows up: on the 100 x 100 case, COLAMD produces 645,750
nonzeros in L and U while the pre-permuted NATURAL run produces 3,933,424, a
factor of 6.1 more fill. SuperLU's COLAMD path does column elimination tree
postordering that the NATURAL path skips, so perm_c on its own does not
reproduce the ordering SuperLU actually eliminated with.

Conclusion: nothing about the factorization is reusable through scipy, so this
class does not pretend otherwise. It always factorizes fresh with COLAMD.

What it does keep is the sparsity pattern fingerprint, which is free and
genuinely useful. It tells the caller whether the pattern changed, which is a
real question during continuation when contacts switch or a mesh is refined,
and it is the hook a backend with a real symbolic split would use. UMFPACK
(scikit-umfpack) and KLU both expose one, and swapping either in is a change
inside this file only.

What the pattern does buy, short of a symbolic factorization
------------------------------------------------------------
The COO to CSC conversion is not free, and unlike the factorization it is
genuinely reusable. Converting means sorting the triplets into column major
order and summing duplicates, and both of those depend only on (rows, cols).
Across Newton steps only the values move, so the sort permutation and the
duplicate grouping are computed once and replayed as a gather plus a segmented
sum on every later call. Measured on the Phase 2 diode, that is about a fifth
of the wall clock of a bias sweep, spent inside scipy's construction and
validation path rather than in any arithmetic.

The replay is the same conversion, not an approximation of it. Duplicates are
accumulated in the order they appear in the triplet arrays, which is what
scipy's canonical form does, so the summed values agree bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import SuperLU, splu

Number = TypeVar("Number", np.float64, np.complex128)
"""The dtype of a right hand side.

Constrained to the two that occur, matching discretize/coupled.py. A DC solve
is float64; complex128 appears only in the Phase 4 AC solve, where the system
is (J_dc + i*omega*M). Writing it as a TypeVar rather than a union is what
keeps a float64 solve typed as returning float64.
"""


@dataclass(frozen=True)
class _CSCPattern:
    """The part of a COO to CSC conversion that depends only on the pattern."""

    rows: npt.NDArray[np.int64]
    """The triplet row indices this pattern was built from."""

    cols: npt.NDArray[np.int64]
    """The triplet column indices this pattern was built from."""

    shape: tuple[int, int]
    """Shape of the matrix."""

    order: npt.NDArray[np.intp]
    """Permutation putting the triplets into column major order."""

    group: npt.NDArray[np.intp]
    """For each triplet in `order`, which CSC entry it lands in."""

    n_entries: int
    """Number of distinct (row, col) pairs, the length of the CSC data."""

    def matches(
        self,
        rows: npt.NDArray[np.integer],
        cols: npt.NDArray[np.integer],
        shape: tuple[int, int],
    ) -> bool:
        """Whether these triplets have the pattern this was built from."""
        return (
            shape == self.shape
            and np.array_equal(rows, self.rows)
            and np.array_equal(cols, self.cols)
        )

    def data(self, values: npt.NDArray[Number]) -> npt.NDArray[Number]:
        """The CSC data array for these values, duplicates summed.

        Real values go through np.bincount, which is the fast path and the one
        every Newton step takes. Complex values cannot: bincount refuses a
        complex weights array outright rather than silently dropping the
        imaginary part, which is the better of the two failures but still
        leaves the AC solve with nowhere to go. np.add.at does the same
        accumulation for any dtype.
        """
        gathered = values[self.order]

        if np.iscomplexobj(gathered):
            summed = np.zeros(self.n_entries, dtype=gathered.dtype)
            np.add.at(summed, self.group, gathered)
            return cast("npt.NDArray[Number]", summed)

        # The branch above has already ruled out complex, so the weights are
        # real here. The stubs cannot see that, hence the cast rather than a
        # runtime conversion.
        return cast(
            "npt.NDArray[Number]",
            np.bincount(
                self.group,
                weights=cast("npt.NDArray[np.float64]", gathered),
                minlength=self.n_entries,
            ),
        )


def _build_pattern(
    rows: npt.NDArray[np.integer],
    cols: npt.NDArray[np.integer],
    shape: tuple[int, int],
) -> tuple[_CSCPattern, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Work out the CSC structure of a set of triplets.

    Returns the replayable pattern together with the CSC indices and indptr.
    lexsort with cols last makes columns the primary key and rows the
    secondary one, which is exactly column major order, and it is stable, so
    duplicates keep the order they had in the triplet arrays.
    """
    n_columns = shape[1]
    order = np.lexsort((rows, cols))
    sorted_rows = rows[order]
    sorted_cols = cols[order]

    # A triplet starts a new CSC entry unless it repeats the (row, col) before
    # it. Counting the starts gives each triplet the entry it belongs to.
    starts = np.empty(order.size, dtype=bool)
    starts[0] = True
    starts[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (
        sorted_cols[1:] != sorted_cols[:-1]
    )
    group = np.cumsum(starts) - 1

    indices = np.ascontiguousarray(sorted_rows[starts], dtype=np.int32)
    entry_columns = sorted_cols[starts]

    indptr = np.zeros(n_columns + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(entry_columns, minlength=n_columns))

    pattern = _CSCPattern(
        rows=np.array(rows, dtype=np.int64, copy=True),
        cols=np.array(cols, dtype=np.int64, copy=True),
        shape=shape,
        order=order,
        group=np.asarray(group, dtype=np.intp),
        n_entries=int(indices.size),
    )
    return pattern, indices, indptr


class SparseLU:
    """LU factorization of a sparse square matrix.

    Typical Newton use, where the pattern never changes and only the values do:

        solver = SparseLU()
        for step in range(max_steps):
            solver.factorize(rows, cols, jacobian_values, shape)
            delta = solver.solve(-residual)
    """

    def __init__(self) -> None:
        self._lu: SuperLU | None = None
        self._pattern: _CSCPattern | None = None
        self._matrix: sp.csc_matrix | None = None
        self._pattern_unchanged = False
        self._size = 0

    @property
    def pattern_unchanged(self) -> bool:
        """Whether the last factorize saw the same sparsity pattern as before.

        False on the first factorization. Informational only, it never changes
        what the solver does.
        """
        return self._pattern_unchanged

    @property
    def size(self) -> int:
        """Dimension of the factorized matrix."""
        return self._size

    @property
    def fill_nnz(self) -> int:
        """Nonzeros in L plus U, a direct measure of ordering quality."""
        if self._lu is None:
            raise RuntimeError("no factorization available, call factorize first")
        return int(self._lu.L.nnz + self._lu.U.nnz)

    def factorize(
        self,
        rows: npt.NDArray[np.integer],
        cols: npt.NDArray[np.integer],
        values: npt.NDArray[np.floating],
        shape: tuple[int, int],
    ) -> None:
        """Assemble COO triplets into CSC and factorize with COLAMD ordering.

        Duplicate (row, col) entries are summed, which is what assembly wants.
        """
        if shape[0] != shape[1]:
            raise ValueError(f"matrix must be square, got shape {shape}")

        shape = (int(shape[0]), int(shape[1]))

        # The dtype is whatever came in, so that the Phase 4 AC solve can hand
        # this a complex system. Anything not already floating or complex is
        # promoted, which keeps an integer Jacobian working as it always did.
        entries = np.asarray(values)
        if not np.issubdtype(entries.dtype, np.inexact):
            entries = entries.astype(np.float64)

        pattern = self._pattern
        unchanged = (
            pattern is not None
            and self._matrix is not None
            # The dtype is part of what counts as unchanged. The replay path
            # writes into the matrix already built for this pattern, and
            # writing complex values into a float64 buffer discards the
            # imaginary part with only a warning. A DC solve followed by an AC
            # solve on the same solver is exactly that sequence.
            and self._matrix.dtype == entries.dtype
            and pattern.matches(rows, cols, shape)
        )

        if unchanged:
            # Same structure, new numbers. Replay the conversion into the
            # matrix already built for this pattern, so nothing is sorted,
            # allocated or validated a second time.
            assert pattern is not None and self._matrix is not None
            matrix = self._matrix
            matrix.data[:] = pattern.data(entries)
        elif entries.size == 0:
            # No triplets at all. There is nothing to cache and the matrix is
            # singular by construction, so let scipy build it and say so.
            matrix = sp.coo_matrix(
                (entries, (rows, cols)), shape=shape
            ).tocsc()
            pattern = None
        else:
            pattern, indices, indptr = _build_pattern(
                np.asarray(rows), np.asarray(cols), shape
            )
            matrix = sp.csc_matrix(
                (pattern.data(entries), indices, indptr), shape=shape
            )
            # lexsort put the rows in ascending order within every column, so
            # SuperLU can be told not to check.
            matrix.has_sorted_indices = True

        try:
            self._lu = splu(matrix, permc_spec="COLAMD")
        except RuntimeError as error:
            self._lu = None
            raise RuntimeError(
                f"LU factorization failed, the matrix is singular or nearly so: {error}"
            ) from error

        self._pattern_unchanged = unchanged
        self._pattern = pattern
        self._matrix = matrix
        self._size = shape[0]

    def solve(self, b: npt.NDArray[Number]) -> npt.NDArray[Number]:
        """Solve A x = b using the stored factorization.

        Dtype preserving in the same sense as factorize: a real system returns
        float64 exactly as before, and a complex one returns complex128 rather
        than throwing the imaginary part away on the way out.
        """
        if self._lu is None:
            raise RuntimeError("no factorization available, call factorize first")

        rhs = np.asarray(b)
        if not np.issubdtype(rhs.dtype, np.inexact):
            rhs = rhs.astype(np.float64)

        if rhs.shape[0] != self._size:
            raise ValueError(
                f"right hand side has length {rhs.shape[0]}, "
                f"expected {self._size} to match the factorized matrix"
            )

        return cast("npt.NDArray[Number]", np.asarray(self._lu.solve(rhs)))
