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
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import SuperLU, splu


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
        self._fingerprint: tuple[tuple[int, int], bytes, bytes] | None = None
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

        matrix = sp.coo_matrix(
            (np.asarray(values, dtype=np.float64), (rows, cols)), shape=shape
        ).tocsc()

        fingerprint = (
            (int(shape[0]), int(shape[1])),
            matrix.indptr.tobytes(),
            matrix.indices.tobytes(),
        )

        try:
            self._lu = splu(matrix, permc_spec="COLAMD")
        except RuntimeError as error:
            self._lu = None
            raise RuntimeError(
                f"LU factorization failed, the matrix is singular or nearly so: {error}"
            ) from error

        self._pattern_unchanged = fingerprint == self._fingerprint
        self._fingerprint = fingerprint
        self._size = int(shape[0])

    def solve(self, b: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Solve A x = b using the stored factorization."""
        if self._lu is None:
            raise RuntimeError("no factorization available, call factorize first")

        rhs = np.asarray(b, dtype=np.float64)
        if rhs.shape[0] != self._size:
            raise ValueError(
                f"right hand side has length {rhs.shape[0]}, "
                f"expected {self._size} to match the factorized matrix"
            )

        return np.asarray(self._lu.solve(rhs), dtype=np.float64)
