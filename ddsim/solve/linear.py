from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast
import numpy as np; import numpy.typing as npt, scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, SuperLU, gmres, spilu, splu
Number = TypeVar('Number', np.float64, np.complex128)
Ordering = Literal["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]


@dataclass(frozen=True)
class _CSCPattern:
    rows: npt.NDArray[np.int64]

    cols:npt.NDArray[np.int64]

    shape: tuple[int, int]
    order: npt.NDArray[np.intp]

    group: npt.NDArray[np.intp]

    n_entries: int

    def matches(self, rows:npt.NDArray[np.integer], cols: npt.NDArray[np.integer], shape: tuple[int, int],) -> bool:
        return (
            shape == self.shape
            and np.array_equal(rows, self.rows)
            and np.array_equal(cols, self.cols)
        )

    def data(self, values:npt.NDArray[Number]) -> npt.NDArray[Number]:
        foo = values[self.order]
        if np.iscomplexobj(foo):
            arr = np.zeros(self.n_entries, dtype=foo.dtype)

            np.add.at(arr, self.group, foo)
            return cast('npt.NDArray[Number]', arr)

        return cast(
            "npt.NDArray[Number]",
            np.bincount(
                self.group,
                weights=cast("npt.NDArray[np.float64]", foo),
                minlength=self.n_entries,
            ),
        )


def _build_pattern(
    rows: npt.NDArray[np.integer],
    cols: npt.NDArray[np.integer],
    shape:tuple[int, int],
) -> tuple[_CSCPattern, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    tmp = shape[1]
    f = np.lexsort((rows, cols))
    cc = rows[f]
    tt = cols[f]
    s = np.empty(f.size, dtype=bool)
    s[0] = True
    s[1:] = (cc[1:] != cc[:-1]) | (tt[1:] != tt[:-1])

    val = np.cumsum(s) - 1
    item = np.ascontiguousarray(cc[s], dtype=np.int32)
    kk = tt[s]

    info = np.zeros(tmp + 1,
         dtype=np.int32)
    info[1:] = np.cumsum(np.bincount(kk, minlength=tmp))

    u = _CSCPattern(rows=np.array(rows, dtype=np.int64, copy=True), cols=np.array(cols, dtype=np.int64, copy=True), shape=shape, order=f, group=np.asarray(val, dtype=np.intp), n_entries=int(item.size),)
    return u, item, info

class SparseLU:
    def __init__(self, ordering: Ordering = "COLAMD") -> None:
        self._ordering = ordering
        self._lu: SuperLU|None = None
        self._pattern:_CSCPattern | None = None
        self._matrix: sp.csc_matrix | None = None
        self._pattern_unchanged = False

        self._size = 0

    @property
    def pattern_unchanged(self) -> bool:
        return self._pattern_unchanged

    @property
    def size(self) -> int:
        return self._size

    @property
    def fill_nnz(self) -> int:
        if self._lu is None:
            raise RuntimeError('no factorization available, call factorize first')
        return int(self._lu.L.nnz + self._lu.U.nnz)

    def factorize(
        self,
        rows:npt.NDArray[np.integer],
        cols:npt.NDArray[np.integer],
        values:npt.NDArray[np.floating],
        shape: tuple[int, int],
    ) -> None:
        if shape[0] != shape[1]:
            raise ValueError(f"matrix must be square, got shape {shape}")
        shape = (int(shape[0]), int(shape[1]))

        a = np.asarray(values)

        if not np.issubdtype(a.dtype, np.inexact):
            a = a.astype(np.float64)
        j = self._pattern

        v = (j is not None and self._matrix is not None and self._matrix.dtype == a.dtype and j.matches(rows, cols, shape))
        if v:
            assert j is not None and self._matrix is not None
            d = self._matrix
            d.data[:] = j.data(a)
        elif a.size == 0:
            d = sp.coo_matrix((a, (rows, cols)), shape=shape).tocsc()
            j = None
        else:
            j, u, num = _build_pattern(np.asarray(rows), np.asarray(cols), shape)
            d = sp.csc_matrix(
                (j.data(a), u, num), shape=shape
            )
            d.has_sorted_indices = True
        try:
            self._lu = splu(d,
                 permc_spec=self._ordering)
        except RuntimeError as bar:
            self._lu = None
            raise RuntimeError(
                f"LU factorization failed, the matrix is singular or nearly so: {bar}"
            ) from bar
        self._pattern_unchanged = v
        self._pattern = j
        self._matrix = d
        self._size = shape[0]
    def solve(self, b:npt.NDArray[Number], trans: str = "N") -> npt.NDArray[Number]:
        if self._lu is None:
            raise RuntimeError("no factorization available, call factorize first")

        m = np.asarray(b)
        if not np.issubdtype(m.dtype, np.inexact):
            m = m.astype(np.float64)

        if m.shape[0] != self._size:
            raise ValueError(
                f"right hand side has length {m.shape[0]}, "
                f"expected {self._size} to match the factorized matrix"
            )
        return cast('npt.NDArray[Number]', np.asarray(cast(Any, self._lu).solve(m, trans=trans)))


class GmresILU:
    def __init__(self, drop_tol: float = 1e-3, fill_factor: float = 10.0, eta_max: float = 0.1, eta_min: float = 1e-12, restart: int = 60, max_restarts: int = 20, tightenings: int = 2) -> None:
        self.drop_tol = drop_tol
        self.tightenings = tightenings
        self.fill: list[float] = []
        self.fill_factor = fill_factor
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.restart = restart
        self.max_restarts = max_restarts
        self._matrix: sp.csc_matrix | None = None
        self._ilu: SuperLU | None = None
        self._rows: npt.NDArray[np.float64] | None = None
        self._cols: npt.NDArray[np.float64] | None = None
        self._last_norm: float | None = None
        self._eta = eta_max
        self.iterations: list[int] = []
        self.forcing: list[float] = []

    def factorize(self, rows: npt.NDArray[np.integer], cols: npt.NDArray[np.integer], values: npt.NDArray[np.floating], shape: tuple[int, int]) -> None:
        if shape[0] != shape[1]:
            raise ValueError(f"matrix must be square, got shape {shape}")
        a = sp.csr_matrix((np.asarray(values, dtype=np.float64), (rows, cols)), shape=shape)
        big = np.asarray(abs(a).max(axis=1).todense()).ravel()
        if np.any(big == 0.0):
            raise RuntimeError(f"row {int(np.flatnonzero(big == 0.0)[0])} of the Jacobian is all zeros, so the matrix is singular")
        self._rows = 1.0 / big
        b = sp.csc_matrix(sp.diags(self._rows) @ a)
        tall = np.asarray(abs(b).max(axis=0).todense()).ravel()
        if np.any(tall == 0.0):
            raise RuntimeError(f"column {int(np.flatnonzero(tall == 0.0)[0])} of the Jacobian is all zeros, so the matrix is singular")
        self._cols = 1.0 / tall
        d = sp.csc_matrix(b @ sp.diags(self._cols))
        self._matrix = d
        self._ilu = self._incomplete(d, self.drop_tol, self.fill_factor)

    def _incomplete(self, d: sp.csc_matrix, drop_tol: float, fill_factor: float) -> SuperLU:
        try:
            ilu = spilu(d, drop_tol=drop_tol, fill_factor=fill_factor, permc_spec="MMD_AT_PLUS_A")
        except RuntimeError as bar:
            self._ilu = None
            raise RuntimeError(f"the incomplete LU broke down, the matrix is singular or nearly so: {bar}") from bar
        self.fill.append((ilu.L.nnz + ilu.U.nnz) / d.nnz)
        return ilu

    def _next_eta(self, size: float) -> float:
        if self._last_norm is None or self._last_norm == 0.0:
            eta = self.eta_max
        else:
            eta = 0.9 * (size / self._last_norm) ** 2
            floor = 0.9 * self._eta**2
            if floor > 0.1:
                eta = max(eta, floor)
        return min(max(eta, self.eta_min), self.eta_max)

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._ilu is None or self._matrix is None or self._rows is None or self._cols is None:
            raise RuntimeError("no factorization available, call factorize first")
        rhs = self._rows * np.asarray(b, dtype=np.float64)
        size = float(np.linalg.norm(rhs))
        self._eta = self._next_eta(size)
        self._last_norm = size
        drop, fill = self.drop_tol, self.fill_factor
        for attempt in range(self.tightenings + 1):
            m = LinearOperator(self._matrix.shape, matvec=self._ilu.solve, dtype=np.float64)
            count = [0]

            def tick(_: object, count: list[int] = count) -> None:
                count[0] += 1

            x, info = gmres(self._matrix, rhs, rtol=self._eta, atol=0.0, restart=self.restart, maxiter=self.max_restarts, M=m, callback=tick, callback_type="pr_norm")
            self.iterations.append(count[0])
            self.forcing.append(self._eta)
            got = float(np.linalg.norm(self._matrix @ x - rhs))
            if got <= max(self._eta, 1e-8) * size or (info == 0 and got <= 10 * self._eta * size):
                return np.asarray(self._cols * x)
            if attempt < self.tightenings:
                drop, fill = drop * 1e-2, fill * 2
                self._ilu = self._incomplete(self._matrix, drop, fill)
        raise RuntimeError(f"GMRES reached {got / size:.3e} of the right hand side in {count[0]} iterations against a forcing term of {self._eta:.3e}, with the incomplete LU already tightened to drop_tol {drop:.0e}. The Jacobian is too badly conditioned for this preconditioner.")
