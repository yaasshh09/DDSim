"""Tests for solve/linear.py.

This package must stay free of semiconductor knowledge. continuation.py is
meant to be lifted into the SPICE layer unchanged, and that only works if
nothing in solve/ knows what a carrier density is. One of the tests below
walks the import graph and enforces that.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

from ddsim.solve.linear import SparseLU


def tridiagonal(n: int, off: float = -1.0, diag: float = 2.0) -> sp.coo_matrix:
    """A well conditioned tridiagonal matrix, the 1D Laplacian shape."""
    rows, cols, values = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        values.append(diag)
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            values.append(off)
        if i < n - 1:
            rows.append(i)
            cols.append(i + 1)
            values.append(off)
    return sp.coo_matrix(
        (np.array(values), (np.array(rows), np.array(cols))), shape=(n, n)
    )


Triplets = tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]


def as_arrays(matrix: sp.coo_matrix) -> Triplets:
    """COO triplets plus shape, the form SparseLU.factorize takes."""
    return matrix.row, matrix.col, matrix.data, matrix.shape


# ------------------------------------------------------------------- solving


def test_solves_a_small_system() -> None:
    matrix = sp.coo_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    solver = SparseLU()
    solver.factorize(*as_arrays(matrix))
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(solver.solve(b), np.linalg.solve(matrix.toarray(), b))


def test_solves_a_tridiagonal_system_against_a_dense_reference() -> None:
    matrix = tridiagonal(50)
    solver = SparseLU()
    solver.factorize(*as_arrays(matrix))
    b = np.linspace(1.0, 2.0, 50)
    expected = np.linalg.solve(matrix.toarray(), b)
    np.testing.assert_allclose(solver.solve(b), expected, rtol=1e-12)


def test_solves_a_nonsymmetric_system() -> None:
    """The real Jacobians are strongly non-symmetric. Do not assume otherwise."""
    matrix = tridiagonal(30, off=-1.0)
    dense = matrix.toarray()
    dense[0, -1] = 5.0
    dense[-1, 0] = -3.0
    solver = SparseLU()
    solver.factorize(*as_arrays(sp.coo_matrix(dense)))
    b = np.ones(30)
    np.testing.assert_allclose(solver.solve(b), np.linalg.solve(dense, b), rtol=1e-12)


def test_solves_multiple_right_hand_sides_with_one_factorization() -> None:
    matrix = tridiagonal(20)
    solver = SparseLU()
    solver.factorize(*as_arrays(matrix))
    dense = matrix.toarray()
    for b in (np.ones(20), np.arange(20.0), np.linspace(-1.0, 1.0, 20)):
        expected = np.linalg.solve(dense, b)
        np.testing.assert_allclose(solver.solve(b), expected, rtol=1e-12)


def test_duplicate_coo_entries_are_summed() -> None:
    """Assembly emits one triplet per contribution, so duplicates are normal."""
    rows = np.array([0, 0, 1])
    cols = np.array([0, 0, 1])
    values = np.array([1.0, 3.0, 2.0])
    solver = SparseLU()
    solver.factorize(rows, cols, values, (2, 2))
    # The matrix is diag(4, 2), not diag(3, 2) or diag(1, 2).
    np.testing.assert_allclose(solver.solve(np.array([8.0, 2.0])), [2.0, 1.0])


# ----------------------------------------------------------- pattern tracking


def test_first_factorization_reports_a_new_pattern() -> None:
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(10)))
    assert solver.pattern_unchanged is False


def test_same_pattern_with_new_values_is_recognised() -> None:
    """Across Newton steps the pattern never changes, only the values."""
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(10)))
    solver.factorize(*as_arrays(tridiagonal(10, diag=3.0)))
    assert solver.pattern_unchanged is True


def test_a_changed_pattern_is_recognised() -> None:
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(10)))
    solver.factorize(*as_arrays(tridiagonal(12)))
    assert solver.pattern_unchanged is False


def test_a_changed_pattern_at_the_same_size_is_recognised() -> None:
    """Same shape, different stencil. Comparing shapes alone is not enough."""
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(10)))
    dense = tridiagonal(10).toarray()
    dense[0, 9] = 1.0
    solver.factorize(*as_arrays(sp.coo_matrix(dense)))
    assert solver.pattern_unchanged is False


def test_refactorizing_with_new_values_gives_the_new_solution() -> None:
    """The pattern cache must not leak stale numbers into the next solve.

    This is the test that matters. A cache that accidentally reused the old
    numerical factorization would give a plausible wrong answer.
    """
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(15, diag=2.0)))
    first = solver.solve(np.ones(15))

    updated = tridiagonal(15, diag=10.0)
    solver.factorize(*as_arrays(updated))
    second = solver.solve(np.ones(15))

    expected = np.linalg.solve(updated.toarray(), np.ones(15))
    np.testing.assert_allclose(second, expected, rtol=1e-12)
    assert not np.allclose(first, second)


def test_pattern_tracking_never_changes_the_answer() -> None:
    """Tracking is bookkeeping. A warm solver and a cold one must agree bitwise."""
    matrix = tridiagonal(40, diag=5.0)
    warm = SparseLU()
    warm.factorize(*as_arrays(tridiagonal(40)))
    warm.factorize(*as_arrays(matrix))

    cold = SparseLU()
    cold.factorize(*as_arrays(matrix))

    b = np.linspace(0.0, 1.0, 40)
    np.testing.assert_array_equal(warm.solve(b), cold.solve(b))


def test_ordering_is_colamd_not_natural() -> None:
    """Guards against reintroducing the pre-permutation pessimization.

    Feeding perm_c back through permc_spec="NATURAL" looks like ordering
    reuse but produces 6.1x more fill on a 2D 5-point stencil, because
    SuperLU's COLAMD path does column elimination tree postordering that the
    NATURAL path skips. Measured 14x to 46x slower overall. See the docstring
    in solve/linear.py.
    """
    side = 30
    laplacian = sp.kron(
        sp.eye(side),
        sp.diags([[-1.0] * (side - 1), [4.0] * side, [-1.0] * (side - 1)], [-1, 0, 1]),
    ) + sp.kron(
        sp.diags([[-1.0] * (side - 1), [0.0] * side, [-1.0] * (side - 1)], [-1, 0, 1]),
        sp.eye(side),
    )
    solver = SparseLU()
    solver.factorize(*as_arrays(sp.coo_matrix(laplacian)))
    fill_first = solver.fill_nnz
    solver.factorize(*as_arrays(sp.coo_matrix(laplacian)))
    assert solver.pattern_unchanged is True
    assert solver.fill_nnz == fill_first


# ------------------------------------------------------------------- failures


def test_singular_matrix_raises_an_informative_error() -> None:
    rows = np.array([0, 1])
    cols = np.array([0, 1])
    values = np.array([1.0, 0.0])
    solver = SparseLU()
    with pytest.raises(RuntimeError, match="singular"):
        solver.factorize(rows, cols, values, (2, 2))


def test_solving_before_factorizing_raises() -> None:
    solver = SparseLU()
    with pytest.raises(RuntimeError, match="factorize"):
        solver.solve(np.ones(3))


def test_non_square_matrix_raises() -> None:
    solver = SparseLU()
    with pytest.raises(ValueError, match="square"):
        solver.factorize(np.array([0]), np.array([0]), np.array([1.0]), (2, 3))


def test_right_hand_side_of_wrong_length_raises() -> None:
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(5)))
    with pytest.raises(ValueError, match="length"):
        solver.solve(np.ones(4))


# ------------------------------------------------------------ module boundary


def test_solve_package_imports_nothing_semiconductor_specific() -> None:
    """docs/03-architecture.md: solve/ knows nothing about semiconductors.

    continuation.py gets reused by the SPICE layer unchanged. That only works
    if a carrier density never leaks into this package. Enforced here rather
    than left as a comment nobody reads.
    """
    forbidden = {
        "ddsim.core.constants",
        "ddsim.physics",
        "ddsim.device",
        "ddsim.discretize",
    }
    package = pathlib.Path(__file__).parents[2] / "ddsim" / "solve"

    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            for name in names:
                for banned in forbidden:
                    assert not name.startswith(banned), (
                        f"{source.name} imports {name}, which breaks the "
                        "solve/ module boundary"
                    )


def test_fill_nnz_before_factorizing_raises() -> None:
    solver = SparseLU()
    with pytest.raises(RuntimeError, match="factorize"):
        _ = solver.fill_nnz


def test_size_reports_the_factorized_dimension() -> None:
    solver = SparseLU()
    solver.factorize(*as_arrays(tridiagonal(7)))
    assert solver.size == 7


# ------------------------------------------------- pattern cached conversion


def scattered_with_duplicates(n: int) -> tuple:
    """Triplets in no useful order, with the same entry contributed twice.

    That is what assembly actually emits: one triplet per contribution, in
    whatever order the terms were written, and a diagonal built from several
    of them. The conversion has to sort them and sum the duplicates.
    """
    rows, cols, values = [], [], []
    for i in reversed(range(n)):
        if i < n - 1:
            rows += [i, i + 1]
            cols += [i + 1, i]
            values += [-1.0, -1.0]
    # The diagonal arrives in two pieces, out of order, as a stencil plus a
    # source term would.
    for i in range(n):
        rows.append(i)
        cols.append(i)
        values.append(1.5)
    for i in reversed(range(n)):
        rows.append(i)
        cols.append(i)
        values.append(2.5)
    return (
        np.array(rows, dtype=np.int64),
        np.array(cols, dtype=np.int64),
        np.array(values),
        (n, n),
    )


def test_the_conversion_matches_scipy_on_the_first_call_and_on_a_replay() -> None:
    """The cached pattern is a shortcut through the conversion, not a new one.

    Reusing the sort and the duplicate grouping across Newton steps is only
    safe if the replay lands on exactly what scipy would have built from the
    same triplets, summation order of duplicates included. Compared bit for
    bit rather than to a tolerance: a replay that merely agrees closely is a
    replay that is doing different arithmetic.
    """
    rows, cols, values, shape = scattered_with_duplicates(9)
    solver = SparseLU()

    solver.factorize(rows, cols, values, shape)
    expected = sp.coo_matrix((values, (rows, cols)), shape=shape).tocsc()
    np.testing.assert_array_equal(solver._matrix.indptr, expected.indptr)
    np.testing.assert_array_equal(solver._matrix.indices, expected.indices)
    np.testing.assert_array_equal(solver._matrix.data, expected.data)

    # New numbers on the same pattern, which is every Newton step after the
    # first. This is the path that skips scipy entirely.
    moved = values * 3.0 + 0.5
    solver.factorize(rows, cols, moved, shape)
    assert solver.pattern_unchanged is True

    expected = sp.coo_matrix((moved, (rows, cols)), shape=shape).tocsc()
    np.testing.assert_array_equal(solver._matrix.indptr, expected.indptr)
    np.testing.assert_array_equal(solver._matrix.indices, expected.indices)
    np.testing.assert_array_equal(solver._matrix.data, expected.data)


def test_a_system_with_no_triplets_is_reported_as_singular() -> None:
    """An empty Jacobian is singular, and saying so beats an index error."""
    solver = SparseLU()
    empty = np.array([], dtype=np.int64)

    with pytest.raises(RuntimeError, match="singular"):
        solver.factorize(empty, empty, np.array([]), (3, 3))
