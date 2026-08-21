"""Tests for solve/newton.py.

solve/ knows nothing about semiconductors, so everything here is tested on pure
mathematics. That is not a stylistic choice: docs/03-architecture.md wants
continuation.py lifted into the SPICE layer unchanged, and the import graph
test in test_linear.py enforces it for every file in the package.

Convergence criteria follow docs/02-numerics.md: both the update norm and the
residual norm have to pass, not just one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from ddsim.solve.newton import NewtonResult, newton_solve


@dataclass(frozen=True)
class System:
    """The minimal shape newton_solve accepts, matching PoissonAssembly."""

    residual: npt.NDArray[np.float64]
    rows: npt.NDArray[np.int64]
    cols: npt.NDArray[np.int64]
    values: npt.NDArray[np.float64]
    shape: tuple[int, int]


def diagonal_system(residual: np.ndarray, derivative: np.ndarray) -> System:
    """Wrap a diagonal Jacobian into the assembly shape."""
    n = residual.size
    index = np.arange(n, dtype=np.int64)
    return System(
        residual=residual, rows=index, cols=index, values=derivative, shape=(n, n)
    )


def square_root_problem(target: np.ndarray):
    """F(x) = x^2 - target, root sqrt(target). Smooth and quadratic."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x * x - target, 2.0 * x)

    return assemble


def exponential_problem(target: float):
    """F(x) = exp(x) - target, root ln(target).

    Stiff in the same way Poisson is: a Newton step from far away overshoots
    by an enormous amount unless it is limited.
    """

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(np.exp(x) - target, np.exp(x))

    return assemble


# ------------------------------------------------------------------- solving


def test_one_newton_step_solves_a_linear_system_exactly() -> None:
    """Newton is exact on a linear problem, so the residual is zero after one
    step. Convergence is declared on the second, because docs/02-numerics.md
    requires the update norm to be small as well as the residual, and the
    first step is necessarily large. That confirming step is the price of
    checking both criteria, and it is worth paying: a small update with a
    large residual is a stalled solve, not a converged one."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(3.0 * x - 6.0, np.full(x.size, 3.0))

    result = newton_solve(assemble, np.zeros(4))
    assert result.converged
    assert result.residual_history[1] == 0.0, "one step must land on the root"
    assert result.iterations == 2
    np.testing.assert_allclose(result.x, 2.0, rtol=1e-14)


def test_solves_a_nonlinear_system() -> None:
    target = np.array([2.0, 9.0, 16.0])
    result = newton_solve(square_root_problem(target), np.full(3, 1.0))
    assert result.converged
    np.testing.assert_allclose(result.x, np.sqrt(target), rtol=1e-12)


def test_converges_quadratically() -> None:
    """The acceptance criterion in phases/PHASE-1.md, as an assertion.

    Quadratic means the residual roughly squares each step, so the number of
    correct digits doubles. Checked as r_{k+1} <= C * r_k^2 over the tail.
    """
    result = newton_solve(square_root_problem(np.array([2.0])), np.array([1.0]))
    history = np.array(result.residual_history)
    tail = history[history > 1e-14]

    assert len(tail) >= 3, "need a few iterations to see the tail"
    for previous, current in zip(tail[:-1], tail[1:], strict=True):
        assert current <= 10.0 * previous**2 + 1e-15


def test_reports_the_residual_history() -> None:
    result = newton_solve(square_root_problem(np.array([2.0])), np.array([1.0]))
    assert len(result.residual_history) == result.iterations + 1
    assert result.residual_history[0] > result.residual_history[-1]


def test_reports_the_update_history() -> None:
    result = newton_solve(square_root_problem(np.array([2.0])), np.array([1.0]))
    assert len(result.update_history) == result.iterations


def test_starting_at_the_solution_takes_no_iterations() -> None:
    result = newton_solve(square_root_problem(np.array([4.0])), np.array([2.0]))
    assert result.converged
    assert result.iterations == 0


# ------------------------------------------------------------- step limiting


def test_step_limiting_caps_the_update() -> None:
    """docs/02-numerics.md: 5 * V_T per step, which is 5.0 scaled."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - 1000.0, np.ones(x.size))

    result = newton_solve(assemble, np.zeros(1), max_step=5.0, max_iterations=3)
    assert not result.converged
    assert result.x[0] == pytest.approx(15.0, rel=1e-14)


def test_step_limiting_preserves_the_update_direction() -> None:
    """Clamping the magnitude must not flip a sign."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x + 1000.0, np.ones(x.size))

    result = newton_solve(assemble, np.zeros(1), max_step=5.0, max_iterations=1)
    assert result.x[0] == pytest.approx(-5.0, rel=1e-14)


def test_step_limiting_scales_the_whole_vector_together() -> None:
    """Clamping each entry separately would rotate the search direction."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([100.0, 50.0]), np.ones(2))

    result = newton_solve(assemble, np.zeros(2), max_step=5.0, max_iterations=1)
    assert result.x[0] / result.x[1] == pytest.approx(2.0, rel=1e-14)
    assert np.max(np.abs(result.x)) == pytest.approx(5.0, rel=1e-14)


def test_step_limiting_rescues_a_stiff_exponential() -> None:
    """Undamped Newton on exp(x) = c from far below diverges. Limited, it does not."""
    result = newton_solve(exponential_problem(1.0), np.array([-40.0]), max_step=5.0)
    assert result.converged
    assert result.x[0] == pytest.approx(0.0, abs=1e-10)


def test_unlimited_newton_on_the_same_problem_diverges() -> None:
    """Shows the limiter is doing something, rather than passing for free.

    From x = -40 the undamped step is exp(40), about 2.4e17, which sends the
    next exp straight past the double range. The overflow is the phenomenon
    under test, so it is allowed here rather than raised.
    """
    with np.errstate(over="ignore"):
        unlimited = newton_solve(
            exponential_problem(1.0), np.array([-40.0]), max_step=None
        )
    assert not unlimited.converged
    assert "diverged" in unlimited.message

    limited = newton_solve(exponential_problem(1.0), np.array([-40.0]), max_step=5.0)
    assert limited.converged


# ---------------------------------------------------------------- termination


def test_gives_up_after_max_iterations() -> None:
    """A step limited walk that cannot reach the root in the budget."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - 1000.0, np.ones(x.size))

    result = newton_solve(assemble, np.zeros(1), max_step=5.0, max_iterations=12)
    assert not result.converged
    assert result.iterations == 12
    assert "did not converge" in result.message


def test_a_problem_with_no_root_stops_rather_than_looping() -> None:
    """F(x) = exp(x) + 1 has no root.

    Newton walks left until exp underflows, at which point the Jacobian is
    exactly zero and the linear solve fails. It stops there, after 3 steps,
    rather than burning the whole iteration budget.
    """

    def assemble(x: np.ndarray) -> System:
        with np.errstate(over="ignore", under="ignore"):
            return diagonal_system(np.exp(x) + 1.0, np.exp(x))

    result = newton_solve(assemble, np.zeros(1), max_iterations=50)
    assert not result.converged
    assert result.iterations < 50
    assert result.message != ""


def test_both_convergence_criteria_must_pass() -> None:
    """docs/02-numerics.md asks for the update norm and the residual norm.

    A tiny update with a large residual is a stalled solve, not a converged
    one, and must not be reported as success.
    """

    def assemble(x: np.ndarray) -> System:
        # Jacobian is enormous, so every update is negligible while the
        # residual stays stubbornly large.
        return diagonal_system(np.full(1, 5.0), np.full(1, 1e14))

    result = newton_solve(assemble, np.zeros(1), max_iterations=5)
    assert not result.converged


def test_singular_jacobian_is_reported_not_raised() -> None:
    def assemble(x: np.ndarray) -> System:
        return diagonal_system(np.ones(2), np.zeros(2))

    result = newton_solve(assemble, np.zeros(2), max_iterations=3)
    assert not result.converged
    assert result.message != ""


def test_result_is_immutable() -> None:
    result = newton_solve(square_root_problem(np.array([4.0])), np.array([2.0]))
    with pytest.raises(AttributeError):
        result.converged = False  # type: ignore[misc]


def test_result_repr_mentions_convergence_and_iterations() -> None:
    result = newton_solve(square_root_problem(np.array([4.0])), np.array([1.0]))
    text = repr(result)
    assert "converged" in text.lower()
    assert str(result.iterations) in text


def test_does_not_mutate_the_initial_guess() -> None:
    x0 = np.full(3, 1.0)
    original = x0.copy()
    newton_solve(square_root_problem(np.array([2.0, 9.0, 16.0])), x0)
    np.testing.assert_array_equal(x0, original)


def test_returns_a_newton_result() -> None:
    result = newton_solve(square_root_problem(np.array([4.0])), np.array([1.0]))
    assert isinstance(result, NewtonResult)


# ------------------------------------------------- residual tolerance scaling


def scaled_by(assemble, factor: float):
    """The same problem with its residual and Jacobian multiplied through.

    Mathematically identical: the Newton step is unchanged, since the factor
    cancels between the residual and the Jacobian. Only the numbers are
    bigger, exactly as they are on a heavily doped device.
    """

    def wrapped(x: np.ndarray) -> System:
        inner = assemble(x)
        return System(
            residual=inner.residual * factor,
            rows=inner.rows,
            cols=inner.cols,
            values=inner.values * factor,
            shape=inner.shape,
        )

    return wrapped


def test_convergence_is_invariant_under_scaling_the_residual() -> None:
    """The bug this guards against, stated as an invariant.

    A residual with an absolute tolerance is not scale free. Multiplying the
    whole system by 1e8 changes nothing about the mathematics, but it lifts
    the roundoff floor of the residual by 1e8 as well, so a fixed 1e-10
    threshold becomes unreachable. Real devices do exactly this: the Poisson
    residual is proportional to the doping, so a solve that converges at
    1e16 cm^-3 fails at 1e18 for no physical reason.
    """
    problem = square_root_problem(np.array([2.0]))
    base = newton_solve(problem, np.array([1.0]))
    scaled = newton_solve(scaled_by(problem, 1e8), np.array([1.0]))

    assert base.converged
    assert scaled.converged, scaled.message
    assert scaled.iterations == base.iterations
    np.testing.assert_allclose(scaled.x, base.x, rtol=1e-14)


def test_relative_residual_tolerance_still_rejects_a_stalled_solve() -> None:
    """Loosening the residual test must not let a stalled solve through.

    Both criteria are required, so a persistently large update still fails
    even when the relative residual threshold is generous.
    """

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(np.full(1, 1e8), np.full(1, 1.0))

    result = newton_solve(assemble, np.zeros(1), max_step=1.0, max_iterations=5)
    assert not result.converged


def test_a_problem_that_starts_at_zero_residual_still_converges() -> None:
    """A relative threshold must not collapse to zero when F_0 is zero."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(np.zeros(1), np.ones(1))

    result = newton_solve(assemble, np.zeros(1))
    assert result.converged
    assert result.iterations == 0


def test_a_non_finite_residual_is_reported_as_divergence() -> None:
    """An iterate that overflows is a diverged solve, not a crash.

    Starting at x = -700 the Jacobian is exp(-700), about 1e-304, so the
    Newton step is 1e304. That is finite, so the update passes its own check,
    and only the next residual overflows. This is the path that separates a
    bad iterate from a bad step.
    """
    with np.errstate(over="ignore"):
        result = newton_solve(
            exponential_problem(1.0), np.array([-700.0]), max_iterations=5
        )
    assert not result.converged
    assert "diverged" in result.message
    assert result.residual_history[-1] == float("inf")


def test_a_non_finite_newton_update_is_reported() -> None:
    """A finite residual over a zero-ish Jacobian gives an infinite step.

    Distinct from the diverged-residual case above: here the step itself is
    already unusable, so it is caught before it is ever applied.
    """

    def assemble(x: np.ndarray) -> System:
        with np.errstate(divide="ignore", over="ignore"):
            return diagonal_system(np.ones(1), np.full(1, 5e-324))

    result = newton_solve(assemble, np.zeros(1), max_iterations=3)
    assert not result.converged
    assert "non-finite" in result.message


# ------------------------------------------------------------ residual scale


def floored_problem(floor: float):
    """A residual stuck at `floor`, with a Jacobian too large to move it.

    Stands in for a real assembly whose residual cannot fall below the size of
    the terms it differences. Every warm started solve is in this position: it
    arrives at the answer already, and the residual it reports is the roundoff
    left over from cancelling terms of size 1e9 against each other.
    """

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(np.full_like(x, floor), np.full_like(x, 1e18))

    return assemble


def test_a_solve_started_at_its_own_floor_cannot_converge_without_a_scale() -> None:
    """The default threshold is relative to the initial residual.

    Starting at the floor, the threshold lands a decade below it and is
    unreachable however correct the iterate is. Pinned here because inside a
    Gummel cycle it looks exactly like a broken solver, and because the repair
    below only makes sense next to the failure it repairs.
    """
    result = newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations=5)

    assert not result.converged
    assert result.update_history[-1] < 1e-20


def test_an_explicit_residual_scale_lets_a_warm_start_converge() -> None:
    """The scale comes from the size of the terms, not from where it started."""
    result = newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        residual_scale=1.0,
        max_iterations=5,
    )

    assert result.converged


def test_a_residual_scale_still_rejects_a_genuinely_stalled_solve() -> None:
    """The point of the residual criterion survives the change.

    A solve that stops moving while its residual is still large next to the
    scale of the problem is a failure and has to keep being reported as one.
    Otherwise the scale would have turned the criterion into decoration.
    """
    result = newton_solve(
        floored_problem(0.5), np.array([1.0]), residual_scale=1.0, max_iterations=5
    )

    assert not result.converged
