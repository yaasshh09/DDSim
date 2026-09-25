"""Tests for solve/newton.py.

solve/ knows nothing about semiconductors, so everything here is tested on pure
mathematics. That is not a stylistic choice: docs/03-architecture.md wants
continuation.py lifted into the SPICE layer unchanged, and the import graph
test in test_linear.py enforces it for every file in the package.

Convergence criteria follow docs/02-numerics.md: both the update norm and the
residual norm have to pass, not just one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import numpy.typing as npt
import pytest

from ddsim.solve.linear import SparseLU
from ddsim.solve.newton import NewtonIteration, NewtonResult, newton_solve


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
        result.converged = False


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


def test_an_assembly_that_overflows_is_reported_as_divergence() -> None:
    """A row scaled by its own terms can overflow before the residual does,
    and the assembly raises FloatingPointError saying so. That is the same
    diverged iterate as a non-finite residual and ends the same way, rather
    than escaping and taking a continuation with it."""
    calls = []

    def assemble(x: np.ndarray) -> System:
        calls.append(x)
        if len(calls) > 1:
            raise FloatingPointError("the iterate has diverged")
        return diagonal_system(x - 1.0, np.full_like(x, 1e-3))

    result = newton_solve(assemble, np.array([0.0]), max_iterations=5)

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


def test_a_reused_solver_gives_the_identical_answer() -> None:
    """Handing the same factorization back in is an optimisation only.

    Every Gummel cycle solves the same Poisson system on the same mesh, so
    the caller keeps one SparseLU rather than paying for the sparsity pattern
    on every cycle. That has to be invisible in the answer: a solver that had
    leaked any part of a previous numerical factorization would give a
    plausible wrong result rather than an obviously wrong one, so this
    compares bitwise across three consecutive solves.
    """
    target = np.array([2.0, 9.0, 16.0])
    shared = SparseLU()

    for _ in range(3):
        warm = newton_solve(square_root_problem(target), np.full(3, 5.0), solver=shared)
        cold = newton_solve(square_root_problem(target), np.full(3, 5.0))

        assert warm.converged and cold.converged
        assert warm.iterations == cold.iterations
        np.testing.assert_array_equal(warm.x, cold.x)


def test_a_reused_solver_follows_a_changed_pattern() -> None:
    """A shared solver must not pin the caller to the first problem's shape."""
    shared = SparseLU()

    newton_solve(square_root_problem(np.array([4.0, 9.0])), np.full(2, 3.0),
                 solver=shared)
    bigger = newton_solve(
        square_root_problem(np.array([4.0, 9.0, 25.0])), np.full(3, 3.0),
        solver=shared,
    )

    assert bigger.converged
    np.testing.assert_allclose(bigger.x, [2.0, 3.0, 5.0], rtol=1e-12)


def test_a_frozen_residual_with_a_settled_update_stops_early() -> None:
    """Spinning the whole budget on a solve that has stopped moving is waste.

    `floored_problem` is the shape of a real stall: the residual sits on its
    own arithmetic floor and the Jacobian is far too large for any update to
    shift it. Once the residual has not changed in the last bit for several
    steps running and the update is already inside its tolerance, nothing
    later in the budget can change the answer.
    """
    result = newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations=50)

    assert not result.converged
    assert result.iterations < 10
    assert "stopped moving" in result.message


def test_the_stagnation_message_carries_both_numbers() -> None:
    """A stall and a slow solve look identical without the residual and the
    threshold side by side. That was what made the low doping convergence bug
    slow to find, so the message keeps carrying them.
    """
    result = newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations=50)

    assert "1.000e-11" in result.message
    assert "threshold" in result.message


def test_stagnation_does_not_fire_while_the_update_is_still_large() -> None:
    """A flat residual is not a stall on its own.

    Here the residual is bit for bit identical every step while the iterate
    marches off by one unit each time. That is a solver making real steps that
    happen not to help, which is a different failure and must not be cut short
    by a guard aimed at settled iterates.
    """

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(np.ones(1), np.ones(1))

    result = newton_solve(assemble, np.zeros(1), max_iterations=12)

    assert not result.converged
    assert result.iterations == 12
    assert "did not converge" in result.message


def test_stagnation_does_not_fire_on_a_healthy_solve() -> None:
    """Quadratic convergence changes the residual every step until it lands."""
    result = newton_solve(square_root_problem(np.array([2.0])), np.array([1.0]))

    assert result.converged
    assert result.x == pytest.approx(np.sqrt([2.0]))


def test_the_stagnation_guard_can_be_switched_off() -> None:
    """Off restores the old behaviour exactly: run the budget, then report."""
    result = newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        max_iterations=50,
        stagnation_window=None,
    )

    assert not result.converged
    assert result.iterations == 50
    assert "did not converge" in result.message


def test_a_converged_solve_is_never_turned_into_a_stall() -> None:
    """The guard runs after the convergence check, not before it.

    A warm started solve arrives with a frozen residual and a zero update by
    construction, which is precisely the stagnation pattern. Given a scale it
    has to still be reported as the success it is.
    """
    result = newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        residual_scale=1.0,
        max_iterations=50,
    )

    assert result.converged


def test_a_limit_callable_replaces_the_uniform_scaling() -> None:
    """The coupled system needs a limit on one component, not on the vector.

    max_step scales everything by max|dx|, which on a coupled solve is
    dominated by the density updates: n is 1e6 in scaled units and psi is a
    few, so a cap of 5 on the whole vector shrinks the potential update by
    five decades and nothing moves.
    """

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([100.0, 2.0]), np.ones(2))

    def limit(delta: np.ndarray) -> np.ndarray:
        capped = delta.copy()
        capped[0] = np.clip(capped[0], -5.0, 5.0)
        return capped

    result = newton_solve(assemble, np.zeros(2), limit=limit, max_iterations=1)

    assert result.x[0] == pytest.approx(5.0)
    assert result.x[1] == pytest.approx(2.0)


def test_a_limit_that_changes_nothing_is_not_counted_as_limited() -> None:
    """limited_steps has to keep meaning what it means.

    A converged solve is supposed to end with several unlimited steps. If a
    limiter that never fires still counted, that diagnostic would read as a
    solve permanently against its cap.
    """

    def limit(delta: np.ndarray) -> np.ndarray:
        return delta

    result = newton_solve(
        square_root_problem(np.array([4.0])), np.array([1.0]), limit=limit
    )

    assert result.converged
    assert result.limited_steps == 0


def test_a_limit_that_fires_is_counted() -> None:
    """And one that does fire has to show up in the diagnostic."""

    def limit(delta: np.ndarray) -> np.ndarray:
        return np.clip(delta, -5.0, 5.0)

    result = newton_solve(
        exponential_problem(1.0), np.array([-40.0]), limit=limit
    )

    assert result.converged
    assert result.limited_steps > 0


def test_a_limit_rescues_the_stiff_exponential_like_max_step_does() -> None:
    """Same rescue, different mechanism, so the two are known to agree."""

    def limit(delta: np.ndarray) -> np.ndarray:
        return np.clip(delta, -5.0, 5.0)

    result = newton_solve(exponential_problem(1.0), np.array([-40.0]), limit=limit)

    assert result.converged
    assert result.x[0] == pytest.approx(0.0, abs=1e-9)


def test_passing_both_a_limit_and_a_max_step_is_rejected() -> None:
    """Two damping rules on one update is a caller who means one of them.

    Silently letting one win would make the other look ineffective, which is
    a long debugging session over a keyword argument.
    """
    with pytest.raises(ValueError, match="max_step"):
        newton_solve(
            square_root_problem(np.array([4.0])),
            np.array([1.0]),
            max_step=5.0,
            limit=lambda delta: delta,
        )


def test_a_limit_that_returns_the_wrong_shape_is_rejected() -> None:
    """A limiter that drops entries would silently freeze those unknowns."""

    def limit(delta: np.ndarray) -> np.ndarray:
        return delta[:1]

    with pytest.raises(ValueError, match="shape"):
        newton_solve(
            square_root_problem(np.array([4.0, 9.0])),
            np.array([1.0, 1.0]),
            limit=limit,
        )


def test_an_update_norm_callable_replaces_max_abs_delta() -> None:
    """max |dx| is the wrong measure when the unknowns differ by decades.

    A coupled solve carries a potential of order 10 next to a density of
    order 1e6 in scaled units. Once the density is converged to the last bit
    its update is still 1e-10 in absolute terms, so max |dx| has a floor six
    decades above the potential's, and a threshold that suits psi can never
    be met. docs/02-numerics.md asks for the carrier change relative to
    n + n_i for exactly this reason.
    """

    assemble = square_root_problem(np.array([1.0, 2e16]))
    start = np.array([1.0, 1e7])

    def relative(delta: np.ndarray, x: np.ndarray) -> float:
        return float(np.max(np.abs(delta) / (np.abs(x) + 1.0)))

    strict = newton_solve(assemble, start, update_tol=1e-10)
    relaxed = newton_solve(
        assemble, start, update_tol=1e-10, update_norm=relative
    )

    assert not strict.converged
    assert relaxed.converged
    assert relaxed.x[1] == pytest.approx(np.sqrt(2e16), rel=1e-15)


def test_the_update_norm_sees_the_iterate_before_the_step() -> None:
    """A relative measure divides by where the solve is, not where it lands."""
    seen: list[float] = []

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([4.0]), np.ones(1))

    def record(delta: np.ndarray, x: np.ndarray) -> float:
        seen.append(float(x[0]))
        return float(np.max(np.abs(delta)))

    newton_solve(
        assemble, np.array([1.0]), update_norm=record, max_iterations=1
    )

    assert seen[0] == 1.0


def test_the_update_norm_is_what_gets_recorded_in_the_history() -> None:
    """Otherwise the reported history and the applied criterion disagree."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([8.0]), np.ones(1))

    def halved(delta: np.ndarray, x: np.ndarray) -> float:
        return float(np.max(np.abs(delta))) / 2.0

    result = newton_solve(
        assemble, np.zeros(1), update_norm=halved, max_iterations=1
    )

    assert result.update_history[0] == pytest.approx(4.0)


def test_the_update_norm_is_measured_after_the_limiter() -> None:
    """A limited step is smaller than the raw one, and it is the one taken."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([100.0]), np.ones(1))

    result = newton_solve(
        assemble,
        np.zeros(1),
        limit=lambda delta: np.clip(delta, -5.0, 5.0),
        update_norm=lambda delta, x: float(np.max(np.abs(delta))),
        max_iterations=1,
    )

    assert result.update_history[0] == pytest.approx(5.0)


def linear_problem(target: float):
    """F(x) = x - target, J = 1. One step from anywhere, of known size."""

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([target]), np.ones(1))

    return assemble


def collect() -> tuple[list[NewtonIteration], Callable[[NewtonIteration], None]]:
    """A callback and the list it fills."""
    frames: list[NewtonIteration] = []
    return frames, frames.append


def test_a_frame_arrives_for_every_residual_evaluation() -> None:
    """The point of the hook is a live picture of residual_history, so it has
    to carry the same number of entries the history ends up with."""
    frames, watch = collect()

    result = newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration=watch,
    )

    assert result.converged
    assert len(frames) == len(result.residual_history)
    assert [frame.iteration for frame in frames] == list(range(len(frames)))


def test_the_frame_residual_is_the_one_in_the_history() -> None:
    """Bit for bit, not close. A telemetry number that disagrees with the
    number the solve was judged on is worse than no telemetry."""
    frames, watch = collect()

    result = newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration=watch,
    )

    assert [frame.residual for frame in frames] == result.residual_history


def test_the_frame_update_is_the_one_in_the_history() -> None:
    """Same argument as the residual, on the other history."""
    frames, watch = collect()

    result = newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration=watch,
    )

    assert [frame.update for frame in frames[1:]] == result.update_history


def test_the_first_frame_has_no_update_because_no_step_has_been_taken() -> None:
    """It reports the residual at x0. Calling that an update of zero would
    put a point on the update plot that no step produced."""
    frames, watch = collect()

    newton_solve(linear_problem(8.0), np.zeros(1), on_iteration=watch)

    assert frames[0].iteration == 0
    assert frames[0].update is None
    assert frames[0].damping is None
    assert not frames[0].limited
    assert frames[0].residual == pytest.approx(8.0)


def test_a_limited_step_reports_the_factor_that_was_applied() -> None:
    """Newton asks for 100, max_step allows 5, so the step taken is one
    twentieth of the one requested."""
    frames, watch = collect()

    newton_solve(
        linear_problem(100.0),
        np.zeros(1),
        max_step=5.0,
        max_iterations=1,
        on_iteration=watch,
    )

    assert frames[1].limited
    assert frames[1].damping == pytest.approx(0.05)


def test_an_unlimited_step_reports_a_damping_factor_of_one() -> None:
    """Otherwise a healthy solve looks damped on the plot."""
    frames, watch = collect()

    newton_solve(
        linear_problem(1.0),
        np.zeros(1),
        max_step=5.0,
        max_iterations=1,
        on_iteration=watch,
    )

    assert not frames[1].limited
    assert frames[1].damping == pytest.approx(1.0)


def test_a_caller_supplied_limiter_reports_the_ratio_it_applied() -> None:
    """The coupled solve damps psi alone, so there is no single factor in the
    limiter itself. The ratio of the step taken to the step asked for is the
    honest scalar summary of what it did."""
    frames, watch = collect()

    newton_solve(
        linear_problem(100.0),
        np.zeros(1),
        limit=lambda delta: delta * 0.25,
        max_iterations=1,
        on_iteration=watch,
    )

    assert frames[1].limited
    assert frames[1].damping == pytest.approx(0.25)


def test_a_diverged_iterate_emits_a_final_frame() -> None:
    """Phase 7 wants a diverged solve to show the divergence rather than the
    stream going quiet. The infinite residual is the last thing sent."""
    frames, watch = collect()

    with np.errstate(over="ignore"):
        result = newton_solve(
            exponential_problem(1.0),
            np.array([-700.0]),
            max_iterations=5,
            on_iteration=watch,
        )

    assert not result.converged
    assert frames[-1].residual == float("inf")
    assert len(frames) == len(result.residual_history)


def test_watching_a_solve_does_not_change_it() -> None:
    """The inertness argument the edge list refactor used. A callback that
    only reads cannot move a solved number, and this is what says so."""
    frames, watch = collect()

    quiet = newton_solve(
        exponential_problem(1e6),
        np.array([0.0]),
        max_step=5.0,
    )
    watched = newton_solve(
        exponential_problem(1e6),
        np.array([0.0]),
        max_step=5.0,
        on_iteration=watch,
    )

    assert np.array_equal(quiet.x, watched.x)
    assert quiet.residual_history == watched.residual_history
    assert quiet.update_history == watched.update_history
    assert quiet.iterations == watched.iterations
    assert quiet.limited_steps == watched.limited_steps
    assert quiet.converged == watched.converged
    assert quiet.message == watched.message
    assert len(frames) == len(quiet.residual_history)


def test_an_exception_from_the_callback_stops_the_solve() -> None:
    """This is how cancellation works. The callback is the only place the
    solve looks up from the arithmetic, so raising there is what stops it,
    and swallowing the exception would make a cancel button do nothing."""

    def refuse(frame: NewtonIteration) -> None:
        if frame.iteration == 2:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        newton_solve(
            exponential_problem(1e6),
            np.array([0.0]),
            max_step=5.0,
            on_iteration=refuse,
        )


def test_the_frame_is_frozen_so_a_client_cannot_edit_the_record() -> None:
    """The API layer hands these to a serializer. A mutable frame invites a
    client side edit that makes the transcript disagree with the solve."""
    frames, watch = collect()

    newton_solve(linear_problem(8.0), np.zeros(1), on_iteration=watch)

    with pytest.raises(FrozenInstanceError):
        frames[0].residual = 0.0


def test_damping_reports_a_rule_that_only_damps_part_of_the_vector() -> None:
    """The coupled solve's limiter caps psi and takes the density updates in
    full, and in scaled units the densities are six decades larger, so they
    set max |dx| on every step. A factor measured as damped max over raw max
    would read exactly 1.0 on every limited MOSFET step, which is the one
    place the number is worth having. It reports the strongest factor applied
    to any component instead."""
    frames, watch = collect()

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([10.0, 1e6]), np.ones(2))

    def cap_the_first(delta: np.ndarray) -> np.ndarray:
        limited = delta.copy()
        limited[0] = delta[0] * 0.1
        return limited

    newton_solve(
        assemble,
        np.zeros(2),
        limit=cap_the_first,
        max_iterations=1,
        on_iteration=watch,
    )

    assert frames[1].limited
    assert frames[1].damping == pytest.approx(0.1)


def test_damping_ignores_components_newton_did_not_ask_to_move() -> None:
    """A zero entry in the raw update has no factor to report, and dividing
    by it would make the whole frame nan."""
    frames, watch = collect()

    def assemble(x: np.ndarray) -> System:
        return diagonal_system(x - np.array([0.0, 100.0]), np.ones(2))

    newton_solve(
        assemble,
        np.zeros(2),
        max_step=5.0,
        max_iterations=1,
        on_iteration=watch,
    )

    assert frames[1].damping == pytest.approx(0.05)
