"""Tests for solve/gummel.py.

The driver knows nothing about semiconductors. It takes a state, a list of
block steps, and cycles them until the largest update in a cycle falls below a
tolerance. Everything here is therefore tested on small algebraic fixed point
problems rather than on a device, which is the point: if a test in this file
needed a carrier density, the module boundary in docs/03-architecture.md would
already be broken.

The model problem is a 2x2 linear system solved by block Gauss-Seidel:

    x = (b1 - c*y) / a1
    y = (b2 - c*x) / a2

which converges linearly at rate c^2/(a1*a2). That is the same convergence
character real Gummel has, and it degrades the same way as the coupling grows,
which is what makes it a fair stand in.
"""

from __future__ import annotations

import pytest

from ddsim.solve.gummel import GummelResult, gummel_solve

State = tuple[float, float]


def gauss_seidel_steps(
    a1: float, a2: float, c: float, b1: float = 1.0, b2: float = 1.0
) -> list:
    """Two block steps for the 2x2 system, each returning its own update size."""

    def solve_x(state: State) -> tuple[State, float]:
        x, y = state
        updated = (b1 - c * y) / a1
        return (updated, y), abs(updated - x)

    def solve_y(state: State) -> tuple[State, float]:
        x, y = state
        updated = (b2 - c * x) / a2
        return (x, updated), abs(updated - y)

    return [solve_x, solve_y]


def exact_solution(a1: float, a2: float, c: float) -> State:
    """The fixed point, for comparison."""
    determinant = a1 * a2 - c * c
    return ((a2 - c) / determinant, (a1 - c) / determinant)


# ------------------------------------------------------------- convergence


def test_converges_on_a_weakly_coupled_system() -> None:
    result = gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))

    assert result.converged
    expected = exact_solution(4.0, 5.0, 1.0)
    assert abs(result.state[0] - expected[0]) < 1e-10
    assert abs(result.state[1] - expected[1]) < 1e-10


def test_convergence_is_linear() -> None:
    """Successive updates fall by a constant factor, not a squaring one.

    Worth pinning, because it is the reason Phase 3 exists. If this ever
    started looking quadratic, something would have quietly become a Newton
    solve.
    """
    result = gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))
    history = result.update_history

    ratios = [
        later / earlier
        for earlier, later in zip(history[2:-1], history[3:], strict=True)
        if earlier > 0.0
    ]
    assert ratios, "no usable ratios in the history"
    for ratio in ratios:
        assert abs(ratio - ratios[0]) < 1e-6


def test_stronger_coupling_takes_more_iterations() -> None:
    """Gummel degrades as the equations couple, which is its whole weakness."""
    weak = gummel_solve((0.0, 0.0), gauss_seidel_steps(10.0, 10.0, 1.0))
    strong = gummel_solve((0.0, 0.0), gauss_seidel_steps(10.0, 10.0, 9.0))

    assert weak.converged and strong.converged
    assert strong.iterations > weak.iterations


def test_reports_failure_when_the_iteration_diverges() -> None:
    """Coupling above the diagonal diverges, and must be reported, not returned.

    A diverged Gummel that reports success is the worst outcome available,
    because the state looks like a solution.
    """
    result = gummel_solve(
        (0.0, 0.0), gauss_seidel_steps(1.0, 1.0, 2.0), max_iterations=30
    )

    assert not result.converged
    assert "did not converge" in result.message


def test_stops_early_on_a_non_finite_update() -> None:
    """An overflowed update ends the solve rather than burning the budget."""

    def explode(state: State) -> tuple[State, float]:
        return state, float("inf")

    result = gummel_solve((0.0, 0.0), [explode], max_iterations=100)

    assert not result.converged
    assert result.iterations == 1
    assert "not finite" in result.message


def test_an_exact_step_converges_in_one_cycle() -> None:
    """A step that lands on the answer and then stops moving is converged.

    The second cycle is what proves it: the update is zero only because the
    state stopped changing, not because the first cycle was skipped.
    """

    def land(state: State) -> tuple[State, float]:
        x, _ = state
        return (1.0, 0.0), abs(1.0 - x)

    result = gummel_solve((0.0, 0.0), [land])

    assert result.converged
    assert result.iterations == 2
    assert result.state == (1.0, 0.0)


# ---------------------------------------------------------------- mechanics


def test_steps_run_in_the_order_given() -> None:
    """Poisson, then n, then p. Reordering changes the answer path."""
    order: list[str] = []

    def record(label: str):
        def step(state: State) -> tuple[State, float]:
            order.append(label)
            return state, 0.0

        return step

    gummel_solve((0.0, 0.0), [record("psi"), record("n"), record("p")])

    assert order[:3] == ["psi", "n", "p"]


def test_update_history_records_one_entry_per_cycle() -> None:
    result = gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))

    assert len(result.update_history) == result.iterations


def test_the_cycle_update_is_the_largest_of_its_steps() -> None:
    """One slow block must not be hidden by a fast one."""

    def big(state: State) -> tuple[State, float]:
        return state, 7.0

    def small(state: State) -> tuple[State, float]:
        return state, 0.1

    result = gummel_solve((0.0, 0.0), [small, big], max_iterations=1)

    assert result.update_history[0] == 7.0


def test_the_original_state_is_not_mutated() -> None:
    """The driver threads state through rather than editing in place."""
    start = (0.0, 0.0)
    gummel_solve(start, gauss_seidel_steps(4.0, 5.0, 1.0))

    assert start == (0.0, 0.0)


def test_max_iterations_is_respected() -> None:
    result = gummel_solve(
        (0.0, 0.0), gauss_seidel_steps(1.0, 1.0, 2.0), max_iterations=7
    )

    assert result.iterations == 7


def test_an_empty_step_list_raises() -> None:
    """A cycle with no blocks would report convergence having done nothing."""
    with pytest.raises(ValueError, match="at least one"):
        gummel_solve((0.0, 0.0), [])


def test_a_non_positive_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0), update_tol=0.0)


def test_repr_says_whether_it_converged() -> None:
    result = gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))

    assert "converged" in repr(result)
    assert str(result.iterations) in repr(result)


def test_result_is_generic_over_the_state_type() -> None:
    """Any state object works, which is what keeps this reusable."""

    def append(state: list[int]) -> tuple[list[int], float]:
        return [*state, len(state)], 0.0

    result: GummelResult[list[int]] = gummel_solve([], [append])

    assert result.state == [0]
