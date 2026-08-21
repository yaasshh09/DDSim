"""Tests for solve/continuation.py.

The bias ramp driver, and like everything in solve/ it knows nothing about
semiconductors. It walks a scalar parameter from where it is to where it needs
to be, growing the step when a solve succeeds and halving it when one fails,
reusing the previous solution as the next initial guess.

docs/02-numerics.md gives the algorithm and docs/05-pitfalls.md gives the two
traps: growing by 2 overshoots into non-convergence repeatedly and wastes more
time than it saves, so growth is 1.5 and capped, and there is no such thing as
a good initial guess at 1 V forward bias, so the previous solution is the only
starting point that works.

The solves here are fakes. A fake that refuses to converge outside a window is
a better test of the halving logic than any real device, because the window
edge is exactly where the driver has to behave.
"""

from __future__ import annotations

import pytest

from ddsim.solve.continuation import continue_to


def always(value: float) -> float:
    """A solve that always converges, returning the parameter as the solution."""
    return value


def record_calls(calls: list[tuple[float, float]]):
    """A solve that records the parameter and the guess it was handed."""

    def solve(parameter: float, guess: float) -> float:
        calls.append((parameter, guess))
        return parameter

    return solve


def fails_beyond(limit: float, minimum_step: float):
    """Converges only when the step from the last accepted point is small enough.

    Stands in for a real solver near high injection, where a large bias jump
    lands outside the basin of attraction and a small one does not.
    """
    accepted = [0.0]

    def solve(parameter: float, guess: float) -> float | None:
        if parameter > limit and abs(parameter - accepted[-1]) > minimum_step:
            return None
        accepted.append(parameter)
        return parameter

    return solve


# ---------------------------------------------------------------- the happy path


def test_reaches_the_target_exactly() -> None:
    """The last step is clipped, so the target is hit and not merely passed.

    Float accumulation of 0.05 twenty times does not land on 1.0, and a bias
    sweep that reports 0.9999999999 V is a nuisance in every plot downstream.
    """
    result = continue_to(
        lambda value, guess: always(value),
        start=0.0,
        target=1.0,
        initial=0.0,
        step=0.05,
    )

    assert result.converged
    assert result.parameter == 1.0
    assert result.solution == 1.0


def test_never_overshoots_the_target() -> None:
    calls: list[tuple[float, float]] = []
    continue_to(
        record_calls(calls), start=0.0, target=0.3, initial=0.0, step=0.11
    )

    assert max(parameter for parameter, _ in calls) <= 0.3


def test_walks_downward_too() -> None:
    """Reverse bias ramps run the other way, and the sign is the driver's job."""
    calls: list[tuple[float, float]] = []
    result = continue_to(
        record_calls(calls), start=0.0, target=-1.0, initial=0.0, step=0.25
    )

    assert result.converged
    assert result.parameter == -1.0
    assert all(parameter <= 0.0 for parameter, _ in calls)


def test_hands_each_solve_the_previous_solution() -> None:
    """The entire reason continuation works, from docs/02-numerics.md."""
    calls: list[tuple[float, float]] = []
    continue_to(record_calls(calls), start=0.0, target=1.0, initial=0.0, step=0.25)

    for previous, current in zip(calls, calls[1:], strict=False):
        assert current[1] == previous[0]


def test_a_target_equal_to_the_start_does_nothing() -> None:
    calls: list[tuple[float, float]] = []
    result = continue_to(
        record_calls(calls), start=0.5, target=0.5, initial=7.0, step=0.1
    )

    assert result.converged
    assert result.solution == 7.0
    assert calls == []


# ------------------------------------------------------------- step control


def test_the_step_grows_by_the_growth_factor() -> None:
    calls: list[tuple[float, float]] = []
    continue_to(
        record_calls(calls),
        start=0.0,
        target=100.0,
        initial=0.0,
        step=1.0,
        growth=1.5,
        max_step=1e9,
    )

    positions = [0.0, *[parameter for parameter, _ in calls]]
    taken = [
        later - earlier
        for earlier, later in zip(positions[:-1], positions[1:], strict=True)
    ]
    assert taken[0] == pytest.approx(1.0)
    assert taken[1] == pytest.approx(1.5)
    assert taken[2] == pytest.approx(2.25)


def test_the_step_is_capped() -> None:
    calls: list[tuple[float, float]] = []
    continue_to(
        record_calls(calls),
        start=0.0,
        target=100.0,
        initial=0.0,
        step=1.0,
        growth=1.5,
        max_step=2.0,
    )

    positions = [0.0, *[parameter for parameter, _ in calls]]
    taken = [
        later - earlier
        for earlier, later in zip(positions[:-1], positions[1:], strict=True)
    ]
    assert max(taken) <= 2.0 + 1e-12


def test_a_failed_step_is_halved_and_retried() -> None:
    """The requirement phases/PHASE-2.md states in as many words."""
    result = continue_to(
        fails_beyond(0.5, 0.2), start=0.0, target=1.0, initial=0.0, step=0.5
    )

    assert result.converged
    assert result.parameter == 1.0

    rejected = [event for event in result.events if not event.converged]
    assert rejected, "nothing was rejected, so the halving path never ran"
    for event in rejected:
        assert "halved" in event.message


def test_every_attempt_is_logged() -> None:
    """A logged event per attempt, accepted or not."""
    result = continue_to(
        fails_beyond(0.5, 0.2), start=0.0, target=1.0, initial=0.0, step=0.5
    )

    assert len(result.events) > len(result.accepted)
    assert [event.parameter for event in result.events if event.converged] == list(
        result.accepted
    )


def test_gives_up_below_the_minimum_step() -> None:
    """Failing loudly beats halving forever."""
    result = continue_to(
        lambda value, guess: None,
        start=0.0,
        target=1.0,
        initial=0.0,
        step=0.1,
        min_step=0.01,
    )

    assert not result.converged
    assert result.parameter == 0.0
    assert result.solution == 0.0
    assert "minimum step" in result.message


def test_the_partial_solution_survives_a_failure() -> None:
    """Whatever was reached is returned, because it is worth inspecting."""
    result = continue_to(
        fails_beyond(0.4, 1e-9),
        start=0.0,
        target=1.0,
        initial=0.0,
        step=0.1,
        min_step=0.01,
    )

    assert not result.converged
    assert 0.0 < result.parameter <= 0.4
    assert result.solution == result.parameter


def test_running_out_of_attempts_is_reported() -> None:
    result = continue_to(
        lambda value, guess: always(value),
        start=0.0,
        target=1e6,
        initial=0.0,
        step=1.0,
        max_step=1.0,
        max_attempts=5,
    )

    assert not result.converged
    assert "attempts" in result.message


# ------------------------------------------------------------- input checks


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step": 0.0}, "step must be positive"),
        ({"step": -0.1}, "step must be positive"),
        ({"step": 0.1, "growth": 1.0}, "growth"),
        ({"step": 0.1, "min_step": 0.5}, "min_step"),
        ({"step": 0.1, "max_step": 0.05}, "max_step"),
        ({"step": 0.1, "max_attempts": 0}, "max_attempts"),
    ],
)
def test_bad_arguments_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        continue_to(
            lambda value, guess: always(value),
            start=0.0,
            target=1.0,
            initial=0.0,
            **kwargs,
        )


def test_repr_reports_where_it_got_to() -> None:
    result = continue_to(
        lambda value, guess: always(value), start=0.0, target=1.0, initial=0.0, step=0.5
    )

    assert "1" in repr(result)
    assert "converged" in repr(result)
