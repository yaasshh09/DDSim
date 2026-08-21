"""Generic continuation, the bias ramp driver.

Never jump to the target. Ramp, reusing the previous converged solution as the
initial guess for the next point. That reuse is the entire reason continuation
works: docs/05-pitfalls.md is blunt that there is no such thing as a good
initial guess at 1 V forward bias.

    value = start
    while value != target:
        try to solve at value + step, from the solution at value
        if it converged:  accept it, grow the step by 1.5, capped
        otherwise:        halve the step and try again
        if the step falls below the floor: stop and say so

Growth of 1.5 rather than 2 is deliberate, from docs/05-pitfalls.md: doubling
overshoots into non-convergence repeatedly and wastes more time than it saves.

Nothing here knows what the parameter means. It is a bias in this project and a
source stepping factor in the SPICE layer, and this file is meant to be lifted
into that project unchanged, which an import graph test enforces.

The solve callback returns None to mean it did not converge. An exception would
be more expressive, but generic code cannot catch a specific exception type
without knowing what the caller raises, and catching everything would swallow
real bugs. Returning None is the narrow contract, so the caller decides what
counts as a failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

SolutionT = TypeVar("SolutionT")
"""Whatever the callback returns. The driver only ever passes it back in."""

SolveStep = Callable[[float, SolutionT], "SolutionT | None"]
"""Solve at a parameter value, starting from a previous solution.

Returns the new solution, or None if it did not converge.
"""


@dataclass(frozen=True)
class ContinuationEvent:
    """One attempt, accepted or rejected. The log the phase asks for."""

    parameter: float
    """The parameter value attempted."""

    step: float
    """The step size used to reach it, always positive."""

    converged: bool
    """Whether the solve at this value succeeded."""

    message: str = ""
    """What was done about it, when it failed."""

    def __repr__(self) -> str:
        outcome = "ok" if self.converged else "failed"
        return f"{self.parameter:+.6g} step {self.step:.3g} {outcome}"


@dataclass(frozen=True)
class ContinuationResult(Generic[SolutionT]):
    """Where the ramp got to, and how it went."""

    parameter: float
    """The last parameter value with a converged solution."""

    solution: SolutionT
    """The solution there. The partial result is kept even on failure, because
    the state at the point where a ramp stalls is exactly what you want to
    look at."""

    converged: bool
    """Whether the target was reached."""

    events: tuple[ContinuationEvent, ...] = ()
    """Every attempt, in order."""

    message: str = ""
    """Why the ramp stopped, when it did not reach the target."""

    @property
    def accepted(self) -> tuple[float, ...]:
        """The parameter values that converged, in order."""
        return tuple(event.parameter for event in self.events if event.converged)

    def __repr__(self) -> str:
        state = "converged" if self.converged else "stalled"
        return (
            f"ContinuationResult {state} at {self.parameter:+.6g} "
            f"after {len(self.events)} attempts"
        )


def continue_to(
    solve: SolveStep[SolutionT],
    *,
    start: float,
    target: float,
    initial: SolutionT,
    step: float,
    min_step: float | None = None,
    max_step: float | None = None,
    growth: float = 1.5,
    max_attempts: int = 200,
) -> ContinuationResult[SolutionT]:
    """Ramp the parameter from start to target, adapting the step size.

    Args:
        solve: given a parameter value and the previous solution, returns a
            new solution or None if it did not converge.
        start: parameter value that `initial` was solved at.
        target: parameter value wanted. May be above or below start.
        initial: the solution at start.
        step: first step size, always positive. Direction comes from the sign
            of target minus start, not from this.
        min_step: give up when the step would fall below this. Defaults to a
            thousandth of the first step.
        max_step: cap on step growth. None means the only cap is the target.
        growth: factor the step grows by after a success. 1.5 by default.
        max_attempts: total solve calls allowed, failures included.

    Returns a ContinuationResult rather than raising, matching newton_solve.
    A stalled ramp is a measurement, not an accident: phases/PHASE-2.md asks
    for the bias at which Gummel gives up to be documented, and that number
    comes out of this function.

    The last step is clipped so the target is landed on exactly rather than
    passed. A failed step is halved from the size actually attempted, not from
    the nominal one, which is what guarantees the retry is a different point
    when the attempt was already clipped.
    """
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if growth <= 1.0:
        raise ValueError(f"growth must be above 1, got {growth}")
    if min_step is None:
        min_step = step * 1e-3
    if min_step <= 0.0 or min_step > step:
        raise ValueError(
            f"min_step must be positive and no larger than step, got {min_step}"
        )
    if max_step is not None and max_step < step:
        raise ValueError(
            f"max_step={max_step} is below the starting step={step}"
        )
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    value = start
    solution = initial
    events: list[ContinuationEvent] = []

    if target == start:
        return ContinuationResult(
            parameter=value, solution=solution, converged=True, events=()
        )

    direction = 1.0 if target > start else -1.0
    step_size = step
    message = ""

    while value != target and len(events) < max_attempts:
        trial = value + direction * step_size
        if direction * (trial - target) > 0.0:
            trial = target
        attempted = abs(trial - value)

        candidate = solve(trial, solution)

        if candidate is not None:
            value = trial
            solution = candidate
            events.append(ContinuationEvent(trial, attempted, True))
            step_size = attempted * growth
            if max_step is not None:
                step_size = min(step_size, max_step)
            continue

        step_size = attempted / 2.0
        events.append(
            ContinuationEvent(
                trial,
                attempted,
                False,
                f"did not converge, step halved to {step_size:.4g}",
            )
        )
        if step_size < min_step:
            message = (
                f"stalled at {value:+.6g} on the way to {target:+.6g}: the step "
                f"fell below the minimum step of {min_step:.4g}. Reducing it "
                "further will not help, the solver has left its basin of "
                "attraction."
            )
            break

    if not message and value != target:
        message = (
            f"stalled at {value:+.6g} on the way to {target:+.6g}: ran out of "
            f"attempts after {len(events)} of them."
        )

    return ContinuationResult(
        parameter=value,
        solution=solution,
        converged=value == target,
        events=tuple(events),
        message=message,
    )
