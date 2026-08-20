"""Damped Newton with step limiting.

Knows nothing about semiconductors, and must not learn. It takes a callable
that returns a residual and a Jacobian and drives it to convergence. That is
what lets this and continuation.py be lifted into the SPICE layer unchanged.
An import graph test enforces it for every file in this package.

Convergence, from docs/02-numerics.md, needs both of:

    1. update norm    max |dx| < update_tol
    2. residual norm  max |F|  < residual_tol

Checking only the update norm reports a stalled solve as a success, which is
worse than reporting a failure, because the stalled answer looks plausible.

Step limiting, rather than a line search. docs/02-numerics.md prescribes
5 * V_T per Newton step, which is 5.0 in scaled units. The update direction is
preserved by scaling the whole vector by one factor. Clamping entries
individually would rotate the search direction and destroy the quadratic tail.

Damping only ever slows convergence, it does not fix a wrong sign. If a solve
diverges, check signs before touching max_step. docs/05-pitfalls.md and
CLAUDE.md both say so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ddsim.solve.linear import SparseLU


class Assembly(Protocol):
    """A residual and a Jacobian in COO form.

    Structural, so that an assembly type from another package satisfies it
    without this module importing that package.
    """

    residual: npt.NDArray[np.float64]
    rows: npt.NDArray[np.int64]
    cols: npt.NDArray[np.int64]
    values: npt.NDArray[np.float64]
    shape: tuple[int, int]


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of a Newton solve, including the history needed to judge it."""

    x: npt.NDArray[np.float64]
    """The final iterate, converged or not."""

    converged: bool
    """Whether both convergence criteria were met."""

    iterations: int
    """Number of Newton steps taken."""

    residual_history: list[float] = field(default_factory=list)
    """max |F| before each step, plus once more at the end. Length is
    iterations + 1. Plot it on a log scale to see the quadratic tail."""

    update_history: list[float] = field(default_factory=list)
    """max |dx| for each step. Length is iterations."""

    limited_steps: int = 0
    """How many steps hit the step limit. A converged solve should end with
    several unlimited steps, otherwise the tail is not really quadratic."""

    message: str = ""
    """Why the solve stopped, when it did not converge."""

    def __repr__(self) -> str:
        state = "converged" if self.converged else "did not converge"
        final = self.residual_history[-1] if self.residual_history else float("nan")
        return (
            f"NewtonResult {state} in {self.iterations} iterations, "
            f"final residual {final:.3e}"
        )


def newton_solve(
    assemble: Callable[[npt.NDArray[np.float64]], Assembly],
    x0: npt.NDArray[np.float64],
    max_step: float | None = None,
    residual_tol: float = 1e-10,
    update_tol: float = 1e-10,
    max_iterations: int = 50,
) -> NewtonResult:
    """Solve F(x) = 0 by damped Newton.

    Args:
        assemble: given x, returns the residual and Jacobian at x.
        x0: initial guess. Not modified.
        max_step: largest allowed max |dx| per step, or None for no limit.
            5.0 is the scaled value docs/02-numerics.md prescribes for psi.
        residual_tol: convergence threshold on max |F|.
        update_tol: convergence threshold on max |dx|.
        max_iterations: give up after this many steps.

    Returns a NewtonResult rather than raising, including when the Jacobian is
    singular. A failed solve is information the caller usually wants to inspect
    rather than an exception to catch.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    solver = SparseLU()

    residual_history: list[float] = []
    update_history: list[float] = []
    limited_steps = 0
    message = ""

    system = assemble(x)
    residual_norm = float(np.max(np.abs(system.residual)))
    residual_history.append(residual_norm)

    if residual_norm < residual_tol:
        return NewtonResult(
            x=x,
            converged=True,
            iterations=0,
            residual_history=residual_history,
            update_history=update_history,
        )

    for iteration in range(1, max_iterations + 1):
        try:
            solver.factorize(system.rows, system.cols, system.values, system.shape)
            delta = solver.solve(-system.residual)
        except RuntimeError as error:
            message = f"linear solve failed at iteration {iteration}: {error}"
            break

        if not np.all(np.isfinite(delta)):
            message = f"non-finite Newton update at iteration {iteration}"
            break

        step_norm = float(np.max(np.abs(delta)))
        if max_step is not None and step_norm > max_step:
            # Scale the whole vector by one factor. Clamping entry by entry
            # would rotate the direction and break quadratic convergence.
            delta = delta * (max_step / step_norm)
            step_norm = max_step
            limited_steps += 1

        x = x + delta
        update_history.append(step_norm)

        system = assemble(x)
        if not np.all(np.isfinite(system.residual)):
            # A diverged iterate, not a crash. Undamped Newton on a stiff
            # exponential overshoots far enough to overflow exp in one step.
            message = (
                f"residual became non-finite at iteration {iteration}, "
                "the iterate has diverged. Try a smaller max_step, but check "
                "signs before reaching for damping."
            )
            residual_history.append(float("inf"))
            break

        residual_norm = float(np.max(np.abs(system.residual)))
        residual_history.append(residual_norm)

        if step_norm < update_tol and residual_norm < residual_tol:
            return NewtonResult(
                x=x,
                converged=True,
                iterations=iteration,
                residual_history=residual_history,
                update_history=update_history,
                limited_steps=limited_steps,
            )

    if not message:
        message = (
            f"did not converge in {max_iterations} iterations, "
            f"final residual {residual_history[-1]:.3e}, "
            f"final update {update_history[-1] if update_history else float('nan'):.3e}"
        )

    return NewtonResult(
        x=x,
        converged=False,
        iterations=len(update_history),
        residual_history=residual_history,
        update_history=update_history,
        limited_steps=limited_steps,
        message=message,
    )
