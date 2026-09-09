"""Damped Newton with step limiting.

Knows nothing about semiconductors, and must not learn. It takes a callable
that returns a residual and a Jacobian and drives it to convergence. That is
what lets this and continuation.py be lifted into the SPICE layer unchanged.
An import graph test enforces it for every file in this package.

Convergence, from docs/02-numerics.md, needs both of:

    1. update norm    max |dx| < update_tol
    2. residual norm  max |F|  < residual_atol + residual_rtol * max |F_0|

Checking only the update norm reports a stalled solve as a success, which is
worse than reporting a failure, because the stalled answer looks plausible.

The residual threshold is relative to a scale, not absolute. An absolute
threshold is not scale free, and the Poisson residual is proportional to the
doping. Its roundoff floor is proportional to the doping too, so a fixed 1e-10
that works at 1e16 cm^-3 is unreachable at 1e18 and a perfectly good solve gets
reported as a failure. The update norm needs no such treatment, because psi is
measured in units of V_T whatever the doping.

The scale defaults to the initial residual, which is right for a cold start and
wrong for a warm one. Handed a solution it has already found, the solve starts
at its own roundoff floor, and a threshold set a decade below that floor can
never be met however correct the answer is. That is not hypothetical: it is
exactly what a Gummel cycle does on every iteration after the first, and at
thermal equilibrium it does it on the first one too. Callers that start warm
pass residual_scale explicitly, taken from the size of the terms the residual
is built from rather than from where the iteration happened to begin.

Step limiting, rather than a line search. docs/02-numerics.md prescribes
5 * V_T per Newton step, which is 5.0 in scaled units. The update direction is
preserved by scaling the whole vector by one factor. Clamping entries
individually would rotate the search direction and destroy the quadratic tail.

Damping only ever slows convergence, it does not fix a wrong sign. If a solve
diverges, check signs before touching max_step. docs/05-pitfalls.md says the
same thing.
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

    The members are read-only properties rather than plain attributes. A
    mutable protocol member is invariant, and a frozen dataclass cannot
    satisfy it, which would rule out exactly the immutable assembly types this
    is meant to accept.
    """

    @property
    def residual(self) -> npt.NDArray[np.float64]:
        """F(x), one entry per unknown."""

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        """Jacobian row indices."""

    @property
    def cols(self) -> npt.NDArray[np.int64]:
        """Jacobian column indices."""

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Jacobian values."""

    @property
    def shape(self) -> tuple[int, int]:
        """Jacobian shape."""


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
    """The residual size before each step, plus once more at the end, measured
    by whatever residual_norm the solve was given. Length is iterations + 1.
    Plot it on a log scale to see the quadratic tail."""

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
    limit: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
    residual_atol: float = 1e-12,
    residual_rtol: float = 1e-10,
    residual_scale: float | None = None,
    residual_norm: (
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float] | None
    ) = None,
    update_tol: float = 1e-10,
    update_norm: (
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float] | None
    ) = None,
    max_iterations: int = 50,
    stagnation_window: int | None = 4,
    solver: SparseLU | None = None,
) -> NewtonResult:
    """Solve F(x) = 0 by damped Newton.

    Args:
        assemble: given x, returns the residual and Jacobian at x.
        x0: initial guess. Not modified.
        max_step: largest allowed max |dx| per step, or None for no limit.
            Scales the whole update by one factor, which preserves the Newton
            direction. Right for a single unknown per node; wrong for a system
            whose components differ in size by decades, because max |dx| is
            then set by the largest component and the cap shrinks every other
            one along with it.
        limit: a caller supplied damping rule, given the raw update and
            returning the damped one. Mutually exclusive with max_step. This
            is what a coupled solve needs: docs/02-numerics.md prescribes
            capping the psi update at 5*V_T and taking the full n and p
            updates, and no single scalar over the whole vector expresses
            that. A limiter that returns its argument unchanged is not
            counted in limited_steps.
        residual_atol: absolute floor on the residual threshold, for problems
            that start at or near zero residual.
        residual_rtol: residual threshold relative to residual_scale.
        residual_norm: how to measure the size of a residual, given the
            residual vector and the iterate it was evaluated at. Defaults to
            max |F|, which is right when every row is measured against the
            same thing and wrong when they are not. A row is divided by the
            terms it is assembled from, and on a graded device those terms
            span 11.5 decades within one equation family, so a residual
            divided by one number per family says nothing at all about the
            rows where the terms are small. Whatever this returns is what
            lands in residual_history, so the reported number and the applied
            criterion cannot drift apart.
        residual_scale: the size of the terms the residual is built from. The
            initial residual is used when this is None, which is right for a
            cold start and wrong for a warm one: a solve handed the answer
            already starts at the roundoff floor, and a threshold a decade
            below that floor can never be met. Any caller that starts from a
            previous solution should pass a scale that does not depend on the
            starting iterate.
        update_tol: convergence threshold on the update measure below.
        update_norm: how to measure the size of an update, given the damped
            update and the iterate it is about to be added to. Defaults to
            max |dx|, which is right when every unknown is the same kind of
            quantity and wrong when they are not. A coupled solve carries psi
            of order ten beside a density of order 1e6 in scaled units, so
            max |dx| bottoms out six decades above the potential's own floor
            and no threshold suits both. docs/02-numerics.md asks for the
            carrier change as max |dn| / (n + n_i) for that reason. Whatever
            this returns is also what lands in update_history, so the reported
            number and the applied criterion cannot drift apart.
        max_iterations: give up after this many steps.
        stagnation_window: give up early once the residual has been identical
            to the last bit across this many consecutive entries of the
            history while the update is already inside update_tol. Both halves
            are load bearing. A frozen residual on its own is a solver taking
            real steps that happen not to help, which is a different failure
            and gets the whole budget. A frozen residual underneath a settled
            iterate means the residual has hit its own arithmetic floor, and no
            number of further steps can move it: on a 1e12 cm^-3 bar the
            Poisson residual reached its floor at step three and sat there,
            unchanged to the last bit, for the remaining forty-seven. None
            disables the guard and runs the full budget. This never turns a
            success into a failure, because the check runs after the
            convergence test and reports converged=False either way.
        solver: a factorization to reuse across calls. A fresh one is built
            when this is None, which is right for a one-off solve. A caller
            that solves the same system over and over, as every Gummel cycle
            does, should keep one and hand it back in: the sparsity pattern is
            identical every time, and SparseLU keeps the part of the COO to
            CSC conversion that depends only on the pattern. It never reuses
            numbers, so a stale factorization cannot leak into a later solve.

    Returns a NewtonResult rather than raising, including when the Jacobian is
    singular. A failed solve is information the caller usually wants to inspect
    rather than an exception to catch.
    """
    if max_step is not None and limit is not None:
        raise ValueError(
            "max_step and limit are two damping rules for one update. Pass "
            "one. Letting either win silently makes the other look ineffective."
        )

    x = np.array(x0, dtype=np.float64, copy=True)
    if solver is None:
        solver = SparseLU()

    residual_history: list[float] = []
    update_history: list[float] = []
    limited_steps = 0
    message = ""

    def measure(system: Assembly, at: npt.NDArray[np.float64]) -> float:
        if residual_norm is None:
            return float(np.max(np.abs(system.residual)))
        return float(residual_norm(system.residual, at))

    system = assemble(x)
    residual_size = measure(system, x)
    residual_history.append(residual_size)

    # Fixed once, so that the threshold cannot drift as the iteration proceeds.
    reference = residual_size if residual_scale is None else abs(residual_scale)
    residual_threshold = residual_atol + residual_rtol * reference

    if residual_size < residual_threshold:
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

        raw_norm = float(np.max(np.abs(delta)))
        step_norm = raw_norm
        if max_step is not None and step_norm > max_step:
            # Scale the whole vector by one factor. Clamping entry by entry
            # would rotate the direction and break quadratic convergence.
            delta = delta * (max_step / step_norm)
            step_norm = max_step
            limited_steps += 1
        elif limit is not None:
            limited = np.asarray(limit(delta), dtype=np.float64)
            if limited.shape != delta.shape:
                raise ValueError(
                    f"limit returned shape {limited.shape} for an update of "
                    f"shape {delta.shape}. Dropping entries would freeze "
                    "those unknowns at their starting values."
                )
            if not np.array_equal(limited, delta):
                limited_steps += 1
            delta = limited

        # Measured on the damped update, which is the one actually taken, and
        # against the iterate it is being added to rather than the one it
        # produces. A relative measure divides by where the solve is now.
        step_norm = (
            float(np.max(np.abs(delta)))
            if update_norm is None
            else float(update_norm(delta, x))
        )

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

        residual_size = measure(system, x)
        residual_history.append(residual_size)

        if step_norm < update_tol and residual_size < residual_threshold:
            return NewtonResult(
                x=x,
                converged=True,
                iterations=iteration,
                residual_history=residual_history,
                update_history=update_history,
                limited_steps=limited_steps,
            )

        if (
            stagnation_window is not None
            and step_norm < update_tol
            and len(residual_history) >= stagnation_window
            and len(set(residual_history[-stagnation_window:])) == 1
        ):
            message = (
                f"the residual stopped moving at iteration {iteration}: "
                f"{residual_size:.3e} unchanged over the last "
                f"{stagnation_window} evaluations, against a threshold of "
                f"{residual_threshold:.3e}, with the update already down to "
                f"{step_norm:.3e}. The residual is on its arithmetic floor "
                "and the remaining budget cannot move it. Either the "
                "threshold is below that floor, in which case pass a "
                "residual_scale built from the size of the terms, or the "
                "Jacobian is wrong."
            )
            break

    if not message:
        # The threshold belongs in the message. A residual that stops moving
        # while the update is already tiny means the threshold is below the
        # arithmetic floor of the residual, and without the number to compare
        # against that reads exactly like a solve that is merely slow.
        message = (
            f"did not converge in {max_iterations} iterations, "
            f"final residual {residual_history[-1]:.3e} "
            f"against a threshold of {residual_threshold:.3e}, "
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
