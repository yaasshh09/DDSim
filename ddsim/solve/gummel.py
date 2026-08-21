"""Decoupled block iteration, the Gummel map.

Knows nothing about semiconductors, like everything else in this package. It
takes a state and an ordered list of block steps and cycles them until the
largest update in a cycle falls below a tolerance. What the state is, and what
each block does to it, is entirely the caller's business.

That generality is not decoration. docs/03-architecture.md requires solve/ to
be liftable into the SPICE layer unchanged, and an import graph test enforces
it for every file here. The semiconductor content of Gummel iteration, which is
that the blocks are Poisson, then electron continuity, then hole continuity,
lives in device/ where it belongs.

    1. Solve nonlinear Poisson for psi, holding the quasi-Fermi levels fixed
    2. Solve the electron continuity equation for n
    3. Solve the hole continuity equation for p
    4. Check the update norm, repeat

Convergence is linear, from docs/02-numerics.md. It is robust at low bias and
degrades badly at high injection, where the coupling between the three
equations is strong. That degradation is expected and is the entire reason
Phase 3 exists. Do not fight it here.

Each block step returns its own update size, because only the block knows what
a meaningful measure of change is for its own variable. A potential is measured
in units of V_T, while a carrier density spanning twenty decades is measured by
the shift in its quasi-Fermi level rather than by any relative change of the
density itself.

A block that cannot proceed raises. The driver does not catch, because generic
code catching arbitrary exceptions would swallow real bugs. A caller that wants
a failed solve reported rather than raised wraps its own steps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
"""Whatever the caller iterates on. The driver only ever passes it along."""

BlockStep = Callable[[StateT], tuple[StateT, float]]
"""One block: takes the state, returns the new state and the update size."""


@dataclass(frozen=True)
class GummelResult(Generic[StateT]):
    """Outcome of a Gummel solve, with the history needed to judge it."""

    state: StateT
    """The final state, converged or not."""

    converged: bool
    """Whether the cycle update fell below the tolerance."""

    iterations: int
    """Number of complete cycles run."""

    update_history: list[float] = field(default_factory=list)
    """The largest block update in each cycle. One entry per iteration. On a
    log scale it should fall along a straight line, since Gummel converges
    linearly. A flattening tail means the coupling has taken over."""

    message: str = ""
    """Why the solve stopped, when it did not converge."""

    def __repr__(self) -> str:
        state = "converged" if self.converged else "did not converge"
        last = self.update_history[-1] if self.update_history else float("nan")
        return (
            f"GummelResult {state} in {self.iterations} iterations, "
            f"final update {last:.3e}"
        )


def gummel_solve(
    state: StateT,
    steps: Sequence[BlockStep[StateT]],
    update_tol: float = 1e-8,
    max_iterations: int = 200,
) -> GummelResult[StateT]:
    """Cycle the blocks until the largest update in a cycle is small.

    Args:
        state: the starting state. Not modified. Each step returns a new one.
        steps: the blocks, run in this order, once per cycle.
        update_tol: convergence threshold on the largest update in a cycle.
        max_iterations: give up after this many cycles.

    Returns a GummelResult rather than raising when it fails to converge. A
    failed solve is information the caller wants to inspect, and at high
    injection failing to converge is the expected outcome rather than an
    exceptional one.

    The cycle update is the largest of the block updates, never their sum or
    their average, so that one slow block cannot be hidden by a fast one.
    """
    if not steps:
        raise ValueError(
            "a Gummel cycle needs at least one block step, otherwise it would "
            "report convergence having done nothing"
        )
    if update_tol <= 0.0:
        raise ValueError(f"update_tol must be positive, got {update_tol}")

    update_history: list[float] = []
    message = ""

    for iteration in range(1, max_iterations + 1):
        cycle_update = 0.0
        for step in steps:
            state, update = step(state)
            cycle_update = max(cycle_update, update)

        update_history.append(cycle_update)

        if not isfinite(cycle_update):
            message = (
                f"update was not finite at iteration {iteration}, the "
                "iteration has diverged. Check signs before reaching for "
                "damping, per docs/05-pitfalls.md."
            )
            break

        if cycle_update < update_tol:
            return GummelResult(
                state=state,
                converged=True,
                iterations=iteration,
                update_history=update_history,
            )

    if not message:
        message = (
            f"did not converge in {max_iterations} iterations, "
            f"final update {update_history[-1]:.3e}"
        )

    return GummelResult(
        state=state,
        converged=False,
        iterations=len(update_history),
        update_history=update_history,
        message=message,
    )
