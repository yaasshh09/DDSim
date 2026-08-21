"""Terminal current and I-V sweeps. Post processing only, no solving.

Sign convention, fixed here and inherited by everything downstream: a terminal
current is positive when conventional current flows from the contact into the
device. A forward biased diode has a positive anode current.

Where a terminal current comes from
-----------------------------------
Not from the edge flux next to the contact, but from the continuity residual at
the contact node before its row is replaced. The discrete balance at a contact
node c is

    (sum of interior fluxes out of c) - In_c = R_c * volume_c

because the contact face carries a current In_c into the cell that no interior
edge accounts for. The electron residual is written as
F_n = R*volume - div(Jn), so In_c is exactly -F_n[c]. The same argument on the
hole equation, where div(Jp) = -R, gives Ip_c = +F_p[c]. So

    I_c = -F_n[c] + F_p[c]

Two things fall out of writing it this way. The recombination in the contact
half cell cancels between the two carriers, which it must, since an electron
and a hole recombine as a pair and carry no net charge away. And the terminal
currents sum to zero identically: summing either residual over every node
telescopes the divergence to nothing and leaves the integrated recombination,
which then cancels between the two carriers.

Accuracy
--------
Jn is the difference of two edge terms of size (Dn/h)*n, and near equilibrium
they cancel almost completely, which costs digits. The contact is the best
place in the device to pay that cost: the mesh is coarsest there and the
majority carrier flux terms sit around 1e8 in scaled units against a forward
current of 1e3, so about eleven digits survive. Next to the junction, where h
is a hundred times smaller, far fewer would.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias
from ddsim.discretize.continuity import (
    assemble_electron_continuity,
    assemble_hole_continuity,
    electron_current,
    hole_current,
)
from ddsim.solve.continuation import continue_to


def current_densities(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
) -> tuple[Field, Field]:
    """(Jn, Jp) on every edge [A/cm^2], physical units.

    Args:
        device: the device the state was solved on.
        state: a solved state.
        models: the transport models used for the solve. Built from the device
            if None, which matches what solve_bias does by default.

    The pair is what the current continuity invariant is measured on: with
    recombination off, Jn + Jp is the same number on every edge.
    """
    if models is None:
        models = TransportModels.for_device(device)

    scale = device.scale
    h = device.mesh.h / scale.x_0

    Jn = electron_current(h, models.Dn, state.psi.data, state.n.data)
    Jp = hole_current(h, models.Dp, state.psi.data, state.p.data)

    return (
        Field(Jn, "A/cm^2", ScalingState.SCALED, Location.EDGE, name="Jn").to_physical(
            scale
        ),
        Field(Jp, "A/cm^2", ScalingState.SCALED, Location.EDGE, name="Jp").to_physical(
            scale
        ),
    )


def continuity_residuals(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The two continuity residuals with no boundary rows applied [1], scaled.

    Interior entries are zero for a converged solution. Contact entries are the
    terminal currents, which is the whole reason to look at them.
    """
    if models is None:
        models = TransportModels.for_device(device)

    electrons = assemble_electron_continuity(
        device.mesh,
        state.psi,
        state.n,
        state.p,
        models.recombination,
        device.scale,
        models.Dn,
    )
    holes = assemble_hole_continuity(
        device.mesh,
        state.psi,
        state.n,
        state.p,
        models.recombination,
        device.scale,
        models.Dp,
    )
    return electrons.residual, holes.residual


def terminal_currents(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
) -> dict[str, float]:
    """Current into the device at each contact [A/cm^2], by contact name.

    Positive means conventional current flowing from the contact into the
    device, so a forward biased diode has a positive anode current. The values
    sum to zero for a converged solution.
    """
    electron_residual, hole_residual = continuity_residuals(device, state, models)

    return {
        contact.name: float(
            (-electron_residual[contact.node] + hole_residual[contact.node])
            * device.scale.J_0
        )
        for contact in device.contacts
    }


def total_current(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
    contact: str | None = None,
) -> float:
    """The current through the device [A/cm^2], measured at one contact.

    Defaults to the first contact, which for a diode built by pn_diode is the
    anode. In steady state the total current is divergence free, so every
    contact reports the same magnitude with opposite signs.
    """
    currents = terminal_currents(device, state, models)
    if contact is None:
        contact = device.contacts[0].name
    return currents[contact]


@dataclass(frozen=True)
class IVPoint:
    """One bias point of a sweep."""

    voltage: float
    """Applied bias at the swept contact [V]."""

    current: float
    """Current into that contact [A/cm^2]."""

    state: DeviceState
    """The converged solution, kept for band diagrams and profile plots."""


@dataclass(frozen=True)
class IVCurve:
    """A bias sweep, complete or as far as it got."""

    contact: str
    """Name of the swept contact."""

    points: tuple[IVPoint, ...]
    """The converged bias points, in the order they were requested."""

    complete: bool
    """Whether every requested voltage was reached."""

    message: str = ""
    """Why the sweep stopped, when it did not finish."""

    @property
    def voltage(self) -> npt.NDArray[np.float64]:
        """Applied bias at each point [V]."""
        return np.array([point.voltage for point in self.points])

    @property
    def current(self) -> npt.NDArray[np.float64]:
        """Terminal current at each point [A/cm^2]."""
        return np.array([point.current for point in self.points])

    def __repr__(self) -> str:
        state = "complete" if self.complete else "stopped early"
        if not self.points:
            return f"IVCurve {self.contact} empty, {state}"
        return (
            f"IVCurve {self.contact} {len(self.points)} points "
            f"{self.voltage[0]:+.3g} to {self.voltage[-1]:+.3g} V, {state}"
        )


def iv_sweep(
    device: Device,
    contact: str,
    voltages: list[float],
    models: TransportModels | None = None,
    step: float = 0.05,
    start: float = 0.0,
    max_iterations: int = 200,
    update_tol: float = 1e-8,
) -> IVCurve:
    """Sweep one contact through a list of biases, continuing between them.

    Args:
        device: the device. Its own contact biases are overridden by the sweep.
        contact: name of the contact to sweep.
        voltages: the biases wanted [V], in the order they should be walked.
        models: transport models, built from the device if None.
        step: first continuation step between requested points [V].
        start: bias to begin from [V], solved directly rather than ramped to.
        max_iterations: Gummel budget at each point.
        update_tol: Gummel convergence threshold.

    The list is walked in the order given, each point continued from the last,
    so it should be monotone or nearly so. A sweep from -1 V to 0.5 V is a
    single ascending list: the first leg ramps down from the starting bias and
    the rest walk back up, and every leg reuses the solution before it.

    Returns everything it reached. A sweep that stalls is a measurement rather
    than an accident: phases/PHASE-2.md asks for the bias at which Gummel gives
    up, and that number is the last voltage in a curve marked incomplete.
    """
    if not any(existing.name == contact for existing in device.contacts):
        raise KeyError(
            f"no contact named {contact!r} on this device, which has "
            f"{sorted(existing.name for existing in device.contacts)}"
        )

    if models is None:
        models = TransportModels.for_device(device)

    def at_bias(voltage: float, guess: DeviceState | None) -> DeviceState | None:
        biased = device.with_bias(**{contact: voltage})
        solved = solve_bias(
            biased,
            models=models,
            guess=guess,
            update_tol=update_tol,
            max_iterations=max_iterations,
        )
        if solved.gummel is None or not solved.gummel.converged:
            return None
        return solved

    first = at_bias(start, None)
    if first is None:
        raise RuntimeError(
            f"the sweep could not be started: the solve at {start:+g} V did not "
            "converge. Every bias point is continued from this one."
        )

    points: list[IVPoint] = []
    position = start
    state = first

    for target in voltages:
        ramp = continue_to(
            at_bias,
            start=position,
            target=target,
            initial=state,
            step=step,
            max_step=step,
        )
        state = ramp.solution
        position = ramp.parameter

        if not ramp.converged:
            return IVCurve(
                contact=contact,
                points=tuple(points),
                complete=False,
                message=(
                    f"stalled on the way to {target:+g} V. {ramp.message}"
                ),
            )

        points.append(
            IVPoint(
                voltage=target,
                current=total_current(
                    device.with_bias(**{contact: target}), state, models, contact
                ),
                state=state,
            )
        )

    return IVCurve(contact=contact, points=tuple(points), complete=True)
