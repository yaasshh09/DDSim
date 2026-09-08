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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import (
    TransportModels,
    solve_bias,
    solve_bias_newton,
)
from ddsim.discretize.continuity import electron_current, hole_current
from ddsim.discretize.coupled import (
    coupled_residual,
    edge_drop,
    pack,
    unpack,
)
from ddsim.discretize.geometry import EdgeGeometry
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.physics.mobility import diffusivity_at
from ddsim.solve.continuation import continue_to


def _models_at(
    device: Device, state: DeviceState, models: TransportModels | None
) -> TransportModels:
    """The transport models resolved at the state a current is read from.

    A current has to be computed with the diffusivity the solve converged
    with, and for a surface corrected mobility that is a function of the
    state rather than something built once with the device. Reading it off
    the models as they were built uses the uncorrected bulk mobility, which
    is wrong by the whole size of the correction and shows up as a terminal
    current sum an order of magnitude further from zero than it should be.

    Field dependence is not handled here. That one is resolved edge by edge
    where it is used, because it needs the potential drop across each edge and
    this function has no reason to know about edges.

    Returns the models untouched where there is no surface model, which is
    every device before Phase 5, so nothing recorded earlier moves.
    """
    if models is None:
        models = TransportModels.for_device(device)
    return models.at_state(device, state.psi.data, state.n.data, state.p.data)


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

    The pair is what the current continuity invariant is measured on. In 1D,
    with recombination off, Jn + Jp is the same number on every edge. In 2D it
    is not, and it should not be: the current spreads out, so what is constant
    is the total crossing every cut through the device rather than the density
    on any one edge. See tests/invariant/test_current_continuity_2d.py.

    A density and not a flux, in every dimension. The flux kernels multiply by
    the face the carrier crosses, because that is what a continuity equation
    wants, and it is divided back out here so that the number carries the unit
    its name claims. In 1D that face is exactly 1.0 and the division is exact,
    so nothing measured before this existed moves.

    An edge with no semiconductor face reports exactly zero rather than a
    quotient of two zeros. There is no current density in an insulator to
    report, and zero is the answer every sum over edges wants.
    """
    models = _models_at(device, state, models)

    scale = device.scale
    mesh = device.scaled_mesh

    # A field dependent diffusivity is a function of this state, so it is
    # evaluated at it. The current is being read off a converged solution, so
    # this is the same diffusivity the solve converged with.
    drop = edge_drop(state.psi.data, mesh.geometry)
    Dn = diffusivity_at(models.Dn, drop, mesh.h)
    Dp = diffusivity_at(models.Dp, drop, mesh.h)

    Jn = _per_unit_face(
        electron_current(mesh.h, Dn, state.psi.data, state.n.data, mesh.geometry),
        mesh.geometry,
    )
    Jp = _per_unit_face(
        hole_current(mesh.h, Dp, state.psi.data, state.p.data, mesh.geometry),
        mesh.geometry,
    )

    return (
        Field(Jn, "A/cm^2", ScalingState.SCALED, Location.EDGE, name="Jn").to_physical(
            scale
        ),
        Field(Jp, "A/cm^2", ScalingState.SCALED, Location.EDGE, name="Jp").to_physical(
            scale
        ),
    )


def _per_unit_face(
    flux: npt.NDArray[np.float64], geometry: EdgeGeometry
) -> npt.NDArray[np.float64]:
    """Turn an edge flux back into a current density [1], scaled.

    Zero where the face is zero, which is every edge an insulator touches.
    """
    face = np.asarray(geometry.carrier_face, dtype=np.float64)
    return np.divide(
        flux, face, out=np.zeros_like(flux), where=face > 0.0
    )


def edge_current_face(device: Device) -> npt.NDArray[np.float64]:
    """The face each edge offers a carrier [cm], physical units.

    What a current density has to be multiplied by to give the current through
    that edge per unit depth. The whole dual face in the bulk, half of it on an
    edge lying along a Si/SiO2 interface, and zero on an edge inside an
    insulator. Summing J times this over the edges crossing a plane is the
    discrete surface integral that current continuity is a statement about.
    """
    face = np.asarray(device.scaled_mesh.geometry.carrier_face, dtype=np.float64)
    power = 0 if isinstance(device.mesh, Mesh1D) else 1
    return np.asarray(face * device.scale.x_0**power)


def continuity_residuals(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The two continuity residuals with no boundary rows applied [1], scaled.

    Interior entries are zero for a converged solution. Contact entries are the
    terminal currents, which is the whole reason to look at them.

    Taken from the coupled residual, which is where the two continuity
    equations are written in a form that does not care how many dimensions it
    is in. They are the same two rows the uncoupled assemblies produce, to the
    bit: the same fluxes, the same recombination rate, accumulated in the same
    order. What differs between the coupled and uncoupled paths is the
    Jacobian, and no Jacobian is wanted here.
    """
    models = _models_at(device, state, models)

    mesh = device.scaled_mesh
    residual = coupled_residual(
        h=mesh.h,
        volume=device.charge_volume_scaled,
        x=pack(state.psi.data, state.n.data, state.p.data),
        net_doping=device.net_doping_scaled.data,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
        geometry=mesh.geometry,
        degeneracy=device.degeneracy,
    )
    _, electrons, holes = unpack(residual)
    return np.asarray(electrons), np.asarray(holes)


def terminal_currents(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
) -> dict[str, float]:
    """Current into the device at each contact, by contact name.

    In 1D the value is a current density [A/cm^2]. In 2D the residual is
    integrated over a dual cell that is an area per unit depth, so it is a
    current per unit depth [A/cm], which for a MOSFET is current per unit gate
    width and is how a drain current is quoted.

    Positive means conventional current flowing from the contact into the
    device, so a forward biased diode has a positive anode current. The values
    sum to zero for a converged solution.

    A plate contact is the sum over every node it covers, because the terminal
    is one piece of metal and the current into it is the current into all of
    it. A point contact is that sum over one node, so the two are the same
    statement and there is no 1D branch here.

    A gate reports exactly zero, and that is a statement rather than a
    placeholder: an ideal insulator passes no DC current, so there is nothing
    to compute. Reading it off the residual would be worse than useless, since
    a gate node sits in the oxide and its two continuity rows are pinned, so
    the number there is whatever the pinning says and not a current.
    """
    electron_residual, hole_residual = continuity_residuals(device, state, models)

    currents = {
        contact.name: float(
            sum(
                -electron_residual[node] + hole_residual[node]
                for node in contact.nodes
            )
            * device.scale.J_0
        )
        for contact in device.semiconductor_contacts
    }
    for contact in device.contacts:
        if contact.name not in currents:
            currents[contact.name] = 0.0
    return currents


def total_current(
    device: Device,
    state: DeviceState,
    models: TransportModels | None = None,
    contact: str | None = None,
) -> float:
    """The current through the device at one contact, units as above.

    Defaults to the first contact that carries current, which for a diode
    built by pn_diode is the anode. In steady state the total current is
    divergence free, so on a two terminal device every contact reports the
    same magnitude with opposite signs.
    """
    currents = terminal_currents(device, state, models)
    if contact is None:
        contact = device.semiconductor_contacts[0].name
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

    measured_at: str = ""
    """Name of the terminal the current was read at.

    The swept one on a two terminal sweep, which is what an I-V curve means.
    A transfer curve is the case where the two differ: the gate is swept and
    the drain is measured, and a curve that did not record which was which
    would be ambiguous exactly where it matters, since the source, drain and
    body currents of a MOSFET are three different curves.
    """

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
        swept = self.contact
        if self.measured_at and self.measured_at != self.contact:
            swept = f"{self.contact} into {self.measured_at}"
        if not self.points:
            return f"IVCurve {swept} empty, {state}"
        return (
            f"IVCurve {swept} {len(self.points)} points "
            f"{self.voltage[0]:+.3g} to {self.voltage[-1]:+.3g} V, {state}"
        )


def _walk_sweep(
    device: Device,
    contact: str,
    measured_at: str,
    voltages: list[float],
    models: TransportModels,
    at_bias: Callable[[float, DeviceState | None], DeviceState | None],
    start: float,
    step: float,
    min_step: float | None,
) -> IVCurve:
    """Walk a list of biases, continuing between them, and record the current.

    The loop both public sweeps share. What differs between them is only how
    one bias point is solved, which arrives as `at_bias`, so the continuation
    policy and the bookkeeping are written once.
    """
    # Two different failures, one meaning: there is nothing to continue from.
    # A solver returns a stalled result rather than raising, but the guess it
    # falls back on when given none is the Phase 1 equilibrium solve, and that
    # one does raise.
    try:
        first = at_bias(start, None)
    except RuntimeError as error:
        raise RuntimeError(
            f"the sweep could not be started: no solution exists at "
            f"{start:+g} V to continue from. {error}"
        ) from error

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
            min_step=min_step,
            max_step=step,
        )
        state = ramp.solution
        position = ramp.parameter

        if not ramp.converged:
            return IVCurve(
                contact=contact,
                measured_at=measured_at,
                points=tuple(points),
                complete=False,
                message=f"stalled on the way to {target:+g} V. {ramp.message}",
            )

        points.append(
            IVPoint(
                voltage=target,
                current=total_current(
                    device.with_bias(**{contact: target}),
                    state,
                    models,
                    measured_at,
                ),
                state=state,
            )
        )

    return IVCurve(
        contact=contact,
        measured_at=measured_at,
        points=tuple(points),
        complete=True,
    )


def iv_sweep(
    device: Device,
    contact: str,
    voltages: list[float],
    models: TransportModels | None = None,
    step: float = 0.05,
    min_step: float | None = None,
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
        min_step: give up once the continuation step falls below this [V].
            None leaves it at the continuation default of a thousandth of
            step, which is about ten halvings. That is the right default for a
            sweep that is expected to succeed, and expensive for one that is
            expected to stall: every one of those ten halvings is a full
            failed solve at the Gummel budget. A caller who already knows the
            sweep may stall, or who only wants to know roughly where, should
            raise this.
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
    if not any(existing.name == contact for existing in device.ohmic_contacts):
        raise KeyError(
            f"no contact named {contact!r} on this device, which has "
            f"{sorted(existing.name for existing in device.ohmic_contacts)}"
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

    return _walk_sweep(
        device=device,
        contact=contact,
        measured_at=contact,
        voltages=voltages,
        models=models,
        at_bias=at_bias,
        start=start,
        step=step,
        min_step=min_step,
    )


def gate_sweep(
    device: Device,
    voltages: list[float],
    contact: str = "gate",
    measure_at: str = "drain",
    models: TransportModels | None = None,
    step: float = 0.1,
    min_step: float | None = None,
    start: float = 0.0,
    max_iterations: int = 30,
) -> IVCurve:
    """Sweep the gate and record the drain current: a transfer curve.

    Args:
        device: the MOSFET, carrying the drain and body biases the curve is
            taken at. Its gate bias is overridden by the sweep.
        voltages: the gate biases wanted [V], in the order to walk them.
        contact: name of the terminal to sweep. The gate.
        measure_at: name of the terminal to read the current at. The drain.
        models: transport models, built from the device if None.
        step: first continuation step between requested points [V].
        min_step: give up once the step falls below this [V].
        start: gate bias to begin from [V], solved directly rather than ramped
            to. Zero, which for an NMOS is off.
        max_iterations: Newton budget at each point.

    Two things separate this from iv_sweep, and both of them are why it is a
    separate function rather than a flag on that one.

    **It solves the coupled system.** iv_sweep runs the hybrid Gummel path,
    whose two uncoupled blocks pin a density at every terminal and so cannot
    take a gate at all. The coupled Newton applies every contact in one pass,
    pinning three unknowns at an ohmic node and one at a gate.

    **What is swept and what is measured are different terminals.** No current
    flows in a gate, so a curve of gate bias against gate current is flat at
    zero. The measurement wanted is the drain.
    """
    known = {existing.name for existing in device.contacts}
    for name in (contact, measure_at):
        if name not in known:
            raise KeyError(
                f"no contact named {name!r} on this device, which has "
                f"{sorted(known)}"
            )

    if models is None:
        models = TransportModels.for_device(device)

    def at_bias(voltage: float, guess: DeviceState | None) -> DeviceState | None:
        biased = device.with_bias(**{contact: voltage})
        solved = solve_bias_newton(
            biased,
            models=models,
            guess=guess,
            max_iterations=max_iterations,
        )
        # solve_bias_newton always attaches a NewtonResult, converged or not.
        assert solved.newton is not None
        return solved if solved.newton.converged else None

    return _walk_sweep(
        device=device,
        contact=contact,
        measured_at=measure_at,
        voltages=voltages,
        models=models,
        at_bias=at_bias,
        start=start,
        step=step,
        min_step=min_step,
    )
