"""The coupled drift diffusion solve at a given bias, by Gummel iteration.

Where the pieces meet. The mesh and doping come from device/, the residuals and
Jacobians from discretize/, the models from physics/, and the iteration from
solve/. Nothing here assembles and nothing here iterates. It only wires, which
is what keeps solve/ free of semiconductor knowledge and keeps every piece
testable on its own.

The cycle, from docs/02-numerics.md:

1. Solve nonlinear Poisson for psi, holding phi_n and phi_p fixed
2. Solve the electron continuity equation for n, holding psi and p fixed
3. Solve the hole continuity equation for p, holding psi and n fixed
4. Check the update norm, repeat

Why the Poisson block holds the quasi-Fermi levels rather than the densities
-----------------------------------------------------------------------------
This is the whole trick, and it is easy to get wrong in a way that still runs.
Freezing n and p and solving a linear Poisson converges badly, because the
charge cannot respond to the potential during the solve. Freezing phi_n and
phi_p instead leaves n = exp(psi - phi_n) inside the equation, so Poisson stays
nonlinear and Newton sees the true exponential response. The Jacobian gains
(n + p)*volume on its diagonal, which is strictly positive and makes the solve
reliable at any doping.

After the potential moves by dpsi at fixed levels, the densities move with it:

    n <- n * exp(dpsi)        p <- p * exp(-dpsi)

which is the same statement as n = exp(psi - phi_n) with phi_n unchanged.

Equilibrium is an exact fixed point of all of this. With n = exp(psi) and
p = exp(-psi) the levels are flat at zero, Poisson is already solved, n*p = 1
kills the recombination rate, and every edge flux cancels identically. A device
at zero bias therefore comes back untouched, which is the sharpest single test
of the wiring.

Measuring convergence
---------------------
Potential updates are measured in units of V_T, which is scale free already.
Densities are measured as max|dn| / (n + 1), where the 1 is n_i in scaled
units. A pure relative change would be dominated by nodes where the density is
1e-15 and physically irrelevant, and an absolute change would be dominated by
the majority carrier. The floor at n_i says, in the only units that matter,
that a carrier below the intrinsic density carries no charge worth converging.

Where it gives up
-----------------
Gummel converges linearly and the rate degrades as the three equations couple,
which they do under forward bias once injection approaches the doping. That
failure is expected, is documented rather than fought, and is the entire reason
Phase 3 exists. See phases/PHASE-2.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.device.equilibrium import (
    frozen_quasi_fermi,
    solve_equilibrium,
    solve_poisson,
)
from ddsim.device.state import DeviceState
from ddsim.discretize.boundary import (
    Carrier,
    apply_ohmic_densities,
    impose_ohmic_densities,
)
from ddsim.discretize.continuity import (
    Diffusivity,
    assemble_electron_continuity,
    assemble_hole_continuity,
)
from ddsim.physics.recombination import (
    RecombinationModel,
    SRHRecombination,
    scharfetter_lifetime,
)
from ddsim.solve.gummel import BlockStep, GummelResult, gummel_solve
from ddsim.solve.linear import SparseLU

DENSITY_REFERENCE = 1.0
"""Floor in the density update norm [1], which is n_i in scaled units.

Not a clamp on any density. It only says that a change from 1e-20 to 1e-19
carries no charge and should not hold up convergence.
"""


class TransportError(RuntimeError):
    """A Gummel block could not produce a usable state.

    Carries the offending state, because the state at the point of failure is
    exactly what wants inspecting. A negative density means a sign error or a
    broken M-matrix, and docs/05-pitfalls.md is clear that the repair is to
    find the cause and never to clamp.
    """

    def __init__(self, message: str, state: DeviceState) -> None:
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class TransportModels:
    """The material models a transport solve needs, all in scaled units."""

    recombination: RecombinationModel
    """Net recombination, built with scaled lifetimes."""

    Dn: Diffusivity
    """Electron diffusivity [1], scaled by D_0."""

    Dp: Diffusivity
    """Hole diffusivity [1], scaled by D_0."""

    @classmethod
    def for_device(
        cls,
        device: Device,
        recombination: RecombinationModel | None = None,
    ) -> TransportModels:
        """Silicon models for a device, scaled to its own scale factors.

        Constant mobility, which is all Phase 2 asks for. Doping dependent
        mobility arrives in Phase 3 and field dependent mobility in Phase 5,
        and both slot in by making Dn and Dp arrays over edges instead of
        scalars, which the assembly already accepts.

        The SRH lifetimes come from the Scharfetter relation evaluated on the
        local doping. It wants the total doping Na + Nd and only the net is
        available, so abs(net) is used. The two agree everywhere except in
        compensated material, and nothing here is compensated yet. When a
        profile that overlaps donors and acceptors arrives, this is the line
        that has to learn about it.
        """
        scale = device.scale
        temperature = device.material.T
        total_doping = np.abs(device.net_doping.data)

        if recombination is None:
            recombination = SRHRecombination(
                tau_n=scharfetter_lifetime(
                    total_doping, tau_max=C.TAU_N_MAX, tau_min=C.TAU_N_MIN
                )
                / scale.t_0,
                tau_p=scharfetter_lifetime(
                    total_doping, tau_max=C.TAU_P_MAX, tau_min=C.TAU_P_MIN
                )
                / scale.t_0,
                ni2=(device.material.n_i / scale.C_0) ** 2,
                n1=device.material.n_i / scale.C_0,
                p1=device.material.n_i / scale.C_0,
            )

        return cls(
            recombination=recombination,
            Dn=C.D_n(temperature) / scale.D_0,
            Dp=C.D_p(temperature) / scale.D_0,
        )


def _node_field(values: npt.NDArray[np.float64], unit: str, name: str) -> Field:
    """A scaled node Field, the form every assembly demands."""
    return Field(values, unit, ScalingState.SCALED, Location.NODE, name=name)


def _density_update(
    old: npt.NDArray[np.float64], new: npt.NDArray[np.float64]
) -> float:
    """max |dn| / (n + n_i), the convergence measure for a density [1]."""
    return float(np.max(np.abs(new - old) / (np.abs(old) + DENSITY_REFERENCE)))


def poisson_block(device: Device) -> BlockStep[DeviceState]:
    """Step 1: nonlinear Poisson at fixed quasi-Fermi levels."""

    def step(state: DeviceState) -> tuple[DeviceState, float]:
        result = solve_poisson(device, state.psi.data, state.phi_n, state.phi_p)
        if not result.converged:
            raise TransportError(
                f"the Poisson block did not converge: {result.message}", state
            )

        shift = result.x - state.psi.data

        # exp is allowed to underflow here. A hole density of 1e-320 in a
        # reverse biased n region is physically zero and numerically harmless.
        # Overflow is not harmless, and means the potential moved so far in one
        # cycle that the Boltzmann densities left the representable range.
        with np.errstate(over="ignore", under="ignore"):
            n = state.n.data * np.exp(shift)
            p = state.p.data * np.exp(-shift)

        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(p))):  # pragma: no cover
            # Unreachable as configured, and deliberately kept. The step
            # limiter allows 5 per Newton iteration and solve_poisson allows
            # 50 of them, so one cycle can move psi by at most 250 and
            # exp(250) is 3.7e108. Raising either number would make this live.
            raise TransportError(
                f"the potential moved by {np.max(np.abs(shift)):.3g} V_T in one "
                "cycle and overflowed the Boltzmann densities. Ramp the bias in "
                "smaller steps.",
                state,
            )

        updated = replace(
            state,
            psi=_node_field(result.x, "V", "psi"),
            n=_node_field(n, "cm^-3", "n"),
            p=_node_field(p, "cm^-3", "p"),
            newton=result,
        )
        return updated, float(np.max(np.abs(shift)))

    return step


def electron_block(
    device: Device, models: TransportModels
) -> BlockStep[DeviceState]:
    """Step 2: electron continuity, linear in n once psi and p are held."""
    doping = device.net_doping_scaled.data
    solver = SparseLU()

    def step(state: DeviceState) -> tuple[DeviceState, float]:
        assembly = assemble_electron_continuity(
            device.mesh,
            state.psi,
            state.n,
            state.p,
            models.recombination,
            device.scale,
            models.Dn,
        )
        assembly = apply_ohmic_densities(
            assembly, state.n.data, doping, device.contacts, Carrier.ELECTRON
        )

        solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_n = impose_ohmic_densities(
            state.n.data + solver.solve(-assembly.residual),
            doping,
            device.contacts,
            Carrier.ELECTRON,
        )

        _check_positive(updated_n, "n", state)
        return (
            replace(state, n=_node_field(updated_n, "cm^-3", "n")),
            _density_update(state.n.data, updated_n),
        )

    return step


def hole_block(device: Device, models: TransportModels) -> BlockStep[DeviceState]:
    """Step 3: hole continuity, linear in p once psi and n are held."""
    doping = device.net_doping_scaled.data
    solver = SparseLU()

    def step(state: DeviceState) -> tuple[DeviceState, float]:
        assembly = assemble_hole_continuity(
            device.mesh,
            state.psi,
            state.n,
            state.p,
            models.recombination,
            device.scale,
            models.Dp,
        )
        assembly = apply_ohmic_densities(
            assembly, state.p.data, doping, device.contacts, Carrier.HOLE
        )

        solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_p = impose_ohmic_densities(
            state.p.data + solver.solve(-assembly.residual),
            doping,
            device.contacts,
            Carrier.HOLE,
        )

        _check_positive(updated_p, "p", state)
        return (
            replace(state, p=_node_field(updated_p, "cm^-3", "p")),
            _density_update(state.p.data, updated_p),
        )

    return step


def _check_positive(
    density: npt.NDArray[np.float64], name: str, state: DeviceState
) -> None:
    """Refuse a non-positive density rather than clamping it.

    The continuity matrix is an M-matrix and the right hand side is
    non-negative, so this cannot happen for a reason that is not a bug. If it
    does, the cause is a sign error or a broken discretization, and clamping
    would hide both while producing a solution that satisfies no equation.
    """
    if np.all(density > 0.0):
        return

    worst = int(np.argmin(density))
    raise TransportError(
        f"{name} came out non-positive at node {worst}, value "
        f"{density[worst]:.3e}. The continuity matrix should be an M-matrix "
        "with a non-negative right hand side, so check signs before anything "
        "else, per docs/05-pitfalls.md. Do not clamp.",
        state,
    )


def initial_state(device: Device) -> DeviceState:
    """A starting guess: Poisson with flat quasi-Fermi levels at the contacts.

    Exact at zero bias and under reverse bias with no recombination, and a
    reasonable guess at low forward bias. It is the Phase 1 solve, reused.
    At any bias worth calling forward it is not good enough on its own, which
    is what continuation is for.
    """
    return solve_equilibrium(device, frozen_quasi_fermi(device))


def solve_bias(
    device: Device,
    models: TransportModels | None = None,
    guess: DeviceState | None = None,
    update_tol: float = 1e-8,
    max_iterations: int = 200,
) -> DeviceState:
    """Solve the coupled system at the device's contact biases.

    Args:
        device: the device, carrying its contact voltages.
        models: recombination and diffusivities. Built from the device if None.
        guess: a previous solution to start from. The single most valuable
            input there is, which is why continuation exists.
        update_tol: convergence threshold on the Gummel cycle update.
        max_iterations: cycle budget.

    Returns the state with its GummelResult attached, converged or not, and
    does not raise on failure. Above roughly 0.6 V forward bias, failing to
    converge is the expected outcome rather than an exceptional one, and
    phases/PHASE-2.md asks for the bias at which that happens to be measured.
    A result that cannot be returned cannot be measured.
    """
    if models is None:
        models = TransportModels.for_device(device)

    start = initial_state(device) if guess is None else guess
    steps = [
        poisson_block(device),
        electron_block(device, models),
        hole_block(device, models),
    ]

    try:
        result = gummel_solve(
            start, steps, update_tol=update_tol, max_iterations=max_iterations
        )
    except TransportError as failure:
        return replace(
            failure.state,
            gummel=GummelResult(
                state=failure.state,
                converged=False,
                iterations=0,
                message=str(failure),
            ),
        )

    return replace(result.state, gummel=result)
