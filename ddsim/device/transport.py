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
from ddsim.discretize.assembly import SparseAssembly
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
from ddsim.discretize.coupled import (
    apply_ohmic_contacts_coupled,
    assemble_coupled_terms,
    coupled_update_norm,
    limit_psi_step,
    pack,
    row_weights,
    scale_rows,
    unpack,
)
from ddsim.physics.mobility import AroraMobility, edge_diffusivity
from ddsim.physics.recombination import (
    AugerRecombination,
    RecombinationModel,
    SRHRecombination,
    SumOfRecombination,
    scharfetter_lifetime,
)
from ddsim.solve.gummel import BlockStep, GummelResult, gummel_solve
from ddsim.solve.linear import SparseLU
from ddsim.solve.newton import newton_solve

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
        mobility: str = "constant",
        auger: bool = False,
    ) -> TransportModels:
        """Silicon models for a device, scaled to its own scale factors.

        Args:
            device: the device, for its doping, temperature and scaling.
            recombination: an explicit model, which overrides both the SRH
                default and the auger flag.
            mobility: "constant" for the Phase 1 and 2 value, or "arora" for
                the doping dependent model docs/01-physics.md puts in Phase 3.
            auger: add band to band Auger alongside SRH. Off by default
                because it changes nothing measurable below high injection
                and every Phase 2 number was taken without it.

        Doping dependent mobility slots in by making Dn and Dp arrays over
        edges instead of scalars, which every assembly already accepts, and it
        adds nothing to the Jacobian because the doping does not change during
        a solve. Field dependent mobility in Phase 5 will not be free in the
        same way: it depends on the potential difference across the edge and
        so puts a dmu/dpsi term into every flux derivative.

        The SRH lifetimes come from the Scharfetter relation evaluated on the
        local doping. It wants the total doping Na + Nd and only the net is
        available, so abs(net) is used. The two agree everywhere except in
        compensated material, and nothing here is compensated yet. When a
        profile that overlaps donors and acceptors arrives, this is the line
        that has to learn about it.
        """
        scale = device.scale
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
            if auger:
                # Scaled by C_0^2 * t_0 rather than by C_0: the coefficient
                # multiplies a triple product, so it carries two powers of the
                # density scale and one of time.
                recombination = SumOfRecombination(
                    (
                        recombination,
                        AugerRecombination(
                            C_n=C.AUGER_C_N * scale.C_0**2 * scale.t_0,
                            C_p=C.AUGER_C_P * scale.C_0**2 * scale.t_0,
                            ni2=(device.material.n_i / scale.C_0) ** 2,
                        ),
                    )
                )

        return cls(
            recombination=recombination,
            Dn=_scaled_diffusivity(device, Carrier.ELECTRON, mobility),
            Dp=_scaled_diffusivity(device, Carrier.HOLE, mobility),
        )


def _scaled_diffusivity(
    device: Device, carrier: Carrier, mobility: str
) -> Diffusivity:
    """Scaled diffusivity for one carrier, from the chosen mobility model.

    Constant comes back as a scalar and Arora as one value per edge. The
    assembly takes either, so the two are not different code paths anywhere
    downstream; only this function knows which was asked for.
    """
    scale = device.scale
    temperature = device.material.T
    electrons = carrier is Carrier.ELECTRON

    if mobility == "constant":
        constant = C.D_n(temperature) if electrons else C.D_p(temperature)
        return constant / scale.D_0

    if mobility != "arora":
        raise ValueError(
            f"unknown mobility model {mobility!r}. Use 'constant' or 'arora'."
        )

    model = (
        AroraMobility.electrons(temperature)
        if electrons
        else AroraMobility.holes(temperature)
    )
    # Total doping, of which only the net is available. Same caveat and same
    # reason as the Scharfetter lifetime above.
    nodal = model(np.abs(device.net_doping.data))
    edge_nodes = device.scaled_mesh.geometry.edge_nodes
    return edge_diffusivity(nodal, C.V_T(temperature), edge_nodes) / scale.D_0


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
    solver = SparseLU()

    def step(state: DeviceState) -> tuple[DeviceState, float]:
        result = solve_poisson(
            device, state.psi.data, state.phi_n, state.phi_p, solver=solver
        )
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
            device.mesh_1d,
            state.psi,
            state.n,
            state.p,
            models.recombination,
            device.scale,
            models.Dn,
        )
        assembly = apply_ohmic_densities(
            assembly, state.n.data, doping, device.ohmic_contacts, Carrier.ELECTRON
        )

        solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_n = impose_ohmic_densities(
            state.n.data + solver.solve(-assembly.residual),
            doping,
            device.ohmic_contacts,
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
            device.mesh_1d,
            state.psi,
            state.n,
            state.p,
            models.recombination,
            device.scale,
            models.Dp,
        )
        assembly = apply_ohmic_densities(
            assembly, state.p.data, doping, device.ohmic_contacts, Carrier.HOLE
        )

        solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_p = impose_ohmic_densities(
            state.p.data + solver.solve(-assembly.residual),
            doping,
            device.ohmic_contacts,
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


def solve_bias_newton(
    device: Device,
    models: TransportModels | None = None,
    guess: DeviceState | None = None,
    max_psi_step: float = 5.0,
    max_iterations: int = 30,
    residual_rtol: float = 1e-10,
    update_tol: float = 1e-10,
) -> DeviceState:
    """Solve the coupled system at the device's biases by full Newton.

    Args:
        device: the device, carrying its contact voltages.
        models: recombination and diffusivities. Built from the device if None.
        guess: a previous solution to start from. Continuation lives on this.
        max_psi_step: cap on the potential update per step [1], scaled.
            5.0 is the 5*V_T that docs/02-numerics.md prescribes.
        max_iterations: Newton budget. Small on purpose: a coupled Newton that
            needs thirty steps from a decent guess is not converging
            quadratically and the budget should not hide that.
        residual_rtol: residual threshold, relative to each equation family's
            own term scale after row scaling.
        update_tol: threshold on the update, measured per family by
            coupled.coupled_update_norm rather than as max |dx|.

    The same equations as solve_bias, solved together instead of in a cycle.
    Returns the state with its NewtonResult attached, converged or not, and
    does not raise: a failed solve is what continuation reads to decide to
    halve its step.

    Works in either dimension. Everything that knew it was in 1D is now asked
    of the device: the mesh scales itself by the right power of x_0, the
    charge volume is zero where there is no semiconductor, and the geometry
    carries the edge list, the permittivity and the face a carrier is allowed
    to cross. The Gummel path in this module is still 1D and says so.

    Three things this does that a textbook Newton loop does not.

    **The rows are scaled by their own terms.** See coupled.residual_term_scales.
    Without it one threshold has to serve a charge and a current, and on a
    1e16 device those differ by six decades.

    **The scales are measured at each iterate, not frozen at the guess.** The
    threshold itself never moves: it stays residual_rtol against a scale of
    one. What is re-measured is the size of the terms the residual is made of,
    which is a property of the state and not of how converged it is. Freezing
    it at the guess made a 1e20 / 1e14 junction report failure at 2.8e-9 while
    it was converged to 4.5e-15, because its electron term scale grows by
    660000 between equilibrium and 1 V. See coupled.residual_term_scales.

    **Only psi is damped.** docs/02-numerics.md and docs/05-pitfalls.md both
    say to cap the potential update and take the density updates in full.
    """
    if models is None:
        models = TransportModels.for_device(device)

    start = initial_state(device) if guess is None else guess
    scale = device.scale
    mesh = device.scaled_mesh

    h = mesh.h
    volume = device.charge_volume_scaled
    geometry = mesh.geometry
    net_doping = device.net_doping_scaled.data
    carrier_free = device.carrier_free_nodes

    x0 = pack(start.psi.data, start.n.data, start.p.data)

    def assemble(x: npt.NDArray[np.float64]) -> SparseAssembly:
        assembly, scales = assemble_coupled_terms(
            h,
            volume,
            x,
            net_doping,
            models.Dn,
            models.Dp,
            models.recombination,
            geometry,
        )
        # Contacts before the scaling, so a pinned row becomes the identity
        # and then gets divided like any other. Scaling first would leave the
        # pinned rows at one while everything around them moved.
        assembly = apply_ohmic_contacts_coupled(
            assembly, x, net_doping, device.ohmic_contacts, scale, carrier_free
        )
        return scale_rows(assembly, row_weights(scales, mesh.n_nodes))

    result = newton_solve(
        assemble,
        x0,
        limit=lambda delta: limit_psi_step(delta, max_psi_step),
        residual_scale=1.0,
        residual_rtol=residual_rtol,
        update_tol=update_tol,
        update_norm=coupled_update_norm,
        max_iterations=max_iterations,
    )

    psi, n, p = unpack(result.x)
    return DeviceState(
        psi=_node_field(psi.copy(), "V", "psi"),
        n=_node_field(n.copy(), "cm^-3", "n"),
        p=_node_field(p.copy(), "cm^-3", "p"),
        newton=result,
    )


def _gummel_prelude(
    device: Device,
    models: TransportModels,
    state: DeviceState,
    cycles: int,
) -> DeviceState:
    """Run a fixed number of Gummel cycles, ignoring whether they converged.

    A prelude is not a solve. It only has to move the iterate into the basin
    Newton can finish from, and stopping it early on a convergence test would
    defeat the point, so the update tolerance is set below anything reachable
    and the cycle count is the only thing that ends it.

    A block that fails outright still ends it, and the state at that point is
    returned rather than raised. Newton is about to be handed whatever this
    produced and is allowed to fail on it in its own way.
    """
    if cycles <= 0:
        return state

    steps = [
        poisson_block(device),
        electron_block(device, models),
        hole_block(device, models),
    ]
    try:
        result = gummel_solve(
            state, steps, update_tol=1e-300, max_iterations=cycles
        )
    except TransportError as failure:
        return failure.state
    return replace(result.state, gummel=result)


def solve_bias_hybrid(
    device: Device,
    models: TransportModels | None = None,
    guess: DeviceState | None = None,
    gummel_cycles: int = 3,
    retry_cycles: int = 5,
    max_psi_step: float = 5.0,
    max_iterations: int = 30,
) -> DeviceState:
    """Gummel for a few cycles to reach the basin, then full Newton.

    Args:
        device: the device, carrying its contact voltages.
        models: recombination and diffusivities. Built from the device if None.
        guess: a previous solution to start from.
        gummel_cycles: prelude length. docs/02-numerics.md says 3 to 5.
        retry_cycles: extra cycles to run before one second Newton attempt,
            when the first fails. Zero disables the retry.
        max_psi_step: cap on the potential update per Newton step [1], scaled.
        max_iterations: Newton budget per attempt.

    The strategy docs/02-numerics.md prescribes, built against a case where it
    is measurably needed rather than on principle. On a 1e15 diode at 1.2 V on
    41 nodes, Newton from the Poisson guess diverges: thirty steps, twenty
    eight of them against the limiter, forty two nodes with a negative
    density, final residual 1.3e5. Two Gummel cycles first turn that into an
    eight step solve with a clean quadratic tail. Three give seven, five give
    six.

    Tightening the damping does not fix that case, which is worth knowing
    before reaching for it. Capping the potential update at 2 V_T instead of 5
    does converge, in forty seven steps with forty of them limited; capping at
    1 or at 0.5 does not converge at all inside sixty. docs/05-pitfalls.md
    says damping hides problems rather than solving them, and here it does not
    even hide it.

    Continuation remains the better answer where it is available: the same
    device reaches 1.2 V in seven solves with no retries and no prelude at
    all. The hybrid is for the case where there is no ramp to come up, which
    is every first solve of one.

    The retry restarts from the state before Newton ran, never from the state
    Newton diverged to. A diverged iterate has negative densities in it, and
    handing those to a Gummel block whose whole positivity argument assumes
    non-negative input would produce a second failure with a different cause.
    """
    if models is None:
        models = TransportModels.for_device(device)

    start = initial_state(device) if guess is None else guess

    def newton_from(state: DeviceState) -> DeviceState:
        return solve_bias_newton(
            device,
            models=models,
            guess=state,
            max_psi_step=max_psi_step,
            max_iterations=max_iterations,
        )

    warmed = _gummel_prelude(device, models, start, gummel_cycles)
    result = newton_from(warmed)

    # solve_bias_newton always attaches a NewtonResult, converged or not.
    assert result.newton is not None
    if result.newton.converged or retry_cycles <= 0:
        return replace(result, gummel=warmed.gummel)

    # From the pre-Newton state, not the diverged one.
    warmed = _gummel_prelude(device, models, warmed, retry_cycles)
    return replace(newton_from(warmed), gummel=warmed.gummel)


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
