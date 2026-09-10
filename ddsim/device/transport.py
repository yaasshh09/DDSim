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

from collections.abc import Callable
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
    apply_contacts_coupled,
    assemble_coupled_terms,
    coupled_update_norm,
    edge_drop,
    limit_psi_step,
    pack,
    residual_measure,
    residual_term_scales,
    row_weights,
    scale_rows,
    unpack,
)
from ddsim.mesh.mesh2d import Mesh2D, normal_field
from ddsim.physics.mobility import (
    AroraMobility,
    CaugheyThomas,
    ConstantMobility,
    EdgeDiffusivity,
    LombardiSurface,
    diffusivity_at,
    edge_diffusivity,
)
from ddsim.physics.recombination import (
    AugerRecombination,
    RecombinationModel,
    SRHRecombination,
    SumOfRecombination,
    scharfetter_lifetime,
)
from ddsim.solve.continuation import continue_to
from ddsim.solve.gummel import BlockStep, GummelResult, gummel_solve
from ddsim.solve.linear import SparseLU
from ddsim.solve.newton import NewtonResult, newton_solve

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
class SurfaceScattering:
    """What it takes to rebuild the diffusivities once the state has moved.

    Lombardi surface mobility reads the field normal to the Si/SiO2 interface
    and the carrier density at the same node, so unlike Arora it is not a
    property of the device that can be worked out once. Unlike Caughey-Thomas
    it also cannot be evaluated inside the assembly, because the normal field
    on a horizontal channel edge lives on the vertical edges above and below
    its endpoints rather than on the edge itself, and carrying that dependence
    exactly would widen the Jacobian stencil past the edge based pattern every
    assembly in `discretize/` is built on.

    So it is frozen instead, and an outer loop turns the freezing into a fixed
    point. See `solve_bias_newton`. Within one Newton solve the surface
    correction is a constant array, which means the residual and the Jacobian
    are assembled from exactly the same mobility and the nine block complex
    step verification is untouched. At the outer fixed point the frozen
    correction is the one the answer implies, so the converged state solves the
    true equations. What is given up is the rate, not the answer.

    This holds the pieces that do not move, so a refresh is one evaluation of
    the model rather than a rebuild of the device.
    """

    electrons: LombardiSurface
    """The model for electrons, with its own parameter set."""

    holes: LombardiSurface
    """The model for holes."""

    mu_bulk_n: npt.NDArray[np.float64]
    """Electron mobility before any surface correction [cm^2/(V s)], per node.

    Whatever the chosen bulk model gives, so `constant` and `arora` both land
    here and the surface term does not know which it corrected.
    """

    mu_bulk_p: npt.NDArray[np.float64]
    """Hole mobility before any surface correction [cm^2/(V s)], per node."""

    total_doping: npt.NDArray[np.float64]
    """Na + Nd at each node [cm^-3], floored at n_i.

    Only the net doping is available, same caveat and same reason as the
    Scharfetter lifetime. The floor is separate and it is load bearing: the
    roughness exponent carries N^(-eta), so a node with exactly zero doping
    raises it to a negative power and returns an infinity. Every node in the
    oxide has exactly zero doping, because `Device.net_doping` zeroes it
    wherever there is no semiconductor. Below the intrinsic density there are
    no scattering centres left worth counting, so n_i is where the count
    stops, and no node of any device built here sits near it anyway.
    """

    semiconductor: npt.NDArray[np.bool_]
    """Which nodes hold semiconductor, one per node.

    The correction is applied at these and nowhere else. An insulator has no
    surface mobility, and an oxide node left carrying one is not harmless:
    an edge running from the interface into the oxide averages the two ends,
    so a bad value there reaches the interface row, which is the channel.
    """

    def corrected(
        self,
        device: Device,
        psi: npt.NDArray[np.float64],
        n: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Nodal mobilities [cm^2/(V s)] with the surface term folded in.

        Args:
            device: for the mesh, the scale factors and nothing else.
            psi: potential at every node [1], scaled.
            n: electron density at every node [1], scaled.
            p: hole density at every node [1], scaled.

        The state arrives scaled and the model's parameters are in physical
        units, so everything is put back before it is used. That conversion is
        the one place this could go quietly wrong: a normal field short by a
        factor of the Debye length over the thermal voltage would still be
        positive, still monotone, and still produce a mobility that looked
        like a mobility.
        """
        mesh = device.mesh
        if not isinstance(mesh, Mesh2D):
            raise TypeError(
                "surface mobility needs a direction normal to the interface "
                f"and a {type(mesh).__name__} has none. Build the device on a "
                "Mesh2D, which is what a MOSFET is on."
            )

        scale = device.scale
        E_perp = normal_field(mesh, psi * scale.psi_0)
        carriers = (n + p) * scale.C_0

        return (
            np.where(
                self.semiconductor,
                self.electrons(self.mu_bulk_n, E_perp, self.total_doping, carriers),
                self.mu_bulk_n,
            ),
            np.where(
                self.semiconductor,
                self.holes(self.mu_bulk_p, E_perp, self.total_doping, carriers),
                self.mu_bulk_p,
            ),
        )


@dataclass(frozen=True)
class TransportModels:
    """The material models a transport solve needs, all in scaled units."""

    recombination: RecombinationModel
    """Net recombination, built with scaled lifetimes."""

    Dn: EdgeDiffusivity
    """Electron diffusivity [1], scaled by D_0.

    A number, one value per edge, or a model the assembly evaluates at the
    state it is assembling at. See physics/mobility.py.
    """

    Dp: EdgeDiffusivity
    """Hole diffusivity [1], scaled by D_0."""

    surface: SurfaceScattering | None = None
    """Surface scattering, or None where the device has no interface.

    Present rather than folded into Dn and Dp because it has to be refreshed
    as the state moves. `solve_bias_newton` reads it to decide whether to run
    the outer fixed point at all, so None is what makes every solve before
    Phase 5 take exactly the path it took.
    """

    field_dependent: bool = False
    """Whether Dn and Dp are wrapped in Caughey-Thomas.

    Carried so that a refresh can rebuild the same shape of model it replaced.
    Reading it off Dn's type instead would work today and would break the
    first time another edge model exists.
    """

    def at_state(
        self,
        device: Device,
        psi: npt.NDArray[np.float64],
        n: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
    ) -> TransportModels:
        """These models with the surface correction taken at this state.

        Returns self unchanged where there is no surface model, so a caller
        does not have to ask first and a device without an interface is not a
        separate code path.
        """
        if self.surface is None:
            return self

        mu_n, mu_p = self.surface.corrected(device, psi, n, p)
        return replace(
            self,
            Dn=_from_nodal_mobility(
                device, Carrier.ELECTRON, mu_n, self.field_dependent
            ),
            Dp=_from_nodal_mobility(device, Carrier.HOLE, mu_p, self.field_dependent),
        )

    @classmethod
    def for_device(
        cls,
        device: Device,
        recombination: RecombinationModel | None = None,
        mobility: str = "constant",
        auger: bool = False,
        field_dependent: bool = False,
        surface: bool = False,
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
            field_dependent: wrap the chosen low field model in
                Caughey-Thomas, which is what produces velocity saturation.
                Off by default for the same reason auger is: every result
                recorded before Phase 5 was taken without it.
            surface: add Lombardi scattering off the Si/SiO2 interface. Needs
                a Mesh2D, since it reads the field normal to that interface
                and a line has no normal. Off by default, and a device with
                no interface has no use for it.

        Doping dependent mobility slots in by making Dn and Dp arrays over
        edges instead of scalars, which every assembly already accepts, and it
        adds nothing to the Jacobian because the doping does not change during
        a solve. Field dependence is not free in the same way: it reads the
        potential difference across an edge, so the coupled assembly evaluates
        it at each iterate and the Jacobian carries its tangent.

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
            Dn=_scaled_diffusivity(
                device, Carrier.ELECTRON, mobility, field_dependent
            ),
            Dp=_scaled_diffusivity(
                device, Carrier.HOLE, mobility, field_dependent
            ),
            surface=(
                _surface_scattering(device, mobility, total_doping)
                if surface
                else None
            ),
            field_dependent=field_dependent,
        )


def _scaled_diffusivity(
    device: Device, carrier: Carrier, mobility: str, field_dependent: bool = False
) -> EdgeDiffusivity:
    """Scaled diffusivity for one carrier, from the chosen mobility model.

    Constant comes back as a scalar and Arora as one value per edge. The
    assembly takes either, so the two are not different code paths anywhere
    downstream; only this function knows which was asked for.

    With field_dependent set, whichever of those was chosen becomes the low
    field limit of Caughey-Thomas and the result is a model rather than an
    array. The wrapping order is the only one that makes sense: the low field
    mobility is a nodal quantity and is averaged onto the edge first, and the
    field factor is applied afterwards with that edge's own drop, because the
    field is an edge quantity and has no value at a node.
    """
    scale = device.scale
    temperature = device.material.T
    electrons = carrier is Carrier.ELECTRON

    if mobility == "constant":
        constant = C.D_n(temperature) if electrons else C.D_p(temperature)
        low_field: Diffusivity = constant / scale.D_0
    elif mobility == "arora":
        model = (
            AroraMobility.electrons(temperature)
            if electrons
            else AroraMobility.holes(temperature)
        )
        # Total doping, of which only the net is available. Same caveat and
        # same reason as the Scharfetter lifetime above.
        nodal = model(np.abs(device.net_doping.data))
        edge_nodes = device.scaled_mesh.geometry.edge_nodes
        low_field = (
            edge_diffusivity(nodal, C.V_T(temperature), edge_nodes) / scale.D_0
        )
    else:
        raise ValueError(
            f"unknown mobility model {mobility!r}. Use 'constant' or 'arora'."
        )

    return _wrapped_in_saturation(device, carrier, low_field, field_dependent)


def _surface_scattering(
    device: Device, mobility: str, total_doping: npt.NDArray[np.float64]
) -> SurfaceScattering:
    """The Lombardi models and the bulk mobility they correct.

    The bulk mobility is worked out once here and kept, because it is a
    function of the doping alone and the doping does not move during a solve.
    Only the normal field and the carrier densities move, and those are the
    two arguments a refresh supplies.
    """
    temperature = device.material.T

    if mobility == "constant":
        nodal_n = ConstantMobility(C.mu_n(temperature))(total_doping)
        nodal_p = ConstantMobility(C.mu_p(temperature))(total_doping)
    elif mobility == "arora":
        nodal_n = AroraMobility.electrons(temperature)(total_doping)
        nodal_p = AroraMobility.holes(temperature)(total_doping)
    else:
        raise ValueError(
            f"unknown mobility model {mobility!r}. Use 'constant' or 'arora'."
        )

    semiconductor = np.ones(device.mesh.n_nodes, dtype=np.bool_)
    semiconductor[list(device.carrier_free_nodes)] = False

    return SurfaceScattering(
        electrons=LombardiSurface.electrons(temperature),
        holes=LombardiSurface.holes(temperature),
        mu_bulk_n=nodal_n,
        mu_bulk_p=nodal_p,
        total_doping=np.maximum(total_doping, device.material.n_i),
        semiconductor=semiconductor,
    )


def _from_nodal_mobility(
    device: Device,
    carrier: Carrier,
    nodal: npt.NDArray[np.float64],
    field_dependent: bool,
) -> EdgeDiffusivity:
    """A scaled edge diffusivity from a mobility already worked out per node.

    What the surface correction produces. It has already replaced whichever
    bulk model was chosen, so there is no model name left to dispatch on, and
    the rest of the journey onto the edges is the same one Arora takes: an
    arithmetic average onto each edge, the Einstein relation, and the scaling.
    """
    edge_nodes = device.scaled_mesh.geometry.edge_nodes
    V_T = C.V_T(device.material.T)
    low_field = edge_diffusivity(nodal, V_T, edge_nodes) / device.scale.D_0
    return _wrapped_in_saturation(device, carrier, low_field, field_dependent)


def _wrapped_in_saturation(
    device: Device,
    carrier: Carrier,
    low_field: Diffusivity,
    field_dependent: bool,
) -> EdgeDiffusivity:
    """Caughey-Thomas around a low field diffusivity, or that diffusivity.

    The wrapping order is the only one that makes sense and it is the one the
    reference uses: the low field mobility is a nodal quantity, so it is
    corrected for the surface and averaged onto the edge first, and the
    velocity saturation factor is applied afterwards with that edge's own
    parallel drop, because the parallel field is an edge quantity and has no
    value at a node.
    """
    if not field_dependent:
        return low_field

    scale = device.scale
    temperature = device.material.T
    electrons = carrier is Carrier.ELECTRON

    # A velocity is scaled by D_0 / x_0, which is what makes the scaled
    # saturation velocity the number of Debye lengths a saturated carrier
    # crosses per dielectric relaxation time.
    v_sat = C.v_sat_n(temperature) if electrons else C.v_sat_p(temperature)
    return CaugheyThomas(
        low_field=np.broadcast_to(
            np.asarray(low_field, dtype=np.float64), (device.scaled_mesh.h.size,)
        ).copy(),
        v_sat=v_sat * scale.x_0 / scale.D_0,
        beta=C.BETA_N if electrons else C.BETA_P,
    )


def _lagged_diffusivity(
    D: EdgeDiffusivity, device: Device, psi: npt.NDArray[np.float64]
) -> Diffusivity:
    """A field dependent diffusivity frozen at the potential of this cycle.

    What the uncoupled Gummel blocks get. Each of them solves one continuity
    equation with psi held fixed, so within a block the field is a constant
    and the diffusivity with it, which is what lagging a coefficient means and
    is what a Gummel cycle already does to everything else it holds. At the
    fixed point psi is the converged potential, so the frozen diffusivity is
    the converged one and the cycle solves the same equations the coupled path
    does. What is given up is the tangent, which is a rate of convergence and
    not an answer.
    """
    return diffusivity_at(D, edge_drop(psi), device.scaled_mesh.h)


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


def _lagged_effective_potential(
    device: Device, state: DeviceState, carrier: Carrier
) -> Field:
    """The potential one carrier is Boltzmann in, at the incoming state [V].

    state.psi itself under Boltzmann, so nothing before Phase 5 pays anything
    or moves a bit.

    Under Fermi-Dirac the Bernoulli argument depends on the density as well as
    on the potential, which the coupled Newton differentiates properly and a
    Gummel block cannot: its whole premise is that the continuity equation is
    linear in its own carrier once the other two unknowns are held. So the
    correction is lagged at the incoming density, exactly the way this path
    already lags a field dependent diffusivity. It costs a Gummel cycle its
    quadratic convergence, which Gummel never had, and it leaves the fixed
    point alone: at convergence the lagged density is the solved one and the
    equation being satisfied is the degenerate one.
    """
    degeneracy = device.degeneracy
    if degeneracy is None:
        return state.psi
    if carrier is Carrier.ELECTRON:
        values = degeneracy.electron_potential(state.psi.data, state.n.data)
    else:
        values = degeneracy.hole_potential(state.psi.data, state.p.data)
    return _node_field(np.asarray(values), "V", "psi_eff")


def electron_block(
    device: Device, models: TransportModels
) -> BlockStep[DeviceState]:
    """Step 2: electron continuity, linear in n once psi and p are held."""
    doping = device.net_doping_scaled.data
    degeneracy = device.degeneracy
    solver = SparseLU()

    def step(state: DeviceState) -> tuple[DeviceState, float]:
        assembly = assemble_electron_continuity(
            device.mesh_1d,
            _lagged_effective_potential(device, state, Carrier.ELECTRON),
            state.n,
            state.p,
            models.recombination,
            device.scale,
            _lagged_diffusivity(models.Dn, device, state.psi.data),
        )
        assembly = apply_ohmic_densities(
            assembly,
            state.n.data,
            doping,
            device.ohmic_contacts,
            Carrier.ELECTRON,
            degeneracy,
        )

        solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_n = impose_ohmic_densities(
            state.n.data + solver.solve(-assembly.residual),
            doping,
            device.ohmic_contacts,
            Carrier.ELECTRON,
            degeneracy,
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
    degeneracy = device.degeneracy
    solver = SparseLU()

    def step(state: DeviceState) -> tuple[DeviceState, float]:
        assembly = assemble_hole_continuity(
            device.mesh_1d,
            _lagged_effective_potential(device, state, Carrier.HOLE),
            state.n,
            state.p,
            models.recombination,
            device.scale,
            _lagged_diffusivity(models.Dp, device, state.psi.data),
        )
        assembly = apply_ohmic_densities(
            assembly,
            state.p.data,
            doping,
            device.ohmic_contacts,
            Carrier.HOLE,
            degeneracy,
        )

        solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_p = impose_ohmic_densities(
            state.p.data + solver.solve(-assembly.residual),
            doping,
            device.ohmic_contacts,
            Carrier.HOLE,
            degeneracy,
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


def _low_field_models(models: TransportModels) -> TransportModels:
    """The same models with every state dependent mobility taken back out.

    Caughey-Thomas unwraps to the low field diffusivity it was built around,
    and the surface model is dropped. Recombination is untouched, since it was
    never the difficulty.

    What comes back is Phase 3's mobility: a fixed array over edges that the
    assembly reads rather than evaluates. Nothing about the device, the mesh
    or the bias changes.
    """
    unwrapped = tuple(
        D.low_field if isinstance(D, CaugheyThomas) else D
        for D in (models.Dn, models.Dp)
    )
    return replace(
        models,
        Dn=unwrapped[0],
        Dp=unwrapped[1],
        surface=None,
        field_dependent=False,
    )


def _needs_a_low_field_prelude(
    models: TransportModels, guess: DeviceState | None
) -> bool:
    """Whether a cold solve should be walked up to these models.

    Only field dependence, and only cold. Two limits and a reason for each.

    **Field dependence and not surface scattering.** Caughey-Thomas is inside
    the Jacobian and its mobility falls steeply through the critical field, so
    a step that overshoots lands somewhere the linearization did not predict.
    Measured on a 1 um NMOS from the Poisson guess: 60 iterations, 55 of them
    against the step limiter, no convergence. Surface scattering is outside
    the Jacobian and only ever scales the mobility by a few, and the same
    device cold starts through it in six steps.

    **Cold and not warm.** A continuation step already arrives with a guess
    from the neighbouring bias, and that guess is worth more than this one.
    """
    return guess is None and models.field_dependent


def _low_field_edges(D: EdgeDiffusivity) -> npt.NDArray[np.float64]:
    """The per edge low field diffusivity inside whatever D is [1].

    The fixed point is on the mobility, and with velocity saturation switched
    on the mobility is wrapped in a model rather than sitting there as an
    array. Reaching through the wrapper compares the thing that actually moves
    between sweeps: the saturation factor is a function of the iterate and is
    already converged by the Newton solve that just finished.
    """
    if isinstance(D, CaugheyThomas):
        return D.low_field
    return np.asarray(D, dtype=np.float64)


def _surface_moved(before: TransportModels, after: TransportModels) -> float:
    """Largest relative change in either diffusivity between two sweeps [1].

    Both carriers, because the electron channel of an NMOS converging says
    nothing about the hole one, and a fixed point that has only half arrived
    is not one.
    """
    worst = 0.0
    for old, new in ((before.Dn, after.Dn), (before.Dp, after.Dp)):
        a, b = _low_field_edges(old), _low_field_edges(new)
        moved, scale = np.abs(b - a), np.abs(a)

        # A baseline of exactly zero has no relative change to report, and it
        # is reachable: the cold guess puts n = n_i exp((psi - phi_n)/V_T) at
        # the drain, which past about 0.3 V underflows the Lombardi roughness
        # term and leaves an edge with no mobility on it at all. An edge that
        # went from nothing to something moved by everything, so the fixed
        # point has not arrived and inf says so; an edge both sweeps agree has
        # no mobility did not move, so it reads zero rather than nan. Where
        # the baseline is positive this is the division it always was, down to
        # the bit.
        relative = np.divide(
            moved, scale, out=np.where(moved > 0.0, np.inf, 0.0), where=scale > 0.0
        )
        worst = max(worst, float(np.max(relative)))
    return worst


def _surface_fixed_point(
    device: Device,
    models: TransportModels,
    run: Callable[[TransportModels, npt.NDArray[np.float64]], NewtonResult],
    x0: npt.NDArray[np.float64],
    max_sweeps: int,
    rtol: float,
) -> NewtonResult:
    """Newton to convergence, with the surface mobility refreshed between runs.

    Lombardi reads the field normal to the interface, which for a horizontal
    channel edge lives on the vertical edges above and below its two endpoints
    rather than on the edge itself. Carrying that exactly would widen the
    Jacobian stencil past the edge based pattern every assembly in
    `discretize/` is built on, so the correction is frozen inside each Newton
    solve and this loop makes the freezing a fixed point instead.

    Two properties are worth being precise about, because "frozen coefficient"
    is usually a euphemism for an approximation and here it is not one.

    **Inside a sweep nothing is approximated.** The residual and the Jacobian
    are assembled from the same frozen mobility, so they agree exactly and the
    nine block complex step verification is untouched by any of this.

    **At the fixed point nothing is frozen.** The loop ends when refreshing
    the mobility from the answer changes it by less than rtol, which is to say
    the mobility the solve used is the mobility the answer implies. What was
    given up is the rate of convergence and not the converged state.

    The reported result carries the whole cost: iterations summed over the
    sweeps and both histories concatenated. The residual history therefore has
    a sawtooth in it, one tooth per refresh, and that is the honest picture of
    what this method does rather than a defect in it.

    A sweep that fails to converge ends the loop immediately and is returned
    as it is. Continuation reads that to decide to halve its step, and there
    is nothing to be gained by refreshing a mobility from a state Newton could
    not reach.
    """
    active = models.at_state(device, *unpack(x0))
    x = x0
    iterations = 0
    residual_history: list[float] = []
    update_history: list[float] = []
    limited_steps = 0
    sweeps = 0

    while sweeps < max_sweeps:
        sweeps += 1
        result = run(active, x)

        iterations += result.iterations
        residual_history.extend(result.residual_history)
        update_history.extend(result.update_history)
        limited_steps += result.limited_steps
        x = result.x

        combined = replace(
            result,
            iterations=iterations,
            residual_history=residual_history,
            update_history=update_history,
            limited_steps=limited_steps,
        )
        if not result.converged:
            return combined

        refreshed = active.at_state(device, *unpack(x))
        if _surface_moved(active, refreshed) < rtol:
            return combined
        active = refreshed

    return replace(
        combined,
        converged=False,
        message=(
            f"the surface mobility was still moving after {max_sweeps} "
            f"sweeps. The last Newton solve converged; what did not is the "
            f"fixed point between the mobility and the state it is read from."
        ),
    )


def solve_bias_newton(
    device: Device,
    models: TransportModels | None = None,
    guess: DeviceState | None = None,
    max_psi_step: float = 5.0,
    max_iterations: int = 30,
    residual_rtol: float = 1e-10,
    update_tol: float = 1e-10,
    max_surface_sweeps: int = 20,
    surface_rtol: float = 1e-8,
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
        max_surface_sweeps: budget for the surface mobility fixed point.
            Ignored where there is no surface model, which is every device
            before Phase 5.
        surface_rtol: how still the surface corrected diffusivity has to be,
            as a relative change on the edge that moved most, before the
            fixed point counts as reached.

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

    **A cold solve with velocity saturation is walked up to it.** The Poisson
    guess is not in the basin of a Caughey-Thomas solve on a MOSFET: measured,
    60 iterations with 55 of them against the step limiter and no convergence.
    The same solve handed the low field answer converges in six, and at the
    off state in zero, because the field dependence changes nothing where no
    current flows. So a cold field dependent solve runs the low field models
    first and continues from that. It is continuation in the model rather than
    in the bias, it is the same shape as the Gummel prelude below it, and the
    iterations it costs are added to the ones reported. See
    `_needs_a_low_field_prelude` for why surface scattering does not need one.
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

    def assembler(
        active: TransportModels,
    ) -> Callable[[npt.NDArray[np.float64]], SparseAssembly]:
        """The assembly closure, over one frozen set of models.

        Taken as a function of the models rather than reading them from the
        enclosing scope, because the surface fixed point replaces them between
        solves and a closure over a name that is being rebound is the kind of
        bug that produces a converged wrong answer.
        """

        def assemble(x: npt.NDArray[np.float64]) -> SparseAssembly:
            assembly, scales = assemble_coupled_terms(
                h,
                volume,
                x,
                net_doping,
                active.Dn,
                active.Dp,
                active.recombination,
                geometry,
                device.degeneracy,
            )
            # Contacts before the scaling, so a pinned row becomes the identity
            # and then gets divided like any other. Scaling first would leave
            # the pinned rows at one while everything around them moved.
            assembly = apply_contacts_coupled(
                assembly,
                x,
                net_doping,
                device.contacts,
                scale,
                carrier_free,
                device.material.T,
                device.degeneracy,
            )
            return scale_rows(assembly, row_weights(scales, mesh.n_nodes))

        return assemble

    def measured(active: TransportModels) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], float
    ]:
        """The residual size, measured row by row against its own terms.

        Recomputed from the iterate rather than carried out of the assembly,
        for the reason `assembler` takes its models as an argument: the
        surface fixed point rebinds the models between solves, and anything
        remembered across that boundary is a converged wrong answer waiting to
        happen. The term scales are a function of the state, so asking for
        them again is the same answer for a fraction of an assembly.
        """

        def norm(
            residual: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
        ) -> float:
            _, n, p = unpack(x)
            scales = residual_term_scales(
                h,
                volume,
                x,
                net_doping,
                active.Dn,
                active.Dp,
                np.asarray(active.recombination.rate(n, p), dtype=np.float64),
                geometry,
                device.degeneracy,
            )
            return residual_measure(residual, scales, mesh.n_nodes)

        return norm

    def run(
        active: TransportModels, x: npt.NDArray[np.float64]
    ) -> NewtonResult:
        return newton_solve(
            assembler(active),
            x,
            limit=lambda delta: limit_psi_step(delta, max_psi_step),
            residual_scale=1.0,
            residual_norm=measured(active),
            residual_rtol=residual_rtol,
            update_tol=update_tol,
            update_norm=coupled_update_norm,
            max_iterations=max_iterations,
        )

    def solve_with(
        active: TransportModels, x: npt.NDArray[np.float64]
    ) -> NewtonResult:
        if active.surface is None:
            return run(active, x)
        return _surface_fixed_point(
            device, active, run, x, max_surface_sweeps, surface_rtol
        )

    prelude: NewtonResult | None = None
    if _needs_a_low_field_prelude(models, guess):
        prelude = solve_with(_low_field_models(models), x0)
        x0 = prelude.x

    result = solve_with(models, x0)
    if prelude is not None:
        result = replace(
            result,
            iterations=result.iterations + prelude.iterations,
            residual_history=prelude.residual_history + result.residual_history,
            update_history=prelude.update_history + result.update_history,
            limited_steps=result.limited_steps + prelude.limited_steps,
        )

    psi, n, p = unpack(result.x)
    return DeviceState(
        psi=_node_field(psi.copy(), "V", "psi"),
        n=_node_field(n.copy(), "cm^-3", "n"),
        p=_node_field(p.copy(), "cm^-3", "p"),
        newton=result,
        degeneracy=device.degeneracy,
    )


def solve_bias_ramped(
    device: Device,
    models: TransportModels | None = None,
    step: float = 0.25,
    max_iterations: int = 30,
) -> DeviceState:
    """Solve at the device's biases with no guess, ramping them in from zero.

    Args:
        device: the device, carrying the contact voltages wanted.
        models: recombination and diffusivities. Built from the device if None.
        step: first continuation step, as a fraction of the applied bias [1].
        max_iterations: Newton budget at each fraction.

    Returns a DeviceState at the device's own biases, with its NewtonResult
    attached, converged or not. Same contract as solve_bias_newton, and the
    same answer wherever that one converges: the parameter is a fraction of
    the bias already on the device, so a fraction of one is the device itself.

    **Why a cold solve needs this and a warm one does not.** The Poisson guess
    knows nothing about the applied bias, so on a MOSFET with the drain at 1 V
    the first Newton step wants a potential update far larger than max_psi_step
    allows and gets clipped. Measured on a 2835 node 1 um NMOS from the Poisson
    guess: at 0 V one step and nothing clipped, at 0.25 V ten steps and nothing
    clipped, at 1 V twenty two steps with twelve of them clipped. A solve that
    spends half its budget against the limiter never reaches a quadratic tail,
    and whether it arrives at all is decided by rounding. It did not arrive on
    CI, which reported a residual of 9.889e+03 on the solve that converges
    here. See docs/07-decisions.md.

    **It is continuation in the bias, not damping.** The Jacobian is not in
    question. At a fraction the solve reaches from its neighbour, the tail is
    quadratic and nothing is clipped, which is what the tests in
    tests/convergence/test_cold_bias_ramp.py assert.
    """
    if models is None:
        models = TransportModels.for_device(device)

    applied = {contact.name: contact.voltage for contact in device.contacts}

    def at_fraction(
        fraction: float, guess: DeviceState | None
    ) -> DeviceState | None:
        solved = solve_bias_newton(
            device.with_bias(
                **{name: fraction * volts for name, volts in applied.items()}
            ),
            models=models,
            guess=guess,
            max_iterations=max_iterations,
        )
        # solve_bias_newton always attaches a NewtonResult, converged or not.
        assert solved.newton is not None
        return solved if solved.newton.converged else None

    off = solve_bias_newton(
        device.with_bias(**dict.fromkeys(applied, 0.0)),
        models=models,
        max_iterations=max_iterations,
    )
    ramp = continue_to(
        at_fraction, start=0.0, target=1.0, initial=off, step=step
    )

    # Unconditionally, rather than returning ramp.solution when it converged.
    # A stalled ramp holds a converged solve at a fraction nobody asked for,
    # and handing that back would be a wrong answer wearing a converged flag.
    # On a ramp that did reach one this costs a solve that takes no steps,
    # because the residual is already under the threshold.
    return solve_bias_newton(
        device,
        models=models,
        guess=ramp.solution,
        max_iterations=max_iterations,
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
