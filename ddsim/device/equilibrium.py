"""Solve a device at thermal equilibrium.

This is where the pieces meet: the mesh and doping come from device/, the
residual and Jacobian from discretize/, and the iteration from solve/. Nothing
here knows how to assemble or how to iterate, it only wires the two together,
which is what keeps solve/ free of semiconductor knowledge.

Equilibrium means phi_n = phi_p = 0 everywhere, so Boltzmann gives n = exp(psi)
and p = exp(-psi) in scaled units and Poisson closes on psi alone. One unknown
per node, a symmetric M-matrix, and Newton that converges reliably. That is why
Phase 1 comes first.

The initial guess is the charge neutral potential, psi = asinh(N/2), which is
the exact answer in uniform material and a good one everywhere except within a
few Debye lengths of a junction.

Applying a bias
---------------
phases/PHASE-1.md asks for depletion widths at -1 V and -5 V reverse bias, and
that cannot be done with phi_n = phi_p = 0. With the quasi-Fermi levels pinned
at zero the carrier densities are tied absolutely to psi, so a quasi-neutral
region cannot shift its potential without changing p by exp(38.7) per volt. The
applied bias never reaches the junction: it drops across a thin layer at the
contact instead. Measured on a 1e16 diode at -1 V, the whole volt falls across
0.05 um at the contact with a 2e5 V/cm field there, while the junction field
stays at its zero bias value of 3.2e4 V/cm.

The standard resolution, and step one of Gummel iteration in Phase 2, is to
carry fixed quasi-Fermi levels. With no current flowing, phi_n is flat across
the whole device at the bias of the contact in the n-type material, and phi_p
is flat at the bias of the contact in the p-type material. Their separation is
the applied bias, which is what reverse bias means.

They have to be two separate levels rather than one. A single common phi with a
step at the metallurgical junction drives n = exp(psi - phi) to 1e26 cm^-3 on
the p side of the junction, which screens the field and gives a depletion width
three times too small.

frozen_quasi_fermi builds both. It is an approximation, valid at reverse bias
and low forward bias where recombination has not yet bent the levels, and
Phase 2 replaces it by solving for phi_n and phi_p properly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import (
    GateContact,
    apply_contacts,
    apply_dirichlet_nodes,
)
from ddsim.discretize.poisson import assemble_poisson
from ddsim.physics.statistics import (
    n_boltzmann_scaled,
    p_boltzmann_scaled,
    psi_equilibrium_scaled,
)
from ddsim.solve.linear import SparseLU
from ddsim.solve.newton import NewtonResult, newton_solve

MAX_PSI_STEP = 5.0
"""Largest Newton step in psi, in scaled units [1].

docs/02-numerics.md prescribes 5 * V_T. A full undamped step from the charge
neutral guess at a heavily doped junction can be tens of volts, which
overflows exp immediately.
"""

EPS = float(np.finfo(np.float64).eps)
"""Machine epsilon [1], the unit the flux cancellation floor is measured in."""

FLUX_FLOOR_MARGIN = 16.0
"""Headroom over eps times the largest face flux [1].

The residual floor is a few eps times the flux, not exactly one: each node
sums two face fluxes and a charge term, and each face flux is itself a
difference. Measured floors run 1.2 to 1.7 times eps*flux across four decades
of doping, so 16 leaves an order of magnitude of headroom while staying far
below the charge threshold at any doping where that one binds.
"""


def frozen_quasi_fermi(device: Device) -> tuple[Field, Field]:
    """Flat quasi-Fermi levels (phi_n, phi_p) [V], scaled.

    With no current flowing, each level is constant across the whole device.
    phi_n takes the bias of the contact sitting in n-type material and phi_p
    the bias of the contact sitting in p-type material, because those are the
    contacts that fix the majority carrier population at each end. Their
    difference is the applied bias.

    They have to be two separate levels. A single common phi with a step at the
    metallurgical junction drives n = exp(psi - phi) to 1e26 cm^-3 on the p side
    of the junction, which screens the field and shrinks the depletion width by
    a factor of three.

    If every contact sits in material of one type, both levels fall back to
    that contact's bias, which is the right answer for a resistor.

    This is the reverse bias and low injection approximation. Phase 2 solves
    for phi_n and phi_p instead of assuming them flat.
    """
    doping = device.net_doping.data
    n_nodes = device.mesh.n_nodes

    # A gate has no doping under it to read and no quasi-Fermi level of its
    # own: it is metal on an insulator. Only the contacts that touch
    # semiconductor say anything about where phi_n and phi_p sit.
    ohmic = [c for c in device.contacts if not isinstance(c, GateContact)]
    n_side = [c for c in ohmic if doping[c.nodes[0]] >= 0.0]
    p_side = [c for c in ohmic if doping[c.nodes[0]] < 0.0]

    n_bias = n_side[0].voltage if n_side else p_side[0].voltage
    p_bias = p_side[0].voltage if p_side else n_side[0].voltage

    phi_n = Field(
        np.full(n_nodes, n_bias / device.scale.psi_0),
        "V",
        ScalingState.SCALED,
        Location.NODE,
        name="phi_n",
    )
    phi_p = Field(
        np.full(n_nodes, p_bias / device.scale.psi_0),
        "V",
        ScalingState.SCALED,
        Location.NODE,
        name="phi_p",
    )
    return phi_n, phi_p


def insulator_guess(
    device: Device, psi: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Fill in the starting potential wherever there is no semiconductor [1].

    Args:
        device: the device. Returned unchanged if it is all semiconductor.
        psi: the charge neutral starting guess on nodes [1], scaled.

    The charge neutral guess is meaningless in an insulator. psi = asinh(N/2)
    with N = 0 is the intrinsic level of a material that has no carriers to be
    intrinsic about, so the oxide starts at zero while the gate sits tens of
    scaled units away.

    That is expensive rather than merely inelegant. psi is damped to MAX_PSI_STEP
    per Newton step and the damping is one factor over the whole vector, so an
    oxide that has to travel a long way throttles the semiconductor with it.
    Measured cold on the default MOS capacitor, the iteration count grows at
    7.7 per volt of gate bias and exhausts a 50 step budget at about 6.5 V.

    With no charge in it, Poisson in an insulator is Laplace, which is linear,
    so one solve gives the insulator the exact answer to its own equation given
    everything around it. That is written as a solve rather than as an
    interpolation between the gate and the surface on purpose: the solve knows
    nothing about which way the layers stack, and the interpolation would have
    to. On the MOS stack the two agree to 5.7e-14 on values of order 300.

    Everything that is not insulator is pinned where it is, so the semiconductor
    comes back bit for bit unchanged and no device without an insulator pays
    anything at all.
    """
    if device.regions is None or device.regions.oxide_nodes.size == 0:
        return psi

    scale = device.scale
    net_doping = device.net_doping_scaled
    field = Field(psi, "V", ScalingState.SCALED, Location.NODE, name="psi")

    assembly = assemble_poisson(
        device.scaled_mesh,
        field,
        net_doping,
        charge_volume=device.charge_volume_scaled,
    )
    assembly = apply_contacts(
        assembly,
        psi,
        net_doping.data,
        device.contacts,
        scale,
        device.material.T,
    )

    # Hold the semiconductor. The gate is already held by apply_contacts, so
    # what is left free is exactly the interior of the insulator.
    held = np.flatnonzero(device.regions.semiconductor_volume > 0.0)
    assembly = apply_dirichlet_nodes(
        assembly, psi, held.tolist(), psi[held].tolist()
    )

    # Nonsingular by construction, so there is no special case here to write.
    # Laplace on the insulator needs boundary data somewhere on every connected
    # piece of it, and it always has some: build_device refuses a device with
    # no contacts, tensor_mesh_2d only makes connected meshes, and every node
    # of an insulator therefore reaches either a pinned contact or a held
    # semiconductor node through the mesh. A floating insulator island would
    # break that, and none can be built.
    solver = SparseLU()
    solver.factorize(
        assembly.rows, assembly.cols, assembly.values, assembly.shape
    )
    return psi + solver.solve(-assembly.residual)


def solve_poisson(
    device: Device,
    psi_initial: npt.NDArray[np.float64],
    phi_n: Field | None = None,
    phi_p: Field | None = None,
    max_iterations: int = 50,
    residual_rtol: float = 1e-10,
    update_tol: float = 1e-10,
    solver: SparseLU | None = None,
) -> NewtonResult:
    """Solve the nonlinear Poisson equation for psi at fixed quasi-Fermi levels.

    Args:
        device: the device specification, which carries the mesh, the doping
            and the contact biases.
        psi_initial: scaled starting potential on nodes [1].
        phi_n: electron quasi-Fermi potential [V], scaled. None means zero.
        phi_p: hole quasi-Fermi potential [V], scaled. None means zero.
        max_iterations: Newton iteration budget.
        residual_rtol: residual threshold relative to the initial residual [1].
        update_tol: convergence threshold on max |dpsi| [1].
        solver: a factorization to reuse. Every Gummel cycle solves this same
            system on the same mesh, so the caller inside a cycle keeps one
            rather than paying for the sparsity pattern each time.

    This is both the whole of the equilibrium solve and the first block of
    every Gummel cycle. Keeping the densities inside Poisson as exp(psi - phi)
    rather than freezing them is what makes the cycle robust: the exponential
    nonlinearity stays implicit and Newton handles it on a strictly positive
    definite Jacobian.

    Returns the NewtonResult rather than raising, so a caller inside a Gummel
    cycle can decide what a stalled Poisson solve means.
    """
    scale = device.scale
    mesh = device.scaled_mesh
    charge_volume = device.charge_volume_scaled
    net_doping = device.net_doping_scaled
    doping_values = net_doping.data

    def assemble(psi_values: npt.NDArray[np.float64]) -> SparseAssembly:
        psi = Field(psi_values, "V", ScalingState.SCALED, Location.NODE, name="psi")
        assembly = assemble_poisson(
            mesh, psi, net_doping, phi_n, phi_p, charge_volume=charge_volume
        )
        return apply_contacts(
            assembly,
            psi_values,
            doping_values,
            device.contacts,
            scale,
            device.material.T,
        )

    # The residual is a charge balance over each dual cell, so the size of its
    # terms is the doping charge in the largest cell. Taking the threshold from
    # that rather than from the initial residual is what lets a solve that
    # starts at the answer report success: inside a Gummel cycle the potential
    # arrives already converged, its residual already at the roundoff floor,
    # and a threshold relative to that floor is unreachable by construction.
    charge = float(np.max(np.abs(doping_values) * charge_volume))

    # The other half of the residual is a difference of face fluxes, each of
    # size psi/h, and a difference cannot be resolved below machine epsilon
    # times the size of the things being differenced. That is a floor no solve
    # gets under however exactly it satisfies the equation.
    #
    # It has to be in the threshold because the two terms scale differently.
    # The charge falls with the doping while the flux barely moves, psi being
    # logarithmic in it. At 1e16 the charge threshold sits four decades above
    # the floor and this never binds. Below about 1e13 it sinks underneath,
    # and then a perfectly converged solve reports failure having spent its
    # whole iteration budget on a residual that stopped moving at step three.
    # Measured on a 1e12 uniform bar: residual pinned at 6.8e-12 for 47
    # iterations against a threshold of 3.4e-12, update 4.4e-16 throughout.
    left, right = mesh.geometry.ends(mesh.n_edges)
    edge_psi = np.maximum(np.abs(psi_initial[left]), np.abs(psi_initial[right]))
    flux_floor = FLUX_FLOOR_MARGIN * EPS * float(
        np.max(mesh.geometry.weight * edge_psi / mesh.h)
    )

    # Raise the scale only when the floor would otherwise bind, so that every
    # device where the charge already dominates keeps the threshold it had.
    residual_scale = charge
    if residual_rtol > 0.0 and residual_rtol * charge < flux_floor:
        residual_scale = flux_floor / residual_rtol

    return newton_solve(
        assemble,
        psi_initial,
        max_step=MAX_PSI_STEP,
        residual_rtol=residual_rtol,
        residual_scale=residual_scale,
        update_tol=update_tol,
        max_iterations=max_iterations,
        solver=solver,
    )


def solve_equilibrium(
    device: Device,
    quasi_fermi: tuple[Field, Field] | None = None,
    max_iterations: int = 50,
    residual_rtol: float = 1e-10,
    update_tol: float = 1e-10,
) -> DeviceState:
    """Solve nonlinear Poisson at equilibrium.

    Args:
        device: the device specification.
        quasi_fermi: fixed (phi_n, phi_p) [V], scaled. None means true thermal
            equilibrium, both zero everywhere. Pass frozen_quasi_fermi(device)
            to solve a reverse biased junction.
        max_iterations: Newton iteration budget.
        residual_rtol: residual threshold relative to the initial residual
            [1]. Relative rather than absolute, because the Poisson residual
            scales with the doping and so does its roundoff floor.
        update_tol: convergence threshold on max |dpsi| [1].

    Raises RuntimeError if Newton does not converge. An unconverged solution
    that is returned quietly is the worst outcome available here, because it
    looks like a converged one and every number downstream inherits the error.
    """
    phi_n, phi_p = (None, None) if quasi_fermi is None else quasi_fermi
    doping_values = device.net_doping_scaled.data

    # Charge neutral guess, shifted by the quasi-Fermi level of the local
    # majority carrier so that a biased region starts near the potential its
    # contact demands rather than a whole volt away from it.
    initial = np.asarray(psi_equilibrium_scaled(doping_values), dtype=np.float64)
    if phi_n is not None and phi_p is not None:
        majority = np.where(doping_values >= 0.0, phi_n.data, phi_p.data)
        initial = initial + majority

    # Where there is no semiconductor the neutral guess says nothing, and the
    # damping makes that expensive rather than merely inaccurate.
    initial = insulator_guess(device, initial)

    result = solve_poisson(
        device,
        initial,
        phi_n,
        phi_p,
        max_iterations=max_iterations,
        residual_rtol=residual_rtol,
        update_tol=update_tol,
    )

    if not result.converged:
        raise RuntimeError(
            f"equilibrium solve did not converge: {result.message}. "
            f"Residual history: {result.residual_history}"
        )

    n_level = 0.0 if phi_n is None else phi_n.data
    p_level = 0.0 if phi_p is None else phi_p.data
    psi = Field(result.x, "V", ScalingState.SCALED, Location.NODE, name="psi")
    return DeviceState(
        psi=psi,
        n=Field(
            np.asarray(n_boltzmann_scaled(result.x, n_level)),
            "cm^-3",
            ScalingState.SCALED,
            Location.NODE,
            name="n",
        ),
        p=Field(
            np.asarray(p_boltzmann_scaled(result.x, p_level)),
            "cm^-3",
            ScalingState.SCALED,
            Location.NODE,
            name="p",
        ),
        newton=result,
    )
