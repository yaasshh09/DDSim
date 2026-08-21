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
from ddsim.discretize.boundary import apply_ohmic_contacts
from ddsim.discretize.poisson import assemble_poisson
from ddsim.physics.statistics import (
    n_boltzmann_scaled,
    p_boltzmann_scaled,
    psi_equilibrium_scaled,
)
from ddsim.solve.newton import NewtonResult, newton_solve

MAX_PSI_STEP = 5.0
"""Largest Newton step in psi, in scaled units [1].

docs/02-numerics.md prescribes 5 * V_T. A full undamped step from the charge
neutral guess at a heavily doped junction can be tens of volts, which
overflows exp immediately.
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

    n_side = [c for c in device.contacts if doping[c.node] >= 0.0]
    p_side = [c for c in device.contacts if doping[c.node] < 0.0]

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


def solve_poisson(
    device: Device,
    psi_initial: npt.NDArray[np.float64],
    phi_n: Field | None = None,
    phi_p: Field | None = None,
    max_iterations: int = 50,
    residual_rtol: float = 1e-10,
    update_tol: float = 1e-10,
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

    This is both the whole of the equilibrium solve and the first block of
    every Gummel cycle. Keeping the densities inside Poisson as exp(psi - phi)
    rather than freezing them is what makes the cycle robust: the exponential
    nonlinearity stays implicit and Newton handles it on a strictly positive
    definite Jacobian.

    Returns the NewtonResult rather than raising, so a caller inside a Gummel
    cycle can decide what a stalled Poisson solve means.
    """
    mesh = device.mesh
    scale = device.scale
    net_doping = device.net_doping_scaled
    doping_values = net_doping.data

    def assemble(psi_values: npt.NDArray[np.float64]) -> SparseAssembly:
        psi = Field(psi_values, "V", ScalingState.SCALED, Location.NODE, name="psi")
        assembly = assemble_poisson(mesh, psi, net_doping, scale, phi_n, phi_p)
        return apply_ohmic_contacts(
            assembly, psi_values, doping_values, device.contacts, scale
        )

    # The residual is a charge balance over each dual cell, so the size of its
    # terms is the doping charge in the largest cell. Taking the threshold from
    # that rather than from the initial residual is what lets a solve that
    # starts at the answer report success: inside a Gummel cycle the potential
    # arrives already converged, its residual already at the roundoff floor,
    # and a threshold relative to that floor is unreachable by construction.
    residual_scale = float(np.max(np.abs(doping_values) * mesh.volume / scale.x_0))

    return newton_solve(
        assemble,
        psi_initial,
        max_step=MAX_PSI_STEP,
        residual_rtol=residual_rtol,
        residual_scale=residual_scale,
        update_tol=update_tol,
        max_iterations=max_iterations,
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
