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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.discretize.boundary import apply_ohmic_contacts
from ddsim.discretize.poisson import PoissonAssembly, assemble_poisson
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


@dataclass(frozen=True)
class DeviceState:
    """A converged solution. All fields are scaled and live on nodes."""

    psi: Field
    """Electrostatic potential [V], scaled by V_T."""

    n: Field
    """Electron density [cm^-3], scaled by C_0."""

    p: Field
    """Hole density [cm^-3], scaled by C_0."""

    newton: NewtonResult
    """The solver history, including the residual tail."""


def solve_equilibrium(
    device: Device,
    max_iterations: int = 50,
    residual_rtol: float = 1e-10,
    update_tol: float = 1e-10,
) -> DeviceState:
    """Solve nonlinear Poisson at equilibrium.

    Args:
        device: the device specification.
        max_iterations: Newton iteration budget.
        residual_rtol: residual threshold relative to the initial residual
            [1]. Relative rather than absolute, because the Poisson residual
            scales with the doping and so does its roundoff floor.
        update_tol: convergence threshold on max |dpsi| [1].

    Raises RuntimeError if Newton does not converge. An unconverged solution
    that is returned quietly is the worst outcome available here, because it
    looks like a converged one and every number downstream inherits the error.
    """
    mesh = device.mesh
    scale = device.scale
    net_doping = device.net_doping_scaled
    doping_values = net_doping.data

    def assemble(psi_values: npt.NDArray[np.float64]) -> PoissonAssembly:
        psi = Field(psi_values, "V", ScalingState.SCALED, Location.NODE, name="psi")
        assembly = assemble_poisson(mesh, psi, net_doping, scale)
        return apply_ohmic_contacts(
            assembly, psi_values, doping_values, device.contacts, scale
        )

    initial = np.asarray(psi_equilibrium_scaled(doping_values), dtype=np.float64)

    result = newton_solve(
        assemble,
        initial,
        max_step=MAX_PSI_STEP,
        residual_rtol=residual_rtol,
        update_tol=update_tol,
        max_iterations=max_iterations,
    )

    if not result.converged:
        raise RuntimeError(
            f"equilibrium solve did not converge: {result.message}. "
            f"Residual history: {result.residual_history}"
        )

    psi = Field(result.x, "V", ScalingState.SCALED, Location.NODE, name="psi")
    return DeviceState(
        psi=psi,
        n=Field(
            np.asarray(n_boltzmann_scaled(result.x)),
            "cm^-3",
            ScalingState.SCALED,
            Location.NODE,
            name="n",
        ),
        p=Field(
            np.asarray(p_boltzmann_scaled(result.x)),
            "cm^-3",
            ScalingState.SCALED,
            Location.NODE,
            name="p",
        ),
        newton=result,
    )
