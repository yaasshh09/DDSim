"""Boundary conditions. Ohmic contacts for Phase 1, the rest later.

An ohmic contact imposes charge neutrality and thermal equilibrium at the
contact node simultaneously, from docs/01-physics.md:

    p - n + Nd - Na = 0        and        n * p = n_i^2

Solving that pair fixes psi at the contact:

    psi_contact = V_applied + V_T * asinh(N / (2 * n_i))

In scaled units with C_0 = n_i this is

    psi_contact = V_applied / V_T + asinh(N / 2)

Use asinh, never V_T*ln(N/n_i). The log form is -inf at N = 0 and NaN for net
acceptor doping, and it is wrong anywhere compensation brings N near zero.
docs/01-physics.md calls this out as a frequent bug source, so it is stated
once here and used everywhere.

Every boundary that is not a contact is reflecting, and reflecting is what the
Poisson assembly already produces by having no face on the outward side. So
there is nothing to do for those.

Dirichlet is applied by row replacement: the contact row becomes the identity
and its residual becomes psi - target. That breaks the symmetry of the Poisson
matrix, which costs nothing with a direct solver and keeps the code obvious.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly


@dataclass(frozen=True)
class OhmicContact:
    """An ideal ohmic contact at a single node."""

    name: str
    """Terminal name, for reporting terminal currents later."""

    node: int
    """Mesh node index the contact sits on."""

    voltage: float
    """Applied bias [V]. Physical volts, converted to scaled units on use."""


def ohmic_psi_scaled(net_doping: float, applied: float) -> float:
    """Potential at an ohmic contact [1].

    Args:
        net_doping: scaled net doping at the contact node [1].
        applied: applied bias in scaled units [1], that is V_applied / V_T.

    psi = applied + asinh(N / 2), from neutrality plus mass action.
    """
    return applied + math.asinh(net_doping / 2.0)


def apply_dirichlet(
    assembly: SparseAssembly,
    psi: npt.NDArray[np.float64],
    node: int,
    target: float,
) -> SparseAssembly:
    """Pin psi at one node, returning a new assembly.

    Args:
        assembly: the assembled Poisson system.
        psi: the current scaled potential [1], used to form the residual.
        node: mesh node index to pin.
        target: scaled potential to pin it to [1].

    The row becomes the identity and the residual becomes psi[node] - target,
    so a Newton step lands exactly on the target. The input assembly is not
    modified.
    """
    n_nodes = assembly.shape[0]
    if not 0 <= node < n_nodes:
        raise IndexError(f"node {node} is outside the mesh, which has {n_nodes} nodes")

    keep = assembly.rows != node
    rows = np.concatenate([assembly.rows[keep], np.array([node], dtype=np.int64)])
    cols = np.concatenate([assembly.cols[keep], np.array([node], dtype=np.int64)])
    values = np.concatenate([assembly.values[keep], np.array([1.0])])

    residual = assembly.residual.copy()
    residual[node] = psi[node] - target

    return SparseAssembly(
        residual=residual,
        rows=rows,
        cols=cols,
        values=values,
        shape=assembly.shape,
    )


def apply_ohmic_contacts(
    assembly: SparseAssembly,
    psi: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    contacts: tuple[OhmicContact, ...],
    scale: ScaleFactors,
) -> SparseAssembly:
    """Apply every ohmic contact to an assembled system.

    Args:
        assembly: the assembled Poisson system.
        psi: current scaled potential on nodes [1].
        net_doping: scaled net doping on nodes [1].
        contacts: the contacts to apply.
        scale: scale factors, used to convert contact voltages from V.

    Each contact reads the doping at its own node, so a diode with a p side
    anode and an n side cathode gets the right potential at each end without
    the caller having to work them out.
    """
    names = [contact.name for contact in contacts]
    if len(set(names)) != len(names):
        raise ValueError(f"contact names must be unique, got {names}")

    result = assembly
    for contact in contacts:
        applied = contact.voltage / scale.psi_0
        target = ohmic_psi_scaled(float(net_doping[contact.node]), applied)
        result = apply_dirichlet(result, psi, contact.node, target)
    return result
