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

The same pair fixes n and p there, from Phase 2 onward, and the two densities
are pinned at their equilibrium values whatever the terminal voltage. The bias
goes into psi and not into the densities. That is exactly what an ideal ohmic
contact is: a perfect sink, where any excess carrier recombines with infinite
velocity, which is why a diode with no bulk recombination at all still passes
current.

Every boundary that is not a contact is reflecting, and reflecting is what the
Poisson assembly already produces by having no face on the outward side. So
there is nothing to do for those.

Dirichlet is applied by eliminating both the row and the column: the contact
row becomes the identity, its residual becomes psi - target, and every other
row that referenced the pinned unknown has that coefficient folded into its own
residual. Row replacement alone is the textbook shortcut and it leaves the
pinned unknown coupled into its neighbours, which costs accuracy exactly where
a pinned value is many decades away from the rest of the solution. Both break
the symmetry of the Poisson matrix, which costs nothing with a direct solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.physics.statistics import equilibrium_densities_scaled


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
    value: npt.NDArray[np.float64],
    node: int,
    target: float,
) -> SparseAssembly:
    """Pin one unknown at one node, returning a new assembly.

    Args:
        assembly: the assembled system, Poisson or either continuity equation.
        value: the current scaled unknown [1], used to form the residual.
        node: mesh node index to pin.
        target: scaled value to pin it to [1].

    The row becomes the identity and its residual becomes value - target, so
    the update at that node is exactly target - value.

    **The column is eliminated as well, not only the row.** Every other row
    that referenced this unknown has its coefficient folded into its residual,
    which is exact because the update at a pinned node is known before the
    solve. Leaving the column in place is the textbook shortcut and it is
    subtly wrong in floating point: the pinned unknown stays coupled into every
    neighbouring equation, so the factorization mixes it with the rest of the
    solution and the pinned value comes back only to within the conditioning of
    the whole system.

    That is not hypothetical. The minority carrier density at a contact is 1e-6
    in scaled units while the majority density elsewhere is 1e7, thirteen
    decades apart in one linear system. Solved with the column left in, a cold
    start at 0.9 V forward bias returned n = -1.02e-6 at the contact, a pinned
    value that came back negative. Eliminating the column makes it exact
    instead, since the unknown no longer appears anywhere the factorization can
    reach.

    The input assembly is not modified.
    """
    n_nodes = assembly.shape[0]
    if not 0 <= node < n_nodes:
        raise IndexError(f"node {node} is outside the mesh, which has {n_nodes} nodes")

    # The update this node will take. Known exactly, before any solve.
    correction = target - value[node]

    in_row = assembly.rows == node
    in_column = (assembly.cols == node) & ~in_row

    # J*delta = -F, and delta at this node is fixed, so its column moves to
    # the right hand side: F_i becomes F_i + J[i, node] * correction.
    residual = assembly.residual.copy()
    np.add.at(
        residual,
        assembly.rows[in_column],
        assembly.values[in_column] * correction,
    )
    residual[node] = value[node] - target

    keep = ~in_row & ~in_column
    rows = np.concatenate([assembly.rows[keep], np.array([node], dtype=np.int64)])
    cols = np.concatenate([assembly.cols[keep], np.array([node], dtype=np.int64)])
    values = np.concatenate([assembly.values[keep], np.array([1.0])])

    return SparseAssembly(
        residual=residual,
        rows=rows,
        cols=cols,
        values=values,
        shape=assembly.shape,
    )


class Carrier(Enum):
    """Which continuity equation a boundary condition is being applied to."""

    ELECTRON = "electron"
    HOLE = "hole"


def ohmic_density_scaled(net_doping: float, carrier: Carrier) -> float:
    """One carrier density at an ohmic contact [1].

    Neutrality plus mass action, n - p = N and n*p = 1, solved for whichever
    carrier is asked for. The minority one comes from the reciprocal rather
    than from the quadratic formula, so mass action is exact rather than
    merely close. See physics/statistics.py.
    """
    n_contact, p_contact = equilibrium_densities_scaled(net_doping)
    return float(n_contact if carrier is Carrier.ELECTRON else p_contact)


def impose_ohmic_densities(
    density: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    contacts: tuple[OhmicContact, ...],
    carrier: Carrier,
) -> npt.NDArray[np.float64]:
    """Write the contact densities into a solved profile, exactly.

    A Dirichlet condition is a statement about the solution, so the right way
    to end up with it is to impose it rather than to arrive at it. Building the
    contact value as old + update instead cancels whenever the two are far
    apart, and at a diode contact they are thirteen decades apart on the first
    forward biased cycle: the update is -6.6e8 and the answer is 1e-6.

    This is not clamping. Nothing here inspects the solved value or moves it
    toward anything, and every interior node is left alone. See
    docs/05-pitfalls.md on why the difference matters.
    """
    imposed = density.copy()
    for contact in contacts:
        imposed[contact.node] = ohmic_density_scaled(
            float(net_doping[contact.node]), carrier
        )
    return imposed


def apply_ohmic_densities(
    assembly: SparseAssembly,
    density: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    contacts: tuple[OhmicContact, ...],
    carrier: Carrier,
) -> SparseAssembly:
    """Pin one carrier density at every contact to its equilibrium value.

    Args:
        assembly: the assembled continuity system.
        density: the current scaled density on nodes [1], n or p.
        net_doping: scaled net doping on nodes [1].
        contacts: the contacts to apply.
        carrier: which of n and p is being solved for.

    An ohmic contact is in thermal equilibrium whatever the terminal voltage,
    so both densities are pinned at the values that solve neutrality together
    with mass action. The applied bias goes into psi, not into n or p. That is
    what makes the contact a perfect sink: any excess carrier arriving there
    recombines with infinite velocity.

    Consistency is worth checking once by hand. The contact potential is
    V + asinh(N/2) and the quasi-Fermi levels there are both V, so
    n = exp(psi - phi_n) = exp(asinh(N/2)), which is exactly the neutrality
    solution pinned here. The two boundary conditions agree by construction
    rather than by coincidence.
    """
    result = assembly
    for contact in contacts:
        target = ohmic_density_scaled(float(net_doping[contact.node]), carrier)
        result = apply_dirichlet(result, density, contact.node, target)
    return result


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
