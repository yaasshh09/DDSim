"""Carrier mobility models.

docs/01-physics.md orders these by phase and is blunt about the stakes:
"Mobility is where a device simulator earns or loses its quantitative
accuracy. The PDE solve can be perfect and the answer still wrong by 3x if
mobility is wrong. Treat these models as first-class code, not as fudge
factors."

Phase 1 and 2 use a constant. Phase 3, this module, adds the doping
dependence. Phase 5 adds the field dependence that produces velocity
saturation, and the surface scattering an inversion layer needs.

Why doping dependence is free and field dependence will not be
------------------------------------------------------------
Arora is a function of the doping, and the doping does not change during a
solve. So the diffusivity is a constant array over edges, the Jacobian gains
no new terms, and nothing in discretize/ has to learn anything: the assemblies
already accept Dn and Dp as per edge arrays rather than scalars.

Caughey-Thomas in Phase 5 depends on the field, which is a difference of the
unknown potential across the edge. That puts mobility inside the Jacobian and
adds a dmu/dpsi term to every flux derivative. It is a much larger change and
it is deliberately not started here.

Total doping, not net
---------------------
The model wants the concentration of ionized scattering centres, which is
Na + Nd. Only the net is available from a composed profile, so abs(net) is
used, exactly as the Scharfetter lifetime in physics/recombination.py does.
The two agree everywhere except in compensated material, and nothing here is
compensated yet. When a profile that overlaps donors and acceptors arrives,
this and the lifetime are the two places that have to learn about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C

Doping = float | npt.NDArray[np.float64]
"""A doping concentration [cm^-3], scalar or per node."""


@runtime_checkable
class MobilityModel(Protocol):
    """What a transport solve needs from a mobility model.

    A protocol rather than a base class, so that Masetti, or Caughey-Thomas
    wrapped around one of these, slots in without touching anything here.
    """

    def __call__(self, total_doping: Doping) -> npt.NDArray[np.float64]:
        """Mobility [cm^2/(V s)] at each node, from the total doping."""
        ...


@dataclass(frozen=True)
class ConstantMobility:
    """Mobility that ignores the doping. The Phase 1 and 2 model.

    Kept once the doping dependent model exists so that the seam has one
    shape, and so that a comparison between the two is a change of model
    rather than a change of code path.
    """

    value: float
    """Mobility [cm^2/(V s)]."""

    def __call__(self, total_doping: Doping) -> npt.NDArray[np.float64]:
        """The same value at every node.

        One value per node rather than a bare scalar, because a scalar would
        broadcast into an edge average and silently collapse it.
        """
        return np.full(np.shape(total_doping), self.value, dtype=np.float64)


@dataclass(frozen=True)
class AroraMobility:
    """Doping dependent mobility, Arora.

        mu = mu_min + mu_d / (1 + (N / N_ref)^A)

    Four limits, all checkable by hand and all tested:

        N -> 0      mu -> mu_min + mu_d
        N = N_ref   mu = mu_min + mu_d/2
        N -> inf    mu -> mu_min
        dmu/dN < 0  everywhere, since A > 0

    Parameters from docs/06-constants.md, each with its own temperature
    exponent. Build them with the electrons and holes constructors rather than
    by hand.

    Its N -> 0 limit is not the tabulated undoped mobility. Arora gives 1340
    for electrons against a table value of 1417, and 461.3 for holes against
    470. Both numbers are measured; they come from different fits, and Arora
    is fitted over the doped range where it gets used rather than at an
    intrinsic limit nobody measures. Switching a device from the constant
    model to this one therefore moves its current by five percent even at
    doping low enough that the model should be doing nothing, which is worth
    knowing before it reads as a bug.
    """

    mu_min: float
    """Mobility floor at very high doping [cm^2/(V s)]."""

    mu_d: float
    """Lattice scattering contribution, lost as doping rises [cm^2/(V s)]."""

    N_ref: float
    """Doping at which half of mu_d has been lost [cm^-3]."""

    exponent: float
    """The exponent A. Sets how sharply mobility falls through N_ref."""

    @classmethod
    def electrons(cls, T: float = C.T_ROOM) -> AroraMobility:
        """Arora parameters for electrons in silicon at temperature T [K]."""
        ratio = T / C.T_ROOM
        return cls(
            mu_min=88.0 * ratio**-0.57,
            mu_d=1252.0 * ratio**-2.33,
            N_ref=1.432e17 * ratio**2.546,
            exponent=0.88 * ratio**-0.146,
        )

    @classmethod
    def holes(cls, T: float = C.T_ROOM) -> AroraMobility:
        """Arora parameters for holes in silicon at temperature T [K].

        The mu_d exponent is -2.23 here against -2.33 for electrons. They look
        like a typo for one another and they are not; both are in the source
        table and in the literature.
        """
        ratio = T / C.T_ROOM
        return cls(
            mu_min=54.3 * ratio**-0.57,
            mu_d=407.0 * ratio**-2.23,
            N_ref=2.67e17 * ratio**2.546,
            exponent=0.88 * ratio**-0.146,
        )

    def __call__(self, total_doping: Doping) -> npt.NDArray[np.float64]:
        """Mobility [cm^2/(V s)] at each node.

        The doping is taken in absolute value, so a net doping array can be
        passed straight in. See the module docstring on total against net.
        """
        N = np.abs(np.asarray(total_doping, dtype=np.float64))
        return np.asarray(
            self.mu_min + self.mu_d / (1.0 + (N / self.N_ref) ** self.exponent)
        )


def edge_diffusivity(
    mobility: npt.NDArray[np.float64],
    V_T: float,
    edge_nodes: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float64]:
    """Diffusivity on every edge [cm^2/s], from mobility on every node.

    Args:
        mobility: mobility at each node [cm^2/(V s)], length n_nodes.
        V_T: thermal voltage [V].
        edge_nodes: shape (n_edges, 2), the two nodes of each edge. None means
            the contiguous 1D chain, where edge e joins node e and node e+1.
            A 2D mesh has to say, because its edges are not contiguous and a
            column of it is not a slice.

    Two steps, both worth stating.

    **The Einstein relation, D = V_T * mu.** It holds under Boltzmann
    statistics, which docs/01-physics.md keeps through Phase 4. It stops
    holding in degenerate material, which is one more reason the 1e20 results
    carry no quantitative claim.

    **The arithmetic mean of the two endpoints.** Scharfetter-Gummel derives
    its edge flux by assuming the current and the coefficients are constant
    along the edge, so a single value per edge is what the scheme asks for and
    the only question is which one. The mean is the honest reading of a
    quantity that is genuinely varying. On a mesh graded to a junction the two
    endpoints of an edge sit in nearly the same doping anyway, so the choice
    only shows up on a coarse mesh across an abrupt profile, where every part
    of the answer is already mesh limited.

    The edge list form gathers the same two endpoints the slices name, in the
    same order, so a 1D device gets the identical arithmetic and the identical
    bits whichever branch it takes. The argument is taken as a plain array
    rather than an EdgeGeometry to keep physics/ from importing discretize/.
    """
    if edge_nodes is None:
        return np.asarray(V_T * 0.5 * (mobility[:-1] + mobility[1:]))
    left, right = edge_nodes[:, 0], edge_nodes[:, 1]
    return np.asarray(V_T * 0.5 * (mobility[left] + mobility[right]))
