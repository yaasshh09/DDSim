"""Carrier mobility models.

docs/01-physics.md orders these by phase and is blunt about the stakes:
"Mobility is where a device simulator earns or loses its quantitative
accuracy. The PDE solve can be perfect and the answer still wrong by 3x if
mobility is wrong. Treat these models as first-class code, not as fudge
factors."

Phase 1 and 2 use a constant. Phase 3 adds the doping dependence, Arora.
Phase 5 adds the field dependence that produces velocity saturation,
Caughey-Thomas, and still owes the surface scattering an inversion layer needs.

Why doping dependence was free and field dependence is not
---------------------------------------------------------
Arora is a function of the doping, and the doping does not change during a
solve. So the diffusivity is a constant array over edges, the Jacobian gains
no new terms, and nothing in discretize/ had to learn anything: the assemblies
already accept Dn and Dp as per edge arrays rather than scalars.

Caughey-Thomas depends on the field, which is a difference of the unknown
potential across an edge, so it belongs inside the Jacobian. What that costs
turns out to be one term per carrier and no new blocks. A flux already carries
a factor of D, so differentiating it with respect to the drop across the edge
adds J times dlnD/dX to a derivative that was there already. The seam is
EdgeMobilityModel below: an assembly handed one of these evaluates it at the
state it is assembling at, the same way it already treats recombination.

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
from typing import Any, Protocol, runtime_checkable

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


# ----------------------------------------------------- field dependent, Phase 5


@runtime_checkable
class EdgeMobilityModel(Protocol):
    """A diffusivity that depends on the potential drop across its edge.

    The seam that lets field dependence into the assemblies without them
    learning any semiconductor physics, exactly as RecombinationModel does for
    the rate. An assembly that is handed one of these evaluates it at the
    state it is assembling at, instead of reading a fixed array.
    """

    def __call__(
        self, X: npt.NDArray[Any], h: npt.NDArray[np.float64]
    ) -> npt.NDArray[Any]:
        """Diffusivity on every edge, at potential drop X across it."""
        ...

    def derivative(
        self, X: npt.NDArray[np.float64], h: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """dD/dX on every edge, for the Jacobian."""
        ...


def _magnitude(X: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """|X|, written so a complex step through it still differentiates.

    abs() of a complex number is its modulus, which is not holomorphic, so a
    residual containing one returns an imaginary part of zero and the block
    verification would report every field dependent term as missing. The
    square root of the square is holomorphic away from the origin and its
    complex step returns sign(X), which is the derivative of the absolute
    value and the thing wanted.

    The real path stays np.abs. sqrt(X*X) agrees with it to within an ulp over
    the range a scaled potential drop occupies, but it squares first, so it
    would return zero for an X below the square root of the smallest normal
    number and infinity above its reciprocal, and neither is a rounding error.
    Same shape as the complex branch in physics/bernoulli.py, and there for
    the same reason.
    """
    if np.iscomplexobj(X):
        return np.asarray(np.sqrt(X * X))
    return np.abs(X)


@dataclass(frozen=True)
class CaugheyThomas:
    """Field dependent mobility on mesh edges, producing velocity saturation.

        mu(E) = mu_0 / (1 + (mu_0 |E| / v_sat)^beta)^(1/beta)

    docs/01-physics.md gives this as the Phase 5 model and is specific about
    two things.

    **E is the component along the current direction, which on a box
    integration mesh is the potential drop across an edge divided by its
    length.** Using the magnitude of the full field vector is the common
    shortcut and it is wrong, by more the more a mesh is graded: a node where
    a 2 nm edge meets a 200 nm one has one field, and the two edges leaving it
    do not see the same one.

    **beta is 2 for electrons and 1 for holes.** The two carriers approach
    saturation differently and one exponent each is how the model says so.

    Why this one is not free where Arora was
    ----------------------------------------
    Arora is a function of the doping, which does not move during a solve, so
    the diffusivity is a constant array and the Jacobian gains nothing. This
    one is a function of the unknown potential, so it belongs inside the
    Jacobian. The term it adds is small and exactly one line per carrier: the
    flux already carries a factor of D, so differentiating it with respect to
    the drop adds J * dlnD/dX to the derivative that was already there.

    Everything here is unit free. The low field diffusivity, the saturation
    velocity and the potential drop have to be in one consistent system, and
    device/transport.py passes scaled ones. In scaled units a diffusivity and
    a mobility are the same number, since D_0 = V_T * mu_0 is exactly the
    Einstein relation the scaling was built on.
    """

    low_field: npt.NDArray[np.float64]
    """Diffusivity on every edge with no field applied. What Arora or the
    constant model gives on the nodes, averaged to the edge."""

    v_sat: float
    """Saturation velocity, in the units low_field and X/h imply."""

    beta: float
    """2 for electrons, 1 for holes."""

    def __post_init__(self) -> None:
        if self.v_sat <= 0.0:
            raise ValueError(f"v_sat must be positive, got {self.v_sat}")
        if self.beta <= 0.0:
            raise ValueError(f"beta must be positive, got {self.beta}")

    def _ratio(
        self, X: npt.NDArray[Any], h: npt.NDArray[np.float64]
    ) -> npt.NDArray[Any]:
        """mu_0 |E| / v_sat on every edge, the argument of the bracket."""
        return np.asarray(
            self.low_field * _magnitude(X) / (h * self.v_sat)
        )

    def __call__(
        self, X: npt.NDArray[Any], h: npt.NDArray[np.float64]
    ) -> npt.NDArray[Any]:
        """Diffusivity on every edge, at potential drop X across it.

        Analytic in X, so a complex step through the residual differentiates
        it. See _magnitude.
        """
        u = self._ratio(X, h)
        return np.asarray(self.low_field / (1.0 + u**self.beta) ** (1.0 / self.beta))

    def derivative(
        self, X: npt.NDArray[np.float64], h: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """dD/dX on every edge. Real only, like the Bernoulli tangent.

            dD/dX = -mu_0 k u^(beta-1) (1 + u^beta)^(-(1+beta)/beta) sign(X)

        with k = mu_0 / (h v_sat), so that u = k|X|.

        At X = 0 with beta = 1 the model has a kink and no two sided
        derivative. sign(0) is zero, which takes the value from neither side
        but from the symmetry between them, and it is what a complex step
        through the square root of a square returns, so the Jacobian and the
        thing that checks it agree there.
        """
        u = self._ratio(X, h)
        k = self.low_field / (h * self.v_sat)
        return np.asarray(
            -np.sign(X)
            * self.low_field
            * k
            * u ** (self.beta - 1.0)
            * (1.0 + u**self.beta) ** (-(1.0 + self.beta) / self.beta)
        )


EdgeDiffusivity = float | npt.NDArray[np.float64] | EdgeMobilityModel
"""A diffusivity, either fixed or a function of the drop across its edge.

A number or an array is the state independent case, which is everything
through Phase 4: a constant, or Arora evaluated on doping that does not move
during a solve. An EdgeMobilityModel is Caughey-Thomas, which reads the
potential drop and so has to be evaluated at whatever state is being solved.
"""


def diffusivity_at(
    D: EdgeDiffusivity, X: npt.NDArray[Any], h: npt.NDArray[np.float64]
) -> float | npt.NDArray[Any]:
    """One diffusivity per edge at potential drop X across each edge [1].

    Dtype preserving through the model, so a complex step through a residual
    picks the field dependence up instead of silently missing it.
    """
    if isinstance(D, EdgeMobilityModel):
        return D(X, h)
    return D


def diffusivity_tangent(
    D: EdgeDiffusivity,
    X: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    """dD/dX on every edge [1], or None where D does not depend on X.

    None rather than an array of zeros, so an assembly with no field dependent
    model does exactly the arithmetic it did before there was one. Adding a
    zero is not free of consequence in a project that claims earlier results
    are unchanged bit for bit; not adding it is.
    """
    if isinstance(D, EdgeMobilityModel):
        return D.derivative(X, h)
    return None


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
