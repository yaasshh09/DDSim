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


# ------------------------------------------------- surface scattering, Phase 5


@dataclass(frozen=True)
class LombardiSurface:
    """Surface scattering at the Si/SiO2 interface, by Matthiessen's rule.

        1/mu = 1/mu_bulk + 1/mu_ac + 1/mu_sr

        mu_ac = B/E_perp + C N^tau E_perp^(-1/3) / (T/300)^kappa
        mu_sr = delta E_perp^(-gamma)
        gamma = A + alpha (n + p) N^(-eta)

    docs/01-physics.md calls this not optional and says what it is worth:
    without it the inversion layer mobility is too high by a factor of 2 to 3
    and the drain current is wrong by the same factor. A test asserts that
    factor at a channel condition rather than taking it on trust.

    Why it needs no layer thickness
    -------------------------------
    Both surface terms diverge as E_perp falls, so their reciprocals vanish
    and Matthiessen hands back mu_bulk untouched. That is what lets the model
    be evaluated over a whole region instead of inside a surface layer whose
    depth somebody would have to pick, and picking one is exactly the sort of
    fitting this phase's success criterion forbids.

    E_perp is a magnitude
    ---------------------
    The field normal to the interface, not a signed component and not the
    length of the full field vector. On the tensor product mesh this project
    uses, the interface is a horizontal line, so the normal direction is y and
    the magnitude is abs(E_y). A negative argument is refused rather than
    quietly absolute valued, because a signed difference that reached here
    would produce a plausible looking mobility and no other symptom.

    Provenance
    ----------
    This is the enhanced Lombardi, sometimes the Darwish model, in the form
    DEVSIM ships in its Klaassen.py, with DEVSIM's parameter values. It is the
    1988 Lombardi model with the surface roughness exponent made a function of
    the carrier density instead of fixed at 2. Matching the model the tier 4
    reference actually runs is what gives the MOSFET regressions a chance of
    agreeing, and none of these parameters is in docs/06-constants.md, so a
    test pins every one of them against the reference's own source.

    The floor on E_perp is DEVSIM's too, and it is load bearing rather than
    cosmetic: both terms divide by E_perp, so an unfloored zero is an infinity
    in mu_ac and a nan as soon as it meets the reciprocal sum.

    The parameter names are the reference's own, which is what makes the
    pinning test readable against DEVSIM's source, with one exception. Its C
    is C_ac here, because a field called C would shadow this module's alias
    for the constants and a reader should not have to work out which of the
    two a bare C meant.
    """

    B: float
    """Coulomb term of the acoustic phonon mobility [V/s]."""

    C_ac: float
    """Doping term of the acoustic phonon mobility, units to suit tau.

    DEVSIM calls this C_e and C_h. The suffix here says which of the two
    surface terms it belongs to, and keeps it from shadowing this module's
    alias for the constants."""

    tau: float
    """Doping exponent of the acoustic phonon term [1]."""

    delta: float
    """Surface roughness prefactor, units to suit gamma."""

    A: float
    """Surface roughness exponent with no carriers present [1].

    The 1988 Lombardi model fixes the exponent at 2 and stops here. This is
    where the two models agree.
    """

    alpha: float
    """How fast the roughness exponent grows with carrier density [cm^3]."""

    eta: float
    """Doping exponent damping that growth [1]."""

    kappa: float
    """Temperature exponent of the acoustic phonon term [1]."""

    T: float = C.T_ROOM
    """Lattice temperature [K]."""

    E_floor: float = 1.0e2
    """Smallest normal field the model is evaluated at [V/cm].

    Not a physical cutoff. Below it both terms are already so large that they
    contribute nothing through Matthiessen, and the only thing still changing
    is how close to dividing by zero the arithmetic gets.
    """

    @classmethod
    def electrons(cls, T: float = C.T_ROOM) -> LombardiSurface:
        """Parameters for electrons, from DEVSIM's Klaassen.py."""
        return cls(
            B=3.61e7,
            C_ac=1.70e4,
            tau=0.0233,
            delta=3.58e18,
            A=2.58,
            alpha=6.85e-21,
            eta=0.0767,
            kappa=1.7,
            T=T,
        )

    @classmethod
    def holes(cls, T: float = C.T_ROOM) -> LombardiSurface:
        """Parameters for holes, from DEVSIM's Klaassen.py.

        delta is three decades below the electron value and that is not a
        transcription slip. Holes sit further from the interface and scatter
        off its roughness far less, which is why their surface mobility
        degrades more gently than their bulk mobility would suggest.
        """
        return cls(
            B=1.51e7,
            C_ac=4.18e3,
            tau=0.0119,
            delta=4.10e15,
            A=2.18,
            alpha=7.82e-21,
            eta=0.123,
            kappa=0.9,
            T=T,
        )

    def _floored(self, E_perp: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """E_perp, refused if signed and held above the floor [V/cm]."""
        E = np.asarray(E_perp, dtype=np.float64)
        if np.any(E < 0.0):
            raise ValueError(
                "E_perp is the magnitude of the field normal to the "
                "interface and cannot be negative. A signed difference "
                "reached here without its absolute value."
            )
        return np.maximum(E, self.E_floor)

    def acoustic(
        self, E_perp: npt.NDArray[np.float64], total_doping: Doping
    ) -> npt.NDArray[np.float64]:
        """Acoustic phonon limited mobility [cm^2/(V s)].

            mu_ac = B/E_perp + C N^tau E_perp^(-1/3) / (T/300)^kappa

        Two terms with different powers of the field, so which one dominates
        moves with bias. The 1/E term rules at low field and the E^(-1/3) term
        takes over in inversion, which is where the doping dependence starts
        to matter.
        """
        E = self._floored(E_perp)
        N = np.abs(np.asarray(total_doping, dtype=np.float64))
        temperature = (self.T / C.T_ROOM) ** self.kappa
        return np.asarray(
            self.B / E + self.C_ac * N**self.tau * E ** (-1.0 / 3.0) / temperature
        )

    def gamma(
        self, total_doping: Doping, carriers: Doping
    ) -> npt.NDArray[np.float64]:
        """Surface roughness exponent [1].

            gamma = A + alpha (n + p) N^(-eta)

        The one place a carrier density enters the model, and the whole of
        what distinguishes this from the 1988 form. A heavier inversion layer
        sits closer to the interface and sees a rougher one.
        """
        N = np.abs(np.asarray(total_doping, dtype=np.float64))
        return np.asarray(
            self.A + self.alpha * np.asarray(carriers, dtype=np.float64) * N**-self.eta
        )

    def roughness(
        self,
        E_perp: npt.NDArray[np.float64],
        total_doping: Doping,
        carriers: Doping,
    ) -> npt.NDArray[np.float64]:
        """Surface roughness limited mobility [cm^2/(V s)].

            mu_sr = delta E_perp^(-gamma)

        The steeper of the two surface terms, and the one that ends up setting
        the inversion layer mobility at high gate bias.
        """
        E = self._floored(E_perp)
        return np.asarray(self.delta * E ** -self.gamma(total_doping, carriers))

    def __call__(
        self,
        mu_bulk: npt.NDArray[np.float64],
        E_perp: npt.NDArray[np.float64],
        total_doping: Doping,
        carriers: Doping,
    ) -> npt.NDArray[np.float64]:
        """Mobility [cm^2/(V s)] with surface scattering folded in.

        Args:
            mu_bulk: the doping dependent mobility this corrects [cm^2/(V s)].
            E_perp: magnitude of the field normal to the interface [V/cm].
            total_doping: Na + Nd at each node [cm^-3].
            carriers: n + p at each node [cm^-3].

        Every argument is a nodal quantity, and so is the answer. Nothing here
        knows about edges; averaging onto them happens where the geometry is
        known. See device/transport.py.
        """
        mu_ac = self.acoustic(E_perp, total_doping)
        mu_sr = self.roughness(E_perp, total_doping, carriers)
        bulk = np.asarray(mu_bulk, dtype=np.float64)
        return np.asarray(1.0 / (1.0 / bulk + 1.0 / mu_ac + 1.0 / mu_sr))


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
