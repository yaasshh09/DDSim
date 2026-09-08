"""Carrier statistics. Boltzmann through Phase 4, Fermi-Dirac deferred.

Pure functions over arrays. No mesh, no solver state, no device knowledge, so
every function here is directly testable against a textbook formula.

Boltzmann statistics, from docs/01-physics.md:

    n = n_i * exp((psi - phi_n) / V_T)
    p = n_i * exp((phi_p - psi) / V_T)

In de Mari scaled units with C_0 = n_i, psi is measured in units of V_T and
densities in units of n_i, so both collapse to

    n = exp(psi - phi_n)
    p = exp(phi_p - psi)

with no constants at all. That is the form the solver uses. The physical forms
are kept alongside for testing against textbook numbers and for reporting.

Sign convention, fixed by docs/01-physics.md: psi increases toward n-type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C

Scalar = float | npt.NDArray[np.float64]


# ------------------------------------------------------------ Boltzmann, scaled


def n_boltzmann_scaled(psi: Scalar, phi_n: Scalar = 0.0) -> Scalar:
    """Electron density [1], in units of C_0 = n_i.

    n = exp(psi - phi_n). Equals 1 in intrinsic material at equilibrium.
    """
    return np.asarray(np.exp(np.asarray(psi) - np.asarray(phi_n)))


def p_boltzmann_scaled(psi: Scalar, phi_p: Scalar = 0.0) -> Scalar:
    """Hole density [1], in units of C_0 = n_i.

    p = exp(phi_p - psi). Equals 1 in intrinsic material at equilibrium.
    """
    return np.asarray(np.exp(np.asarray(phi_p) - np.asarray(psi)))


def dn_dpsi_scaled(psi: Scalar, phi_n: Scalar = 0.0) -> Scalar:
    """dn/dpsi [1]. Equal to n itself, which is why Poisson stays well behaved."""
    return n_boltzmann_scaled(psi, phi_n)


def dp_dpsi_scaled(psi: Scalar, phi_p: Scalar = 0.0) -> Scalar:
    """dp/dpsi [1]. Equal to -p."""
    return -p_boltzmann_scaled(psi, phi_p)


# ---------------------------------------------------------- Boltzmann, physical


def n_boltzmann(psi: Scalar, phi_n: Scalar, n_i: float, V_T: float) -> Scalar:
    """Electron density [cm^-3] from potentials in volts.

    Args:
        psi: electrostatic potential [V].
        phi_n: electron quasi-Fermi potential [V].
        n_i: intrinsic density [cm^-3].
        V_T: thermal voltage [V].
    """
    return np.asarray(n_i * np.exp((np.asarray(psi) - np.asarray(phi_n)) / V_T))


def p_boltzmann(psi: Scalar, phi_p: Scalar, n_i: float, V_T: float) -> Scalar:
    """Hole density [cm^-3] from potentials in volts."""
    return np.asarray(n_i * np.exp((np.asarray(phi_p) - np.asarray(psi)) / V_T))


# ------------------------------------------------------- equilibrium from doping


def psi_equilibrium_scaled(net_doping: Scalar) -> Scalar:
    """Equilibrium potential [1] for a given net doping [1].

    Solves charge neutrality together with mass action:

        p - n + N = 0        and       n * p = 1

    Substituting n = exp(psi) and p = exp(-psi) gives 2*sinh(psi) = N, so

        psi = asinh(N / 2)

    Use asinh, never V_T*ln(N/n_i). The log form is -inf at N = 0 and NaN for
    net acceptor doping, and it is wrong wherever compensation brings N near
    zero. docs/01-physics.md calls this out as a frequent bug source.

    Also the initial guess for the nonlinear Poisson solve.
    """
    return np.asarray(np.arcsinh(np.asarray(net_doping) / 2.0))


def equilibrium_densities_scaled(
    net_doping: Scalar,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Equilibrium (n, p) [1] for a given net doping [1].

    Same two conditions as psi_equilibrium_scaled, solved directly:

        n - p = N        and       n * p = 1

    The quadratic formula gives n = (N + sqrt(N^2 + 4))/2, and it is safe for
    the majority carrier only. Taking the minority carrier from the same
    formula subtracts two nearly equal numbers: at N = 1e6 it computes
    (sqrt(1e12 + 4) - 1e6)/2 to get 1e-6, losing twelve digits. So the majority
    carrier comes from the quadratic formula and the minority carrier comes
    from n*p = 1, which makes mass action exact rather than merely close.
    """
    original = np.asarray(net_doping, dtype=np.float64)
    N = np.atleast_1d(original)
    root = np.sqrt(N * N + 4.0)

    n = np.empty_like(N)
    p = np.empty_like(N)

    donors = N >= 0.0
    acceptors = ~donors

    # Majority carrier first, from the quadratic formula. Adding two positive
    # numbers, so no cancellation.
    n[donors] = 0.5 * (N[donors] + root[donors])
    p[acceptors] = 0.5 * (root[acceptors] - N[acceptors])

    # Minority carrier from mass action. Exact rather than merely close.
    p[donors] = 1.0 / n[donors]
    n[acceptors] = 1.0 / p[acceptors]

    return n.reshape(original.shape), p.reshape(original.shape)


# ------------------------------------------------------------------ Fermi-Dirac
#
# Everything below is written in terms of u = n/Nc, dimensionless, with Nc the
# conduction band effective density of states. The hole side is the same
# functions with p and Nv, which is why nothing here says "electron".
#
# The normalisation is fixed by wanting u -> exp(eta) in the nondegenerate
# limit, so
#
#     u = n / Nc = F_{1/2}(eta) / Gamma(3/2),      eta = (E_F - E_c) / kT
#
# and F_{1/2} is the plain integral that the tables carry. docs/01-physics.md
# writes the same relation with an explicit 2/sqrt(pi), which is 1/Gamma(3/2).
#
# Why this file does not simply replace n = n_i exp(psi - phi_n) everywhere
# -------------------------------------------------------------------------
# It could not. Writing n = Nc * F_{1/2} directly makes the intrinsic density
# come out as sqrt(Nc Nv) exp(-Eg/2V_T) = 1.0757e10, and this project anchors
# n_i at 1.0e10 instead. Row 97 of docs/07-decisions.md records that the two
# are 7.6 percent apart and cannot both be primary. So Fermi-Dirac enters as a
# correction to the Boltzmann form rather than as a replacement for it:
#
#     n = n_i * exp(psi - phi_n) * gamma_n
#
# with gamma_n = degeneracy_factor(u), which is exactly 1 at u = 0. Nothing
# nondegenerate moves, the n_i inconsistency stays where it was, and the
# degenerate correction is the only new physics. See the decisions log.


EXP_LIMIT = 700.0
"""Largest exponent evaluated inside the Fermi occupancy [1].

1/(1 + exp(z)) is below 1e-304 at z = 700 and is 1 to the last bit at
z = -700, so clipping there changes no representable result and is what stops
exp from overflowing on an eta hundreds of kT below the band edge.
"""

QUADRATURE_ORDER = 96
"""Gauss-Legendre nodes per panel [1].

Two panels, so 192 evaluations. Measured against the alternating series and
against adaptive quadrature, this returns F_{1/2} to 5e-15 relative over
eta in [-40, 40]. 48 nodes reaches 3e-11, which is enough for the 1 percent
criterion but not enough to grade Joyce-Dixon by.
"""

QUADRATURE_TAIL = 60.0
"""How far past the Fermi level the quadrature integrates [1].

The occupancy is exp(-60) = 8.8e-27 of its plateau value beyond this, and the
x^(1/2) in front cannot rescue that.
"""

JOYCE_DIXON_COEFFICIENTS = (
    1.0 / np.sqrt(8.0),
    3.0 / 16.0 - np.sqrt(3.0) / 9.0,
    1.48386e-4,
    -4.42563e-6,
)
"""A1 to A4 of the Joyce-Dixon series, Joyce and Dixon 1977 [1].

    eta = ln(u) + A1 u + A2 u^2 + A3 u^3 + A4 u^4

A1 = 1/sqrt(8) = 3.53553e-1 and A2 = 3/16 - sqrt(3)/9 = -4.95009e-3 have
closed forms. A3 and A4 are the published decimals and have no short one.

**A2 is negative.** docs/01-physics.md wrote this term with a minus sign in
front of the same bracket, which flips it. Checked against Brent inversion of
the integral: with A2 as written here the series lands 1.0e-4 from the true
eta at u = 4, and with the sign flipped it lands 1.6e-1 away, three decades
worse. See docs/07-decisions.md.
"""

JOYCE_DIXON_MAX_U = 8.0
"""Largest n/Nc the series is allowed at [1].

The series is a fit and it eventually turns over. Measured against Brent
inversion of the integral, the relative error in n and in the Einstein ratio:

    u      error in n     error in D/(mu V_T)
    4      4.4e-5         2.2e-4
    8      9.0e-4         4.4e-3
    10     2.3e-3         1.1e-2
    20     4.1e-2         1.9e-1
    35     3.5e-1         ratio has gone negative

Eight is where both are still inside the 1 percent docs/04-validation.md asks
for. It is 2.29e20 cm^-3, which clears the 1e20 source and drain of
phases/PHASE-5.md with room. Past twenty the Einstein ratio turns over and
goes negative, which is a diffusivity pointing the wrong way, so this refuses
rather than extrapolates.
"""


def _fermi_dirac_integral(eta: Scalar, order: float) -> npt.NDArray[np.float64]:
    """F_order(eta) = int_0^inf x^order / (1 + exp(x - eta)) dx [1].

    Fixed order Gauss-Legendre on the substitution x = t^2, which turns
    dx = 2 t dt and removes the square root branch point at the origin that
    would otherwise cost the rule most of its order:

        F_order(eta) = int_0^inf 2 t^(2 order + 1) / (1 + exp(t^2 - eta)) dt

    Two panels, split at t = sqrt(eta), because that is where the occupancy
    falls from one to zero and a single panel would have to resolve a step in
    the middle of its interval. When eta <= 0 the first panel is empty, its
    half width is exactly zero, and it contributes exactly zero.

    Deliberately not adaptive. The result has to be the same bits every time
    it is called at the same eta, because a Jacobian entry derived from it is
    compared against complex step at 1e-10.
    """
    eta_array = np.atleast_1d(np.asarray(eta, dtype=np.float64)).ravel()
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)

    fermi_level = np.sqrt(np.maximum(eta_array, 0.0))
    edges = (
        (np.zeros_like(eta_array), fermi_level),
        (fermi_level, np.sqrt(np.maximum(eta_array, 0.0) + QUADRATURE_TAIL)),
    )

    total = np.zeros_like(eta_array)
    for low, high in edges:
        half_width = 0.5 * (high - low)
        centre = 0.5 * (high + low)
        t = centre[:, None] + half_width[:, None] * nodes[None, :]
        exponent = np.clip(t * t - eta_array[:, None], -EXP_LIMIT, EXP_LIMIT)
        integrand = 2.0 * t ** (2.0 * order + 1.0) / (1.0 + np.exp(exponent))
        total += half_width * (integrand @ weights)

    return np.asarray(total.reshape(np.shape(eta)))


def fermi_dirac_half(eta: Scalar) -> npt.NDArray[np.float64]:
    """F_{1/2}(eta) [1], the integral the tables carry.

    Args:
        eta: (E_F - E_c)/kT for electrons, (E_v - E_F)/kT for holes [1].

    F_{1/2}(0) = 0.678094, and the density follows as n = Nc F_{1/2}/Gamma(3/2).
    """
    return _fermi_dirac_integral(eta, 0.5)


def fermi_dirac_minus_half(eta: Scalar) -> npt.NDArray[np.float64]:
    """F_{-1/2}(eta) [1].

    Args:
        eta: (E_F - E_c)/kT for electrons, (E_v - E_F)/kT for holes [1].

    Equal to twice the slope of F_{1/2}, since dF_s/deta = s F_(s-1). That
    identity is what turns the generalized Einstein relation into a ratio of
    two integrals rather than a numerical derivative.
    """
    return _fermi_dirac_integral(eta, -0.5)


def _checked_u(u: Scalar, strictly_positive: bool) -> npt.NDArray[np.float64]:
    """n/Nc as a float array, with the range checks done once [1]."""
    ratio = np.asarray(u, dtype=np.float64)

    if strictly_positive and np.any(ratio <= 0.0):
        raise ValueError(
            "n/Nc must be positive to take its logarithm, and the smallest "
            f"value given is {float(np.min(ratio)):g}"
        )
    if np.any(ratio < 0.0):
        raise ValueError(
            f"n/Nc cannot be negative, and the smallest value given is "
            f"{float(np.min(ratio)):g}"
        )
    if np.any(ratio > JOYCE_DIXON_MAX_U):
        raise ValueError(
            f"the Joyce-Dixon series is validated to n/Nc = {JOYCE_DIXON_MAX_U:g} "
            f"and the largest value given is {float(np.max(ratio)):g}. Past "
            "that it turns over and the Einstein ratio changes sign. Use a "
            "rational approximation instead if the material is really that "
            "degenerate."
        )
    return ratio


def _joyce_dixon_correction(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The part of eta that is not ln(u) [1]. Zero at u = 0, exactly."""
    correction = np.zeros_like(u)
    for power, coefficient in enumerate(JOYCE_DIXON_COEFFICIENTS, start=1):
        correction = correction + coefficient * u**power
    return correction


def joyce_dixon_eta(u: Scalar) -> npt.NDArray[np.float64]:
    """Reduced Fermi level [1] from a carrier density.

    Args:
        u: n/Nc for electrons, p/Nv for holes [1]. Positive, at most
            JOYCE_DIXON_MAX_U.

    The inverse of u = F_{1/2}(eta)/Gamma(3/2), which has no closed form, as
    the Joyce-Dixon series

        eta = ln(u) + A1 u + A2 u^2 + A3 u^3 + A4 u^4

    The correction is positive: filling the band means a given density needs a
    higher Fermi level than Boltzmann predicts. It is 0.3 mV at 1e18, 3.2 mV
    at 1e19 and 30.5 mV at 1e20, in silicon at 300 K.
    """
    ratio = _checked_u(u, strictly_positive=True)
    return np.asarray(np.log(ratio) + _joyce_dixon_correction(ratio))


def degeneracy_factor(u: Scalar) -> npt.NDArray[np.float64]:
    """gamma = n / (Nc exp(eta)) [1], Fermi-Dirac over Boltzmann at fixed eta.

    Args:
        u: n/Nc for electrons, p/Nv for holes [1]. Non-negative, at most
            JOYCE_DIXON_MAX_U.

    This is the number that lets Fermi-Dirac enter as a correction rather than
    as a rewrite. With eta from Joyce-Dixon,

        gamma = u / exp(eta) = exp(-(A1 u + A2 u^2 + A3 u^3 + A4 u^4))

    so the logarithm cancels analytically and gamma comes out as exp of a
    polynomial with no constant term. At u = 0 that polynomial is exactly zero
    and gamma is exactly 1.0, not 1.0 to rounding, which is what makes every
    nondegenerate result in this project survive unchanged.

    gamma < 1 always, and 1/gamma is the factor by which Boltzmann
    overestimates the density at a given Fermi level: 1.04 at 1e19 and 3.26 at
    1e20.
    """
    return np.asarray(np.exp(-_joyce_dixon_correction(_checked_u(u, False))))


def einstein_ratio(u: Scalar) -> npt.NDArray[np.float64]:
    """D / (mu V_T) [1], the generalized Einstein relation.

    Args:
        u: n/Nc for electrons, p/Nv for holes [1]. Non-negative, at most
            JOYCE_DIXON_MAX_U.

    docs/01-physics.md gives it as F_{1/2}(eta)/F_{-1/2}(eta) in the
    Gamma-normalised convention, which is 2 F_{1/2}/F_{-1/2} in the plain one.
    Evaluating that directly would need two quadratures per node per assembly
    and a numerical eta to put them at.

    It has a closed form instead. D/mu = (1/q) n dE_F/dn, so

        D / (mu V_T) = u * deta/du = 1 + A1 u + 2 A2 u^2 + 3 A3 u^3 + 4 A4 u^4

    which is the term by term derivative of the same series joyce_dixon_eta
    uses. That matters beyond speed: the ratio and the inversion are then one
    approximation rather than two, they share an eta exactly, and the tangent
    of one is the other. Two separately fitted forms would not.

    Exactly 1.0 at u = 0, so a device with no degenerate material anywhere
    pays nothing for switching statistics on. 1.34 at n = Nc, and 2.13 at
    1e20, where degeneracy has doubled the diffusivity.
    """
    ratio = _checked_u(u, strictly_positive=False)
    total = np.ones_like(ratio)
    for power, coefficient in enumerate(JOYCE_DIXON_COEFFICIENTS, start=1):
        total = total + power * coefficient * ratio**power
    return np.asarray(total)


# ------------------------------------------------- Fermi-Dirac, wired for use
#
# Everything above is arithmetic on u = n/Nc. What follows is the object a
# solver holds: the same series, in the scaled units the assemblies work in,
# with the four questions transport actually asks of it.
#
# The four questions
# ------------------
# 1. What potential makes this density exactly Boltzmann? That is the
#    effective potential of docs/07-decisions.md, psi_eff = psi + ln(gamma),
#    and it is the only thing the Scharfetter-Gummel argument changes.
# 2. How does that potential move when the density does? The Bernoulli
#    argument now depends on n as well as psi, so dF_n/dn grows the same
#    stencil dF_n/dpsi already has.
# 3. What density belongs to this potential? The inverse of the series, which
#    the equilibrium Poisson solve needs, because there n and p are not
#    unknowns but functions of psi.
# 4. What does an ohmic contact hold? Neutrality and mass action again, with
#    the mass action product no longer 1.
#
# The sign, once, so it is not re-derived at four call sites. gamma < 1, so
# ln(gamma) < 0, so the effective potential is below psi where electrons are
# degenerate and above psi where holes are. Both statements say the same
# thing: a filled band pushes its own carriers out, which is the enhanced
# diffusion the generalized Einstein ratio describes.


INVERSION_STEPS = 6
"""Newton steps taken to invert the Joyce-Dixon series [1].

The iteration is Newton on w = ln(u) against a function whose derivative is
the Einstein ratio and therefore never below 1, started from the Boltzmann
answer, which lies on the convex side of the root. That makes it monotone and
quadratic with no safeguarding. Six steps reach the roundoff floor from the
worst start in range, u = 8, where the correction it has to undo is 2.5.

Fixed rather than tolerance driven, for the same reason the quadrature is
fixed order: the result has to be the same bits every time it is called at the
same argument, because a Jacobian entry derived from it is compared against
complex step at 1e-10.

The same count serves the contact solve below, which is a substitution rather
than a Newton and converges faster still.
"""

LOG_MAX_U = float(np.log(JOYCE_DIXON_MAX_U))
"""ln of the largest n/Nc the series is allowed at [1]."""


def _cap(values: Scalar, ceiling: float) -> npt.NDArray[np.float64]:
    """min(values, ceiling), branching on the real part [1].

    np.minimum orders complex numbers lexicographically and np.clip refuses
    them outright, and either would break the complex step verification of
    everything downstream. Comparing the real part is the same function on
    real input and the analytic continuation of it on a perturbed one.
    """
    array = np.asarray(values)
    return np.asarray(np.where(np.real(array) > ceiling, ceiling, array))


def _joyce_dixon_slope(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """d(correction)/du [1], the term by term derivative of the series.

    Related to the Einstein ratio by ratio = 1 + u * slope, which is how the
    same four coefficients end up describing the density, the diffusivity and
    the tangent of the inversion.
    """
    total = np.zeros_like(u)
    for power, coefficient in enumerate(JOYCE_DIXON_COEFFICIENTS, start=1):
        total = total + power * coefficient * u ** (power - 1)
    return total


@dataclass(frozen=True)
class Degeneracy:
    """Fermi-Dirac corrections for one material, in scaled units.

    Args:
        Nc: conduction band effective density of states over C_0 [1].
        Nv: valence band effective density of states over C_0 [1].

    Frozen and hashable, because ohmic contact values are cached on it.

    A device holding None instead of one of these is a Boltzmann device, and
    every path below is written so that the two agree bit for bit at zero
    density rather than merely to rounding. See the module comment.
    """

    Nc: float
    Nv: float

    def __post_init__(self) -> None:
        for name, states in (("Nc", self.Nc), ("Nv", self.Nv)):
            if states <= 0.0:
                raise ValueError(f"{name} must be positive, got {states}")

    @classmethod
    def for_silicon(cls, C_0: float, T: float = C.T_ROOM) -> Degeneracy:
        """Silicon at temperature T [K], with both densities scaled by C_0."""
        return cls(Nc=C.Nc(T) / C_0, Nv=C.Nv(T) / C_0)

    # ------------------------------------------------ the effective potential

    def _u(self, density: Scalar, states: float) -> npt.NDArray[np.float64]:
        """n/Nc, capped at the last density the series is validated to [1].

        Capped rather than refused, unlike the bare functions above. Those are
        asked about a state someone chose; this is asked about whatever a
        Newton step landed on, and a transient overshoot next to a 1e20
        contact would otherwise abort a whole continuation sweep. Above the
        cap the correction is held constant, which keeps the density monotone
        in the potential and the inversion single valued. Nothing this project
        reports sits above the cap, and a test pins that.
        """
        return _cap(np.asarray(density) / states, JOYCE_DIXON_MAX_U)

    @staticmethod
    def _slope(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """d(correction)/du at an already capped u [1], zero above the cap.

        The cap holds the correction constant, so its derivative there is zero
        and not the slope the series would have had. Getting that wrong is not
        cosmetic: the inverse below is a Newton whose tangent is exactly this,
        and a tangent three times too steep turns a step that should land on
        the answer into linear convergence that never arrives. Measured before
        this was a single helper: the round trip at n = 2e20 came back 7.3e-3
        out after six steps instead of at the roundoff floor.
        """
        capped = np.real(u) >= JOYCE_DIXON_MAX_U
        return np.asarray(np.where(capped, 0.0, _joyce_dixon_slope(u)))

    def electron_potential(self, psi: Scalar, n: Scalar) -> npt.NDArray[np.float64]:
        """psi + ln(gamma_n) [1], the potential electrons are Boltzmann in.

        n = exp(psi_eff - phi_n) exactly, so the Scharfetter-Gummel
        exponential fit stays exactly valid in psi_eff and the flux keeps the
        ordinary form with D = mu V_T. Below psi, because gamma_n < 1.
        """
        return np.asarray(psi) - _joyce_dixon_correction(self._u(n, self.Nc))

    def hole_potential(self, psi: Scalar, p: Scalar) -> npt.NDArray[np.float64]:
        """psi - ln(gamma_p) [1], the potential holes are Boltzmann in.

        p = exp(phi_p - psi_eff), so the mirror of the electron case puts the
        correction on with the opposite sign. Above psi, for the same reason
        the electron one is below it.
        """
        return np.asarray(psi) + _joyce_dixon_correction(self._u(p, self.Nv))

    def d_electron_potential_dn(self, n: Scalar) -> npt.NDArray[np.float64]:
        """d(psi_eff_n)/dn [1]. Negative, and zero where the cap is active."""
        return np.asarray(-self._slope(self._u(n, self.Nc)) / self.Nc)

    def d_hole_potential_dp(self, p: Scalar) -> npt.NDArray[np.float64]:
        """d(psi_eff_p)/dp [1]. Positive, and zero where the cap is active."""
        return np.asarray(self._slope(self._u(p, self.Nv)) / self.Nv)

    # ----------------------------------------------------------- the inverse

    def _density(self, exponent: Scalar, states: float) -> npt.NDArray[np.float64]:
        """The density solving x = exp(exponent) * gamma(x/states) [1].

        Substituting w = ln(x/states) turns that into

            w + correction(exp(w)) = exponent - ln(states)

        which is joyce_dixon_eta(u) = eta, so this is the inverse of the same
        series the forward direction uses rather than a second approximation
        of the same physics. Newton on w, whose derivative is exactly the
        Einstein ratio and so is never below 1.

        An exponent of -inf marks a node with no carriers in it and comes back
        as exactly zero, rather than as a nan out of inf minus inf.
        """
        target = np.asarray(exponent) - float(np.log(states))
        alive = np.isfinite(target)
        target = np.where(alive, target, -EXP_LIMIT)

        w = target
        for _ in range(INVERSION_STEPS):
            u = np.where(
                np.real(w) > LOG_MAX_U,
                JOYCE_DIXON_MAX_U,
                np.exp(_cap(w, LOG_MAX_U)),
            )
            w = w - (w + _joyce_dixon_correction(u) - target) / (
                1.0 + u * self._slope(u)
            )
        return np.asarray(np.where(alive, states * np.exp(w), 0.0))

    def electron_density(self, exponent: Scalar) -> npt.NDArray[np.float64]:
        """n [1] from the Boltzmann exponent psi - phi_n."""
        return self._density(exponent, self.Nc)

    def hole_density(self, exponent: Scalar) -> npt.NDArray[np.float64]:
        """p [1] from the Boltzmann exponent phi_p - psi."""
        return self._density(exponent, self.Nv)

    def dn_dpsi(self, n: Scalar) -> npt.NDArray[np.float64]:
        """dn/dpsi at fixed phi_n [1], which is n over the Einstein ratio.

        Boltzmann returns n itself and this returns less, because filling the
        band means a given rise in the Fermi level buys less density. It is
        the same ratio einstein_ratio reports, and it appears here because the
        Poisson diagonal is exactly this derivative.
        """
        u = self._u(n, self.Nc)
        return np.asarray(np.asarray(n) / (1.0 + u * self._slope(u)))

    def dp_dpsi(self, p: Scalar) -> npt.NDArray[np.float64]:
        """-dp/dpsi at fixed phi_p [1], the hole mirror. Returned positive."""
        u = self._u(p, self.Nv)
        return np.asarray(np.asarray(p) / (1.0 + u * self._slope(u)))

    # ---------------------------------------------------------- the contacts

    def equilibrium_densities(
        self, net_doping: Scalar
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Equilibrium (n, p) [1] at a given net doping, degenerate.

        Neutrality is unchanged, n - p = N. Mass action is not: with each
        density carrying its own gamma,

            n * p = gamma_n(n) * gamma_p(p)

        so the product is no longer 1 and depends on the answer. Solved by
        substitution, starting from the Boltzmann answer and re-solving the
        same quadratic against the product it implies. The product is set
        almost entirely by the majority carrier, so this converges in a
        handful of passes, and the majority carrier still comes from the
        quadratic formula while the minority still comes from the product,
        which is what keeps mass action exact rather than merely close.
        """
        original = np.asarray(net_doping, dtype=np.float64)
        N = np.atleast_1d(original)
        n, p = equilibrium_densities_scaled(N)

        donors = N >= 0.0
        for _ in range(INVERSION_STEPS):
            product = degeneracy_factor(self._u(n, self.Nc)) * degeneracy_factor(
                self._u(p, self.Nv)
            )
            root = np.sqrt(N * N + 4.0 * product)
            majority = np.where(donors, 0.5 * (N + root), 0.5 * (root - N))
            minority = product / majority
            n = np.where(donors, majority, minority)
            p = np.where(donors, minority, majority)

        return n.reshape(original.shape), p.reshape(original.shape)

    def equilibrium_psi(self, net_doping: Scalar) -> npt.NDArray[np.float64]:
        """Equilibrium potential [1] at a given net doping, degenerate.

        n = exp(psi) gamma_n inverts to psi = ln(n) + correction(n/Nc), and
        the hole relation gives psi = -ln(p) - correction(p/Nv). The two agree
        exactly whenever the pair above satisfies its own mass action, which
        is the same consistency the Boltzmann contact has and matters for the
        same reason: the potential and the two densities pinned at one contact
        node have to be one state, not three conditions that nearly agree.

        Read off the majority carrier, whose density carries no cancellation.
        """
        original = np.asarray(net_doping, dtype=np.float64)
        N = np.atleast_1d(original)
        n, p = self.equilibrium_densities(N)

        psi = np.where(
            N >= 0.0,
            np.log(n) + _joyce_dixon_correction(self._u(n, self.Nc)),
            -np.log(p) - _joyce_dixon_correction(self._u(p, self.Nv)),
        )
        return np.asarray(psi.reshape(original.shape))
