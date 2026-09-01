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

import numpy as np
import numpy.typing as npt

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
