"""Independent references for the Fermi-Dirac integrals and their inverse.

`ddsim.physics.statistics` evaluates F_{1/2} and F_{-1/2} by fixed order
Gauss-Legendre quadrature on a squared abscissa, and inverts F_{1/2} by the
Joyce-Dixon series. Neither of those may check itself. This module supplies
three references that share no arithmetic with either.

1. The alternating series. For eta < 0,

       F_s(eta) = Gamma(s+1) * sum_{k>=1} (-1)^(k+1) exp(k*eta) / k^(s+1)

   which is the polylogarithm expanded about zero. Every term is exact, the
   series is alternating with terms falling geometrically, so 400 terms are
   machine precision for any eta below about -0.05. It says nothing at all
   about eta > 0, which is where the degenerate material lives, so it is
   necessary and not sufficient.

2. Adaptive quadrature, `scipy.integrate.quad`, over the untransformed
   integrand. A different rule, a different abscissa and a different error
   control from the shipped one, so agreement between them is evidence rather
   than a tautology. Its absolute error control makes it useless below about
   1e-12, which is exactly the range the alternating series owns.

3. Brent inversion of reference 2, which gives eta(u) with no series in it at
   all and is what Joyce-Dixon has to be graded against.

Everything here is in units of kT, so eta = (E_F - E_c)/kT for electrons and
(E_v - E_F)/kT for holes, and u = n/Nc is dimensionless [1].
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

SERIES_TERMS = 400
"""Terms in the alternating series [1]. At eta = -0.05 the 400th term is
exp(-20)/400^1.5 = 2.6e-13 of the leading one, and it alternates, so the
truncation error is below machine precision on the whole range this is used."""

SERIES_MAX_ETA = -0.05
"""Largest eta the alternating series is trusted at [1]."""

TAIL = 60.0
"""How far past the Fermi level the adaptive quadrature integrates [1]. The
integrand is below exp(-60) = 8.8e-27 of its peak beyond that."""


def F_series(eta: float, order: float = 0.5) -> float:
    """F_order(eta) [1] by the alternating series. Only for eta < 0."""
    if eta > SERIES_MAX_ETA:
        raise ValueError(
            f"the alternating series converges for eta < 0 and is only "
            f"trusted below {SERIES_MAX_ETA}, got {eta}"
        )
    k = np.arange(1.0, SERIES_TERMS + 1.0)
    terms = ((-1.0) ** (k + 1.0)) * np.exp(k * eta) / k ** (order + 1.0)
    return float(math.gamma(order + 1.0) * np.sum(terms))


def F_quad(eta: float, order: float = 0.5) -> float:
    """F_order(eta) = int_0^inf x^order / (1 + exp(x - eta)) dx [1].

    Adaptive, with the Fermi level handed to the integrator as a break point
    so the rule does not have to discover the shoulder for itself.
    """
    shoulder = max(eta, 0.0)

    def integrand(x: float) -> float:
        return x**order / (1.0 + math.exp(x - eta))

    value, _ = quad(
        integrand, 0.0, shoulder + TAIL, limit=400, points=[shoulder]
    )
    return float(value)


def u_reference(eta: float) -> float:
    """u = n/Nc [1] at a given eta, from reference quadrature.

    n = Nc * F_{1/2}(eta) / Gamma(3/2), the normalisation that makes u -> e^eta
    in the nondegenerate limit.
    """
    return F_quad(eta, 0.5) / math.gamma(1.5)


def eta_reference(u: float) -> float:
    """eta [1] at a given u = n/Nc, by Brent inversion of `u_reference`.

    No series anywhere in it. This is what Joyce-Dixon is graded against.
    """
    return float(
        brentq(lambda e: u_reference(e) - u, -80.0, 80.0, xtol=1e-14, rtol=8.9e-16)
    )


def einstein_reference(u: float) -> float:
    """D / (mu * V_T) [1] at a given u = n/Nc, from the integral definition.

    D/mu = (1/q) * n * (dE_F/dn), and with n = Nc*F_{1/2}(eta)/Gamma(3/2) and
    dF_{1/2}/deta = F_{-1/2}/2 that is

        D / (mu * V_T) = 2 * F_{1/2}(eta) / F_{-1/2}(eta)

    which is 1 in the nondegenerate limit, where F_{1/2}/F_{-1/2} -> 1/2.
    """
    eta = eta_reference(u)
    return 2.0 * F_quad(eta, 0.5) / F_quad(eta, -0.5)
