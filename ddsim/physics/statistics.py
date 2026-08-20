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
