"""Silicon and fundamental constants, the single source of truth.

Mirrors docs/06-constants.md. If a physical number appears anywhere else in the
codebase, that is a bug.

Convention: lengths in cm, concentrations in cm^-3, matching the semiconductor
literature.

Every temperature dependent quantity is a function of T, never a module level
float, so that room temperature is the default rather than an assumption baked
into the code.
"""

from __future__ import annotations

import math
from typing import Protocol

# ---------------------------------------------------------------- fundamental
# These are genuinely constant, so they are module level values.

q: float = 1.602176634e-19
"""Elementary charge [C]."""

k_B: float = 1.380649e-23
"""Boltzmann constant [J/K]."""

eps_0: float = 8.8541878128e-14
"""Vacuum permittivity [F/cm]. Note the cm, not m."""

h: float = 6.62607015e-34
"""Planck constant [J s]."""

m_0: float = 9.1093837015e-31
"""Free electron mass [kg]."""

T_ROOM: float = 300.0
"""Reference temperature [K]. The default argument everywhere, not a hardcoded
assumption."""


# ------------------------------------------------------------ thermal voltage


def V_T(T: float = T_ROOM) -> float:
    """Thermal voltage kT/q [V]. 0.0258520 V at 300 K."""
    return k_B * T / q


def SS_min(T: float = T_ROOM) -> float:
    """Thermodynamic subthreshold swing floor [V/decade].

    V_T * ln(10). 0.059526 V/decade at 300 K. No transistor beats this without
    a mechanism outside drift diffusion.
    """
    return V_T(T) * math.log(10.0)


# ------------------------------------------------------------------- band gap

_EG_0: float = 1.1696
"""Varshni zero temperature gap for silicon [eV]."""

_EG_ALPHA: float = 4.73e-4
"""Varshni alpha for silicon [eV/K]."""

_EG_BETA: float = 636.0
"""Varshni beta for silicon [K]."""


def Eg(T: float = T_ROOM) -> float:
    """Silicon band gap [eV] from the Varshni relation.

    Eg(T) = 1.1696 - 4.73e-4 * T^2 / (T + 636)

    Gives 1.124119 eV at 300 K. The table in docs/06-constants.md quotes 1.1242,
    which is this value rounded.
    """
    return _EG_0 - _EG_ALPHA * T * T / (T + _EG_BETA)


# ------------------------------------------------- band edge density of states


class BandDensityModel(Protocol):
    """Supplies the conduction and valence band effective densities of state.

    This exists so that Nc and Nv can later be computed from effective masses
    by the band structure layer without touching a single call site. Swapping
    the model means replacing the module level BAND_DENSITY object.
    """

    def Nc(self, T: float) -> float:
        """Conduction band effective density of states [cm^-3]."""
        ...

    def Nv(self, T: float) -> float:
        """Valence band effective density of states [cm^-3]."""
        ...


class TabulatedBandDensity:
    """Nc and Nv anchored to the measured 300 K values in docs/06-constants.md.

    Temperature dependence is the free carrier T^(3/2) scaling, which assumes
    the effective masses themselves are temperature independent.
    """

    NC_300: float = 2.86e19
    """Conduction band effective density of states at 300 K [cm^-3]."""

    NV_300: float = 3.10e19
    """Valence band effective density of states at 300 K [cm^-3]."""

    def Nc(self, T: float) -> float:
        """Conduction band effective density of states [cm^-3]."""
        return float(self.NC_300 * (T / T_ROOM) ** 1.5)

    def Nv(self, T: float) -> float:
        """Valence band effective density of states [cm^-3]."""
        return float(self.NV_300 * (T / T_ROOM) ** 1.5)


class EffectiveMassBandDensity:
    """Nc and Nv computed from effective masses.

    Nc = 2 * (2*pi * m_e* * k * T / h^2)^(3/2) * M_c
    Nv = 2 * (2*pi * m_h* * k * T / h^2)^(3/2)

    Not used yet. It is here so the seam is visible and so the formula lives
    next to the model it will replace. Wire it up when the band structure layer
    exists, by assigning it to BAND_DENSITY.
    """

    def __init__(self, m_e: float, m_h: float, M_c: int = 6) -> None:
        """Effective masses in units of m_0 [1], M_c is the valley count [1]."""
        self.m_e = m_e
        self.m_h = m_h
        self.M_c = M_c

    def _density(self, m_star: float, T: float) -> float:
        """Effective density of states for one band [cm^-3].

        The 1e-6 converts m^-3 to cm^-3, since the masses and h are SI.
        """
        m = m_star * m_0
        return float(2.0 * (2.0 * math.pi * m * k_B * T / (h * h)) ** 1.5 * 1e-6)

    def Nc(self, T: float) -> float:
        """Conduction band effective density of states [cm^-3]."""
        return self._density(self.m_e, T) * self.M_c

    def Nv(self, T: float) -> float:
        """Valence band effective density of states [cm^-3]."""
        return self._density(self.m_h, T)


BAND_DENSITY: BandDensityModel = TabulatedBandDensity()
"""The active band density model. Replace this one object to swap in computed
values from the band structure layer."""


def Nc(T: float = T_ROOM) -> float:
    """Conduction band effective density of states [cm^-3]. 2.86e19 at 300 K."""
    return BAND_DENSITY.Nc(T)


def Nv(T: float = T_ROOM) -> float:
    """Valence band effective density of states [cm^-3]. 3.10e19 at 300 K."""
    return BAND_DENSITY.Nv(T)


# --------------------------------------------------------- intrinsic density

N_I_300: float = 1.0e10
"""Silicon intrinsic carrier density at 300 K [cm^-3].

Chosen over 9.65e9 (Sproul and Green) and 1.45e10 (older literature) because it
matches the textbook worked examples used as analytic validation targets. See
docs/06-constants.md and the decision log in PROGRESS.md.

This value is anchored, not derived. sqrt(Nc * Nv) * exp(-Eg / (2 V_T)) with the
Nc, Nv and Eg above gives 1.0757e10, which is 7.6 percent higher. That deviation
is recorded in PROGRESS.md and is deliberate.
"""


def n_i(T: float = T_ROOM) -> float:
    """Silicon intrinsic carrier density [cm^-3].

    Anchored to exactly N_I_300 at 300 K, with the temperature dependence taken
    from the physics:

        n_i(T) = n_i(300) * (T/300)^(3/2) * exp(Eg(300)/(2 V_T(300))
                                                - Eg(T)/(2 V_T(T)))

    The (T/300)^(3/2) carries the Nc * Nv scaling and the exponential carries
    the gap. Both agree with sqrt(Nc Nv) exp(-Eg/2kT) up to the constant offset
    introduced by the anchoring, so n_i^2 / (Nc Nv exp(-Eg/V_T)) is exactly
    temperature independent.
    """
    gap_term = Eg(T_ROOM) / (2.0 * V_T(T_ROOM)) - Eg(T) / (2.0 * V_T(T))
    return float(N_I_300 * (T / T_ROOM) ** 1.5 * math.exp(gap_term))


# ---------------------------------------------------------------- permittivity

EPS_R_SI: float = 11.7
"""Relative permittivity of silicon [1]."""

EPS_R_OX: float = 3.9
"""Relative permittivity of silicon dioxide [1]."""


def eps_Si() -> float:
    """Permittivity of silicon [F/cm]."""
    return EPS_R_SI * eps_0


def eps_ox() -> float:
    """Permittivity of silicon dioxide [F/cm]."""
    return EPS_R_OX * eps_0


# ------------------------------------------------------------------ transport
# Constant stubs only. The real mobility models are Phase 2 and live in
# physics/mobility.py. Do not add doping or field dependence here.

MU_N_300: float = 1417.0
"""Undoped silicon electron mobility at 300 K [cm^2/(V s)]."""

MU_P_300: float = 470.0
"""Undoped silicon hole mobility at 300 K [cm^2/(V s)]."""

V_SAT_N_300: float = 1.07e7
"""Electron saturation velocity at 300 K [cm/s]."""

V_SAT_P_300: float = 8.3e6
"""Hole saturation velocity at 300 K [cm/s]."""


def mu_n(T: float = T_ROOM) -> float:
    """Electron mobility [cm^2/(V s)], constant stub.

    Temperature independent in Phase 0. Arora and Masetti arrive in Phase 2.
    """
    return MU_N_300


def mu_p(T: float = T_ROOM) -> float:
    """Hole mobility [cm^2/(V s)], constant stub."""
    return MU_P_300


def v_sat_n(T: float = T_ROOM) -> float:
    """Electron saturation velocity [cm/s], constant stub."""
    return V_SAT_N_300


def v_sat_p(T: float = T_ROOM) -> float:
    """Hole saturation velocity [cm/s], constant stub."""
    return V_SAT_P_300


def D_n(T: float = T_ROOM) -> float:
    """Electron diffusivity [cm^2/s] from the Einstein relation D = V_T * mu."""
    return V_T(T) * mu_n(T)


def D_p(T: float = T_ROOM) -> float:
    """Hole diffusivity [cm^2/s] from the Einstein relation D = V_T * mu."""
    return V_T(T) * mu_p(T)


# ------------------------------------------------------------ SRH lifetimes
# Numbers only. The Scharfetter model that consumes them lives in
# physics/recombination.py, because a model is not a constant.

TAU_N_MAX: float = 1e-5
"""Electron lifetime in undoped silicon [s], from docs/06-constants.md."""

TAU_P_MAX: float = 3e-6
"""Hole lifetime in undoped silicon [s]."""

TAU_N_MIN: float = 0.0
"""Electron lifetime floor at very high doping [s]."""

TAU_P_MIN: float = 0.0
"""Hole lifetime floor at very high doping [s]."""

N_REF_SRH: float = 5e16
"""Doping at which the Scharfetter lifetime is halfway to its floor [cm^-3]."""

GAMMA_SRH: float = 1.0
"""Sharpness of the Scharfetter lifetime transition [1]."""
