"""The benchmark definitions and the model parameters both codes must share.

This module is deliberately free of any devsim import, and free of any ddsim
import. The generator runs it under a separate interpreter that has devsim but
not ddsim; the regression test runs it under the project interpreter, which has
ddsim but not devsim. It is the one file both sides read, so the device
geometry, the doping and the physical constants cannot drift apart between
them.

The constants below are literal copies of `ddsim.core.constants`. That
duplication is not an accident and it is not allowed to rot:
`tests/regression/test_devsim_diodes.py` asserts every one of them against the
ddsim value it mirrors, so a change on the ddsim side fails the suite until the
golden data is regenerated.

Units follow the project convention: lengths in cm, concentrations in cm^-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------ constants
# Mirrors of ddsim.core.constants, pinned by test_constants_mirror_ddsim.

Q = 1.602176634e-19
"""Elementary charge [C]."""

K_B = 1.380649e-23
"""Boltzmann constant [J/K]."""

EPS_0 = 8.8541878128e-14
"""Vacuum permittivity [F/cm]."""

T = 300.0
"""Temperature [K]."""

EPS_R_SI = 11.7
"""Relative permittivity of silicon [1]."""

N_I = 1.0e10
"""Intrinsic carrier density at 300 K [cm^-3]."""

MU_N = 1417.0
"""Electron mobility [cm^2/(V s)], the constant model."""

MU_P = 470.0
"""Hole mobility [cm^2/(V s)], the constant model."""

TAU_N_MAX = 1e-5
"""Electron lifetime in undoped silicon [s]."""

TAU_P_MAX = 3e-6
"""Hole lifetime in undoped silicon [s]."""

TAU_N_MIN = 0.0
"""Electron lifetime floor at very high doping [s]."""

TAU_P_MIN = 0.0
"""Hole lifetime floor at very high doping [s]."""

N_REF_SRH = 5e16
"""Doping at which the Scharfetter lifetime is halfway to its floor [cm^-3]."""

GAMMA_SRH = 1.0
"""Sharpness of the Scharfetter lifetime transition [1]."""

V_T = K_B * T / Q
"""Thermal voltage [V]. 0.02585199 V at 300 K."""


# --------------------------------------------------------------------- devices


@dataclass(frozen=True)
class DiodeBenchmark:
    """One PN diode from the tier 4 benchmark set of docs/04-validation.md."""

    name: str
    """Short name, also the golden file stem."""

    number: int
    """Row in the docs/04-validation.md benchmark table."""

    Na: float
    """Acceptor concentration on the p side [cm^-3], positive."""

    Nd: float
    """Donor concentration on the n side [cm^-3], positive."""

    length: float
    """Device length [cm]."""

    junction: float
    """Junction position [cm]. Doping is right continuous here, so a node
    sitting exactly on it is n-type. Both codes are told the same thing."""

    voltages: tuple[float, ...]
    """Anode biases to record [V], in sweep order."""

    tolerance: float
    """Agreement required on terminal current [1], as a fraction."""

    n_nodes: int = 201
    """Node count for the ddsim mesh."""

    h_min: float = 1e-7
    """ddsim mesh spacing at the junction [cm]."""

    devsim_h_junction: float = 2e-7
    """devsim mesh spacing at the junction [cm]."""

    devsim_h_bulk: float = 2e-6
    """devsim mesh spacing at the outer contacts [cm]."""

    notes: str = ""
    """What this device is for, carried into the golden file header."""


CURRENT_FLOOR = 1e-10
"""Below this current magnitude a curve carries no information [A/cm^2].

At zero bias the terminal current is the difference of two drift and diffusion
fluxes of order 1e4 A/cm^2, so the exact answer, zero, arrives as whatever
rounding leaves behind: about 2e-12 from devsim and about 9e-24 from ddsim on
the symmetric diode. Neither number means anything, and a relative comparison
between them is meaningless rather than strict. Points under the floor are
checked against the floor instead of against each other.

The floor sits two decades below the smallest current either code produces at a
bias that is actually biased, which on device 1 is 3.4e-9 A/cm^2 at -0.1 V.
"""


def _forward(stop: float, step: float = 0.05) -> tuple[float, ...]:
    """Biases from 0 to stop inclusive [V], on a fixed step."""
    count = round(stop / step)
    return tuple(round(i * step, 10) for i in range(count + 1))


REVERSE = (-1.0, -0.75, -0.5, -0.25, -0.1)
"""Reverse biases for device 1 [V]. Ascending, so the sweep is monotone."""

BENCHMARKS: tuple[DiodeBenchmark, ...] = (
    DiodeBenchmark(
        name="diode_1e16_1e16",
        number=1,
        Na=1e16,
        Nd=1e16,
        length=1e-4,
        junction=0.5e-4,
        voltages=REVERSE + _forward(0.7),
        tolerance=0.02,
        notes=(
            "Symmetric junction, the clean analytic target. Reverse bias is "
            "generation limited and forward bias is diffusion limited, so the "
            "one sweep exercises both SRH branches."
        ),
    ),
    DiodeBenchmark(
        name="diode_1e18_1e16",
        number=2,
        Na=1e18,
        Nd=1e16,
        length=1e-4,
        junction=0.5e-4,
        voltages=_forward(0.7),
        tolerance=0.03,
        notes=(
            "Asymmetric junction. The p side is two decades heavier, so almost "
            "all the injection is into the n side and the Scharfetter lifetime "
            "differs by a factor of twenty across the junction."
        ),
    ),
    DiodeBenchmark(
        name="diode_1e20_1e15",
        number=3,
        Na=1e20,
        Nd=1e15,
        length=1e-4,
        junction=0.5e-4,
        voltages=_forward(0.7),
        tolerance=0.05,
        notes=(
            "P+N. docs/04-validation.md lists this one as the degeneracy test. "
            "Both codes are run in Boltzmann statistics here, so what it "
            "actually measures is agreement at a doping where Boltzmann is "
            "already wrong, which is a code comparison and not a physics "
            "check. Fermi-Dirac is deferred with the rest of it."
        ),
    ),
)

BY_NAME: dict[str, DiodeBenchmark] = {b.name: b for b in BENCHMARKS}


# --------------------------------------------------------------------- helpers


def scharfetter_lifetime(
    N_total: float, tau_max: float, tau_min: float = 0.0
) -> float:
    """Doping dependent lifetime [s], the scalar form of the ddsim model.

    tau = tau_min + (tau_max - tau_min) / (1 + N_total / N_ref)

    Used only for reporting in the golden file header. The generator hands
    devsim the same expression symbolically so it is evaluated per node.
    """
    return tau_min + (tau_max - tau_min) / (1.0 + (N_total / N_REF_SRH) ** GAMMA_SRH)


MODEL_SUMMARY: tuple[str, ...] = (
    "statistics:      Boltzmann",
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu",
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    "recombination:   SRH only, no Auger, no band to band, no impact ionisation",
    "SRH lifetimes:   Scharfetter, tau = tau_max / (1 + |N| / N_ref), with "
    f"tau_n_max = {TAU_N_MAX} s, tau_p_max = {TAU_P_MAX} s, N_ref = {N_REF_SRH} cm^-3",
    "SRH trap level:  midgap, n1 = p1 = n_i",
    "contacts:        ideal ohmic, psi from charge neutrality, densities "
    "pinned at equilibrium",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, n_i = {N_I:.6e} cm^-3, T = {T} K",
)
"""The model choices, verbatim into every golden file header.

docs/04-validation.md is blunt that an unmatched model makes the comparison
meaningless, so the match is written down where the data is rather than in a
document somewhere else. Each line is `key: value` so that `read_golden` can
pull the models out and a test can assert on them, rather than the header being
prose that only a human ever checks.
"""


@dataclass
class GoldenCurve:
    """A golden I-V curve read back from disk."""

    name: str
    """Device name."""

    header: dict[str, str] = field(default_factory=dict)
    """Key and value pairs from the commented header."""

    voltage: list[float] = field(default_factory=list)
    """Anode bias [V]."""

    current: list[float] = field(default_factory=list)
    """Terminal current into the anode [A/cm^2]."""

    cathode_current: list[float] = field(default_factory=list)
    """Terminal current into the cathode [A/cm^2]."""

    def imbalance(self, index: int) -> float:
        """How badly the two terminals fail to cancel at one point [1].

        In steady state the anode and cathode currents sum to zero exactly, so
        whatever is left is the golden datum's own numerical uncertainty. It is
        not a solver tolerance: at reverse bias the terminal current is a
        twelve decade cancellation between the drift and the diffusion term, so
        a converged solve still leaves about a percent behind. A regression
        test has no business demanding that ddsim match a number more closely
        than that number agrees with itself.
        """
        anode = self.current[index]
        cathode = self.cathode_current[index]
        scale = max(abs(anode), abs(cathode))
        if scale == 0.0:
            return 0.0
        return abs(anode + cathode) / scale


def read_golden(path: str) -> GoldenCurve:
    """Read one golden CSV, header comments and all.

    Kept here rather than in the test so the generator and the reader agree on
    the format by construction.
    """
    curve = GoldenCurve(name="")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#"):
                body = line[1:].strip()
                if ":" in body:
                    key, _, value = body.partition(":")
                    curve.header.setdefault(key.strip(), value.strip())
                continue
            if not line or line.startswith("voltage"):
                continue
            v, i, c = line.split(",")[:3]
            curve.voltage.append(float(v))
            curve.current.append(float(i))
            curve.cathode_current.append(float(c))
    curve.name = curve.header.get("device", "")
    return curve
