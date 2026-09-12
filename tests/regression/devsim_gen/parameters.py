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

import math
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

EPS_R_OX: float = 3.9
"""Relative permittivity of silicon dioxide [1]."""

CHI_SI: float = 4.05
"""Electron affinity of silicon [eV], vacuum level to conduction band edge."""

_EG_0: float = 1.1696
_EG_ALPHA: float = 4.73e-4
_EG_BETA: float = 636.0

EG: float = _EG_0 - _EG_ALPHA * T * T / (T + _EG_BETA)
"""Silicon band gap at T [eV], Varshni. 1.124119 eV at 300 K."""

PHI_M_N_POLY: float = CHI_SI
"""Work function of n+ polysilicon [eV]. Fermi level at the conduction edge."""

PHI_M_MIDGAP: float = CHI_SI + EG / 2.0
"""Work function of a midgap metal [eV].

Also the work function of intrinsic silicon, which is why the gate potential
can be written without reference to the substrate: psi is measured from the
intrinsic level in both codes, so psi_gate = V_gate + (PHI_M_MIDGAP - Phi_M).
"""

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


@dataclass(frozen=True)
class MosBenchmark:
    """One MOS capacitor from the tier 4 benchmark set of docs/04-validation.md.

    An ideal capacitor: no fixed oxide charge, no poly depletion, an ideal metal
    gate described by a work function alone. Both codes are told the same thing.
    """

    name: str
    """Short name, also the golden file stem."""

    number: int
    """Row in the docs/04-validation.md benchmark table."""

    substrate_doping: float
    """Net doping of the substrate [cm^-3], negative for p-type."""

    t_ox: float
    """Oxide thickness [cm]."""

    t_si: float
    """Silicon thickness [cm]. Several times the maximum depletion width, so
    the body contact sits in neutral material."""

    work_function: float
    """Gate metal work function [eV]."""

    voltages: tuple[float, ...]
    """Gate biases to record [V], ascending and evenly spaced.

    Evenly spaced on purpose: the capacitance is taken from the charge by a
    central difference applied identically to both codes, and an uneven grid
    would make that two different operators.
    """

    tolerance: float
    """Agreement required on gate charge and capacitance [1], as a fraction."""

    n_silicon: int = 121
    """ddsim node count through the silicon."""

    n_oxide: int = 5
    """ddsim node count through the oxide."""

    h_min: float = 5e-8
    """ddsim mesh spacing at the silicon surface [cm]."""

    devsim_h_surface: float = 5e-9
    """devsim mesh spacing at the silicon surface [cm].

    0.05 nm. The inversion layer is a nanometre or so thick and is the only
    structure on this device that a mesh can miss, so this is the one spacing
    the answer is actually sensitive to. Measured on a refinement ladder at
    +2 V, where the layer is thinnest: 8e-8 cm is 4.3e-4 off the converged
    charge, 2e-8 is 9e-5, 5e-9 is 6.8e-5, and 1.25e-9 is where it stops moving.
    5e-9 is two hundred times under the tolerance being asserted and still
    coarser than a lattice constant, which is as far as a continuum model has
    any business being refined.
    """

    devsim_h_bulk: float = 1e-6
    """devsim mesh spacing at the body contact [cm].

    Resolves the depletion edge, which is the only thing out here that moves.
    Contributes 3.6e-6 at the next halving, against 1.2e-5 at 2e-6.
    """

    devsim_oxide_cells: int = 8
    """devsim cells through the oxide.

    With no charge in it the oxide potential is a straight line, which any
    number of cells resolves exactly. Measured rather than assumed: the gate
    charge is identical from 2 cells to 32 to within 8e-15, which is round off.
    Kept at 8 because it costs nothing.
    """

    notes: str = ""
    """What this device is for, carried into the golden file header."""


def _gate_sweep(low: float, high: float, step: float) -> tuple[float, ...]:
    """Gate biases from low to high inclusive [V], evenly spaced."""
    count = round((high - low) / step)
    return tuple(round(low + index * step, 10) for index in range(count + 1))


MOS_BENCHMARKS: tuple[MosBenchmark, ...] = (
    MosBenchmark(
        name="mos_cap_5nm",
        number=4,
        substrate_doping=-1e16,
        t_ox=5e-7,
        t_si=2e-4,
        work_function=PHI_M_N_POLY,
        voltages=_gate_sweep(-2.0, 2.0, 0.1),
        tolerance=0.02,
        notes=(
            "Thin oxide, so the oxide drop is small and most of the bias lands "
            "on the silicon surface. That puts the weight of the comparison on "
            "the semiconductor charge rather than on the parallel plate."
        ),
    ),
    MosBenchmark(
        name="mos_cap_20nm",
        number=5,
        substrate_doping=-1e16,
        t_ox=2e-6,
        t_si=2e-4,
        work_function=PHI_M_N_POLY,
        voltages=_gate_sweep(-2.0, 2.0, 0.1),
        tolerance=0.02,
        notes=(
            "Four times the oxide of device 4 and otherwise identical, so the "
            "pair separates an error in the oxide from an error in the "
            "silicon: only the first moves with t_ox."
        ),
    ),
)

MOS_MODEL_SUMMARY: tuple[str, ...] = (
    "statistics:      Boltzmann",
    "carriers:        equilibrium, phi_n = phi_p = body bias, no transport",
    "oxide:           Poisson only, no carriers, no fixed interface charge",
    "interface:       continuity of normal D, which box integration gives for "
    "free once each edge carries its own permittivity",
    "gate:            ideal metal, Dirichlet on psi at V_gate + "
    "(PHI_M_MIDGAP - Phi_M), no poly depletion",
    "body:            ideal ohmic, psi from charge neutrality",
    "capacitance:     dQ_gate/dV_gate by central difference on the charge, "
    "applied identically to both codes",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K, chi = {CHI_SI} eV, Eg = {EG:.6f} eV",
)
"""The model choices for the MOS benchmarks, verbatim into the golden header."""


@dataclass
class MosGoldenCurve:
    """A golden C-V curve read back from disk."""

    name: str
    header: dict[str, str] = field(default_factory=dict)
    gate_voltage: list[float] = field(default_factory=list)
    charge: list[float] = field(default_factory=list)

    @property
    def tolerance(self) -> float:
        """The agreement the header asks for [1]."""
        return float(self.header["tolerance"])


def read_mos_golden(path: str) -> MosGoldenCurve:
    """Read one golden C-V CSV, header comments and all."""
    curve = MosGoldenCurve(name="")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#"):
                body = line[1:].strip()
                if ":" in body:
                    key, _, value = body.partition(":")
                    curve.header.setdefault(key.strip(), value.strip())
                continue
            if not line or line.startswith("gate_voltage"):
                continue
            v, q = line.split(",")[:2]
            curve.gate_voltage.append(float(v))
            curve.charge.append(float(q))
    curve.name = curve.header.get("device", "")
    return curve


def central_difference(
    voltage: list[float] | tuple[float, ...],
    charge: list[float] | tuple[float, ...],
) -> tuple[list[float], list[float]]:
    """dQ/dV at the interior points of an evenly spaced sweep [V, F/cm^2].

    The same operator on both codes, so its truncation error cancels out of the
    comparison instead of being one more thing to argue about. The endpoints
    have no centred neighbour and are dropped.
    """
    midpoints: list[float] = []
    slopes: list[float] = []
    for index in range(1, len(voltage) - 1):
        span = voltage[index + 1] - voltage[index - 1]
        midpoints.append(voltage[index])
        slopes.append((charge[index + 1] - charge[index - 1]) / span)
    return midpoints, slopes


# --------------------------------------------------------------------- MOSFETs


def erfcinv(target: float) -> float:
    """Inverse of `math.erfc` on (0, 2), by bisection [1].

    `scipy.special.erfcinv` is what ddsim's `device/mosfet.py` calls, and the
    devsim interpreter has no scipy. Bisection on [-30, 30] matches it to
    machine precision across the whole of (0, 2). The bracket has to reach
    below zero because erfc passes through 1 at the origin, even though an
    implant profile only ever asks for twice the ratio of two doping
    concentrations and so stays well inside (0, 1).
    `tests/regression/test_devsim_mosfet.py` pins the two against each other.
    """
    if not 0.0 < target < 2.0:
        raise ValueError(f"erfc maps onto (0, 2), so target must too, got {target}")
    low, high = -30.0, 30.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if math.erfc(middle) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


MOSFET_PROCESS: dict[str, float] = {
    "substrate_doping": -1e18,
    "sd_peak": 1e20,
    "x_j": 2.5e-6,
    "lateral_diffusion": 1.0e-6,
    "t_ox": 2e-7,
    "sd_length": 4e-5,
    "contact_length": 2e-5,
    "t_si": 1e-4,
}
"""The vertical process every MOSFET benchmark is drawn on.

A literal mirror of `ddsim.extract.rolloff.SHORT_CHANNEL_PROCESS`, pinned key by
key by `test_devsim_mosfet.py::test_process_mirrors_ddsim`. Benchmarks 6 to 8
are that one process drawn at three gate lengths, which is what makes the set a
roll-off measurement rather than three unrelated devices.
"""


def implant_shape(process: dict[str, float]) -> tuple[float, float]:
    """The two implant lengths ddsim's `nmos` derives from a process [cm, cm].

    Returns `(sigma, edge)`: the depth standard deviation of the gaussian and
    the characteristic length of the lateral erfc. Both are closed form in
    `ddsim/device/mosfet.py` and are reproduced here rather than imported,
    because the generator runs under an interpreter that has no ddsim. The
    mirror test compares the pair against the values ddsim computes.
    """
    Na = -process["substrate_doping"]
    peak = process["sd_peak"]
    sigma = process["x_j"] / math.sqrt(2.0 * math.log(peak / Na))
    edge = process["lateral_diffusion"] / erfcinv(2.0 * Na / peak)
    return sigma, edge


H_DEPTH_SIGMAS = 0.25
"""Row spacing through the implant, as a fraction of the implant sigma [1].

The source and drain profile is a Gaussian of width `implant_shape(...)[0]`,
and the row spacing that resolves it has to be derived from that width rather
than picked. The old value was a flat 1e-6 cm, which on this process is 1.21
sigma: more than one standard deviation per row, which puts the metallurgical
junction in the wrong place. Measured on the 1 um device at 50 mV, halving
every spacing from there moved the drain current 4.6 percent at zero gate and
2.9 percent at 0.1 V, and the split showed the lateral columns contributed
0.036 percent of it and these rows almost all the rest.

A quarter of a sigma is where it stops mattering. Going from 1.21 to 0.25 sigma
takes the same halving check to 0.53 percent at zero gate and under 0.08
percent everywhere else, and going on to 0.125 sigma moves the answer a further
0.015 percent, so the profile is resolved. ddsim needed no equivalent change:
it grades its own rows and already samples this implant at 0.21 sigma at the
junction depth, which is what `test_ddsim_resolves_the_implant` holds it to.
"""

H_DEPTH = implant_shape(MOSFET_PROCESS)[0] * H_DEPTH_SIGMAS
"""Row spacing at the implant depth line [cm]. See `H_DEPTH_SIGMAS`."""

@dataclass(frozen=True)
class MosfetBenchmark:
    """One NMOS from the tier 4 benchmark set of docs/04-validation.md."""

    name: str
    """Short name, also the golden file stem."""

    number: int
    """Row in the docs/04-validation.md benchmark table."""

    L_gate: float
    """Drawn gate length [cm]. 1e-4 is 1 um."""

    gate_voltages: tuple[float, ...]
    """Gate biases to record [V], ascending from the off state."""

    drain_low: float
    """Drain bias of the linear curve [V], the one thresholds are read off."""

    drain_high: float
    """Drain bias of the saturated curve [V], the one DIBL is read off."""

    tolerance: float
    """Agreement required on drain current [1], as a fraction."""

    devsim_h_junction: float = 5e-7
    """devsim column spacing at each metallurgical junction [cm].

    The default is the 1 um value. The two short devices set their own, because
    a column spacing of half a micron is wider than their whole gate.
    """

    devsim_h_channel: float = 2e-6
    """devsim column spacing in the middle of the channel [cm].

    The default is the 1 um value, for the same reason `devsim_h_junction` is.
    """

    devsim_h_contact: float = 4e-6
    """devsim column spacing under each contact plate [cm]."""

    devsim_h_surface: float = 6.25e-9
    """devsim row spacing at the silicon surface [cm].

    The inversion layer is a nanometre or so thick and is the only structure on
    this device that a mesh can miss, so this is the spacing the drain current
    is actually sensitive to, and it is the one spacing that is not coarsened.
    Measured on the 1 um device at zero gate and 50 mV of drain, where the
    current is drain junction leakage and so at its most mesh sensitive,
    relaxing it alone from 5e-8 to 2e-7 moved devsim's answer from 1.5612e-6 to
    2.1849e-6 A/cm, which is 46 percent. No other spacing on this device moves
    the answer anything like that far.

    It is 6.25e-9 rather than that 5e-8 because at 5e-8 neither code was
    converged. ddsim's own ladder, in
    tests/convergence/test_mosfet_mesh_convergence.py, still moved 5.0 percent
    subthreshold when the vertical mesh was halved from there, and the two
    codes are matched at the same surface spacing on purpose, so a reference
    left at 5e-8 would have agreed with ddsim to a fraction of a percent while
    both sat 7 percent from the limit. This value is three halvings down, where
    the move is 0.12 percent, inside the tenth of tolerance
    test_devsim_mosfet.py::test_golden_reference_is_converged demands.
    """

    devsim_h_depth: float = H_DEPTH
    """devsim row spacing at the implant depth [cm]."""

    devsim_oxide_cells: int = 32
    """devsim cells through the oxide.

    The oxide holds no charge, so its potential is a straight line and any
    number of cells resolves a straight line exactly. What fixes this number is
    the other side of the interface: t_ox / devsim_oxide_cells is the cell the
    silicon surface spacing meets at the seam, and on this process, 2 nm of
    oxide, 32 cells makes that 6.25e-9 and the seam ratio exactly 1. Refining
    devsim_h_surface without following it here reopens the seam, which reads as
    a converging refinement and is partly a widening discontinuity. See
    docs/05-pitfalls.md.
    """

    notes: str = ""
    """What this device is for, carried into the golden file header."""


CURRENT_FLOOR_MOSFET = 1e-12
"""Below this current magnitude a transfer curve carries no information [A/cm].

The off state of these devices is reverse drain junction leakage, which both
codes compute as a difference of much larger fluxes. Points under the floor are
checked against the floor rather than against each other, which is the rule the
diode benchmarks already use for the same reason.
"""


def _gate_range(low: float, high: float, step: float) -> tuple[float, ...]:
    """Gate biases from low to high inclusive [V], evenly spaced."""
    count = round((high - low) / step)
    return tuple(round(low + index * step, 10) for index in range(count + 1))


MOSFET_BENCHMARKS: tuple[MosfetBenchmark, ...] = (
    MosfetBenchmark(
        name="nmos_1um",
        number=6,
        L_gate=1e-4,
        gate_voltages=_gate_range(0.0, 1.5, 0.1),
        drain_low=0.05,
        drain_high=1.0,
        tolerance=0.05,
        notes=(
            "The long device. At 1 um this process has no short channel effect "
            "left in it, so it is the reference the shorter ones are measured "
            "against and the one place where a disagreement is about the 2D "
            "transport rather than about a barrier."
        ),
    ),
    MosfetBenchmark(
        name="nmos_180nm",
        number=7,
        L_gate=1.8e-5,
        gate_voltages=_gate_range(0.0, 1.5, 0.1),
        drain_low=0.05,
        drain_high=1.0,
        tolerance=0.05,
        devsim_h_junction=4.5e-7,
        devsim_h_channel=9e-7,
        notes=(
            "The same process drawn at 180 nm. The lateral encroachment leaves "
            "160 nm of metallurgical channel, so roll-off has started but the "
            "device is still comfortably long channel."
        ),
    ),
    MosfetBenchmark(
        name="nmos_65nm",
        number=8,
        L_gate=6.5e-6,
        gate_voltages=_gate_range(-0.2, 1.5, 0.1),
        drain_low=0.05,
        drain_high=1.0,
        tolerance=0.08,
        devsim_h_junction=1.625e-7,
        devsim_h_channel=3.25e-7,
        notes=(
            "45 nm of metallurgical channel. This is the DIBL row: the point of "
            "it is the gap between the two curves, so the gate sweep starts "
            "below zero to hold the off state of both."
        ),
    ),
)

MOSFET_BY_NAME: dict[str, MosfetBenchmark] = {b.name: b for b in MOSFET_BENCHMARKS}

MOSFET_MODEL_SUMMARY: tuple[str, ...] = (
    "statistics:      Boltzmann",
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu",
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    "recombination:   SRH only, no Auger, no band to band, no impact ionisation",
    "SRH lifetimes:   Scharfetter, tau = tau_max / (1 + |N| / N_ref), with "
    f"tau_n_max = {TAU_N_MAX} s, tau_p_max = {TAU_P_MAX} s, N_ref = {N_REF_SRH} cm^-3",
    "SRH trap level:  midgap, n1 = p1 = n_i",
    "source/drain:    ideal ohmic plates on the silicon surface, psi from "
    "charge neutrality, densities pinned at equilibrium",
    "gate:            ideal metal, Dirichlet on psi at V_gate + "
    "(PHI_M_MIDGAP - Phi_M), no poly depletion, no gate overlap",
    "oxide:           Poisson only, no carriers, no fixed interface charge",
    "body:            ideal ohmic plate over the whole bottom edge, at 0 V",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K, chi = {CHI_SI} eV, Eg = {EG:.6f} eV",
)
"""The model choices for the MOSFET benchmarks, verbatim into every header.

Constant mobility rather than the Phase 5 stack, and Boltzmann rather than
Fermi-Dirac. Both codes are run that way, so the comparison is still a
comparison, but what it measures is the 2D transport, the geometry and the
electrostatics and not the mobility models. See the dated row in
docs/07-decisions.md.
"""


@dataclass
class MosfetGoldenCurve:
    """The two transfer curves of one MOSFET, read back from disk."""

    name: str
    header: dict[str, str] = field(default_factory=dict)
    gate_voltage: list[float] = field(default_factory=list)
    drain_low: list[float] = field(default_factory=list)
    source_low: list[float] = field(default_factory=list)
    drain_high: list[float] = field(default_factory=list)
    source_high: list[float] = field(default_factory=list)

    @property
    def tolerance(self) -> float:
        """The agreement the header asks for [1]."""
        return float(self.header["tolerance"])

    def imbalance(self, index: int, high: bool) -> float:
        """How badly drain and source fail to cancel at one point [1].

        The body current is many decades below either of them on a device with
        no impact ionisation, so in steady state the two surface terminals sum
        to zero and whatever is left is the golden datum's own numerical
        uncertainty. A regression test has no business demanding that ddsim
        match a number more closely than that number agrees with itself.
        """
        drain = (self.drain_high if high else self.drain_low)[index]
        source = (self.source_high if high else self.source_low)[index]
        scale = max(abs(drain), abs(source))
        return 0.0 if scale == 0.0 else abs(drain + source) / scale


def read_mosfet_golden(path: str) -> MosfetGoldenCurve:
    """Read one golden transfer pair, header comments and all."""
    curve = MosfetGoldenCurve(name="")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#"):
                body = line[1:].strip()
                if ":" in body:
                    key, _, value = body.partition(":")
                    curve.header.setdefault(key.strip(), value.strip())
                continue
            if not line or line.startswith("gate_voltage"):
                continue
            columns = line.split(",")
            curve.gate_voltage.append(float(columns[0]))
            curve.drain_low.append(float(columns[1]))
            curve.source_low.append(float(columns[2]))
            curve.drain_high.append(float(columns[3]))
            curve.source_high.append(float(columns[4]))
    curve.name = curve.header.get("device", "")
    return curve
