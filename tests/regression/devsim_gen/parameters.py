from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Q = 1.602176634e-19


K_B = 1.380649e-23


EPS_0 = 8.8541878128e-14

T = 300.0
EPS_R_SI = 11.7


N_I = 1.0e10

MU_N = 1417.0


MU_P = 470.0

TAU_N_MAX = 1e-5

TAU_P_MAX = 3e-6


TAU_N_MIN = 0.0


TAU_P_MIN = 0.0


N_REF_SRH = 5e16

GAMMA_SRH = 1.0
EPS_R_OX:float = 3.9


CHI_SI: float = 4.05

_EG_0: float = 1.1696
_EG_ALPHA: float = 4.73e-4; _EG_BETA: float = 636.0
EG:float = _EG_0 - _EG_ALPHA * T*T / (T + _EG_BETA)
PHI_M_N_POLY: float = CHI_SI

PHI_M_MIDGAP: float = CHI_SI + EG / 2.0


V_T = K_B * T / Q


def eg_at(T: float) -> float:
    return _EG_0 - _EG_ALPHA * T * T / (T + _EG_BETA)


def n_i_at(T: float) -> float:
    return float(N_I * (T / 300.0) ** 1.5 * math.exp(eg_at(300.0) / (2.0 * (K_B * 300.0 / Q)) - eg_at(T) / (2.0 * (K_B * T / Q))))

ARORA_N: tuple[float, float, float, float] = (88.0, 1252.0, 1.432e17, 0.88)
ARORA_P: tuple[float, float, float, float] = (54.3, 407.0, 2.67e17, 0.88)

LOMBARDI_N: dict[str, float] = {
    "B": 3.61e7,
    "C": 1.70e4,
    "tau": 0.0233,
    "delta": 3.58e18,
    "A": 2.58,
    "alpha": 6.85e-21,
    "eta": 0.0767,
    "kappa": 1.7,
}

LOMBARDI_P:dict[str, float] = {
    'B': 1.51e7,
    'C': 4.18e3,
    "tau": 0.0119,
    "delta": 4.10e15,
    "A":2.18,
    'alpha': 7.82e-21,
    "eta": 0.123,
    'kappa': 0.9,
}

E_PERP_FLOOR = 1.0e2

V_SAT_N = 1.07e7


V_SAT_P = 8.3e6

BETA_N = 2.0

BETA_P = 1.0


NC_300 = 2.86e19

NV_300 = 3.10e19


JOYCE_DIXON = (
    1.0 / math.sqrt(8.0),
    3.0 / 16.0 - math.sqrt(3.0) / 9.0,
    1.48386e-4,
    -4.42563e-6,
)

JOYCE_DIXON_MAX_U = 8.0


@dataclass(frozen=True)
class DiodeBenchmark:
    name: str

    number: int

    Na: float

    Nd: float

    length: float

    junction:float

    voltages: tuple[float, ...]

    tolerance: float
    n_nodes: int = 201
    h_min: float = 1e-7

    devsim_h_junction: float = 2e-7

    devsim_h_bulk: float = 2e-6

    notes: str = ""

    T: float = 300.0


CURRENT_FLOOR = 1e-10


def _forward(stop:float, step: float = 0.05) -> tuple[float, ...]:
    cur = round(stop / step)
    return tuple(round(h * step, 10) for h in range(cur + 1))
REVERSE = (-1.0, -0.75, -0.5, -0.25, -0.1)

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
            'Symmetric junction, the clean analytic target. Reverse bias is '
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
            'Asymmetric junction. The p side is two decades heavier, so almost '
            'all the injection is into the n side and the Scharfetter lifetime '
            "differs by a factor of twenty across the junction."
        ),
    ),
    DiodeBenchmark(
        name='diode_1e20_1e15',
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
            'already wrong, which is a code comparison and not a physics '
            "check. Fermi-Dirac is deferred with the rest of it."
        ),
    ),
    DiodeBenchmark(
        name="diode_1e16_1e16_400K",
        number=1,
        Na=1e16,
        Nd=1e16,
        length=1e-4,
        junction=0.5e-4,
        voltages=REVERSE + _forward(0.7),
        tolerance=0.02,
        T=400.0,
        notes=(
            "Benchmark 1 at 400 K, the Phase 9 temperature check. n_i, V_T and "
            "the bandgap follow T; mobility and lifetimes stay at their 300 K "
            "values, the same in both codes."
        ),
    ),
)
BY_NAME: dict[str, DiodeBenchmark] = {b.name: b for b in BENCHMARKS}


def scharfetter_lifetime(
    N_total: float, tau_max: float, tau_min: float = 0.0
) -> float:
    return float(tau_min + (tau_max - tau_min) / (1.0 + (N_total / N_REF_SRH) ** GAMMA_SRH))


MODEL_SUMMARY:tuple[str, ...] = (
    'statistics:      Boltzmann',
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu",
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    "recombination:   SRH only, no Auger, no band to band, no impact ionisation",
    'SRH lifetimes:   Scharfetter, tau = tau_max / (1 + |N| / N_ref), with '
    f"tau_n_max = {TAU_N_MAX} s, tau_p_max = {TAU_P_MAX} s, N_ref = {N_REF_SRH} cm^-3",
    'SRH trap level:  midgap, n1 = p1 = n_i',
    'contacts:        ideal ohmic, psi from charge neutrality, densities '
    'pinned at equilibrium',
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, n_i = {N_I:.6e} cm^-3, T = {T} K",
)


@dataclass
class GoldenCurve:
    name: str

    header: dict[str, str] = field(default_factory=dict)

    voltage: list[float] = field(default_factory=list)

    current:list[float] = field(default_factory=list)

    cathode_current: list[float] = field(default_factory=list)

    def imbalance(self, index:int) -> float:
        ys = self.current[index]
        dat = self.cathode_current[index]
        flag = max(abs(ys), abs(dat))
        if flag == 0.0:
            return 0.0
        return abs(ys + dat) / flag

def read_golden(path:str) -> GoldenCurve:
    el = GoldenCurve(name='')
    with open(path, encoding="utf-8") as a:
        for w in a:
            w = w.rstrip("\n")
            if w.startswith("#"):
                nxt = w[1:].strip()
                if ":" in nxt:
                    tmp3, _, g = nxt.partition(':')

                    el.header.setdefault(tmp3.strip(), g.strip())
                continue
            if not w or w.startswith('voltage'):
                continue
            k, aa, r = w.split(",")[:3]
            el.voltage.append(float(k))
            el.current.append(float(aa))
            el.cathode_current.append(float(r))
    el.name = el.header.get("device", ''); return el


@dataclass(frozen=True)
class MosBenchmark:
    name: str

    number:int

    substrate_doping: float

    t_ox: float
    t_si: float

    work_function:float
    voltages: tuple[float, ...]

    tolerance: float

    n_silicon:int = 121

    n_oxide:int = 5

    h_min: float = 5e-8
    devsim_h_surface: float = 5e-9
    devsim_h_bulk: float = 1e-6

    devsim_oxide_cells:int = 8
    notes: str = ''

def _gate_sweep(low: float, high: float, step: float) -> tuple[float, ...]:
    res = round((high - low) / step)

    return tuple(round(low + lst * step, 10) for lst in range(res + 1))

MOS_BENCHMARKS: tuple[MosBenchmark, ...] = (
    MosBenchmark(
        name='mos_cap_5nm',
        number=4,
        substrate_doping=-1e16,
        t_ox=5e-7,
        t_si=2e-4,
        work_function=PHI_M_N_POLY,
        voltages=_gate_sweep(-2.0, 2.0, 0.1),
        tolerance=0.02,
        notes=(
            'Thin oxide, so the oxide drop is small and most of the bias lands '
            'on the silicon surface. That puts the weight of the comparison on '
            "the semiconductor charge rather than on the parallel plate."
        ),
    ),
    MosBenchmark(
        name='mos_cap_20nm',
        number=5,
        substrate_doping=-1e16,
        t_ox=2e-6,
        t_si=2e-4,
        work_function=PHI_M_N_POLY,
        voltages=_gate_sweep(-2.0, 2.0, 0.1),
        tolerance=0.02,
        notes=(
            'Four times the oxide of device 4 and otherwise identical, so the '
            'pair separates an error in the oxide from an error in the '
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
    'capacitance:     dQ_gate/dV_gate by central difference on the charge, '
    "applied identically to both codes",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K, chi = {CHI_SI} eV, Eg = {EG:.6f} eV",
)

@dataclass
class MosGoldenCurve:
    name: str
    header: dict[str, str] = field(default_factory=dict)
    gate_voltage:list[float] = field(default_factory=list)
    charge: list[float] = field(default_factory=list)
    @property
    def tolerance(self) -> float:
        return float(self.header['tolerance'])

def read_mos_golden(path: str) -> MosGoldenCurve:
    e = MosGoldenCurve(name='')
    with open(path, encoding="utf-8") as c:
        for d in c:
            d = d.rstrip("\n")
            if d.startswith("#"):
                r = d[1:].strip()
                if ":" in r:
                    g, _, j = r.partition(':')
                    e.header.setdefault(g.strip(), j.strip())
                continue
            if not d or d.startswith('gate_voltage'):
                continue

            thing, t = d.split(",")[:2]
            e.gate_voltage.append(float(thing))
            e.charge.append(float(t))
    e.name = e.header.get('device', "")
    return e

def central_difference(voltage:list[float] | tuple[float, ...], charge: list[float]|tuple[float, ...],) -> tuple[list[float], list[float]]:
    e: list[float] = []
    flag:list[float] = []
    for xx in range(1,
                   len(voltage) - 1):
        jj = voltage[xx + 1] - voltage[xx - 1]
        e.append(voltage[xx])
        flag.append((charge[xx + 1] - charge[xx - 1]) / jj)
    return e, flag

def erfcinv(target: float) -> float:
    if not 0.0 < target < 2.0:
        raise ValueError(f"erfc maps onto (0, 2), so target must too, got {target}")
    m2, t = -30.0, 30.0
    for _ in range(200):
        i = 0.5*(m2+t)
        if math.erfc(i) > target:
            m2 = i
        else:
            t = i
    return 0.5 * (m2 + t)

MOSFET_PROCESS:dict[str, float] = {
    'substrate_doping': -1e18,
    "sd_peak":1e20,
    'x_j':2.5e-6,
    'lateral_diffusion':1.0e-6,
    "t_ox":2e-7,
    "sd_length":4e-5,
    'contact_length':2e-5,
    "t_si":1e-4,
}


def implant_shape(process:dict[str, float]) -> tuple[float, float]:
    Na = -process["substrate_doping"]
    f = process["sd_peak"]

    bb = process["x_j"] / math.sqrt(2.0 * math.log(f / Na))
    hh = process["lateral_diffusion"]/erfcinv(2.0 * Na/f); return bb, hh

H_DEPTH_SIGMAS = 0.25


H_DEPTH = implant_shape(MOSFET_PROCESS)[0] * H_DEPTH_SIGMAS

REDUCED_MODELS = "reduced"

FULL_MODELS = 'full'


@dataclass(frozen=True)
class MosfetBenchmark:
    name:str

    number:int

    L_gate:float
    gate_voltages: tuple[float, ...]

    drain_low: float
    drain_high:float
    tolerance: float

    devsim_h_junction: float = 5e-7

    devsim_h_channel: float = 2e-6

    devsim_h_contact: float = 4e-6
    devsim_h_surface:float = 6.25e-9
    devsim_h_depth: float = H_DEPTH

    devsim_oxide_cells: int = 32

    models:str = REDUCED_MODELS

    notes: str = ""


CURRENT_FLOOR_MOSFET = 1e-12

def _gate_range(low: float, high: float, step: float) -> tuple[float, ...]:
    f = round((high - low) / step); return tuple(round(low + h*step, 10) for h in range(f + 1))
MOSFET_BENCHMARKS: tuple[MosfetBenchmark, ...] = (MosfetBenchmark(name="nmos_1um", number=6, L_gate=1e-4, gate_voltages=_gate_range(0.0, 1.5, 0.1), drain_low=0.05, drain_high=1.0, tolerance=0.05, notes=('The long device. At 1 um this process has no short channel effect ' "left in it, so it is the reference the shorter ones are measured " 'against and the one place where a disagreement is about the 2D ' "transport rather than about a barrier."),), MosfetBenchmark(name='nmos_180nm', number=7, L_gate=1.8e-5, gate_voltages=_gate_range(0.0, 1.5, 0.1), drain_low=0.05, drain_high=1.0, tolerance=0.05, devsim_h_junction=2.25e-7, devsim_h_channel=4.5e-7, notes=("The same process drawn at 180 nm. The lateral encroachment leaves " '160 nm of metallurgical channel, so roll-off has started but the ' "device is still comfortably long channel. The lateral spacings " 'are half what they first were, because on this device the columns ' "and not the implant rows carry the mesh error: halving them alone " 'moved the drain current 6.9e-3 of a 7.3e-3 total and halving the ' 'rows alone moved it 5.7e-4. At 4.5e-7 the gate spans 40 columns, ' 'against the 20 it had and the 50 the 1 um device gets.'),), MosfetBenchmark(name='nmos_65nm', number=8, L_gate=6.5e-6, gate_voltages=_gate_range(-0.2, 1.5, 0.1), drain_low=0.05, drain_high=1.0, tolerance=0.08, devsim_h_junction=8.125e-8, devsim_h_channel=1.625e-7, notes=("45 nm of metallurgical channel. This is the DIBL row: the point of " 'it is the gap between the two curves, so the gate sweep starts ' 'below zero to hold the off state of both. The lateral spacings are ' 'half what they first were, and the columns carry that alone ' "because the rows cannot take up any slack. Halving one axis at a " 'time from the original mesh: columns alone moved the drain current ' "1.6606e-2 of a 1.964e-2 total, rows alone 2.6789e-3. Halving the " "rows again is not available, since most of their residual is " "devsim_h_surface and that is already 6.25e-9, where the next " 'halving is a third of an angstrom and past where a continuum ' "model means anything. The columns converge at order 2.07, taken " "off the 180 nm device's own before and after pair rather than " "assumed, so one halving takes their 1.6606e-2 to about 3.9e-3 and " "leaves the pair near 6.6e-3. A quartering was tried first and its " 'halved mesh check could not be solved in the memory available, ' "11022 silicon nodes shipping and about four times that to check."),),)

def _rolloff(name:str, L_gate:float) -> MosfetBenchmark:
    return MosfetBenchmark(
        name=name,
        number=9,
        L_gate=L_gate,
        gate_voltages=_gate_range(-0.4, 0.6, 0.05),
        drain_low=0.05,
        drain_high=1.0,
        tolerance=0.10,
        devsim_h_junction=L_gate / 80.0,
        devsim_h_channel=L_gate / 40.0,
        notes=(
            f"Gate length {L_gate * 1e7:g} nm, one point of the benchmark 9 "
            "threshold roll-off trend. Not one of benchmarks 6 to 8: those "
            "compare a drain current at every bias to 5 or 8 percent, and this "
            "one compares an extracted threshold across gate lengths to 10."
        ),
    )


ROLLOFF_BENCHMARKS: tuple[MosfetBenchmark, ...] = (
    _rolloff("rolloff_200nm", 2e-5),
    _rolloff("rolloff_100nm", 1e-5),
    _rolloff('rolloff_70nm', 7e-6),
    _rolloff("rolloff_50nm", 5e-6),
)


def _full_stack(name:str, L_gate: float) -> MosfetBenchmark:
    return MosfetBenchmark(
        name=name,
        number=10,
        L_gate=L_gate,
        gate_voltages=_gate_range(-0.4, 0.6, 0.05),
        drain_low=0.05,
        drain_high=1.0,
        tolerance=0.10,
        devsim_h_junction=min(L_gate / 80.0, 5e-7),
        devsim_h_channel=min(L_gate / 40.0, 2e-6),
        models=FULL_MODELS,
        notes=(
            f"Gate length {L_gate * 1e7:g} nm at the full Phase 5 model stack, "
            "one point of the benchmark 10 trend. The same device as its "
            'benchmark 9 sibling and a different set of models, which is what '
            "makes the pair worth having: a disagreement that is in both is "
            "the geometry, and one that is only here is a mobility model or "
            'the statistics.'
        ),
    )

FULL_STACK_BENCHMARKS: tuple[MosfetBenchmark, ...] = (
    _full_stack("fullstack_1um", 1e-4),
    _full_stack("fullstack_200nm", 2e-5),
    _full_stack('fullstack_100nm', 1e-5),
    _full_stack('fullstack_70nm', 7e-6),
    _full_stack("fullstack_50nm", 5e-6),
)

FULL_STACK_TREND: tuple[str, ...] = tuple(
    x2.name for x2 in FULL_STACK_BENCHMARKS
)

ROLLOFF_TREND: tuple[str, ...] = ('nmos_1um',) + tuple(
    m.name for m in ROLLOFF_BENCHMARKS
)

MOSFET_BY_NAME: dict[str, MosfetBenchmark] = {b.name: b for b in MOSFET_BENCHMARKS + ROLLOFF_BENCHMARKS + FULL_STACK_BENCHMARKS}
MOSFET_MODEL_SUMMARY: tuple[str, ...] = (
    "statistics:      Boltzmann",
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu",
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    'recombination:   SRH only, no Auger, no band to band, no impact ionisation',
    "SRH lifetimes:   Scharfetter, tau = tau_max / (1 + |N| / N_ref), with "
    f"tau_n_max = {TAU_N_MAX} s, tau_p_max = {TAU_P_MAX} s, N_ref = {N_REF_SRH} cm^-3",
    'SRH trap level:  midgap, n1 = p1 = n_i',
    'source/drain:    ideal ohmic plates on the silicon surface, psi from '
    "charge neutrality, densities pinned at equilibrium",
    'gate:            ideal metal, Dirichlet on psi at V_gate + '
    '(PHI_M_MIDGAP - Phi_M), no poly depletion, no gate overlap',
    'oxide:           Poisson only, no carriers, no fixed interface charge',
    "body:            ideal ohmic plate over the whole bottom edge, at 0 V",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K, chi = {CHI_SI} eV, Eg = {EG:.6f} eV",
)
_FULL_STACK_REPLACED = ('statistics:', "transport:", "mobility:", 'source/drain:')

FULL_STACK_MODEL_SUMMARY: tuple[str, ...] = (
    "statistics:      Fermi-Dirac by the Joyce-Dixon series, "
    f"Nc = {NC_300:.3e} and Nv = {NV_300:.3e} cm^-3, series capped at "
    f"n/Nc = {JOYCE_DIXON_MAX_U:g}",
    "transport:       Scharfetter-Gummel in the effective potential "
    "psi - V_t ln(gamma), Einstein relation D = V_t * mu",
    "mobility:        Arora on the doping, corrected by enhanced Lombardi "
    "surface scattering at the node, wrapped in Caughey-Thomas on the edge "
    f"parallel field with v_sat = {V_SAT_N:.3e} and {V_SAT_P:.3e} cm/s and "
    f"beta = {BETA_N:g} and {BETA_P:g}",
    "surface term:    frozen within a solve and taken to a fixed point "
    "across solves, in both codes",
    'source/drain:    ideal ohmic plates on the silicon surface, psi and both '
    'densities from neutrality against the degenerate mass action product',
) + tuple(
    lst
    for lst in MOSFET_MODEL_SUMMARY
    if not lst.startswith(_FULL_STACK_REPLACED)
)

def mosfet_model_summary(models:str) -> tuple[str,
          ...]:
    if models == FULL_MODELS:
        return FULL_STACK_MODEL_SUMMARY
    return MOSFET_MODEL_SUMMARY
@dataclass
class MosfetGoldenCurve:
    name:str
    header: dict[str, str] = field(default_factory=dict); gate_voltage: list[float] = field(default_factory=list)
    drain_low:list[float] = field(default_factory=list)
    source_low:list[float] = field(default_factory=list)
    drain_high: list[float] = field(default_factory=list)

    source_high: list[float] = field(default_factory=list)

    @property
    def tolerance(self) -> float:
        return float(self.header["tolerance"])

    def imbalance(self, index:int, high:bool) -> float:
        t = (self.drain_high if high else self.drain_low)[index]
        x = (self.source_high if high else self.source_low)[index]
        ok = max(abs(t), abs(x))
        return 0.0 if ok == 0.0 else abs(t + x) / ok


def first_resolved_point(current: Sequence[float], target: float) -> int:
    z = [float(w2) for w2 in current]
    v = 0
    for d in range(len(z) - 1, 0, -1):
        if z[d] <= z[d - 1] or z[d - 1] <= 0.0:
            v = d
            break

    if v:
        foo = max(abs(c2) for c2 in z[:v])
        if foo >= 1e-4 * target:
            raise AssertionError(
                f"a dropped point reaches {foo:.3e} A/cm against a target "
                f"of {target:.3e}, which is too close to the current being "
                'extracted at to be the terminal floor. Trimming it would be '
                'hiding a solver problem rather than ignoring roundoff.'
            )

    return v
def read_mosfet_golden(path: str) -> MosfetGoldenCurve:
    x = MosfetGoldenCurve(name='')
    with open(path, encoding='utf-8') as f:
        for d in f:
            d = d.rstrip("\n")
            if d.startswith("#"):
                e = d[1:].strip()

                if ":" in e:
                    i, _, tt = e.partition(":")

                    x.header.setdefault(i.strip(), tt.strip())

                continue
            if not d or d.startswith('gate_voltage'):
                continue
            rows = d.split(",")
            x.gate_voltage.append(float(rows[0]))
            x.drain_low.append(float(rows[1]))
            x.source_low.append(float(rows[2]))
            x.drain_high.append(float(rows[3]))

            x.source_high.append(float(rows[4]))
    x.name = x.header.get('device', '')
    return x


@dataclass(frozen=True)
class AcBenchmark:
    name: str
    number: int
    biases: tuple[float, ...]
    frequencies: tuple[float, ...]
    tolerance: float
    tau: float | None
    devsim_refine: float = 1.0
    devsim_runs: tuple[tuple[int, bool], ...] = ((1, False), (4, False))
    notes: str = ""


AC_DIODE_NA = 1e19
AC_DIODE_ND = 1e18
AC_DIODE_TAU = 1e-3
AC_DIODE_LN = math.sqrt(MU_N * V_T * AC_DIODE_TAU)
AC_DIODE_LP = math.sqrt(MU_P * V_T * AC_DIODE_TAU)
AC_DIODE_LENGTH = 5 * AC_DIODE_LN + 5 * AC_DIODE_LP
AC_DIODE_JUNCTION = 5 * AC_DIODE_LN

BAR_DOPING = 1e16
BAR_LENGTH = 1e-4
BAR_DEVSIM_H = 1e-6

AC_BENCHMARKS: tuple[AcBenchmark, ...] = (
    AcBenchmark(
        name="mos_cap_5nm_ac",
        number=11,
        biases=(-1.0, 0.5, 1.0),
        frequencies=tuple(10 ** (k / 2) for k in range(4, 21)),
        tolerance=0.03,
        tau=1e-12,
        devsim_refine=2.0,
        devsim_runs=((1, True), (2, True), (4, True), (1, False), (4, False)),
        notes=(
            "Benchmark 4's capacitor solved on the coupled system at each frequency. "
            "tau 1e-12 s puts the inversion layer's response near 10 kHz, so the sweep "
            "runs from the quasi-static curve through the transition to the frozen one, "
            "and up into the substrate RC rolloff past 100 MHz. On benchmark 4's own "
            "mesh devsim's low frequency inversion C moves 0.6 percent when halved, so "
            "this golden is solved on that mesh halved. In inversion devsim's answer from the "
            "low frequency end through the transition is roundoff limited: the same state on "
            "1, 2 and 4 MKL threads differs by up to 10 percent in double and about 3 in "
            "extended precision, and double against extended by about 4. The stored values "
            "are extended on one thread, and the spread column is the worst change over all "
            "five runs."
        ),
    ),
    AcBenchmark(
        name="diode_long_ac",
        number=12,
        biases=(0.776,),
        frequencies=tuple(10 ** (k / 2) / (2 * math.pi * AC_DIODE_TAU) for k in range(-4, 5)),
        tolerance=0.03,
        tau=AC_DIODE_TAU,
        notes=(
            "The long p+n diode of tests/analytic/test_ac_diode.py, both sides five "
            "diffusion lengths long, at low injection. Frequencies run omega tau from "
            "1e-2 to 1e2, across the bend in G0 sqrt(1 + i omega tau)."
        ),
    ),
    AcBenchmark(
        name="bar_noise",
        number=13,
        biases=(0.0,),
        frequencies=tuple(10 ** (k / 2) for k in range(18, 29)),
        tolerance=0.05,
        tau=None,
        devsim_refine=2.0,
        notes=(
            "A 1 um bar of 1e16 n type at equilibrium, through its dielectric "
            "relaxation frequency near 3.5e11 Hz. devsim gives the impedance field and "
            "the source 4 q^2 V_t mu n is integrated against its node gradient the way "
            "devsim's testing/noise_res.py does. That recipe reads 4kT Re Z times "
            "1 - h/L, so the golden uses the halved mesh."
        ),
    ),
)
AC_BY_NAME: dict[str, AcBenchmark] = {b.name: b for b in AC_BENCHMARKS}


@dataclass(frozen=True)
class TransientBenchmark:
    name: str
    number: int
    forward: float
    reverse: float
    ratio: float
    steps: int
    span: float
    tolerance: float
    tau: float = AC_DIODE_TAU
    devsim_runs: tuple[tuple[int, bool], ...] = ((1, False), (4, False))
    notes: str = ""

    @property
    def kingston(self) -> float:
        lo, hi = 0.0, 6.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if math.erf(mid) < 1.0 / (1.0 + self.ratio) else (lo, mid)
        return self.tau * lo * lo

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(k * self.span * self.kingston / self.steps for k in range(1, self.steps + 1))


TRANSIENT_BENCHMARKS: tuple[TransientBenchmark, ...] = (
    TransientBenchmark(
        name="diode_recovery",
        number=14,
        forward=0.776,
        reverse=-100.0,
        ratio=1.0,
        steps=200,
        span=2.0,
        tolerance=0.05,
        notes=(
            "Benchmark 12's long p+n diode, tau 1e-3 s, held at its DC state at 0.776 V and "
            "switched at t = 0 to -100 V through a series resistor sized so the first "
            "reverse current equals the forward one. Fixed steps, BDF1 for the first and "
            "BDF2 after, out to twice Kingston's storage time. Junction capacitance puts "
            "the storage time near 1.19 times Kingston on this device, see the 2026-10-18 rows."
        ),
    ),
)
TRANSIENT_BY_NAME: dict[str, TransientBenchmark] = {b.name: b for b in TRANSIENT_BENCHMARKS}

AC_MODEL_SUMMARY: tuple[str, ...] = (
    "statistics:      Boltzmann",
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu",
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    "recombination:   SRH only, midgap traps, tau uniform where the benchmark sets "
    "one, Scharfetter otherwise",
    "small signal:    (J + i omega M) dx = b, M on the continuity rows only, "
    "displacement current through the contact D field",
    "admittance:      Y = dI/dV at the driven contact, everything else held at its "
    "DC bias",
    "noise:           impedance field, diffusion sources 4 q^2 D n for both carriers, "
    "S_V = S_I / |Y|^2",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K",
)


@dataclass
class AcGoldenCurve:
    header: dict[str, str] = field(default_factory=dict)
    rows: list[dict[str, float]] = field(default_factory=list)


def read_ac_golden(path: str) -> AcGoldenCurve:
    x = AcGoldenCurve()
    names: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(":")
                if v:
                    x.header.setdefault(k.strip(), v.strip())
                continue
            if not line:
                continue
            if not names:
                names = line.split(",")
                continue
            x.rows.append(dict(zip(names, map(float, line.split(",")), strict=True)))
    return x


NPN_2D: dict[str, float] = {
    "width": 3e-4,
    "depth": 2e-4,
    "emitter_window": 1e-4,
    "emitter_contact": 8e-5,
    "base_contact": 2e-4,
    "epi_doping": 1e16,
    "buried_peak": 5e18,
    "buried_sigma": 3e-5,
    "base_peak": 1e17,
    "base_sigma": 2.33e-5,
    "emitter_peak": 1e20,
    "emitter_sigma": 3.97e-6,
    "lateral_length": 5e-6,
}


@dataclass(frozen=True)
class BjtBenchmark:
    name: str

    number: int

    v_be: tuple[float, ...]

    i_b: float

    v_ce: tuple[float, ...]

    tolerance: float

    devsim_h: float = 2e-6

    devsim_h_junction: float = 5e-7

    devsim_h_edge: float = 2.5e-7

    notes: str = ""


BJT_BENCHMARKS: tuple[BjtBenchmark, ...] = (
    BjtBenchmark(
        name="npn_2d",
        number=15,
        v_be=tuple(round(0.4 + 0.05 * k, 2) for k in range(9)),
        i_b=1e-5,
        v_ce=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
        tolerance=0.05,
        notes=(
            "A 2D npn with an n+ emitter windowed on the left of the top surface, a p "
            "base implanted across the whole top, and n epi over a buried n+ layer. "
            "The emitter and base plates sit on top and the collector on the bottom. "
            "The Gummel plot drives the emitter down with base and collector at "
            "zero, and the output curve holds the base current while the "
            "collector walks up."
        ),
    ),
)

BJT_BY_NAME: dict[str, BjtBenchmark] = {b.name: b for b in BJT_BENCHMARKS}


NC = 2.86e19

NV = 3.1e19

A_STAR_N = 112.0

A_STAR_P = 32.0


@dataclass(frozen=True)
class SchottkyBenchmark:
    name: str

    number: int

    Nd: float

    length: float

    barrier: float

    voltages: tuple[float, ...]

    tolerance: float

    n_nodes: int = 201

    h_min: float = 1e-7

    devsim_h_contact: float = 1e-7

    devsim_h_bulk: float = 2e-6

    notes: str = ""


SCHOTTKY_BENCHMARKS: tuple[SchottkyBenchmark, ...] = (
    SchottkyBenchmark(
        name="schottky_0p7",
        number=16,
        Nd=1e16,
        length=1e-4,
        barrier=0.7,
        voltages=(-1.0, -0.5, -0.2) + tuple(round(0.05 * k, 2) for k in range(1, 8)),
        tolerance=0.05,
        notes=(
            "A 0.7 eV barrier on 1e16 n-type silicon, 1 um long. DEVSIM has no "
            "Schottky contact, so the generator writes one: the bulk edge "
            "current over the contact edge plus q v (c - c0) / NodeVolume as "
            "the contact node model, since DEVSIM integrates a contact node "
            "model over the node volume. At this mobility v_R / v_D is about "
            "0.05, so the current sits 5 percent under pure thermionic "
            "emission and the diffusion part is exercised too."
        ),
    ),
)
