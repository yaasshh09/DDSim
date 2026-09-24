"""The ddsim side of the Phase 8 scoreboard. See phases/PHASE-8.md.

Run by hand from the repository root, like the DEVSIM generators, and commit
what it writes into `data/scoreboard/`:

    .venv/Scripts/python.exe tools/scoreboard.py cases

`cases` draws the robustness case set: 200 devices with a target bias, drawn
with a fixed seed from the knob ranges the constructors declare in their
docstrings. tests/regression/test_scoreboard.py redraws it and fails if the
committed file differs, so a moved range can't leave a stale case set behind.

`run` solves every case cold, twice: once through the public sweep the API
calls, and once as the fine-step reference, a ramp of every contact from zero
in 2 percent steps. It writes the raw numbers into `ddsim_robustness.csv`. It
refuses to run unless the BLAS thread counts are pinned to 1, since the speed
axis compares one thread against one thread:

    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        .venv/Scripts/python.exe tools/scoreboard.py run

The pass rule is `score`, applied to the raw numbers of both tools, so it can
be argued about without solving anything again. The floors and the balance
lines are the ones Tier 4 already measured and uses:

- A result is valid when it converged and, wherever its benchmark family
  requires balance, its terminal currents cancel to a tenth of the family's
  tolerance. Noise has to sit an order below the tolerance it's judged by.
- Two results agree when both sit under the family floor, or when they differ
  by no more than the family tolerance plus three times the larger imbalance,
  which is the Tier 4 comparison rule.
- The references are the valid fine results of both tools. A driver passes a
  case when it is valid and agrees with every reference. A case with no valid
  reference, or with references that disagree, is dropped and named rather
  than scored for either side.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ddsim.api.devices import build_from_spec, device_parameters
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep, iv_sweep, terminal_currents
from ddsim.extract.rolloff import REFERENCE_CURRENT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "scoreboard"

SEED = 20260924
"""The draw's seed. Changing it is a new case set, not a tweak."""

SPLIT = (("pn_diode", 60), ("mos_cap", 40), ("nmos", 100))
"""How many cases of each device, in the order they're drawn."""

PHYSICAL = {
    "pn_diode": ("Na", "Nd", "length", "junction", "anode_voltage"),
    "mos_cap": ("substrate_doping", "t_ox", "t_si", "work_function", "gate_voltage"),
    "nmos": (
        "L_gate",
        "substrate_doping",
        "sd_peak",
        "x_j",
        "lateral_diffusion",
        "t_ox",
        "work_function",
        "gate_voltage",
        "drain_voltage",
    ),
}
"""The knobs each case draws: doping, geometry, oxide, work function and the
target bias. Mesh knobs, domain padding and the other contacts keep their
defaults, since each tool meshes the device its own way."""

PREFIX = {"pn_diode": "d", "mos_cap": "c", "nmos": "m"}
SIGNIFICANT = 4
"""Digits kept per drawn value, so the case file reads like a device spec."""


@dataclass(frozen=True)
class Case:
    name: str
    device: str
    knobs: dict[str, float]
    redraws: int
    """How many draws before this one DDSim's device check refused."""


def _draw(rng: np.random.Generator, low: float, high: float, axis: str) -> float:
    if axis == "log":
        # A negative log range is an acceptor doping: draw the size, keep the sign.
        sign = -1.0 if high < 0 else 1.0
        a, b = sorted((abs(low), abs(high)))
        value = sign * math.exp(rng.uniform(math.log(a), math.log(b)))
    else:
        value = rng.uniform(low, high)
    return float(f"{value:.{SIGNIFICANT}g}")


def _builds(device: str, knobs: dict[str, Any]) -> bool:
    try:
        build_from_spec(device, knobs)
    except ValueError:
        return False
    return True


def draw_cases() -> list[Case]:
    """The whole case set, drawn in a fixed order from one seeded stream."""
    rng = np.random.default_rng(SEED)
    cases = []
    for device, count in SPLIT:
        ranges = {p.name: p for p in device_parameters(device)}
        for index in range(1, count + 1):
            redraws = 0
            while True:
                knobs = {}
                for name in PHYSICAL[device]:
                    p = ranges[name]
                    assert p.low is not None and p.high is not None, name
                    knobs[name] = _draw(rng, p.low, p.high, p.axis)
                if _builds(device, knobs):
                    break
                redraws += 1
            name = f"{PREFIX[device]}{index:03d}"
            cases.append(Case(name, device, knobs, redraws))
    return cases


def write_cases() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cases.csv"
    lines = [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py cases",
        f"# seed {SEED}, split {SPLIT}, {SIGNIFICANT} significant digits",
        "# knobs not listed keep the constructor default",
        "case,device,knobs,redraws",
    ]
    for case in draw_cases():
        knobs = json.dumps(case.knobs).replace('"', '""')
        lines.append(f'{case.name},{case.device},"{knobs}",{case.redraws}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


THREAD_VARIABLES = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")

FINE_STEP = 0.02
"""The reference ramp's step, as a fraction of every applied bias [1]."""

BIAS = {"pn_diode": "anode_voltage", "mos_cap": "gate_voltage", "nmos": "gate_voltage"}
"""The knob each device's public sweep walks to."""

MEASURED = {"pn_diode": "anode", "nmos": "drain"}
"""The terminal whose current is the case's value. A MOS capacitor's value is
its capacitance instead."""


@dataclass(frozen=True)
class Result:
    case: str
    driver: str
    converged: bool
    value: float
    """Terminal current [A/cm^2 in 1D, A/cm in 2D], or capacitance [F/cm^2]."""
    imbalance: float
    """|sum of terminal currents|, in the value's current unit. Zero for a
    MOS capacitor, whose terminal charges balance identically."""
    largest: float
    """The largest |terminal current|, same unit."""
    seconds: float
    message: str


def read_cases() -> list[Case]:
    lines = (OUT / "cases.csv").read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if not line.startswith("#")]
    return [
        Case(row["case"], row["device"], json.loads(row["knobs"]), int(row["redraws"]))
        for row in csv.DictReader(body)
    ]


def _models(device: Device, kind: str) -> TransportModels:
    if kind == "nmos":
        # The Phase 5 stack benchmark 10 runs, not the constant mobility default.
        return TransportModels.for_device(
            device, mobility="arora", field_dependent=True, surface=True
        )
    return TransportModels.for_device(device)


def _public(device: Device, case: Case, models: TransportModels) -> DeviceState | None:
    """The sweep the API runs, from equilibrium to the case's bias."""
    target = case.knobs[BIAS[case.device]]
    if case.device == "pn_diode":
        curve = iv_sweep(device, "anode", [target], models=models)
    else:
        curve = gate_sweep(device, [target], models=models)
    return curve.points[-1].state if curve.complete else None


def solve_case(case: Case, fine: bool) -> Result:
    driver = "ddsim_fine" if fine else "ddsim"
    device = build_from_spec(case.device, case.knobs)
    started = time.perf_counter()
    try:
        if case.device == "mos_cap":
            # Equilibrium Poisson has one solution, so there is no path to get
            # wrong and the reference is the solve itself.
            cv = cv_sweep(device, "gate", [case.knobs["gate_voltage"]])
            seconds = time.perf_counter() - started
            value = cv.points[-1].capacitance if cv.complete else math.nan
            return Result(
                case.name, driver, cv.complete, value, 0.0, 0.0, seconds, cv.message
            )
        models = _models(device, case.device)
        if fine:
            state: DeviceState | None = solve_bias_ramped(
                device, models, step=FINE_STEP
            )
        else:
            state = _public(device, case, models)
        seconds = time.perf_counter() - started
    except Exception as failure:  # noqa: BLE001 - every failure is a result here
        seconds = time.perf_counter() - started
        return Result(
            case.name,
            driver,
            False,
            math.nan,
            math.nan,
            math.nan,
            seconds,
            str(failure)[:200],
        )
    if state is None:
        return Result(
            case.name,
            driver,
            False,
            math.nan,
            math.nan,
            math.nan,
            seconds,
            "sweep stopped early",
        )
    currents = terminal_currents(device, state, models)
    return Result(
        case.name,
        driver,
        True,
        currents[MEASURED[case.device]],
        abs(sum(currents.values())),
        max(abs(c) for c in currents.values()),
        seconds,
        "",
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def run(names: list[str] | None, path: Path) -> Path:
    unpinned = [v for v in THREAD_VARIABLES if os.environ.get(v) != "1"]
    if unpinned:
        raise SystemExit(
            f"set {', '.join(unpinned)} to 1 first, see the module docstring"
        )
    cases = [c for c in read_cases() if names is None or c.name in names]
    header = [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py run",
        f"# ddsim {_git_sha()}, python {platform.python_version()}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        f"# fine step {FINE_STEP} of every bias, from zero",
        "case,driver,converged,value,imbalance,largest,seconds,message",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for case in cases:
            for fine in (False, True):
                r = solve_case(case, fine)
                message = r.message.replace('"', "'").replace("\n", " ")
                f.write(
                    f"{r.case},{r.driver},{r.converged},{r.value!r},{r.imbalance!r},"
                    f'{r.largest!r},{r.seconds:.3f},"{message}"\n'
                )
                f.flush()
                print(
                    f"{r.case} {r.driver} {r.converged} {r.value:.4g} {r.seconds:.1f}s"
                )
    return path


TOLERANCE = {"pn_diode": 0.02, "mos_cap": 0.02, "nmos": 0.05}
"""Agreement required per family [1]: the Tier 4 targets of benchmarks 1, 4
and 6 to 8."""

DIODE_FLOOR = 1e-10
"""Diode current carrying no information [A/cm^2]. CURRENT_FLOOR in
tests/regression/devsim_gen/parameters.py, measured there."""

MOSFET_FLOOR = 1e-5
"""FULL_STACK_FLOOR of test_devsim_mosfet.py: under this fraction of the
extraction target a drain current is compared against the floor [1]."""

MOSFET_BALANCE = 1e-4
"""LOAD_BEARING of test_devsim_mosfet.py: above this fraction of the target a
drain current has to balance [1]."""

NOISE_FACTOR = 3.0
"""How much of a result's own imbalance counts as allowance [1], as in Tier 4."""

REFERENCE_DRIVERS = ("ddsim_fine", "devsim_fine")


def floor(case: Case) -> float:
    """Under this |value| a result says nothing beyond "about zero"."""
    if case.device == "pn_diode":
        return DIODE_FLOOR
    if case.device == "nmos":
        return MOSFET_FLOOR * REFERENCE_CURRENT / case.knobs["L_gate"]
    return 0.0


def balance_line(case: Case) -> float:
    """Above this |value| a result's terminal currents have to cancel."""
    if case.device == "pn_diode":
        return DIODE_FLOOR
    if case.device == "nmos":
        return MOSFET_BALANCE * REFERENCE_CURRENT / case.knobs["L_gate"]
    return math.inf


def valid(case: Case, result: Result) -> bool:
    if not result.converged or not math.isfinite(result.value):
        return False
    if result.largest < balance_line(case):
        return True
    return result.imbalance <= 0.1 * TOLERANCE[case.device] * result.largest


def agree(case: Case, a: Result, b: Result) -> bool:
    low = floor(case)
    if abs(a.value) < low and abs(b.value) < low:
        return True
    allowed = TOLERANCE[case.device] * max(abs(a.value), abs(b.value))
    allowed += NOISE_FACTOR * max(a.imbalance, b.imbalance)
    return abs(a.value - b.value) <= allowed


@dataclass(frozen=True)
class Score:
    passes: dict[str, int]
    """Cases each driver passed, over the cases not dropped."""
    dropped: dict[str, str]
    """Case name to the reason it counts for nobody."""
    counted: int
    """Cases scored."""


def score(cases: list[Case], results: list[Result]) -> Score:
    by_case: dict[str, dict[str, Result]] = {}
    for r in results:
        by_case.setdefault(r.case, {})[r.driver] = r
    passes: dict[str, int] = {}
    dropped: dict[str, str] = {}
    for case in cases:
        got = by_case.get(case.name, {})
        references = [
            got[d] for d in REFERENCE_DRIVERS if d in got and valid(case, got[d])
        ]
        if not references:
            dropped[case.name] = "no valid reference"
            continue
        if not all(agree(case, a, b) for a in references for b in references):
            dropped[case.name] = "the references disagree"
            continue
        for driver, r in got.items():
            if valid(case, r) and all(agree(case, r, ref) for ref in references):
                passes[driver] = passes.get(driver, 0) + 1
    return Score(passes, dropped, len(cases) - len(dropped))


REFINEMENTS = (1.0, 1.5, 2.25)
"""Mesh refinement factors for the accuracy axis: every spacing divided by
these. A ratio of 1.5 rather than 2 keeps the finest 2D mesh at five times the
reference's nodes instead of sixteen."""


@dataclass(frozen=True)
class Fit:
    limit: float | None
    """The Richardson extrapolated value, None outside the asymptotic range."""
    order: float | None
    """Observed order of convergence in h [1]."""
    error: float | None
    """Relative error of the reference mesh against the limit [1]."""
    nodes_for_one_percent: float | None
    """Silicon nodes at which the error model reaches 1 percent [1]."""


def richardson(levels: list[tuple[float, int, float]], dimension: int) -> Fit:
    """Extrapolate three (refinement, nodes, value) levels to zero spacing.

    Assumes value = limit + C h^p on a constant refinement ratio. If the two
    moves have opposite signs or don't shrink, the meshes aren't in the
    asymptotic range and there is no honest limit to report, so all four
    fields come back None rather than a number that means nothing.
    """
    (r1, n1, q1), (r2, _, q2), (r3, _, q3) = sorted(levels)
    ratio = r2 / r1
    d1, d2 = q1 - q2, q2 - q3
    if d1 * d2 <= 0.0 or abs(d2) >= abs(d1):
        return Fit(None, None, None, None)
    order = math.log(d1 / d2) / math.log(ratio)
    limit = q3 - d2 / (ratio**order - 1.0)
    error = abs(q1 - limit) / abs(limit)
    nodes = n1 * (error / 0.01) ** (dimension / order)
    return Fit(limit, order, error, nodes)


ACCURACY = (
    (1, "diode_1e16_1e16"),
    (2, "diode_1e18_1e16"),
    (3, "diode_1e20_1e15"),
    (4, "mos_cap_5nm"),
    (5, "mos_cap_20nm"),
    (6, "nmos_1um"),
    (7, "nmos_180nm"),
    (8, "nmos_65nm"),
    (9, "rolloff_100nm"),
    (10, "fullstack_100nm"),
)
"""One device per Tier 4 benchmark, by golden file stem. Benchmark 9 is five
devices; its 100 nm one stands for it, as does benchmark 10's."""

ACCURACY_BIAS: dict[str, Any] = {"diode": 0.6, "mos_cap": 0.0, "mosfet": (1.0, 0.05)}
"""Where each family's quantity is read: diode anode current at 0.6 V, MOS
capacitor low frequency capacitance at 0 V of gate, MOSFET drain current at
1.0 V of gate and 50 mV of drain, all above every floor [V]."""


def _scaled(nodes: int, r: float) -> int:
    """A node count with every spacing divided by r."""
    return round((nodes - 1) * r) + 1


def silicon_nodes(device: Device) -> int:
    """Nodes carrying all three unknowns, the count both tools can agree on."""
    if device.regions is None:
        return int(device.mesh.n_nodes)
    return int(np.count_nonzero(device.regions.semiconductor_volume > 0.0))


def accuracy_point(name: str, r: float) -> tuple[int, float]:
    """One benchmark's quantity on its reference mesh refined by r."""
    sys.path.insert(0, str(ROOT))
    import inspect

    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.mosfet import nmos
    from ddsim.device.pn_diode import pn_diode
    from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
    from tests.regression.devsim_gen import parameters as P

    if name.startswith("diode"):
        b = P.BY_NAME[name]
        device = pn_diode(
            Na=b.Na,
            Nd=b.Nd,
            length=b.length,
            junction=b.junction,
            n_nodes=_scaled(b.n_nodes, r),
            h_min=b.h_min / r,
        )
        curve = iv_sweep(device, "anode", [ACCURACY_BIAS["diode"]])
        return silicon_nodes(device), float(curve.current[-1])
    if name.startswith("mos_cap"):
        m = {b.name: b for b in P.MOS_BENCHMARKS}[name]
        device = mos_cap(
            substrate_doping=m.substrate_doping,
            t_ox=m.t_ox,
            t_si=m.t_si,
            n_silicon=_scaled(m.n_silicon, r),
            n_oxide=_scaled(m.n_oxide, r),
            h_min=m.h_min / r,
            work_function=m.work_function,
        )
        cv = cv_sweep(device, "gate", [ACCURACY_BIAS["mos_cap"]])
        # mos_cap solves a 1D stack on a few identical columns. Count one
        # column, the same way DEVSIM's 1D mesh counts it.
        columns = device.mesh.nx  # type: ignore[union-attr]
        return silicon_nodes(device) // columns, float(cv.points[-1].capacitance)
    f = P.MOSFET_BY_NAME[name]
    full = f.models == P.FULL_MODELS
    defaults = inspect.signature(nmos).parameters
    mesh: dict[str, Any] = {
        k: _scaled(defaults[k].default, r)
        for k in ("n_contact", "n_sd", "n_channel", "n_silicon", "n_oxide")
    }
    mesh["h_min_x"] = defaults["h_min_x"].default / r
    mesh["h_min_y"] = defaults["h_min_y"].default / r
    gate, drain = ACCURACY_BIAS["mosfet"]
    device = nmos(
        L_gate=f.L_gate,
        drain_voltage=drain,
        degenerate=full,
        **SHORT_CHANNEL_PROCESS,
        **mesh,
    )
    models = (
        TransportModels.for_device(
            device, mobility="arora", field_dependent=True, surface=True
        )
        if full
        else TransportModels.for_device(device, mobility="constant")
    )
    curve = gate_sweep(device, [gate], models=models)
    return silicon_nodes(device), float(curve.current[-1])


def run_accuracy(path: Path) -> Path:
    header = [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py",
        f"# ddsim {_git_sha()}, refinements {REFINEMENTS}, biases {ACCURACY_BIAS}",
        "benchmark,name,refine,nodes,value,seconds",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for number, name in ACCURACY:
            for r in REFINEMENTS:
                started = time.perf_counter()
                nodes, value = accuracy_point(name, r)
                seconds = time.perf_counter() - started
                f.write(f"{number},{name},{r},{nodes},{value!r},{seconds:.3f}\n")
                f.flush()
                print(f"{name} r={r} {nodes} nodes {value:.8g} {seconds:.1f}s")
    return path


SPEED_RUNS = 5
"""Timed repeats per benchmark sweep. The median is reported."""

SPEED = (
    (1, "diode_1e16_1e16"),
    (2, "diode_1e18_1e16"),
    (3, "diode_1e20_1e15"),
    (4, "mos_cap_5nm"),
    (5, "mos_cap_20nm"),
    (6, "nmos_1um"),
    (7, "nmos_180nm"),
    (8, "nmos_65nm"),
)
"""Benchmark sweeps 1 to 8, exactly as the Tier 4 tests run them."""


def speed_sweep(name: str) -> int:
    """Run one benchmark's full Tier 4 sweep and return the points reached."""
    sys.path.insert(0, str(ROOT))
    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.mosfet import nmos
    from ddsim.device.pn_diode import pn_diode
    from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
    from tests.regression.devsim_gen import parameters as P

    if name.startswith("diode"):
        b = P.BY_NAME[name]
        device = pn_diode(
            Na=b.Na,
            Nd=b.Nd,
            length=b.length,
            junction=b.junction,
            n_nodes=b.n_nodes,
            h_min=b.h_min,
        )
        return len(iv_sweep(device, "anode", list(b.voltages), step=0.05).points)
    if name.startswith("mos_cap"):
        m = {b.name: b for b in P.MOS_BENCHMARKS}[name]
        device = mos_cap(
            substrate_doping=m.substrate_doping,
            t_ox=m.t_ox,
            t_si=m.t_si,
            n_silicon=m.n_silicon,
            n_oxide=m.n_oxide,
            h_min=m.h_min,
            work_function=m.work_function,
        )
        return len(cv_sweep(device, "gate", list(m.voltages)).points)
    f = P.MOSFET_BY_NAME[name]
    points = 0
    for drain in (f.drain_low, f.drain_high):
        device = nmos(
            L_gate=f.L_gate,
            drain_voltage=drain,
            degenerate=False,
            **SHORT_CHANNEL_PROCESS,
        )
        models = TransportModels.for_device(device, mobility="constant")
        points += len(gate_sweep(device, list(f.gate_voltages), models=models).points)
    return points


def run_speed(path: Path) -> Path:
    unpinned = [v for v in THREAD_VARIABLES if os.environ.get(v) != "1"]
    if unpinned:
        raise SystemExit(f"set {', '.join(unpinned)} to 1 first, see the docstring")
    header = [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py",
        f"# ddsim {_git_sha()}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        "benchmark,name,run,points,seconds",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for number, name in SPEED:
            for run_index in range(1, SPEED_RUNS + 1):
                started = time.perf_counter()
                points = speed_sweep(name)
                seconds = time.perf_counter() - started
                f.write(f"{number},{name},{run_index},{points},{seconds:.3f}\n")
                f.flush()
                print(f"{name} run {run_index}: {points} points {seconds:.1f}s")
    return path


def read_results(path: Path) -> list[Result]:
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if not line.startswith("#")]
    return [
        Result(
            row["case"],
            row["driver"],
            row["converged"] == "True",
            float(row["value"]),
            float(row["imbalance"]),
            float(row["largest"]),
            float(row["seconds"]),
            row["message"],
        )
        for row in csv.DictReader(body)
    ]


def _rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader(line for line in lines if not line.startswith("#")))


def read_accuracy(path: Path) -> dict[str, list[tuple[float, int, float]]]:
    levels: dict[str, list[tuple[float, int, float]]] = {}
    for row in _rows(path):
        level = (float(row["refine"]), int(row["nodes"]), float(row["value"]))
        levels.setdefault(row["name"], []).append(level)
    return levels


def dimension(name: str) -> int:
    """Diodes and MOS capacitors are 1D problems, the MOSFETs 2D."""
    return 1 if name.startswith(("diode", "mos_cap")) else 2


def read_speed(path: Path) -> dict[str, float]:
    """Median seconds per converged bias point, per benchmark."""
    per_point: dict[str, list[float]] = {}
    for row in _rows(path):
        seconds = float(row["seconds"]) / int(row["points"])
        per_point.setdefault(row["name"], []).append(seconds)
    return {name: float(np.median(values)) for name, values in per_point.items()}


def _capabilities() -> list[dict[str, str]]:
    return _rows(OUT / "capabilities.csv")


@dataclass(frozen=True)
class Board:
    robustness: Score
    ddsim_fit: dict[str, Fit]
    devsim_fit: dict[str, Fit]
    ddsim_speed: dict[str, float]
    devsim_speed: dict[str, float]
    capabilities: list[dict[str, str]]


def load_board() -> Board:
    results = read_results(OUT / "ddsim_robustness.csv") + read_results(
        OUT / "devsim_robustness.csv"
    )
    fits = []
    for tool_name in ("ddsim", "devsim"):
        levels = read_accuracy(OUT / f"{tool_name}_accuracy.csv")
        fits.append({n: richardson(lv, dimension(n)) for n, lv in levels.items()})
    return Board(
        score(read_cases(), results),
        fits[0],
        fits[1],
        read_speed(OUT / "ddsim_speed.csv"),
        read_speed(OUT / "devsim_speed.csv"),
        _capabilities(),
    )


def _fewer_nodes(board: Board) -> tuple[int, int, int]:
    """Benchmarks where each tool reaches 1 percent on fewer nodes, and ties."""
    ours = theirs = undecided = 0
    for _, name in ACCURACY:
        a = board.ddsim_fit[name].nodes_for_one_percent
        b = board.devsim_fit[name].nodes_for_one_percent
        if a is None or b is None:
            undecided += 1
        elif a < b:
            ours += 1
        else:
            theirs += 1
    return ours, theirs, undecided


def _speed_ratio(board: Board) -> float:
    """Geometric mean over benchmarks 1 to 8 of ddsim's time over DEVSIM's."""
    ratios = [board.ddsim_speed[n] / board.devsim_speed[n] for _, n in SPEED]
    return float(np.exp(np.mean(np.log(ratios))))


def _estimates(board: Board) -> tuple[str, str]:
    row = {r["capability"]: r for r in board.capabilities}
    has = row["Discretization error estimates"]
    return (
        "every result" if has["ddsim"] == "yes" else "none",
        "every result" if has["devsim"] == "yes" else "none",
    )


def headline(board: Board) -> str:
    s = board.robustness
    ours, theirs, undecided = _fewer_nodes(board)
    ratio = _speed_ratio(board)
    caps = board.capabilities
    ddsim_yes = sum(r["ddsim"] == "yes" for r in caps)
    devsim_yes = sum(r["devsim"] == "yes" for r in caps)
    scripted = sum(r["devsim"] == "scripted" for r in caps)
    estimates = _estimates(board)
    speed = f"{ratio:.2g}x DEVSIM's" if ratio >= 1.0 else f"{1.0 / ratio:.2g}x faster"
    lines = [
        "| Axis | DDSim | DEVSIM 2.11 |",
        "|---|---|---|",
        f"| Robustness: cold solves passed, of {s.counted} scored | "
        f"{s.passes.get('ddsim', 0)} | stock ramp {s.passes.get('devsim_stock', 0)}, "
        f"my ramp {s.passes.get('devsim_expert', 0)} |",
        f"| Accuracy: benchmarks reaching 1% on fewer nodes, of {len(ACCURACY)} | "
        f"{ours} | {theirs} ({undecided} not in the asymptotic range) |",
        f"| Error estimates reported | {estimates[0]} | {estimates[1]} |",
        f"| Speed: time per bias point, benchmarks 1 to 8 | {speed} | 1x |",
        f"| Capabilities, of {len(caps)} rows | {ddsim_yes} | {devsim_yes} built in, "
        f"{scripted} if you write the equations |",
    ]
    return "\n".join(lines)


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def details(board: Board) -> str:
    s = board.robustness
    drivers = ("ddsim", "ddsim_fine", "devsim_stock", "devsim_expert", "devsim_fine")
    lines = [
        "# Scoreboard details",
        "",
        "Generated by `tools/scoreboard.py summary` from the CSVs in this folder.",
        "Don't edit it by hand: tests/regression/test_scoreboard.py regenerates",
        "it and fails if this file is stale. The rules are in phases/PHASE-8.md",
        "and in the docstring of tools/scoreboard.py.",
        "",
        "## Robustness",
        "",
        f"{s.counted} of {s.counted + len(s.dropped)} cases scored.",
        "",
        "| Driver | Passed |",
        "|---|---|",
    ]
    lines += [f"| {d} | {s.passes.get(d, 0)} |" for d in drivers]
    lines += ["", "Dropped cases, which count for nobody:", ""]
    lines += [f"- {case}: {reason}" for case, reason in sorted(s.dropped.items())]
    if not s.dropped:
        lines.append("- none")
    lines += [
        "",
        "## Accuracy",
        "",
        "Relative error of each tool's reference mesh against its own Richardson",
        "limit, the observed order, and the silicon nodes its error model needs",
        "for 1 percent. The last column is how far apart the two limits are.",
        "",
        "| # | Device | DDSim nodes | DDSim error | DDSim order | DDSim nodes "
        "for 1% | DEVSIM nodes | DEVSIM error | DEVSIM order | DEVSIM nodes "
        "for 1% | Limits differ by |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ddsim_levels = read_accuracy(OUT / "ddsim_accuracy.csv")
    devsim_levels = read_accuracy(OUT / "devsim_accuracy.csv")
    for number, name in ACCURACY:
        a, b = board.ddsim_fit[name], board.devsim_fit[name]
        gap = (
            None
            if a.limit is None or b.limit is None
            else abs(a.limit - b.limit) / abs(b.limit)
        )
        lines.append(
            f"| {number} | {name} | {min(ddsim_levels[name])[1]} | "
            f"{_number(a.error)} | {_number(a.order)} | "
            f"{_number(a.nodes_for_one_percent)} | {min(devsim_levels[name])[1]} | "
            f"{_number(b.error)} | {_number(b.order)} | "
            f"{_number(b.nodes_for_one_percent)} | {_number(gap)} |"
        )
    lines += [
        "",
        "## Speed",
        "",
        f"Median of {SPEED_RUNS} runs, seconds per converged bias point, one",
        "BLAS thread each.",
        "",
        "| # | Benchmark | DDSim | DEVSIM | Ratio |",
        "|---|---|---|---|---|",
    ]
    for number, name in SPEED:
        ours_s, theirs_s = board.ddsim_speed[name], board.devsim_speed[name]
        lines.append(
            f"| {number} | {name} | {ours_s:.3g} | {theirs_s:.3g} | "
            f"{ours_s / theirs_s:.2g} |"
        )
    lines += ["", "## Capabilities", "", "See capabilities.csv.", ""]
    return "\n".join(lines)


START, END = "<!-- scoreboard:start -->", "<!-- scoreboard:end -->"
"""The markers around the generated table in the repository README."""


def write_summary() -> None:
    board = load_board()
    (OUT / "README.md").write_text(details(board), encoding="utf-8", newline="\n")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(
        before + START + "\n" + headline(board) + "\n" + END + after, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = ["cases", "run", "accuracy", "speed", "summary"]
    parser.add_argument("command", choices=commands)
    parser.add_argument("names", nargs="*", help="run only these cases")
    parser.add_argument("--out", type=Path, default=OUT / "ddsim_robustness.csv")
    args = parser.parse_args()
    if args.command == "cases":
        print(write_cases())
    elif args.command == "accuracy":
        print(run_accuracy(OUT / "ddsim_accuracy.csv"))
    elif args.command == "speed":
        print(run_speed(OUT / "ddsim_speed.csv"))
    elif args.command == "summary":
        write_summary()
    else:
        print(run(args.names or None, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
