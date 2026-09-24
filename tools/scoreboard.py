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
in 2 percent steps. It writes the raw numbers into `ddsim_robustness.csv`. The
pass rule lives in the test, not here, so the rule can be argued about without
solving anything again. It refuses to run unless the BLAS thread counts are
pinned to 1, since the speed axis compares one thread against one thread:

    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        .venv/Scripts/python.exe tools/scoreboard.py run
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["cases", "run"])
    parser.add_argument("names", nargs="*", help="run only these cases")
    parser.add_argument("--out", type=Path, default=OUT / "ddsim_robustness.csv")
    args = parser.parse_args()
    if args.command == "cases":
        print(write_cases())
    else:
        print(run(args.names or None, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
