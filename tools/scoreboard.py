"""The ddsim side of the Phase 8 scoreboard. See phases/PHASE-8.md.

Run by hand from the repository root, like the DEVSIM generators, and commit
what it writes into `data/scoreboard/`:

    .venv/Scripts/python.exe tools/scoreboard.py cases

`cases` draws the robustness case set: 200 devices with a target bias, drawn
with a fixed seed from the knob ranges the constructors declare in their
docstrings. tests/regression/test_scoreboard.py redraws it and fails if the
committed file differs, so a moved range can't leave a stale case set behind.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ddsim.api.devices import build_from_spec, device_parameters

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["cases"])
    args = parser.parse_args()
    if args.command == "cases":
        print(write_cases())
    return 0


if __name__ == "__main__":
    sys.exit(main())
