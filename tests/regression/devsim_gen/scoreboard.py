"""The DEVSIM side of the Phase 8 scoreboard.

Runs under `.venv-devsim`, like the other generators here, and writes into
`data/scoreboard/`. See phases/PHASE-8.md.

    .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/scoreboard.py api

`api` writes `devsim_api.txt`: every public name devsim exports, the solve
types and linear solvers its docstring lists as `solve:<type>` and
`solver:<type>`, the element types its Gmsh import accepts as
`create_gmsh_mesh:<element>`, and every function in its bundled
`python_packages` as `python_packages.<module>.<function>`. The capability
matrix cites these names, and test_scoreboard.py checks each citation against
this file, so a claim about what DEVSIM can do is a claim about this snapshot
rather than about my memory of its manual.

`run` solves the robustness cases in `cases.csv` three ways and writes
`devsim_robustness.csv`, in the same columns as ddsim's file:

- `devsim_stock`: DEVSIM's own `python_packages/ramp.py` walks the bias, one
  solve per step, halving on failure and never growing back.
- `devsim_expert`: the driver that produced the golden data for that device
  family, which is the best I know how to write for DEVSIM.
- `devsim_fine`: the expert driver with its step held ten times smaller, as
  the reference.

All three share the physics, the mesh, the equilibrium start and the solver
tolerances, so what differs between them is only how the bias gets walked.
Every device is deleted after its case, since devsim solves every live device
at once (see README.md here).

    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/scoreboard.py run
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import json
import math
import os
import platform
import re
import sys
import time
from typing import Any

import devsim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "data", "scoreboard")

sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import generate_diodes as GD  # noqa: E402
import generate_mos_cv as GC  # noqa: E402
import generate_mosfet as GM  # noqa: E402
import parameters as P  # noqa: E402
from devsim.python_packages.ramp import rampbias  # noqa: E402


def _choices(doc: str | None, argument: str) -> list[str]:
    """The {a, b, c} choices a devsim docstring lists for one argument."""
    match = re.search(rf"\b{argument} : \{{([^}}]*)\}}", doc or "")
    if match is None:
        raise RuntimeError(f"devsim's docstring no longer lists choices for {argument}")
    return [t.strip().strip("'") for t in match.group(1).split(",")]


def solve_types() -> list[str]:
    """solve's `type` and `solver_type` choices, e.g. solve:ac, solver:iterative."""
    doc = devsim.solve.__doc__
    return [f"solve:{t}" for t in _choices(doc, "type")] + [
        f"solver:{t}" for t in _choices(doc, "solver_type")
    ]


def gmsh_elements() -> list[str]:
    """The element types create_gmsh_mesh's docstring accepts, e.g. tetrahedron."""
    found = re.findall(r"- \d+ (\w+)", devsim.create_gmsh_mesh.__doc__ or "")
    if "triangle" not in found:
        raise RuntimeError("create_gmsh_mesh's docstring no longer lists element types")
    return [f"create_gmsh_mesh:{e}" for e in found]


def package_functions() -> list[str]:
    """Top level functions in devsim/python_packages, read from source."""
    import devsim.python_packages as packages

    folder = os.path.dirname(packages.__file__)
    names = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        module = filename[:-3]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                names.append(f"python_packages.{module}.{node.name}")
    return names


def write_api() -> str:
    public = sorted(n for n in dir(devsim) if not n.startswith("_"))
    today = datetime.date.today().isoformat()
    lines = [
        f"# devsim {devsim.__version__}",
        f"# written {today} by devsim_gen/scoreboard.py api",
        "# public names, solve types, gmsh elements, python_packages functions",
    ]
    lines += public
    lines += solve_types()
    lines += gmsh_elements()
    lines += package_functions()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "devsim_api.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


THREAD_VARIABLES = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
DRIVERS = ("devsim_stock", "devsim_expert", "devsim_fine")
FINE = 10.0
"""How much smaller the reference's step is than the expert's [1]."""

CV_STEP = 0.005
"""Half the gate span of DEVSIM's central difference capacitance [V]. ddsim
takes the derivative exactly, so this is kept small enough that its
truncation error sits far under the 2 percent the comparison allows."""


def read_cases() -> list[dict[str, Any]]:
    with open(os.path.join(OUT, "cases.csv"), encoding="utf-8") as f:
        body = [line for line in f.read().splitlines() if not line.startswith("#")]
    return [
        {
            "case": row["case"],
            "device": row["device"],
            "knobs": json.loads(row["knobs"]),
        }
        for row in csv.DictReader(body)
    ]


def _noop(device: str) -> None:
    return None


def _stock_ramp(device: str, contact: str, target: float, step: float) -> None:
    """DEVSIM's shipped ramp, at the expert driver's tolerances and budget."""
    rampbias(device, contact, target, step, 1e-4, 100, 1e-8, 1e30, _noop)


def _cleanup() -> None:
    """Delete every device and mesh. devsim solves every live device at once."""
    for name in list(devsim.get_device_list()):
        devsim.delete_device(device=name)
    for name in list(devsim.get_mesh_list()):
        devsim.delete_mesh(mesh=name)


def _diode(case: dict[str, Any], driver: str) -> tuple[float, float, float]:
    k = case["knobs"]
    bench = P.DiodeBenchmark(
        name=f"{case['case']}_{driver}",
        number=0,
        Na=k["Na"],
        Nd=k["Nd"],
        length=k["length"],
        junction=k["junction"],
        voltages=(k["anode_voltage"],),
        tolerance=0.02,
    )
    device = bench.name
    target = k["anode_voltage"]
    with GM.quiet():
        GD.build_mesh(bench, device)
        GD.set_doping(bench, device)
        GD.set_silicon_parameters(device)
        GD.build_physics(device)
        if driver == "devsim_stock":
            _stock_ramp(device, GD.ANODE, target, 0.05)
        else:
            step = 0.05 if driver == "devsim_expert" else 0.05 / FINE
            GD.ramp_to(device, target, 0.0, step=step)
    anode = GD.anode_current(device)
    cathode = GD.cathode_current(device)
    return anode, abs(anode + cathode), max(abs(anode), abs(cathode))


def _mos_cap(case: dict[str, Any], driver: str) -> tuple[float, float, float]:
    k = case["knobs"]
    bench = P.MosBenchmark(
        name=f"{case['case']}_{driver}",
        number=0,
        substrate_doping=k["substrate_doping"],
        t_ox=k["t_ox"],
        t_si=k["t_si"],
        work_function=k["work_function"],
        voltages=(k["gate_voltage"],),
        tolerance=0.02,
    )
    device = bench.name
    target = k["gate_voltage"]

    def gate(v: float) -> None:
        devsim.set_parameter(
            device=device, name=f"{GC.GATE}_bias", value=GC.gate_potential(bench, v)
        )
        devsim.solve(
            type="dc",
            absolute_error=1e-10,
            relative_error=1e-12,
            maximum_iterations=100,
        )

    with GM.quiet():
        GC.build_mesh(bench, device)
        GC.set_material_parameters(device)
        GC.build_physics(bench, device)
        start = target - CV_STEP
        if driver == "devsim_stock":
            gate(0.0)
            end = GC.gate_potential(bench, start)
            rampbias(device, GC.GATE, end, 0.1, 1e-4, 100, 1e-12, 1e-10, _noop)
        elif driver == "devsim_expert":
            gate(start)
        else:
            steps = max(1, math.ceil(abs(start) / 0.01))
            for index in range(1, steps + 1):
                gate(start * index / steps)
        charges = []
        for v in (start, target + CV_STEP):
            gate(v)
            charges.append(
                devsim.get_contact_charge(
                    device=device, contact=GC.GATE, equation="PotentialEquation"
                )
            )
    capacitance = (charges[1] - charges[0]) / (2.0 * CV_STEP)
    # A C-V point has no current to balance, and the body contact's charge is
    # not the silicon's, so both columns are zero as they are in ddsim's file.
    return capacitance, 0.0, 0.0


def _nmos(case: dict[str, Any], driver: str) -> tuple[float, float, float]:
    k = case["knobs"]
    process = dict(P.MOSFET_PROCESS)
    for key in ("substrate_doping", "sd_peak", "x_j", "lateral_diffusion", "t_ox"):
        process[key] = k[key]
    L = k["L_gate"]
    h_surface = 6.25e-9
    bench = P.MosfetBenchmark(
        name=f"{case['case']}_{driver}",
        number=0,
        L_gate=L,
        gate_voltages=(k["gate_voltage"],),
        drain_low=k["drain_voltage"],
        drain_high=k["drain_voltage"],
        tolerance=0.05,
        devsim_h_junction=min(L / 80.0, 5e-7),
        devsim_h_channel=min(L / 40.0, 2e-6),
        devsim_h_surface=h_surface,
        devsim_h_depth=P.implant_shape(process)[0] * P.H_DEPTH_SIGMAS,
        devsim_oxide_cells=min(64, max(4, round(process["t_ox"] / h_surface))),
        models=P.FULL_MODELS,
    )
    device = bench.name
    wf = k["work_function"]
    gate = GM.gate_potential(k["gate_voltage"], wf)
    drain = k["drain_voltage"]
    GM._surface_is_live = False
    try:
        with GM.quiet():
            GM.build_mesh(bench, device, process=process)
            GM.set_material_parameters(device)
            GM.set_doping(bench, device, process=process)
            # The golden transfer curves start every device at zero gate, put
            # the drain on, then walk the gate. All three drivers keep that
            # order, so the only thing that differs is the walk itself.
            if driver == "devsim_expert":
                GM.build_physics(device, 0.0, drain, P.FULL_MODELS, wf)
                GM.ramp_to(device, GM.GATE, gate)
            else:
                GM.build_physics(device, 0.0, 0.0, P.FULL_MODELS, wf)
                if driver == "devsim_stock":
                    _stock_ramp(device, GM.DRAIN, drain, 0.1)
                    _stock_ramp(device, GM.GATE, gate, 0.1)
                else:
                    small = 0.1 / FINE
                    GM.ramp_to(device, GM.DRAIN, drain, step=small, max_step=small)
                    GM.ramp_to(device, GM.GATE, gate, step=small, max_step=small)
            GM.settle(device, balance_tol=GM.BALANCE_TOL)
        currents = [
            GM.terminal_current(device, c) for c in (GM.DRAIN, GM.SOURCE, GM.BODY)
        ]
    finally:
        GM._surface_is_live = False
    return currents[0], abs(sum(currents)), max(abs(c) for c in currents)


SOLVERS = {"pn_diode": _diode, "mos_cap": _mos_cap, "nmos": _nmos}


def _done(path: str) -> set[tuple[str, str]]:
    """The (case, driver) pairs a previous run already wrote to path."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        body = [line for line in f.read().splitlines() if not line.startswith("#")]
    return {(row["case"], row["driver"]) for row in csv.DictReader(body)}


def run(names: list[str] | None, path: str, resume: bool = False) -> str:
    unpinned = [v for v in THREAD_VARIABLES if os.environ.get(v) != "1"]
    if unpinned:
        raise SystemExit(f"set {', '.join(unpinned)} to 1 first, see the docstring")
    cases = [c for c in read_cases() if names is None or c["case"] in names]
    done = _done(path) if resume else set()
    today = datetime.date.today().isoformat()
    header = [
        f"# written {today} by devsim_gen/scoreboard.py run",
        f"# devsim {devsim.__version__}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        f"# fine step {1.0 / FINE:g} of the expert step",
        "case,driver,converged,value,imbalance,largest,seconds,message",
    ]
    with open(path, "a" if done else "w", encoding="utf-8", newline="\n") as f:
        if not done:
            f.write("\n".join(header) + "\n")
        for case in cases:
            for driver in DRIVERS:
                if (case["case"], driver) in done:
                    continue
                started = time.perf_counter()
                try:
                    value, imbalance, largest = SOLVERS[case["device"]](case, driver)
                    converged, message = True, ""
                except Exception as failure:  # noqa: BLE001 - a failure is a result
                    value = imbalance = largest = math.nan
                    converged = False
                    message = str(failure).replace("\n", " ").replace('"', "'")[:200]
                finally:
                    _cleanup()
                seconds = time.perf_counter() - started
                f.write(
                    f"{case['case']},{driver},{converged},{value!r},{imbalance!r},"
                    f'{largest!r},{seconds:.3f},"{message}"\n'
                )
                f.flush()
                print(f"{case['case']} {driver} {converged} {value:.4g} {seconds:.1f}s")
    return path


REFINEMENTS = (1.0, 1.5, 2.25)
"""The accuracy axis's refinements, the same three tools/scoreboard.py uses."""

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
"""The same benchmark devices, quantities and biases as tools/scoreboard.py:
diode anode current at 0.6 V, MOS capacitor capacitance at 0 V, MOSFET drain
current at 1.0 V of gate and 50 mV of drain."""


def _silicon_nodes(device: str, region: str) -> int:
    return len(devsim.get_node_model_values(device=device, region=region, name="x"))


def accuracy_point(name: str, r: float) -> tuple[int, float]:
    """One benchmark's quantity on DEVSIM's reference mesh refined by r."""
    if name.startswith("diode"):
        bench = P.BY_NAME[name]
        device = f"{name}_acc"
        with GM.quiet():
            GD.build_mesh(bench, device, refine=r)
            GD.set_doping(bench, device)
            GD.set_silicon_parameters(device)
            GD.build_physics(device)
            GD.ramp_to(device, 0.6, 0.0, step=0.05)
        return _silicon_nodes(device, GD.REGION), GD.anode_current(device)
    if name.startswith("mos_cap"):
        bench = {b.name: b for b in P.MOS_BENCHMARKS}[name]
        device = f"{name}_acc"
        charges = []
        with GM.quiet():
            GC.build_mesh(bench, device, refine=r)
            GC.set_material_parameters(device)
            GC.build_physics(bench, device)
            for v in (-CV_STEP, CV_STEP):
                devsim.set_parameter(
                    device=device,
                    name=f"{GC.GATE}_bias",
                    value=GC.gate_potential(bench, v),
                )
                devsim.solve(
                    type="dc",
                    absolute_error=1e-10,
                    relative_error=1e-12,
                    maximum_iterations=100,
                )
                charges.append(
                    devsim.get_contact_charge(
                        device=device, contact=GC.GATE, equation="PotentialEquation"
                    )
                )
        capacitance = (charges[1] - charges[0]) / (2.0 * CV_STEP)
        return _silicon_nodes(device, GC.SILICON), capacitance
    import dataclasses

    # From zero gate up, the way the golden curves are walked.
    bench = dataclasses.replace(P.MOSFET_BY_NAME[name], gate_voltages=(0.0, 1.0))
    rows, nodes = GM.transfer_curve(bench, 0.05, refine=r)
    return nodes, rows[-1]["drain"]


def run_accuracy(path: str) -> str:
    today = datetime.date.today().isoformat()
    header = [
        f"# written {today} by devsim_gen/scoreboard.py accuracy",
        f"# devsim {devsim.__version__}, refinements {REFINEMENTS}",
        "benchmark,name,refine,nodes,value,seconds",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for number, name in ACCURACY:
            for r in REFINEMENTS:
                started = time.perf_counter()
                try:
                    nodes, value = accuracy_point(name, r)
                finally:
                    _cleanup()
                seconds = time.perf_counter() - started
                f.write(f"{number},{name},{r},{nodes},{value!r},{seconds:.3f}\n")
                f.flush()
                print(f"{name} r={r} {nodes} nodes {value:.8g} {seconds:.1f}s")
    return path


SPEED_RUNS = 5
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
"""Benchmark sweeps 1 to 8, exactly as the golden generators run them, with
no mesh convergence check."""


def speed_sweep(name: str) -> int:
    """Run one benchmark's golden sweep and return the points reached."""
    if name.startswith("diode"):
        with GM.quiet():
            return len(GD.sweep(P.BY_NAME[name]))
    if name.startswith("mos_cap"):
        bench = {b.name: b for b in P.MOS_BENCHMARKS}[name]
        with GM.quiet():
            return len(GC.sweep(bench))
    low, high, _ = GM.sweep(P.MOSFET_BY_NAME[name])
    return len(low) + len(high)


def run_speed(path: str) -> str:
    unpinned = [v for v in THREAD_VARIABLES if os.environ.get(v) != "1"]
    if unpinned:
        raise SystemExit(f"set {', '.join(unpinned)} to 1 first, see the docstring")
    today = datetime.date.today().isoformat()
    header = [
        f"# written {today} by devsim_gen/scoreboard.py speed",
        f"# devsim {devsim.__version__}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        "benchmark,name,run,points,seconds",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for number, name in SPEED:
            for run_index in range(1, SPEED_RUNS + 1):
                started = time.perf_counter()
                try:
                    points = speed_sweep(name)
                finally:
                    _cleanup()
                seconds = time.perf_counter() - started
                f.write(f"{number},{name},{run_index},{points},{seconds:.3f}\n")
                f.flush()
                print(f"{name} run {run_index}: {points} points {seconds:.1f}s")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["api", "run", "accuracy", "speed"])
    parser.add_argument("names", nargs="*", help="run only these cases")
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true", help="keep finished rows")
    args = parser.parse_args()
    if args.command == "api":
        print(write_api())
    elif args.command == "accuracy":
        print(run_accuracy(args.out or os.path.join(OUT, "devsim_accuracy.csv")))
    elif args.command == "speed":
        print(run_speed(args.out or os.path.join(OUT, "devsim_speed.csv")))
    else:
        out = args.out or os.path.join(OUT, "devsim_robustness.csv")
        print(run(args.names or None, out, resume=args.resume))
    return 0


if __name__ == "__main__":
    sys.exit(main())
