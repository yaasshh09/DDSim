from __future__ import annotations
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
from devsim import (
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    circuit_alter,
    circuit_element,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_circuit_node_value,
    node_model,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import CreateSolution
from devsim.python_packages.simple_physics import CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact
from generate_ac import SILICON, dc, devsim_version, drift_diffusion, label, lifetimes, quiet, ramp, silicon

PRELOAD = 1e-6


def anode() -> float:
    return float(get_circuit_node_value(node="anode_bias"))


def current() -> float:
    return -float(get_circuit_node_value(node="V1.I"))


def transient_dc() -> None:
    solve(type="transient_dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=60)


def step(first: bool, dt: float) -> None:
    if first:
        solve(type="transient_bdf1", absolute_error=1e10, relative_error=1e-8, maximum_iterations=60, tdelta=dt, charge_error=1e10)
    else:
        solve(type="transient_bdf2", absolute_error=1e10, relative_error=1e-8, maximum_iterations=60, tdelta=2.0 * dt, gamma=0.5, charge_error=1e10)


def recovery(b: P.TransientBenchmark, refine: float, fine: int) -> dict[str, Any]:
    dev = f"{b.name}_r{refine:g}_s{fine}".replace(".", "_")
    circuit_element(name="V1", n1="source", n2=0, value=0.0)
    circuit_element(name="R1", n1="source", n2="anode_bias", value=PRELOAD)
    create_1d_mesh(mesh=dev)
    add_1d_mesh_line(mesh=dev, pos=0.0, ps=2e-3 / refine, tag="top")
    add_1d_mesh_line(mesh=dev, pos=P.AC_DIODE_JUNCTION, ps=2e-7 / refine, ns=2e-7 / refine, tag="mid")
    add_1d_mesh_line(mesh=dev, pos=P.AC_DIODE_LENGTH, ps=2e-3 / refine, ns=2e-3 / refine, tag="bot")
    add_1d_contact(mesh=dev, name="anode", tag="top", material="metal")
    add_1d_contact(mesh=dev, name="cathode", tag="bot", material="metal")
    add_1d_region(mesh=dev, material="Silicon", region=SILICON, tag1="top", tag2="bot")
    finalize_mesh(mesh=dev)
    create_device(mesh=dev, device=dev)
    silicon(dev)
    node_model(device=dev, region=SILICON, name="NetDoping", equation=f"ifelse(x < {P.AC_DIODE_JUNCTION:.16e}, {-P.AC_DIODE_NA:.16e}, {P.AC_DIODE_ND:.16e})")
    lifetimes(dev, P.AcBenchmark(name=b.name, number=b.number, biases=(), frequencies=(), tolerance=b.tolerance, tau=b.tau), 0.0)
    CreateSolution(dev, SILICON, "Potential")
    CreateSiliconPotentialOnly(dev, SILICON)
    set_parameter(device=dev, name="cathode_bias", value=0.0)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "anode", is_circuit=True)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "cathode")
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
    drift_diffusion(dev, "anode", "cathode")
    ramp(b.forward)
    dc(1e-12)
    forward = current()
    resistance = (b.forward - b.reverse) / (b.ratio * forward)
    source = b.forward + forward * resistance
    circuit_alter(name="R1", value=resistance)
    circuit_alter(name="V1", value=source)
    for _ in range(4):
        transient_dc()
        source += b.forward - anode()
        circuit_alter(name="V1", value=source)
    transient_dc()
    print(f"  refine {refine:g}, {b.steps * fine} steps: I_F {forward:.6e} A/cm^2, R {resistance:.6e} ohm cm^2, anode at {anode():.12f} V", file=sys.stderr)
    circuit_alter(name="V1", value=b.reverse)
    dt = b.span * b.kingston / (b.steps * fine)
    rows: list[dict[str, float]] = []
    began = time.perf_counter()
    for k in range(1, b.steps * fine + 1):
        step(k == 1, dt)
        if k % fine == 0:
            rows.append({"time": k * dt, "voltage": anode(), "current": current()})
    took = time.perf_counter() - began
    return {"forward": forward, "resistance": resistance, "seconds": took, "steps": b.steps * fine, "rows": rows}


def storage(rows: list[dict[str, float]]) -> float:
    for a, c in zip(rows[:-1], rows[1:], strict=True):
        if a["voltage"] > 0.0 >= c["voltage"]:
            return a["time"] + (c["time"] - a["time"]) * a["voltage"] / (a["voltage"] - c["voltage"])
    raise ValueError("the anode voltage never crosses zero going down")


def run(b: P.TransientBenchmark, refine: float, fine: int, c: tuple[int, bool]) -> dict[str, Any]:
    env = dict(os.environ, MKL_NUM_THREADS=str(c[0]))
    k = [sys.executable, os.path.abspath(__file__), "--one", b.name, "--refine", f"{refine:g}", "--fine", str(fine)] + (["--extended"] if c[1] else [])
    p = subprocess.run(k, env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{b.name} at refine {refine:g}, {fine}x steps on {label(c)} failed:\n{p.stderr[-3000:]}")
    for line in p.stderr.splitlines():
        if line.startswith("  "):
            print(line)
    out: dict[str, Any] = json.loads(p.stdout.strip().splitlines()[-1])
    return out


def change(a: dict[str, Any], c: dict[str, Any]) -> tuple[float, float]:
    scale = abs(c["rows"][0]["current"])
    i = max(abs(x["current"] - y["current"]) / scale for x, y in zip(a["rows"], c["rows"], strict=True))
    return abs(storage(a["rows"]) - storage(c["rows"])) / storage(c["rows"]), i


def write_csv(b: P.TransientBenchmark, ref: dict[str, Any], spread: list[float], mesh: tuple[float, float] | None, steps: tuple[float, float] | None, path: str) -> None:
    h = [
        f"# device: {b.name}",
        f"# benchmark: {b.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_transient.py",
        f"# generator: devsim {devsim_version()} on python {sys.version.split()[0]}",
        f"# generated on: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')}",
        f"# tolerance: {b.tolerance}",
        f"# tau: {b.tau}",
        f"# forward voltage: {b.forward}",
        f"# reverse voltage: {b.reverse}",
        f"# forward current: {ref['forward']:.12e}",
        f"# resistance: {ref['resistance']:.12e}",
        f"# steps: {ref['steps']}",
        f"# storage time: {storage(ref['rows']):.12e}",
        f"# kingston: {b.kingston:.12e}",
        f"# devsim seconds: {ref['seconds']:.3f} for the time steps alone, {ref['steps'] / ref['seconds']:.1f} steps per second on {label(b.devsim_runs[0])}",
        f"# devsim runs: {', '.join(label(c) for c in b.devsim_runs)}, the stored values are the first",
        f"# devsim spread: {max(spread):.3e} worst relative change in a sample current, scaled by the first reverse current, between those runs",
    ]
    if mesh is not None:
        h.append(f"# mesh convergence: {mesh[0]:.3e} relative change in storage time and {mesh[1]:.3e} in the worst sample current, scaled by the first reverse current, when every devsim spacing is halved")
    if steps is not None:
        h.append(f"# step convergence: {steps[0]:.3e} relative change in storage time and {steps[1]:.3e} in the worst sample current, scaled by the first reverse current, when the step is halved")
    h.append("# notes: " + b.notes)
    h.append("# models:")
    h.extend("#   " + x for x in P.AC_MODEL_SUMMARY[:4])
    h.append("#   time:            BDF1 on the first step, BDF2 after, fixed step, carrier charge on the continuity rows only")
    h.append("#   circuit:         a voltage source through a series resistor into the anode, cathode grounded")
    h.append("# columns: time [s], anode voltage [V], anode current [A/cm^2], devsim spread in that current over the first reverse current [1]")
    h.append("time,voltage,current,spread")
    for r, s in zip(ref["rows"], spread, strict=True):
        h.append(f"{r['time']:.12e},{r['voltage']:.12e},{r['current']:.12e},{s:.3e}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(h) + "\n")


def main() -> int:
    a = argparse.ArgumentParser(description="Generate the reverse recovery golden curve with DEVSIM.")
    a.add_argument("names", nargs="*", default=None, help="benchmark names to generate, default all")
    a.add_argument("--out", default=os.path.join("data", "golden"), help="output directory for the CSV files")
    a.add_argument("--no-mesh-check", action="store_true", help="skip the halved mesh and halved step reruns")
    a.add_argument("--one", default=None, help=argparse.SUPPRESS)
    a.add_argument("--refine", type=float, default=1.0, help=argparse.SUPPRESS)
    a.add_argument("--fine", type=int, default=1, help=argparse.SUPPRESS)
    a.add_argument("--extended", action="store_true", help=argparse.SUPPRESS)
    o = a.parse_args()
    if o.one:
        b = P.TRANSIENT_BY_NAME[o.one]
        for k in ("extended_solver", "extended_model", "extended_equation"):
            set_parameter(name=k, value=o.extended)
        with quiet():
            got = recovery(b, o.refine, o.fine)
        print(json.dumps(got))
        return 0
    todo = P.TRANSIENT_BENCHMARKS
    if o.names:
        bad = sorted(set(o.names) - set(P.TRANSIENT_BY_NAME))
        if bad:
            print(f"no such benchmark: {bad}, known: {sorted(P.TRANSIENT_BY_NAME)}", file=sys.stderr)
            return 1
        todo = tuple(P.TRANSIENT_BY_NAME[x] for x in o.names)
    os.makedirs(o.out, exist_ok=True)
    for b in todo:
        print(f"[{b.name}] solving on the reference mesh on {', '.join(label(c) for c in b.devsim_runs)}")
        runs = [run(b, 1.0, 1, c) for c in b.devsim_runs]
        ref = runs[0]
        scale = abs(ref["rows"][0]["current"])
        spread = [max((abs(x["rows"][i]["current"] - r["current"]) / scale for x in runs[1:]), default=0.0) for i, r in enumerate(ref["rows"])]
        print(f"[{b.name}] storage time {storage(ref['rows']):.6e} s, {storage(ref['rows']) / b.kingston:.4f} times Kingston, worst spread {max(spread):.3e}")
        mesh = steps = None
        if not o.no_mesh_check:
            mesh = change(run(b, 2.0, 1, b.devsim_runs[0]), ref)
            print(f"[{b.name}] halved mesh moves storage by {mesh[0]:.3e} and a sample by {mesh[1]:.3e}")
            steps = change(run(b, 1.0, 2, b.devsim_runs[0]), ref)
            print(f"[{b.name}] halved step moves storage by {steps[0]:.3e} and a sample by {steps[1]:.3e}")
        p = os.path.join(o.out, f"{b.name}.csv")
        write_csv(b, ref, spread, mesh, steps, p)
        print(f"[{b.name}] wrote {p} with {len(ref['rows'])} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
