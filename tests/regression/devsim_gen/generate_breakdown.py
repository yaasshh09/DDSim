from __future__ import annotations
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
from devsim import add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh, create_device, delete_device, delete_mesh, equation, finalize_mesh, get_contact_current, node_model, set_node_values, set_parameter, solve
from devsim.python_packages.model_create import CreateEdgeModel, CreateEdgeModelDerivatives, CreateSolution
from devsim.python_packages.simple_physics import CreateSiliconDriftDiffusion, CreateSiliconDriftDiffusionAtContact, CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact, ece_name, hce_name
from generate_diodes import REGION, devsim_version, set_silicon_parameters

ANODE = "anode"
CATHODE = "cathode"

FIELD = "(abs(ElectricField) + 1e-3)"


def generation() -> str:
    an = f"{P.II_A_N:.16e}*exp(-{P.II_B_N:.16e}/{FIELD})"
    ap = f"ifelse({FIELD} < {P.II_E_P_SWITCH:.16e}, {P.II_A_P_LOW:.16e}*exp(-{P.II_B_P_LOW:.16e}/{FIELD}), {P.II_A_P_HIGH:.16e}*exp(-{P.II_B_P_HIGH:.16e}/{FIELD}))"
    return f"({an}*abs(ElectronCurrent) + {ap}*abs(HoleCurrent))"


def settle(d: str) -> None:
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=40)


def device(b: P.BreakdownBenchmark, refine: float) -> str:
    d = f"{b.name}_r{refine:g}".replace(".", "_")
    j, f = b.devsim_h_junction / refine, b.devsim_h_bulk / refine
    create_1d_mesh(mesh=d)
    add_1d_mesh_line(mesh=d, pos=0.0, ps=f, tag="top")
    add_1d_mesh_line(mesh=d, pos=b.junction, ps=j, ns=j, tag="mid")
    add_1d_mesh_line(mesh=d, pos=b.length, ps=f, ns=f, tag="bot")
    add_1d_contact(mesh=d, name=ANODE, tag="top", material="metal")
    add_1d_contact(mesh=d, name=CATHODE, tag="bot", material="metal")
    add_1d_region(mesh=d, material="Silicon", region=REGION, tag1="top", tag2="bot")
    finalize_mesh(mesh=d)
    create_device(mesh=d, device=d)
    node_model(device=d, region=REGION, name="NetDoping", equation=f"ifelse(x < {b.junction:.16e}, {-b.Na:.16e}, {b.Nd:.16e})")
    set_silicon_parameters(d)
    for k in ("taun", "taup"):
        set_parameter(device=d, region=REGION, name=k, value=b.tau)
    CreateSolution(d, REGION, "Potential")
    CreateSiliconPotentialOnly(d, REGION)
    for c in (ANODE, CATHODE):
        set_parameter(device=d, name=f"{c}_bias", value=0.0)
        CreateSiliconPotentialOnlyContact(d, REGION, c)
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
    for m, e in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        CreateSolution(d, REGION, m)
        set_node_values(device=d, region=REGION, name=m, init_from=e)
    CreateSiliconDriftDiffusion(d, REGION, mu_n="mu_n", mu_p="mu_p")
    for c in (ANODE, CATHODE):
        CreateSiliconDriftDiffusionAtContact(d, REGION, c)
    settle(d)
    g = generation()
    CreateEdgeModel(d, REGION, "ImpactN", g)
    CreateEdgeModel(d, REGION, "ImpactP", f"-{g}")
    for v in ("Potential", "Electrons", "Holes"):
        CreateEdgeModelDerivatives(d, REGION, "ImpactN", g, v)
        CreateEdgeModelDerivatives(d, REGION, "ImpactP", f"-{g}", v)
    equation(device=d, region=REGION, name=ece_name, variable_name="Electrons", time_node_model="NCharge", edge_model="ElectronCurrent", variable_update="positive", node_model="ElectronGeneration", edge_volume_model="ImpactN")
    equation(device=d, region=REGION, name=hce_name, variable_name="Holes", time_node_model="PCharge", edge_model="HoleCurrent", variable_update="positive", node_model="HoleGeneration", edge_volume_model="ImpactP")
    settle(d)
    return d


def current(d: str, contact: str) -> float:
    return get_contact_current(device=d, contact=contact, equation=ece_name) + get_contact_current(device=d, contact=contact, equation=hce_name)


def sweep(b: P.BreakdownBenchmark, refine: float) -> list[dict[str, float]]:
    d = device(b, refine)
    rows = []
    at, step = 0.0, 0.5
    for v in sorted(b.voltages, reverse=True):
        while abs(at - v) > 1e-12:
            nxt = at + max(-step, min(step, v - at))
            set_parameter(device=d, name=f"{ANODE}_bias", value=nxt)
            try:
                settle(d)
                at, step = nxt, min(2 * step, 0.5)
            except Exception:
                set_parameter(device=d, name=f"{ANODE}_bias", value=at)
                settle(d)
                step /= 4
                if step < 1e-5:
                    raise RuntimeError(f"the anode is stuck at {at:g} V heading to {v:g} V") from None
        rows.append({"voltage": v, "current": current(d, ANODE), "cathode": current(d, CATHODE)})
        print(f"    {v:+.3f} V: {rows[-1]['current']:.6e} A/cm^2", file=sys.stderr, flush=True)
    delete_device(device=d)
    delete_mesh(mesh=d)
    return sorted(rows, key=lambda r: r["voltage"])


def write(b: P.BreakdownBenchmark, rows: list[dict[str, float]], mesh: float | None, path: str) -> None:
    h = [
        f"# device: {b.name}",
        f"# benchmark: {b.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_breakdown.py",
        f"# generator: devsim {devsim_version()} on python {sys.version.split()[0]}",
        f"# generated on: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')}",
        f"# tolerance: {b.tolerance}",
        f"# Na: {b.Na:.6e}",
        f"# Nd: {b.Nd:.6e}",
        f"# length: {b.length:.6e}",
        f"# junction: {b.junction:.6e}",
        f"# tau: {b.tau:.6e}",
        f"# devsim h_junction: {b.devsim_h_junction:.6e}",
        f"# devsim h_bulk: {b.devsim_h_bulk:.6e}",
        "# statistics: Boltzmann",
    ]
    if mesh is not None:
        h.append(f"# mesh convergence: {mesh:.3e} worst relative change in current when every spacing is halved")
    h.append("# notes: " + b.notes)
    h.append("# models: Boltzmann, constant mobility, SRH with a constant 1 ps lifetime, van Overstraeten-de Man impact ionization on the edge field")
    h.append("# columns: anode bias [V], anode current [A/cm^2], cathode current [A/cm^2]")
    h.append("voltage,current,cathode_current")
    h += [f"{r['voltage']:.10g},{r['current']:.12e},{r['cathode']:.12e}" for r in rows]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(h) + "\n")


def main() -> int:
    a = argparse.ArgumentParser(description="Generate the avalanche breakdown golden curve with DEVSIM.")
    a.add_argument("--out", default=os.path.join("data", "golden"), help="output directory for the CSV files")
    a.add_argument("--no-mesh-check", action="store_true", help="skip the halved mesh rerun")
    o = a.parse_args()
    for b in P.BREAKDOWN_BENCHMARKS:
        print(f"[{b.name}] reference mesh", flush=True)
        rows = sweep(b, 1.0)
        mesh = None
        if not o.no_mesh_check:
            print(f"[{b.name}] halved mesh", flush=True)
            fine = sweep(b, 2.0)
            mesh = max(abs(x["current"] - y["current"]) / abs(y["current"]) for x, y in zip(rows, fine, strict=True))
            print(f"[{b.name}] mesh convergence {mesh:.3e}", flush=True)
        p = os.path.join(o.out, f"{b.name}.csv")
        write(b, rows, mesh, p)
        print(f"[{b.name}] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
