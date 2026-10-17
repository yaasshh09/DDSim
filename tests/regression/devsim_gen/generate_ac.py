from __future__ import annotations
import argparse
import contextlib
import datetime
import io
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
from devsim import (
    add_1d_contact,
    add_1d_interface,
    add_1d_mesh_line,
    add_1d_region,
    circuit_alter,
    circuit_element,
    contact_equation,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_circuit_node_value,
    get_contact_current,
    get_node_model_values,
    node_model,
    set_node_values,
    set_parameter,
    solve,
    vector_gradient,
)
from devsim.python_packages.model_create import (
    CreateContactNodeModel,
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateNodeModel,
    CreateSolution,
)
from devsim.python_packages.simple_physics import (
    CreateOxidePotentialOnly,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    ece_name,
    hce_name,
)

SILICON = "bulk"
OXIDE = "oxide"


def devsim_version() -> str:
    try:
        from importlib.metadata import version
        return version("devsim")
    except Exception:
        return "unknown"


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def dc(rel: float = 1e-9) -> None:
    for x in (rel, 1e-8, 1e-6):
        try:
            solve(type="dc", absolute_error=1e10, relative_error=max(x, rel), maximum_iterations=60)
            return
        except Exception:
            if x == 1e-6:
                raise


def silicon(dev: str) -> None:
    v = {"Permittivity": P.EPS_R_SI * P.EPS_0, "ElectronCharge": P.Q, "n_i": P.N_I, "T": P.T, "kT": P.K_B * P.T, "V_t": P.V_T, "mu_n": P.MU_N, "mu_p": P.MU_P, "n1": P.N_I, "p1": P.N_I}
    for k, x in v.items():
        set_parameter(device=dev, region=SILICON, name=k, value=x)


def lifetimes(dev: str, b: P.AcBenchmark, doping: float) -> None:
    tn = b.tau if b.tau is not None else P.scharfetter_lifetime(doping, P.TAU_N_MAX, P.TAU_N_MIN)
    tp = b.tau if b.tau is not None else P.scharfetter_lifetime(doping, P.TAU_P_MAX, P.TAU_P_MIN)
    CreateNodeModel(dev, SILICON, "taun", f"{tn:.16e}")
    CreateNodeModel(dev, SILICON, "taup", f"{tp:.16e}")


def drift_diffusion(dev: str, driven: str, grounded: str | None) -> None:
    for y, r in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        CreateSolution(dev, SILICON, y)
        set_node_values(device=dev, region=SILICON, name=y, init_from=r)
    CreateSiliconDriftDiffusion(dev, SILICON, mu_n="mu_n", mu_p="mu_p")
    if driven:
        CreateSiliconDriftDiffusionAtContact(dev, SILICON, driven, is_circuit=True)
    if grounded:
        CreateSiliconDriftDiffusionAtContact(dev, SILICON, grounded)
    dc()


def ramp(target: float, start: float = 0.0, step: float = 0.05) -> None:
    n = max(1, round(abs(target - start) / step))
    for i in range(1, n + 1):
        circuit_alter(name="V1", value=start + (target - start) * i / n)
        dc()
    for _ in range(3):
        dc()


def admittance(f: float) -> complex:
    solve(type="ac", frequency=f)
    return -complex(get_circuit_node_value(node="V1.I", solution="ssac_real"), get_circuit_node_value(node="V1.I", solution="ssac_imag"))


def diode(b: P.AcBenchmark, refine: float) -> list[dict[str, Any]]:
    dev = f"{b.name}_r{refine:g}".replace(".", "_")
    circuit_element(name="V1", n1="anode_bias", n2=0, value=0.0, acreal=1.0, acimag=0.0)
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
    lifetimes(dev, b, 0.0)
    CreateSolution(dev, SILICON, "Potential")
    CreateSiliconPotentialOnly(dev, SILICON)
    set_parameter(device=dev, name="cathode_bias", value=0.0)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "anode", is_circuit=True)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "cathode")
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
    drift_diffusion(dev, "anode", "cathode")
    out: list[dict[str, Any]] = []
    for v in b.biases:
        ramp(v)
        dc(1e-12)
        i = sum(get_contact_current(device=dev, contact="anode", equation=e) for e in (ece_name, hce_name))
        k = sum(get_contact_current(device=dev, contact="cathode", equation=e) for e in (ece_name, hce_name))
        print(f"  {v:g} V: anode {i:.6e}, cathode {k:.6e} A/cm^2", file=sys.stderr)
        for f in b.frequencies:
            y = admittance(f)
            out.append({"bias": v, "frequency": f, "re_y": y.real, "im_y": y.imag})
    return out


def mos_cap(b: P.AcBenchmark, refine: float) -> list[dict[str, Any]]:
    m = P.MOS_BENCHMARKS[0]
    r = refine * b.devsim_refine
    dev = f"{b.name}_r{refine:g}".replace(".", "_")
    circuit_element(name="V1", n1="gate_bias", n2=0, value=0.0, acreal=1.0, acimag=0.0)
    h = m.t_ox / (m.devsim_oxide_cells * r)
    create_1d_mesh(mesh=dev)
    add_1d_mesh_line(mesh=dev, pos=0.0, ps=m.devsim_h_bulk / r, tag="body")
    add_1d_mesh_line(mesh=dev, pos=m.t_si, ns=m.devsim_h_surface / r, ps=h, tag="iface")
    add_1d_mesh_line(mesh=dev, pos=m.t_si + m.t_ox, ps=h, tag="gate")
    add_1d_contact(mesh=dev, name="body", tag="body", material="metal")
    add_1d_contact(mesh=dev, name="gate", tag="gate", material="metal")
    add_1d_region(mesh=dev, material="Silicon", region=SILICON, tag1="body", tag2="iface")
    add_1d_region(mesh=dev, material="Oxide", region=OXIDE, tag1="iface", tag2="gate")
    add_1d_interface(mesh=dev, name="si_ox", tag="iface")
    finalize_mesh(mesh=dev)
    create_device(mesh=dev, device=dev)
    silicon(dev)
    for k, x in {"Permittivity": P.EPS_R_OX * P.EPS_0, "ElectronCharge": P.Q}.items():
        set_parameter(device=dev, region=OXIDE, name=k, value=x)
    node_model(device=dev, region=SILICON, name="NetDoping", equation=f"{m.substrate_doping:.16e}")
    lifetimes(dev, b, abs(m.substrate_doping))
    for g in (SILICON, OXIDE):
        CreateSolution(dev, g, "Potential")
    CreateSiliconPotentialOnly(dev, SILICON)
    CreateOxidePotentialOnly(dev, OXIDE, "log_damp")
    set_parameter(device=dev, name="body_bias", value=0.0)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "body")
    CreateContactNodeModel(dev, "gate", "gatenodemodel", "Potential - gate_bias")
    CreateContactNodeModel(dev, "gate", "gatenodemodel:Potential", "1")
    CreateContactNodeModel(dev, "gate", "gatenodemodel:gate_bias", "-1")
    CreateEdgeModel(dev, OXIDE, "contactcharge_edge", "Permittivity*ElectricField")
    CreateEdgeModelDerivatives(dev, OXIDE, "contactcharge_edge", "Permittivity*ElectricField", "Potential")
    contact_equation(device=dev, contact="gate", name="PotentialEquation", node_model="gatenodemodel", edge_charge_model="contactcharge_edge", circuit_node="gate_bias")
    CreateSiliconOxideInterface(dev, "si_ox")
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
    drift_diffusion(dev, "", "body")
    out: list[dict[str, Any]] = []
    now = 0.0
    for v in b.biases:
        ramp(v + P.PHI_M_MIDGAP - m.work_function, now)
        now = v + P.PHI_M_MIDGAP - m.work_function
        for f in b.frequencies:
            y = admittance(f)
            out.append({"bias": v, "frequency": f, "re_y": y.real, "im_y": y.imag})
    return out


def bar(b: P.AcBenchmark, refine: float) -> list[dict[str, Any]]:
    r = refine * b.devsim_refine
    dev = f"{b.name}_r{refine:g}".replace(".", "_")
    circuit_element(name="V1", n1="top_bias", n2=0, value=0.0, acreal=1.0, acimag=0.0)
    create_1d_mesh(mesh=dev)
    add_1d_mesh_line(mesh=dev, pos=0.0, ps=P.BAR_DEVSIM_H / r, tag="top")
    add_1d_mesh_line(mesh=dev, pos=P.BAR_LENGTH, ps=P.BAR_DEVSIM_H / r, tag="bot")
    add_1d_contact(mesh=dev, name="top", tag="top", material="metal")
    add_1d_contact(mesh=dev, name="bot", tag="bot", material="metal")
    add_1d_region(mesh=dev, material="Silicon", region=SILICON, tag1="top", tag2="bot")
    finalize_mesh(mesh=dev)
    create_device(mesh=dev, device=dev)
    silicon(dev)
    node_model(device=dev, region=SILICON, name="NetDoping", equation=f"{P.BAR_DOPING:.16e}")
    lifetimes(dev, b, P.BAR_DOPING)
    CreateSolution(dev, SILICON, "Potential")
    CreateSiliconPotentialOnly(dev, SILICON)
    set_parameter(device=dev, name="bot_bias", value=0.0)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "top", is_circuit=True)
    CreateSiliconPotentialOnlyContact(dev, SILICON, "bot")
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)
    drift_diffusion(dev, "top", "bot")
    out: list[dict[str, Any]] = []
    for f in b.frequencies:
        y = admittance(f)
        solve(type="noise", frequency=f, output_node="V1.I")
        s = 0.0
        for eq, carrier, mu in (("ElectronContinuityEquation", "Electrons", "mu_n"), ("HoleContinuityEquation", "Holes", "mu_p")):
            for part in ("real", "imag"):
                vector_gradient(device=dev, region=SILICON, node_model=f"V1.I_{eq}_{part}", calc_type="avoidzero")
            a, c = f"V1.I_{eq}_real_gradx", f"V1.I_{eq}_imag_gradx"
            node_model(device=dev, region=SILICON, name=f"source_{carrier}", equation=f"4*ElectronCharge^2*V_t*{mu}*{carrier}*({a}^2+{c}^2)*NodeVolume")
            s += sum(get_node_model_values(device=dev, region=SILICON, name=f"source_{carrier}"))
        out.append({"bias": 0.0, "frequency": f, "re_y": y.real, "im_y": y.imag, "s_i": s, "s_v": s / abs(y) ** 2})
    return out


SWEEPS = {11: mos_cap, 12: diode, 13: bar}


def label(c: tuple[int, bool]) -> str:
    return f"{c[0]} thread{'s' if c[0] > 1 else ''} {'extended' if c[1] else 'double'}"


def run(b: P.AcBenchmark, refine: float, c: tuple[int, bool]) -> list[dict[str, Any]]:
    env = dict(os.environ, MKL_NUM_THREADS=str(c[0]))
    k = [sys.executable, os.path.abspath(__file__), "--one", b.name, "--refine", f"{refine:g}"] + (["--extended"] if c[1] else [])
    p = subprocess.run(k, env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{b.name} at refine {refine:g} on {label(c)} failed:\n{p.stderr[-3000:]}")
    for line in p.stderr.splitlines():
        if line.startswith("  "):
            print(line)
    rows: list[dict[str, Any]] = json.loads(p.stdout.strip().splitlines()[-1])
    return rows


def difference(a: dict[str, Any], c: dict[str, Any]) -> float:
    ya, yc = complex(a["re_y"], a["im_y"]), complex(c["re_y"], c["im_y"])
    e = abs(ya - yc) / abs(yc)
    if "s_v" in a:
        e = max(e, abs(a["s_v"] - c["s_v"]) / abs(c["s_v"]))
    return e


SOUND = 1e-3


def mesh_change(rows: list[dict[str, Any]], fine: list[dict[str, Any]]) -> tuple[float, float, float, int]:
    worst, at, bias, skipped = 0.0, 0.0, 0.0, 0
    for a, c in zip(rows, fine, strict=True):
        if a["spread"] > SOUND:
            skipped += 1
            continue
        e = difference(a, c)
        if e > worst:
            worst, at, bias = e, a["frequency"], a["bias"]
    return worst, at, bias, skipped


def write_csv(b: P.AcBenchmark, rows: list[dict[str, Any]], mesh_check: tuple[float, float, float, int] | None, path: str) -> None:
    noise = "s_v" in rows[0]
    w = max(rows, key=lambda r: r["spread"])
    h = [
        f"# device: {b.name}",
        f"# benchmark: {b.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_ac.py",
        f"# generator: devsim {devsim_version()} on python {sys.version.split()[0]}",
        f"# generated on: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')}",
        f"# tolerance: {b.tolerance}",
        f"# tau: {b.tau if b.tau is not None else 'scharfetter'}",
        f"# devsim refine: {b.devsim_refine:g}",
        f"# devsim runs: {', '.join(label(c) for c in b.devsim_runs)}, the stored values are the first",
        f"# devsim spread: {w['spread']:.3e} worst relative change in {'Y or S_V' if noise else 'Y'} between those runs, at {w['frequency']:.4e} Hz and {w['bias']:+g} V; {sum(r['spread'] > SOUND for r in rows)} of {len(rows)} points above {SOUND:g}",
    ]
    if mesh_check is not None:
        e, f, v, n = mesh_check
        h.append(f"# mesh convergence: {e:.3e} worst relative change when every devsim spacing is halved, at {f:.4e} Hz and {v:+g} V, over the {len(rows) - n} points whose spread is under {SOUND:g}")
    h.append("# notes: " + b.notes)
    h.append("# models:")
    h.extend("#   " + x for x in P.AC_MODEL_SUMMARY)
    if noise:
        h.append("# columns: bias [V], frequency [Hz], Re Y and Im Y [A/(V cm^2)], short circuit S_I [A^2 cm^-4 / Hz], open circuit S_V [V^2 / Hz], devsim spread [1]")
        h.append("bias,frequency,re_y,im_y,s_i,s_v,spread")
    else:
        h.append("# columns: bias [V], frequency [Hz], Re Y and Im Y at the driven contact [A/(V cm^2)], devsim spread [1]")
        h.append("bias,frequency,re_y,im_y,spread")
    for r in rows:
        x = f"{r['bias']:.10g},{r['frequency']:.10e},{r['re_y']:.12e},{r['im_y']:.12e}"
        if noise:
            x += f",{r['s_i']:.12e},{r['s_v']:.12e}"
        h.append(x + f",{r['spread']:.3e}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(h) + "\n")


def main() -> int:
    a = argparse.ArgumentParser(description="Generate the small signal and noise golden curves with DEVSIM.")
    a.add_argument("names", nargs="*", default=None, help="benchmark names to generate, default all")
    a.add_argument("--out", default=os.path.join("data", "golden"), help="output directory for the CSV files")
    a.add_argument("--no-mesh-check", action="store_true", help="skip the halved mesh rerun, which roughly doubles the runtime")
    a.add_argument("--one", default=None, help=argparse.SUPPRESS)
    a.add_argument("--refine", type=float, default=1.0, help=argparse.SUPPRESS)
    a.add_argument("--extended", action="store_true", help=argparse.SUPPRESS)
    o = a.parse_args()
    if o.one:
        b = P.AC_BY_NAME[o.one]
        for k in ("extended_solver", "extended_model", "extended_equation"):
            set_parameter(name=k, value=o.extended)
        with quiet():
            rows = SWEEPS[b.number](b, o.refine)
        print(json.dumps(rows))
        return 0
    todo = P.AC_BENCHMARKS
    if o.names:
        bad = sorted(set(o.names) - set(P.AC_BY_NAME))
        if bad:
            print(f"no such benchmark: {bad}, known: {sorted(P.AC_BY_NAME)}", file=sys.stderr)
            return 1
        todo = tuple(P.AC_BY_NAME[x] for x in o.names)
    os.makedirs(o.out, exist_ok=True)
    for b in todo:
        print(f"[{b.name}] solving on the reference mesh on {', '.join(label(c) for c in b.devsim_runs)}")
        runs = [run(b, 1.0, c) for c in b.devsim_runs]
        rows = runs[0]
        for i, r in enumerate(rows):
            r["spread"] = max((difference(x[i], r) for x in runs[1:]), default=0.0)
        print(f"[{b.name}] worst spread between runs {max(r['spread'] for r in rows):.3e}")
        check = None
        if not o.no_mesh_check:
            print(f"[{b.name}] solving again on a halved mesh")
            check = mesh_change(rows, run(b, 2.0, b.devsim_runs[0]))
            print(f"[{b.name}] mesh convergence {check[0]:.3e} at {check[1]:.4e} Hz, {check[2]:+g} V, {check[3]} noisy points left out")
        p = os.path.join(o.out, f"{b.name}.csv")
        write_csv(b, rows, check, p)
        print(f"[{b.name}] wrote {p} with {len(rows)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
