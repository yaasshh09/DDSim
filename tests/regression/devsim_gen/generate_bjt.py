from __future__ import annotations
import argparse
import datetime
import math
import os
import sys
import time
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
from devsim import (
    add_2d_contact,
    add_2d_mesh_line,
    add_2d_region,
    create_2d_mesh,
    create_device,
    delete_device,
    delete_mesh,
    finalize_mesh,
    get_contact_current,
    get_contact_list,
    get_node_model_values,
    get_parameter,
    node_model,
    set_node_values,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    ece_name,
    hce_name,
)
from generate_mosfet import quiet

BULK = "bulk"

EMITTER, BASE, COLLECTOR = "emitter", "base", "collector"

AIR = 1e-6

SOLUTIONS = ("Potential", "Electrons", "Holes")

TOLERANCES = (1e-8, 1e-6, 1e-4)

set_parameter(name="threads_available", value=1)


def devsim_version() -> str:
    import devsim
    return str(getattr(devsim, "__version__", "unknown"))


def doping() -> str:
    g = P.NPN_2D
    d = g["depth"]
    return (
        f"{g['epi_doping']:.16e}"
        f" + {g['buried_peak']:.16e} * exp(-pow(y, 2) / (2 * pow({g['buried_sigma']:.16e}, 2)))"
        f" - {g['base_peak']:.16e} * exp(-pow(y - {d:.16e}, 2) / (2 * pow({g['base_sigma']:.16e}, 2)))"
        f" + {g['emitter_peak']:.16e} * 0.5 * erfc((x - {g['emitter_window']:.16e}) / {g['lateral_length']:.16e})"
        f" * exp(-pow(y - {d:.16e}, 2) / (2 * pow({g['emitter_sigma']:.16e}, 2)))"
    )


def net(x: float, y: float) -> float:
    g = P.NPN_2D
    d = g["depth"]
    return (g["epi_doping"] + g["buried_peak"] * math.exp(-y * y / (2 * g["buried_sigma"] ** 2)) - g["base_peak"] * math.exp(-(y - d) ** 2 / (2 * g["base_sigma"] ** 2))
            + g["emitter_peak"] * 0.5 * math.erfc((x - g["emitter_window"]) / g["lateral_length"]) * math.exp(-(y - d) ** 2 / (2 * g["emitter_sigma"] ** 2)))


def crossings(f: Callable[[float], float], span: float) -> list[float]:
    t = [span * k / 4000 for k in range(4001)]
    v = [f(z) for z in t]
    return [t[i] - v[i] * (t[i + 1] - t[i]) / (v[i + 1] - v[i]) for i in range(4000) if (v[i] < 0.0) != (v[i + 1] < 0.0)]


def build(b: P.BjtBenchmark, dev: str, refine: float) -> None:
    g = P.NPN_2D
    w, d = g["width"], g["depth"]
    h, hj, he = b.devsim_h / refine, b.devsim_h_junction / refine, b.devsim_h_edge / refine
    ys = crossings(lambda y: net(0.0, y), d)
    xs = crossings(lambda x: net(x, d), w)
    create_2d_mesh(mesh=dev)
    for x, s in sorted({0.0: h, g["emitter_contact"]: he, xs[0]: hj, g["base_contact"]: he, w: h}.items()):
        add_2d_mesh_line(mesh=dev, dir="x", pos=x, ns=s, ps=s)
    add_2d_mesh_line(mesh=dev, dir="y", pos=-AIR, ns=AIR, ps=AIR)
    for y, s in sorted({0.0: h, ys[0]: hj, ys[1]: hj, d: hj}.items()):
        add_2d_mesh_line(mesh=dev, dir="y", pos=y, ns=s, ps=s)
    add_2d_mesh_line(mesh=dev, dir="y", pos=d + AIR, ns=AIR, ps=AIR)
    add_2d_region(mesh=dev, region=BULK, material="Silicon", yl=0.0, yh=d)
    add_2d_region(mesh=dev, region="air_bot", material="metal", yl=-AIR, yh=0.0)
    add_2d_region(mesh=dev, region="air_top", material="metal", yl=d, yh=d + AIR)
    add_2d_contact(mesh=dev, name=COLLECTOR, material="metal", region=BULK, yl=0.0, yh=0.0)
    add_2d_contact(mesh=dev, name=EMITTER, material="metal", region=BULK, xl=0.0, xh=g["emitter_contact"], yl=d, yh=d)
    add_2d_contact(mesh=dev, name=BASE, material="metal", region=BULK, xl=g["base_contact"], xh=w, yl=d, yh=d)
    finalize_mesh(mesh=dev)
    create_device(mesh=dev, device=dev)
    got = set(get_contact_list(device=dev))
    if got != {EMITTER, BASE, COLLECTOR}:
        raise RuntimeError(f"{dev}: devsim made contacts {sorted(got)}")


def set_material_parameters(dev: str) -> None:
    for k, v in {"Permittivity": P.EPS_R_SI * P.EPS_0, "ElectronCharge": P.Q, "n_i": P.N_I, "T": P.T, "kT": P.K_B * P.T, "V_t": P.V_T, "mu_n": P.MU_N, "mu_p": P.MU_P, "n1": P.N_I, "p1": P.N_I}.items():
        set_parameter(device=dev, region=BULK, name=k, value=v)


def lifetimes(dev: str) -> None:
    for a, v, j in (("taun", P.TAU_N_MAX, P.TAU_N_MIN), ("taup", P.TAU_P_MAX, P.TAU_P_MIN)):
        CreateNodeModel(dev, BULK, a, f"{j:.16e} + ({v:.16e} - {j:.16e}) / (1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})")


def snapshot(dev: str) -> dict[str, list[float]]:
    return {m: list(get_node_model_values(device=dev, region=BULK, name=m)) for m in SOLUTIONS}


def restore(dev: str, s: dict[str, list[float]]) -> None:
    for m, v in s.items():
        set_node_values(device=dev, region=BULK, name=m, values=v)


def solve_once() -> None:
    for j, t in enumerate(TOLERANCES):
        try:
            solve(type="dc", absolute_error=1e30, relative_error=t, maximum_iterations=100, maximum_error=1e40)
            return
        except Exception:
            if j + 1 == len(TOLERANCES):
                raise


def settle(dev: str, potential_only: bool = False, tol: float = 1e-9, passes: int = 400) -> None:
    last = list(get_node_model_values(device=dev, region=BULK, name="Potential"))
    move = math.inf
    for _ in range(passes):
        solve_once()
        now = list(get_node_model_values(device=dev, region=BULK, name="Potential"))
        move = max(abs(a - c) for a, c in zip(now, last, strict=True))
        last = now
        if move < tol:
            if not potential_only:
                for m in ("Electrons", "Holes"):
                    if min(get_node_model_values(device=dev, region=BULK, name=m)) < 0.0:
                        raise RuntimeError("settled on a negative density")
            return
    raise RuntimeError(f"did not settle, last move {move:.3e} V")


def ramp(dev: str, contact: str, target: float, step: float = 0.05, min_step: float = 1e-4) -> None:
    c = get_parameter(device=dev, name=f"{contact}_bias")
    while abs(target - c) > 1e-12:
        s = snapshot(dev)
        m = c + math.copysign(min(step, abs(target - c)), target - c)
        set_parameter(device=dev, name=f"{contact}_bias", value=m)
        try:
            with quiet():
                settle(dev)
        except Exception:
            restore(dev, s)
            set_parameter(device=dev, name=f"{contact}_bias", value=c)
            step *= 0.5
            if step < min_step:
                raise RuntimeError(f"{contact} stuck at {c:g} V heading to {target:g} V") from None
            continue
        c = m
        step = min(2 * step, 0.1)


def current(dev: str, contact: str) -> float:
    return float(get_contact_current(device=dev, contact=contact, equation=ece_name) + get_contact_current(device=dev, contact=contact, equation=hce_name))


def device(b: P.BjtBenchmark, refine: float) -> str:
    dev = f"{b.name}_r{refine:g}".replace(".", "_")
    with quiet():
        build(b, dev, refine)
    set_material_parameters(dev)
    node_model(device=dev, region=BULK, name="NetDoping", equation=doping())
    with quiet():
        CreateSolution(dev, BULK, "Potential")
        CreateSiliconPotentialOnly(dev, BULK)
        node_model(device=dev, region=BULK, name="PotentialNeutral", equation=f"{P.V_T:.16e} * sgn(NetDoping) * log((abs(NetDoping) + pow(NetDoping*NetDoping + 4*{P.N_I:.16e}*{P.N_I:.16e}, 0.5)) / (2*{P.N_I:.16e}))")
        set_node_values(device=dev, region=BULK, name="Potential", init_from="PotentialNeutral")
        for c in (EMITTER, BASE, COLLECTOR):
            set_parameter(device=dev, name=f"{c}_bias", value=0.0)
            CreateSiliconPotentialOnlyContact(dev, BULK, c)
        settle(dev, potential_only=True)
        for m, e in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            CreateSolution(dev, BULK, m)
            set_node_values(device=dev, region=BULK, name=m, init_from=e)
        lifetimes(dev)
        CreateSiliconDriftDiffusion(dev, BULK, mu_n="mu_n", mu_p="mu_p")
        for c in (EMITTER, BASE, COLLECTOR):
            CreateSiliconDriftDiffusionAtContact(dev, BULK, c)
        settle(dev)
    return dev


def done(dev: str) -> None:
    delete_device(device=dev)
    delete_mesh(mesh=dev)


def gummel(b: P.BjtBenchmark, refine: float) -> tuple[list[dict[str, float]], int]:
    dev = device(b, refine)
    rows = []
    for v in b.v_be:
        ramp(dev, EMITTER, -v)
        rows.append({"v_be": v, "v_ce": v, "collector": current(dev, COLLECTOR), "base": current(dev, BASE)})
        print(f"    V_BE {v:.2f}: I_C {rows[-1]['collector']:.6e} I_B {rows[-1]['base']:.6e} A/cm", file=sys.stderr, flush=True)
    k = len(get_node_model_values(device=dev, region=BULK, name="x"))
    done(dev)
    return rows, k


def output(b: P.BjtBenchmark, refine: float) -> list[dict[str, float]]:
    dev = device(b, refine)
    rows = []
    vb = 0.6
    ramp(dev, BASE, vb)
    for u in b.v_ce:
        ramp(dev, COLLECTOR, u)
        lo: float | None = None
        hi: float | None = None
        seen = []
        for _ in range(80):
            ramp(dev, BASE, vb)
            i = current(dev, BASE)
            miss = math.log(i / b.i_b) if i > 0.0 else -50.0
            seen.append((vb, i))
            if abs(miss) < 1e-6 or (lo is not None and hi is not None and hi - lo < 1e-9):
                break
            if miss < 0.0:
                lo = vb
            else:
                hi = vb
            nxt = vb - miss * P.V_T
            if lo is not None and hi is not None and not lo < nxt < hi:
                nxt = 0.5 * (lo + hi)
            vb = nxt
        else:
            tried = ", ".join(f"{x:.9f} V -> {y:.6e}" for x, y in seen[-8:])
            raise RuntimeError(f"no base voltage holds {b.i_b:g} A/cm at V_CE {u:g} V. The last tries: {tried}")
        rows.append({"v_ce": u, "v_be": vb, "collector": current(dev, COLLECTOR), "base": current(dev, BASE)})
        print(f"    V_CE {u:.2f}: V_BE {vb:.6f} I_C {rows[-1]['collector']:.6e} A/cm", file=sys.stderr, flush=True)
    done(dev)
    return rows


def change(a: list[dict[str, float]], c: list[dict[str, float]]) -> float:
    return max(abs(x[k] - y[k]) / abs(y[k]) for x, y in zip(a, c, strict=True) for k in ("collector", "base"))


def write(b: P.BjtBenchmark, g: list[dict[str, float]], o: list[dict[str, float]], nodes: int, mesh: float | None, refine: float, seconds: float, out: str) -> None:
    h = [
        f"# device: {b.name}",
        f"# benchmark: {b.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_bjt.py",
        f"# generator: devsim {devsim_version()} on python {sys.version.split()[0]}",
        f"# generated on: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')}",
        f"# tolerance: {b.tolerance}",
        f"# base current: {b.i_b:.6e}",
        f"# devsim silicon nodes: {nodes}",
        f"# devsim seconds: {seconds:.1f}",
        f"# devsim spacings: {b.devsim_h:.3e} bulk, {b.devsim_h_junction:.3e} at the junctions, {b.devsim_h_edge:.3e} at the contact edges [cm]",
        "# statistics: Boltzmann",
    ]
    h += [f"# {k}: {v:.6e}" for k, v in P.NPN_2D.items()]
    if mesh is not None:
        how = f"divided by {refine:g}" if refine > 1.0 else f"multiplied by {1.0 / refine:g}"
        h.append(f"# mesh convergence: {mesh:.3e} worst relative change in a terminal current when every devsim spacing is {how}")
    h.append("# notes: " + b.notes)
    h.append("# models: Boltzmann, constant mobility, SRH with Scharfetter lifetimes, ohmic contacts")
    h.append("# columns: curve (0 for the Gummel plot, 1 for the output curve), V_BE [V], V_CE [V], collector and base current [A/cm]")
    h.append("curve,v_be,v_ce,collector,base")
    for r in g:
        h.append(f"0,{r['v_be']:.12e},{r['v_ce']:.12e},{r['collector']:.12e},{r['base']:.12e}")
    for r in o:
        h.append(f"1,{r['v_be']:.12e},{r['v_ce']:.12e},{r['collector']:.12e},{r['base']:.12e}")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(h) + "\n")


def main() -> int:
    a = argparse.ArgumentParser(description="Generate the npn golden curves with DEVSIM.")
    a.add_argument("--out", default=os.path.join("data", "golden"), help="output directory for the CSV files")
    a.add_argument("--no-mesh-check", action="store_true", help="skip the refined mesh rerun")
    a.add_argument("--refine", type=float, default=0.5, help="divide every devsim spacing by this for the mesh check, so 0.5 doubles them")
    a.add_argument("--only", type=float, default=None, help="solve once with every spacing divided by this and write that curve alone, for a convergence study")
    o = a.parse_args()
    if o.only is not None:
        for b in P.BJT_BENCHMARKS:
            t = time.perf_counter()
            g, nodes = gummel(b, o.only)
            out = output(b, o.only)
            p = os.path.join(o.out, f"{b.name}_refine_{o.only:g}.csv")
            write(b, g, out, nodes, None, o.only, time.perf_counter() - t, p)
            print(f"[{b.name}] wrote {p}", flush=True)
        return 0
    for b in P.BJT_BENCHMARKS:
        t = time.perf_counter()
        print(f"[{b.name}] Gummel plot", flush=True)
        g, nodes = gummel(b, 1.0)
        print(f"[{b.name}] output curve at I_B {b.i_b:g} A/cm", flush=True)
        out = output(b, 1.0)
        took = time.perf_counter() - t
        p = os.path.join(o.out, f"{b.name}.csv")
        write(b, g, out, nodes, None, o.refine, took, p)
        print(f"[{b.name}] wrote {p} without a mesh check", flush=True)
        if not o.no_mesh_check:
            print(f"[{b.name}] mesh with every spacing divided by {o.refine:g}", flush=True)
            g2, _ = gummel(b, o.refine)
            o2 = output(b, o.refine)
            mesh = max(change(g, g2), change(out, o2))
            print(f"[{b.name}] the refined mesh moves a current by {mesh:.3e}", flush=True)
            write(b, g, out, nodes, mesh, o.refine, took, p)
            print(f"[{b.name}] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
