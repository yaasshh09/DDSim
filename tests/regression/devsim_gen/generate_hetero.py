from __future__ import annotations
import argparse
import datetime
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
from devsim import add_1d_contact, add_1d_mesh_line, add_1d_region, contact_equation, create_1d_mesh, create_device, equation, finalize_mesh, get_contact_current, get_edge_model_values, get_node_model_values, set_node_value, set_node_values, set_parameter, solve
from devsim.python_packages.model_create import CreateContactNodeModel, CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel, CreateNodeModelDerivative, CreateSolution, EnsureEdgeFromNodeModelExists

REGION, ANODE, CATHODE, DEVICE = "bulk", "anode", "cathode", "hetero"
INPUTS = os.path.join("data", "golden", f"{P.HETERO_NAME}_inputs.json")


def build(x: dict[str, Any]) -> None:
    create_1d_mesh(mesh=DEVICE)
    last = len(x["x"]) - 1
    for k, v in enumerate(x["x"]):
        tag = "top" if k == 0 else "bot" if k == last else None
        if tag is None:
            add_1d_mesh_line(mesh=DEVICE, pos=v, ps=1.0, ns=1.0)
        else:
            add_1d_mesh_line(mesh=DEVICE, pos=v, ps=1.0, ns=1.0, tag=tag)
    add_1d_contact(mesh=DEVICE, name=ANODE, tag="top", material="metal")
    add_1d_contact(mesh=DEVICE, name=CATHODE, tag="bot", material="metal")
    add_1d_region(mesh=DEVICE, material="Silicon", region=REGION, tag1="top", tag2="bot")
    finalize_mesh(mesh=DEVICE)
    create_device(mesh=DEVICE, device=DEVICE)


def contact_values(x: dict[str, Any], k: int) -> tuple[float, float, float]:
    V_t = x["k_B"] * x["T"] / x["q"]
    N, n_i = x["doping"][k], x["n_i"]
    nl = n_i * math.exp(0.5 * (x["d_n"][k] + x["d_p"][k]))
    root = math.sqrt(0.25 * N * N + nl * nl)
    if N >= 0.0:
        n = 0.5 * N + root
        p = nl * nl / n
    else:
        p = -0.5 * N + root
        n = nl * nl / p
    return V_t * (math.log(n / n_i) - x["d_n"][k]), n, p


def physics(x: dict[str, Any]) -> None:
    V_t = x["k_B"] * x["T"] / x["q"]
    for name, value in (("ElectronCharge", x["q"]), ("n_i", x["n_i"]), ("V_t", V_t), ("mu_n", x["mu_n"]), ("mu_p", x["mu_p"])):
        set_parameter(device=DEVICE, region=REGION, name=name, value=value)
    for name in ("NetDoping", "dn", "dp", "EpsLeft"):
        CreateSolution(DEVICE, REGION, name)
    for k in range(len(x["x"])):
        set_node_value(device=DEVICE, region=REGION, name="NetDoping", index=k, value=x["doping"][k])
        set_node_value(device=DEVICE, region=REGION, name="dn", index=k, value=x["d_n"][k])
        set_node_value(device=DEVICE, region=REGION, name="dp", index=k, value=x["d_p"][k])
        eps = x["eps_r_edge"][min(k, len(x["eps_r_edge"]) - 1)]
        set_node_value(device=DEVICE, region=REGION, name="EpsLeft", index=k, value=eps * x["eps_0"])
    CreateSolution(DEVICE, REGION, "Potential")
    for name in ("Potential", "dn", "dp", "EpsLeft"):
        EnsureEdgeFromNodeModelExists(DEVICE, REGION, name)
    CreateEdgeModel(DEVICE, REGION, "EdgeEps", "EpsLeft@n0")
    got = get_edge_model_values(device=DEVICE, region=REGION, name="EdgeEps")
    want = [e * x["eps_0"] for e in x["eps_r_edge"]]
    if len(got) != len(want) or any(abs(a / b - 1.0) > 1e-14 for a, b in zip(got, want, strict=True)):
        raise RuntimeError("devsim's edges do not carry ddsim's edge permittivities, so its edges are not ordered as ddsim's")
    CreateNodeModel(DEVICE, REGION, "n_local", "n_i*exp(0.5*(dn + dp))")
    CreateNodeModel(DEVICE, REGION, "taun", f"{P.TAU_N_MIN:.16e} + ({P.TAU_N_MAX:.16e} - {P.TAU_N_MIN:.16e}) / (1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})")
    CreateNodeModel(DEVICE, REGION, "taup", f"{P.TAU_P_MIN:.16e} + ({P.TAU_P_MAX:.16e} - {P.TAU_P_MIN:.16e}) / (1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})")
    CreateNodeModel(DEVICE, REGION, "GuessPotential", "V_t*(asinh(NetDoping/(2*n_local)) - 0.5*(dn - dp))")
    set_node_values(device=DEVICE, region=REGION, name="Potential", init_from="GuessPotential")
    for name, expr in (("ElectricField", "(Potential@n0-Potential@n1)*EdgeInverseLength"), ("PotentialEdgeFlux", "EdgeEps*ElectricField")):
        CreateEdgeModel(DEVICE, REGION, name, expr)
        CreateEdgeModelDerivatives(DEVICE, REGION, name, expr, "Potential")


def potential_only(x: dict[str, Any]) -> None:
    for name, expr in (("EqElectrons", "n_i*exp(Potential/V_t + dn)"), ("EqHoles", "n_i*exp(-Potential/V_t + dp)"), ("EqCharge", "-ElectronCharge*kahan3(EqHoles, -EqElectrons, NetDoping)")):
        CreateNodeModel(DEVICE, REGION, name, expr)
        CreateNodeModelDerivative(DEVICE, REGION, name, expr, "Potential")
    equation(device=DEVICE, region=REGION, name="PotentialEquation", variable_name="Potential", node_model="EqCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
    for contact, k in ((ANODE, 0), (CATHODE, len(x["x"]) - 1)):
        set_parameter(device=DEVICE, name=f"{contact}_bias", value=0.0)
        set_parameter(device=DEVICE, name=f"{contact}_psi0", value=contact_values(x, k)[0])
        CreateContactNodeModel(DEVICE, contact, f"{contact}_potential", f"Potential - {contact}_psi0 - {contact}_bias")
        CreateContactNodeModel(DEVICE, contact, f"{contact}_potential:Potential", "1")
        contact_equation(device=DEVICE, contact=contact, name="PotentialEquation", node_model=f"{contact}_potential", edge_charge_model="PotentialEdgeFlux")
    solve(type="dc", absolute_error=1.0, relative_error=1e-13, maximum_iterations=60)


def drift_diffusion(x: dict[str, Any]) -> None:
    for name, init in (("Electrons", "EqElectrons"), ("Holes", "EqHoles")):
        CreateSolution(DEVICE, REGION, name)
        set_node_values(device=DEVICE, region=REGION, name=name, init_from=init)
        EnsureEdgeFromNodeModelExists(DEVICE, REGION, name)
    pne = "-ElectronCharge*kahan3(Holes, -Electrons, NetDoping)"
    CreateNodeModel(DEVICE, REGION, "PotentialNodeCharge", pne)
    for v in ("Electrons", "Holes"):
        CreateNodeModelDerivative(DEVICE, REGION, "PotentialNodeCharge", pne, v)
    equation(device=DEVICE, region=REGION, name="PotentialEquation", variable_name="Potential", node_model="PotentialNodeCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
    edges = (
        ("vdiff_n", "(Potential@n0 - Potential@n1)/V_t + (dn@n0 - dn@n1)"),
        ("vdiff_p", "(Potential@n0 - Potential@n1)/V_t - (dp@n0 - dp@n1)"),
        ("Bern_n", "B(vdiff_n)"),
        ("Bern_p", "B(vdiff_p)"),
        ("ElectronCurrent", "ElectronCharge*mu_n*EdgeInverseLength*V_t*kahan3(Electrons@n1*Bern_n, Electrons@n1*vdiff_n, -Electrons@n0*Bern_n)"),
        ("HoleCurrent", "-ElectronCharge*mu_p*EdgeInverseLength*V_t*kahan3(Holes@n1*Bern_p, -Holes@n0*Bern_p, -Holes@n0*vdiff_p)"),
    )
    for name, expr in edges:
        CreateEdgeModel(DEVICE, REGION, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateEdgeModelDerivatives(DEVICE, REGION, name, expr, v)
    usrh = "(Electrons*Holes - n_local^2)/(taup*(Electrons + n_local) + taun*(Holes + n_local))"
    for name, expr in (("USRH", usrh), ("ElectronGeneration", "-ElectronCharge*USRH"), ("HoleGeneration", "ElectronCharge*USRH")):
        CreateNodeModel(DEVICE, REGION, name, expr)
        for v in ("Electrons", "Holes"):
            CreateNodeModelDerivative(DEVICE, REGION, name, expr, v)
    equation(device=DEVICE, region=REGION, name="ElectronContinuityEquation", variable_name="Electrons", edge_model="ElectronCurrent", node_model="ElectronGeneration", variable_update="positive")
    equation(device=DEVICE, region=REGION, name="HoleContinuityEquation", variable_name="Holes", edge_model="HoleCurrent", node_model="HoleGeneration", variable_update="positive")
    for contact, k in ((ANODE, 0), (CATHODE, len(x["x"]) - 1)):
        _, n, p = contact_values(x, k)
        CreateContactNodeModel(DEVICE, contact, f"{contact}_n", f"Electrons - {n:.17e}")
        CreateContactNodeModel(DEVICE, contact, f"{contact}_n:Electrons", "1")
        CreateContactNodeModel(DEVICE, contact, f"{contact}_p", f"Holes - {p:.17e}")
        CreateContactNodeModel(DEVICE, contact, f"{contact}_p:Holes", "1")
        contact_equation(device=DEVICE, contact=contact, name="ElectronContinuityEquation", node_model=f"{contact}_n", edge_current_model="ElectronCurrent")
        contact_equation(device=DEVICE, contact=contact, name="HoleContinuityEquation", node_model=f"{contact}_p", edge_current_model="HoleCurrent")
    settle()


def settle() -> None:
    solve(type="dc", absolute_error=1e10, relative_error=1e-13, maximum_iterations=60)


def anode_current() -> float:
    return get_contact_current(device=DEVICE, contact=ANODE, equation="ElectronContinuityEquation") + get_contact_current(device=DEVICE, contact=ANODE, equation="HoleContinuityEquation")


def cathode_current() -> float:
    return get_contact_current(device=DEVICE, contact=CATHODE, equation="ElectronContinuityEquation") + get_contact_current(device=DEVICE, contact=CATHODE, equation="HoleContinuityEquation")


def equilibrium_is_flat() -> None:
    n = get_node_model_values(device=DEVICE, region=REGION, name="Electrons")
    p = get_node_model_values(device=DEVICE, region=REGION, name="Holes")
    nl = get_node_model_values(device=DEVICE, region=REGION, name="n_local")
    worst = max(abs(a * b / (c * c) - 1.0) for a, b, c in zip(n, p, nl, strict=True))
    if worst > 1e-8:
        raise RuntimeError(f"at 0 V n p departs from the local n_i^2 by {worst:.2e}, so this equilibrium is wrong and the reference would be too")
    print(f"    equilibrium n p matches n_local^2 to {worst:.2e}", flush=True)


def sweep(x: dict[str, Any]) -> list[dict[str, float]]:
    rows, present = [], 0.0
    for v in x["voltages"]:
        while present < v - 1e-12:
            present = min(v, present + 0.05)
            set_parameter(device=DEVICE, name=f"{ANODE}_bias", value=present)
            settle()
        settle()
        rows.append({"voltage": v, "current": anode_current(), "cathode": cathode_current()})
        print(f"    V={v:+.3f}  I={rows[-1]['current']:+.10e}  cathode {rows[-1]['cathode']:+.10e} A/cm^2", flush=True)
    return rows


def main() -> int:
    a = argparse.ArgumentParser(description="Generate benchmark 21, a Si(0.8)Ge(0.2) / Si heterojunction diode, with DEVSIM.")
    a.add_argument("--inputs", default=INPUTS)
    a.add_argument("--out", default=os.path.join("data", "golden"))
    o = a.parse_args()
    with open(o.inputs, encoding="utf-8") as f:
        x = json.load(f)
    build(x)
    nodes = len(get_node_model_values(device=DEVICE, region=REGION, name="NodeVolume"))
    if nodes != len(x["x"]):
        raise RuntimeError(f"devsim made {nodes} nodes from {len(x['x'])} mesh lines, so the two codes would not share one mesh")
    physics(x)
    potential_only(x)
    drift_diffusion(x)
    equilibrium_is_flat()
    rows = sweep(x)
    import devsim
    day = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    path = os.path.join(o.out, f"{P.HETERO_NAME}.csv")
    head = [
        f"# device: {P.HETERO_NAME}",
        f"# benchmark: {P.HETERO_NUMBER} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_hetero.py",
        f"# generator: devsim {devsim.__version__} on python {sys.version.split()[0]}",
        f"# generated on: {day}",
        f"# tolerance: {P.HETERO_TOLERANCE}",
        "# inputs: " + o.inputs.replace(os.sep, "/"),
        f"# nodes: {nodes}",
        "# notes: " + P.HETERO_NOTES,
        "# columns: anode bias [V], anode current [A/cm^2], cathode current [A/cm^2]",
        "voltage,current,cathode_current",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(head + [f"{r['voltage']:.10g},{r['current']:.12e},{r['cathode']:.12e}" for r in rows]) + "\n")
    print(f"[{P.HETERO_NAME}] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
