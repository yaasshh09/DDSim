from __future__ import annotations
import argparse
import datetime
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
from devsim import add_1d_contact, add_1d_interface, add_1d_mesh_line, add_1d_region, contact_equation, create_1d_mesh, create_device, equation, finalize_mesh, get_node_model_values, set_node_value, set_parameter, solve
from devsim.python_packages.model_create import CreateContactNodeModel, CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel, CreateNodeModelDerivative, CreateSolution
from devsim.python_packages.simple_physics import CreateOxideContact, CreateOxidePotentialOnly, CreateSiliconOxideInterface

SILICON, OXIDE, GATE, BODY, DEVICE = "bulk", "oxide", "gate", "body", "dg"
INPUTS = os.path.join("data", "golden", f"{P.DG_NAME}_inputs.json")


def build(x: dict[str, Any]) -> None:
    y, top = x["y"], x["interface"]
    create_1d_mesh(mesh=DEVICE)
    for k, v in enumerate(y):
        tag = "body" if k == 0 else "iface" if k == top else "gate" if k == len(y) - 1 else None
        if tag is None:
            add_1d_mesh_line(mesh=DEVICE, pos=v, ps=1.0, ns=1.0)
        else:
            add_1d_mesh_line(mesh=DEVICE, pos=v, ps=1.0, ns=1.0, tag=tag)
    add_1d_contact(mesh=DEVICE, name=BODY, tag="body", material="metal")
    add_1d_contact(mesh=DEVICE, name=GATE, tag="gate", material="metal")
    add_1d_interface(mesh=DEVICE, name="si_ox", tag="iface")
    add_1d_region(mesh=DEVICE, material="Silicon", region=SILICON, tag1="body", tag2="iface")
    add_1d_region(mesh=DEVICE, material="Oxide", region=OXIDE, tag1="iface", tag2="gate")
    finalize_mesh(mesh=DEVICE)
    create_device(mesh=DEVICE, device=DEVICE)


def physics(x: dict[str, Any]) -> None:
    V_t = x["k_B"] * x["T"] / x["q"]
    b0 = x["hbar"] ** 2 / (6 * x["q"] * P.DG_MASS * x["m_0"]) * 1e4
    box0 = x["hbar"] ** 2 / (6 * x["q"] * P.DG_MASS_OX * x["m_0"]) * 1e4
    x_np = x["hbar"] / (2 * P.DG_MASS_TUNNEL * x["m_0"] * x["q"] * P.DG_BARRIER) ** 0.5 * 1e2
    for name, value in (("Permittivity", x["eps_si"]), ("ElectronCharge", x["q"]), ("n_i", x["n_i"]), ("V_t", V_t), ("b0", b0), ("box0", box0), ("x_np", x_np), ("Gamman", 0.0), ("Gammanox", 0.0)):
        set_parameter(device=DEVICE, region=SILICON, name=name, value=value)
    set_parameter(device=DEVICE, region=OXIDE, name="Permittivity", value=x["eps_ox"])
    set_parameter(device=DEVICE, region=OXIDE, name="ElectronCharge", value=x["q"])
    for r in (SILICON, OXIDE):
        CreateSolution(DEVICE, r, "Potential")
    for name in ("Le", "NetDoping", "AtOx"):
        CreateSolution(DEVICE, SILICON, name)
    for k, v in enumerate(x["doping"][: x["interface"] + 1]):
        set_node_value(device=DEVICE, region=SILICON, name="NetDoping", index=k, value=v)
    set_node_value(device=DEVICE, region=SILICON, name="AtOx", index=x["interface"], value=1.0)
    charge = (("Electrons", "n_i*exp((Potential-Le)/V_t)"), ("Holes", "n_i*exp(-Potential/V_t)"), ("SiliconCharge", "Holes - Electrons + NetDoping"), ("PotentialCharge", "-ElectronCharge*SiliconCharge"))
    for name, expr in charge:
        CreateNodeModel(DEVICE, SILICON, name, expr)
        for v in ("Potential", "Le"):
            CreateNodeModelDerivative(DEVICE, SILICON, name, expr, v)
    for name, expr in (("ElectricField", "(Potential@n0-Potential@n1)*EdgeInverseLength"), ("PotentialEdgeFlux", "Permittivity*ElectricField")):
        CreateEdgeModel(DEVICE, SILICON, name, expr)
        CreateEdgeModelDerivatives(DEVICE, SILICON, name, expr, "Potential")
    equation(device=DEVICE, region=SILICON, name="PotentialEquation", variable_name="Potential", node_model="PotentialCharge", edge_model="PotentialEdgeFlux", variable_update="log_damp")
    gradient = (("d_l_n", "EdgeInverseLength*((Potential@n1-Potential@n0)-(Le@n1-Le@n0))/V_t"), ("del_log_n1", "0.5*Gamman*b0*d_l_n"), ("del_log_n2", "0.25*Gamman*b0*d_l_n^2"))
    for name, expr in gradient:
        CreateEdgeModel(DEVICE, SILICON, name, expr)
        for v in ("Potential", "Le"):
            CreateEdgeModelDerivatives(DEVICE, SILICON, name, expr, v)
    CreateEdgeModel(DEVICE, SILICON, "EdgeNodeVolume", "0.5*EdgeCouple*EdgeLength")
    set_parameter(device=DEVICE, region=SILICON, name="edge_node0_volume_model", value="EdgeNodeVolume")
    set_parameter(device=DEVICE, region=SILICON, name="edge_node1_volume_model", value="EdgeNodeVolume")
    wall = "Le - AtOx*Gammanox*box0/(NodeVolume*x_np)"
    CreateNodeModel(DEVICE, SILICON, "Le_eqn", wall)
    CreateNodeModelDerivative(DEVICE, SILICON, "Le_eqn", wall, "Le")
    equation(device=DEVICE, region=SILICON, name="LeEquation", variable_name="Le", node_model="Le_eqn", edge_model="del_log_n1", edge_volume_model="del_log_n2")
    CreateOxidePotentialOnly(DEVICE, OXIDE, "log_damp")
    set_parameter(device=DEVICE, name=f"{BODY}_bias", value=x["body"])
    set_parameter(device=DEVICE, name=f"{GATE}_bias", value=x["gate_potentials"][0])
    CreateContactNodeModel(DEVICE, BODY, "body_potential", f"Potential - {BODY}_bias")
    CreateContactNodeModel(DEVICE, BODY, "body_potential:Potential", "1")
    contact_equation(device=DEVICE, contact=BODY, name="PotentialEquation", node_model="body_potential")
    CreateContactNodeModel(DEVICE, BODY, "body_lambda", "Le")
    CreateContactNodeModel(DEVICE, BODY, "body_lambda:Le", "1")
    contact_equation(device=DEVICE, contact=BODY, name="LeEquation", node_model="body_lambda")
    CreateOxideContact(DEVICE, OXIDE, GATE)
    CreateSiliconOxideInterface(DEVICE, "si_ox")


def settle() -> None:
    solve(type="dc", absolute_error=1e-13, relative_error=1e-6, maximum_iterations=100)


def sheet_charge(x: dict[str, Any]) -> float:
    v = get_node_model_values(device=DEVICE, region=SILICON, name="NodeVolume")
    c = get_node_model_values(device=DEVICE, region=SILICON, name="SiliconCharge")
    return x["q"] * sum(a * b for a, b in zip(v, c, strict=True))


def sweep(x: dict[str, Any]) -> list[dict[str, float]]:
    settle()
    share = 0.0
    while share < 1.0:
        share = min(1.0, share + 0.05)
        set_parameter(device=DEVICE, region=SILICON, name="Gamman", value=P.DG_GAMMA * share)
        set_parameter(device=DEVICE, region=SILICON, name="Gammanox", value=P.DG_GAMMA_OX * share)
        settle()
    rows = []
    for v, g in zip(x["gates"], x["gate_potentials"], strict=True):
        set_parameter(device=DEVICE, name=f"{GATE}_bias", value=g)
        settle()
        rows.append({"gate": v, "charge": sheet_charge(x)})
        print(f"    Vg={v:+.3f}  Qs={rows[-1]['charge']:+.6e} C/cm^2", flush=True)
    return rows


def main() -> int:
    a = argparse.ArgumentParser(description="Generate benchmark 20, density gradient on the 2 nm MOS capacitor, with DEVSIM.")
    a.add_argument("--inputs", default=INPUTS)
    a.add_argument("--out", default=os.path.join("data", "golden"))
    o = a.parse_args()
    with open(o.inputs, encoding="utf-8") as f:
        x = json.load(f)
    build(x)
    n = len(get_node_model_values(device=DEVICE, region=SILICON, name="NodeVolume")) + len(get_node_model_values(device=DEVICE, region=OXIDE, name="NodeVolume")) - 1
    if n != len(x["y"]):
        raise RuntimeError(f"devsim made {n} nodes from {len(x['y'])} mesh lines, so the two codes would not share one mesh")
    physics(x)
    rows = sweep(x)
    import devsim
    day = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    p = os.path.join(o.out, f"{P.DG_NAME}.csv")
    head = [
        f"# device: {P.DG_NAME}",
        f"# benchmark: {P.DG_NUMBER} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_dg.py",
        f"# generator: devsim {devsim.__version__} on python {sys.version.split()[0]}",
        f"# generated on: {day}",
        f"# tolerance: {P.DG_TOLERANCE}",
        f"# gamma: {P.DG_GAMMA}",
        "# inputs: " + o.inputs.replace(os.sep, "/"),
        f"# nodes: {n}",
        "# notes: " + P.DG_NOTES,
        "# columns: gate bias [V], silicon sheet charge [C/cm^2]",
        "gate,charge",
    ]
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(head + [f"{r['gate']:.10g},{r['charge']:.12e}" for r in rows]) + "\n")
    print(f"[{P.DG_NAME}] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
