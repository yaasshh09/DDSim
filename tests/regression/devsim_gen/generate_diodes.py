"""Generate the tier 4 golden diode curves with DEVSIM.

This is the only file in the repository that imports devsim, and it does not
run under the project interpreter. See README.md in this directory for the
environment. Run it from the repository root:

    .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/generate_diodes.py

It writes one CSV per benchmark into `data/golden/`, each carrying a header
that records the devsim version, the model choices and a mesh refinement self
check, so a curve can be read years later without having to guess how it was
made.

The physics is built from devsim's own `simple_physics` helpers, with every
parameter overridden to the ddsim value. The helpers ship with eps_r = 11.1,
q = 1.6e-19 and mu_n = 400, none of which are what ddsim uses, so the override
is the whole point rather than a detail. docs/04-validation.md: match the
models before comparing numbers.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P  # noqa: E402
from devsim import (  # noqa: E402
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    delete_device,
    delete_mesh,
    finalize_mesh,
    get_contact_current,
    node_model,
    set_node_values,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import (  # noqa: E402
    CreateNodeModel,
    CreateSolution,
)
from devsim.python_packages.simple_physics import (  # noqa: E402
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    ece_name,
    hce_name,
)

REGION = "bulk"
"""The single silicon region. Every device here is one material."""

ANODE = "anode"
"""p side contact, the one that gets swept."""

CATHODE = "cathode"
"""n side contact, held at zero."""


def devsim_version() -> str:
    """The installed devsim version [1], for the golden file header."""
    try:
        from importlib.metadata import version

        return version("devsim")
    except Exception:  # pragma: no cover - only if metadata is missing
        return "unknown"


def build_mesh(
    benchmark: P.DiodeBenchmark, device: str, refine: float = 1.0
) -> None:
    """Create and finalise the 1D mesh for one benchmark.

    Args:
        benchmark: the device definition.
        device: devsim device name.
        refine: divide every mesh spacing by this. 2.0 halves the spacing
            everywhere, which is how the mesh convergence self check is run.

    The junction sits on a mesh line, so the abrupt doping step lands on a node
    rather than inside a cell in either code. docs/04-validation.md warns that
    putting it inside a cell moves the metallurgical junction by half a cell.
    """
    mesh = device
    h_j = benchmark.devsim_h_junction / refine
    h_b = benchmark.devsim_h_bulk / refine

    create_1d_mesh(mesh=mesh)
    add_1d_mesh_line(mesh=mesh, pos=0.0, ps=h_b, tag="top")
    add_1d_mesh_line(mesh=mesh, pos=benchmark.junction, ps=h_j, ns=h_j, tag="mid")
    add_1d_mesh_line(mesh=mesh, pos=benchmark.length, ps=h_b, ns=h_b, tag="bot")
    add_1d_contact(mesh=mesh, name=ANODE, tag="top", material="metal")
    add_1d_contact(mesh=mesh, name=CATHODE, tag="bot", material="metal")
    add_1d_region(mesh=mesh, material="Silicon", region=REGION, tag1="top", tag2="bot")
    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=device)


def set_silicon_parameters(device: str) -> None:
    """Push every ddsim constant into devsim, overriding its own defaults.

    Nothing here is left at a devsim default. A parameter that is not written
    down in `parameters.py` is a parameter the two codes are free to disagree
    about.
    """
    values: dict[str, float] = {
        "Permittivity": P.EPS_R_SI * P.EPS_0,
        "ElectronCharge": P.Q,
        "n_i": P.N_I,
        "T": P.T,
        "kT": P.K_B * P.T,
        "V_t": P.V_T,
        "mu_n": P.MU_N,
        "mu_p": P.MU_P,
        # Midgap traps, so both SRH reference densities are n_i.
        "n1": P.N_I,
        "p1": P.N_I,
    }
    for name, value in values.items():
        set_parameter(device=device, region=REGION, name=name, value=value)


def set_doping(benchmark: P.DiodeBenchmark, device: str) -> None:
    """The abrupt junction, right continuous at the junction like ddsim's Step."""
    equation = (
        f"ifelse(x < {benchmark.junction:.16e}, "
        f"{-benchmark.Na:.16e}, {benchmark.Nd:.16e})"
    )
    node_model(device=device, region=REGION, name="NetDoping", equation=equation)


def set_lifetimes(device: str) -> None:
    """Scharfetter doping dependent lifetimes as node models, not parameters.

    devsim's SRH expression names `taun` and `taup`. Creating node models under
    those names shadows the scalar parameters its helpers would otherwise set,
    which is how the doping dependence gets in without touching the helper. The
    lifetimes do not depend on the carrier densities, so the SRH derivatives
    devsim differentiates symbolically are unaffected.

    ddsim feeds the Scharfetter relation abs(net doping) rather than Na + Nd,
    because only the net is available on a step profile. Same choice here, for
    the same reason, so the two agree node for node.
    """
    for name, tau_max, tau_min in (
        ("taun", P.TAU_N_MAX, P.TAU_N_MIN),
        ("taup", P.TAU_P_MAX, P.TAU_P_MIN),
    ):
        equation = (
            f"{tau_min:.16e} + ({tau_max:.16e} - {tau_min:.16e}) / "
            f"(1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})"
        )
        CreateNodeModel(device, REGION, name, equation)


def build_physics(device: str) -> None:
    """Equilibrium solve, then the full drift diffusion system."""
    CreateSolution(device, REGION, "Potential")
    CreateSiliconPotentialOnly(device, REGION)
    for contact in (ANODE, CATHODE):
        set_parameter(device=device, name=f"{contact}_bias", value=0.0)
        CreateSiliconPotentialOnlyContact(device, REGION, contact)

    solve(
        type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60
    )

    for name, source in (
        ("Electrons", "IntrinsicElectrons"),
        ("Holes", "IntrinsicHoles"),
    ):
        CreateSolution(device, REGION, name)
        set_node_values(device=device, region=REGION, name=name, init_from=source)

    set_lifetimes(device)
    CreateSiliconDriftDiffusion(device, REGION, mu_n="mu_n", mu_p="mu_p")
    for contact in (ANODE, CATHODE):
        CreateSiliconDriftDiffusionAtContact(device, REGION, contact)

    solve(
        type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=60
    )


def anode_current(device: str) -> float:
    """Terminal current into the anode [A/cm^2].

    devsim reports the electron and hole contact currents separately and their
    sum already carries ddsim's convention: positive means conventional current
    flowing from the contact into the device, so a forward biased diode is
    positive at the anode and the two terminals sum to zero. That was measured
    against ddsim rather than assumed, on the 1e16 symmetric diode: both codes
    give about +5.2e2 A/cm^2 at 0.7 V and about -8.3e-9 A/cm^2 at -1 V.

    A 1D devsim device has unit cross section, so the number is already a
    density and needs no area division.
    """
    electrons = get_contact_current(device=device, contact=ANODE, equation=ece_name)
    holes = get_contact_current(device=device, contact=ANODE, equation=hce_name)
    return electrons + holes


def cathode_current(device: str) -> float:
    """Terminal current into the cathode [A/cm^2]. Sums to zero with the anode."""
    electrons = get_contact_current(device=device, contact=CATHODE, equation=ece_name)
    holes = get_contact_current(device=device, contact=CATHODE, equation=hce_name)
    return electrons + holes


def ramp_to(device: str, target: float, present: float, step: float) -> float:
    """Walk the anode bias from present to target [V], solving at each step.

    Returns the bias actually reached, which is the target unless a solve
    raised. Continuation exists because a diode solved cold at 0.7 V does not
    converge, in devsim any more than in ddsim.
    """
    if abs(target - present) < 1e-15:
        return present

    direction = 1.0 if target > present else -1.0
    steps = max(1, int(round(abs(target - present) / step)))
    for index in range(1, steps + 1):
        bias = present + direction * step * index
        if index == steps:
            bias = target
        set_parameter(device=device, name=f"{ANODE}_bias", value=bias)
        solve(
            type="dc",
            absolute_error=1e10,
            relative_error=1e-12,
            maximum_iterations=60,
        )
    return target


def sweep(benchmark: P.DiodeBenchmark, refine: float = 1.0) -> list[dict[str, Any]]:
    """Solve one benchmark at every requested bias and return the curve.

    The voltage list is walked outward from zero in each direction, negatives
    descending and then positives ascending, each leg continued from the
    equilibrium solution rather than from the far end of the other leg.
    """
    device = f"{benchmark.name}_r{refine:g}".replace(".", "_")
    build_mesh(benchmark, device, refine=refine)
    set_doping(benchmark, device)
    set_silicon_parameters(device)
    build_physics(device)

    negatives = sorted((v for v in benchmark.voltages if v < 0.0), reverse=True)
    positives = sorted(v for v in benchmark.voltages if v >= 0.0)

    rows: dict[float, dict[str, Any]] = {}
    present = 0.0
    for leg in (negatives, positives):
        present = ramp_to(device, 0.0, present, step=0.05)
        for target in leg:
            present = ramp_to(device, target, present, step=0.05)
            rows[target] = {
                "voltage": target,
                "current": anode_current(device),
                "cathode": cathode_current(device),
            }

    delete_device(device=device)
    delete_mesh(mesh=device)
    return [rows[v] for v in sorted(rows)]


def relative_difference(
    coarse: list[dict[str, Any]], fine: list[dict[str, Any]]
) -> tuple[float, float]:
    """Worst relative current difference between two meshes, and where [1, V].

    Points under P.CURRENT_FLOOR are skipped, since a relative comparison
    between two roundoff residues says nothing about the mesh.

    Refining does not always improve this number and is not expected to. In
    reverse bias the terminal current is a cancellation between drift and
    diffusion terms that scale as 1/h, so halving the spacing doubles the
    quantity being cancelled and costs about a factor of two in the last digits
    that survive. That is a property of Scharfetter-Gummel at low current, not
    of the mesh being too coarse, and it is why the golden mesh is the coarse
    one rather than the finest that would still run.
    """
    worst = 0.0
    where = 0.0
    for a, b in zip(coarse, fine, strict=True):
        if abs(a["current"]) < P.CURRENT_FLOOR and abs(b["current"]) < P.CURRENT_FLOOR:
            continue
        denominator = max(abs(a["current"]), abs(b["current"]))
        difference = abs(a["current"] - b["current"]) / denominator
        if difference > worst:
            worst, where = difference, a["voltage"]
    return worst, where


def write_csv(
    benchmark: P.DiodeBenchmark,
    rows: list[dict[str, Any]],
    mesh_check: tuple[float, float] | None,
    path: str,
) -> None:
    """Write one golden curve, header and all."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    lines: list[str] = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_diodes.py",
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {stamp}",
        f"# tolerance: {benchmark.tolerance}",
        f"# Na: {benchmark.Na:.6e}",
        f"# Nd: {benchmark.Nd:.6e}",
        f"# length: {benchmark.length:.6e}",
        f"# junction: {benchmark.junction:.6e}",
        f"# devsim h_junction: {benchmark.devsim_h_junction:.6e}",
        f"# devsim h_bulk: {benchmark.devsim_h_bulk:.6e}",
    ]
    if mesh_check is not None:
        worst, where = mesh_check
        lines.append(
            f"# mesh convergence: {worst:.3e} worst relative change in current "
            f"when every spacing is halved, at {where:+g} V"
        )
    lines.append("# notes: " + benchmark.notes)
    lines.append("# models:")
    lines.extend("#   " + line for line in P.MODEL_SUMMARY)
    lines.append(
        "# columns: anode bias [V], anode current [A/cm^2], "
        "cathode current [A/cm^2]"
    )
    lines.append("voltage,current,cathode_current")
    for row in rows:
        lines.append(
            f"{row['voltage']:.10g},{row['current']:.12e},{row['cathode']:.12e}"
        )

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    """Generate every benchmark named on the command line, or all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        default=None,
        help="benchmark names to generate, default all",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "golden"),
        help="output directory for the CSV files",
    )
    parser.add_argument(
        "--no-mesh-check",
        action="store_true",
        help="skip the halved mesh rerun, which roughly doubles the runtime",
    )
    args = parser.parse_args()

    chosen = P.BENCHMARKS
    if args.names:
        chosen = tuple(P.BY_NAME[name] for name in args.names)

    os.makedirs(args.out, exist_ok=True)
    for benchmark in chosen:
        print(f"[{benchmark.name}] solving on the reference mesh")
        rows = sweep(benchmark, refine=1.0)

        mesh_check: tuple[float, float] | None = None
        if not args.no_mesh_check:
            print(f"[{benchmark.name}] solving again on a halved mesh")
            fine = sweep(benchmark, refine=2.0)
            mesh_check = relative_difference(rows, fine)
            print(
                f"[{benchmark.name}] mesh convergence {mesh_check[0]:.3e} "
                f"at {mesh_check[1]:+g} V"
            )

        path = os.path.join(args.out, f"{benchmark.name}.csv")
        write_csv(benchmark, rows, mesh_check, path)
        print(f"[{benchmark.name}] wrote {path} with {len(rows)} points")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
