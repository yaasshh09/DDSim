"""Generate the tier 4 MOS capacitor golden data with devsim.

Run under the devsim interpreter, not the project one:

    .venv-devsim/Scripts/python.exe -m tests.regression.devsim_gen.generate_mos_cv

It writes one CSV per benchmark into `data/golden/`, each carrying a header
that records the devsim version, the device, the model choices and the mesh
convergence check, so a golden file can be read years later without guessing
how it was made. See the README in this directory.

Why devsim solves this in one dimension and ddsim in two
--------------------------------------------------------
A MOS capacitor is a one dimensional problem, and devsim is the reference
rather than the thing under test. Solving it in 1D here makes the reference as
simple as it can be, and it makes the comparison stronger rather than weaker:
ddsim's answer comes off a structured 2D mesh with a different grading and a
different assembly, so the two agree because the physics agrees rather than
because the meshes match.

What is compared, and why the charge rather than the capacitance
----------------------------------------------------------------
The gate charge is what each code actually computes. Both get it the same way,
as the flux of D over the contact's own cell, which is the discrete Gauss law
there rather than a second calculation of the same thing.

The capacitance is then a derivative, and the two codes take it differently:
ddsim differentiates its solved system exactly, devsim has no such path here.
Storing the charge and applying one central difference to both keeps the
comparison about the physics instead of about the differentiation, and the
exact derivative is checked against ddsim's own central difference in
tests/analytic/test_mos_cv.py where that is the question being asked.

The potential reference is the same in both codes
-------------------------------------------------
devsim's `IntrinsicElectrons = n_i*exp(Potential/V_t)` puts its zero at the
intrinsic level, which is where ddsim's psi has always been. That is what lets
the gate bias be handed over as `V_gate + (PHI_M_MIDGAP - Phi_M)` with no
further translation, and it is worth stating because a constant offset between
the two references would shift the whole C-V curve while leaving its shape
perfect.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import os
import sys
from typing import Any

from devsim import (  # noqa: E402
    add_1d_contact,
    add_1d_interface,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_charge,
    node_model,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import CreateSolution  # noqa: E402
from devsim.python_packages.simple_physics import (  # noqa: E402
    CreateOxideContact,
    CreateOxidePotentialOnly,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
)

from tests.regression.devsim_gen import parameters as P  # noqa: E402

SILICON = "bulk"
OXIDE = "oxide"
GATE = "gate"
BODY = "body"

GOLDEN = os.path.join("data", "golden")


def devsim_version() -> str:
    """The devsim version string, for the golden file header."""
    import devsim

    return str(getattr(devsim, "__version__", "unknown"))


@contextlib.contextmanager
def quiet():
    """Swallow devsim's per iteration convergence report.

    It writes to stdout, and the generator's stdout is a progress log a person
    reads. The errors still surface: a solve that fails raises.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def build_mesh(
    benchmark: P.MosBenchmark, device: str, refine: float = 1.0
) -> None:
    """Create and finalise the stack for one benchmark.

    Args:
        benchmark: the device definition.
        device: devsim device name.
        refine: divide every spacing by this. 1 is the golden mesh.

    Silicon below, oxide above, the interface on the node line they share. The
    silicon is graded to the surface, where the inversion layer sits; the oxide
    is uniform, because with no charge in it its potential is a straight line
    and uniform cells resolve a straight line exactly.

    devsim grades between consecutive mesh lines, and each line names its
    spacing separately for each direction: `ps` walking in +x, `ns` walking in
    -x. The interface line is the only one with material on both sides, so it
    is the only one where the two differ and the only one where getting them
    the wrong way round is silent. It buys a mesh that looks refined, refines
    the wrong region, and converges to the right answer slowly.
    """
    mesh = f"{benchmark.name}_r{refine:g}"
    h_surface = benchmark.devsim_h_surface / refine
    h_bulk = benchmark.devsim_h_bulk / refine
    h_oxide = benchmark.t_ox / (benchmark.devsim_oxide_cells * refine)

    interface = benchmark.t_si
    top = benchmark.t_si + benchmark.t_ox

    create_1d_mesh(mesh=mesh)
    add_1d_mesh_line(mesh=mesh, pos=0.0, ps=h_bulk, tag="body")
    # ps is the spacing walking in +x from this line, ns the spacing walking
    # in -x. The silicon is below the interface and the oxide above it, so the
    # surface spacing is the ns of this line, not its ps.
    add_1d_mesh_line(
        mesh=mesh, pos=interface, ns=h_surface, ps=h_oxide, tag="iface"
    )
    add_1d_mesh_line(mesh=mesh, pos=top, ps=h_oxide, tag="gate")

    add_1d_contact(mesh=mesh, name=BODY, tag="body", material="metal")
    add_1d_contact(mesh=mesh, name=GATE, tag="gate", material="metal")
    add_1d_region(
        mesh=mesh, material="Silicon", region=SILICON, tag1="body", tag2="iface"
    )
    add_1d_region(
        mesh=mesh, material="Oxide", region=OXIDE, tag1="iface", tag2="gate"
    )
    add_1d_interface(mesh=mesh, name="si_ox", tag="iface")

    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=device)


def set_material_parameters(device: str) -> None:
    """Push every ddsim constant into devsim, overriding its own defaults.

    devsim's own `simple_physics` carries eps_r(Si) = 11.1, q = 1.6e-19 and
    eps_0 = 8.85e-14, all of which are the right numbers rounded. Left alone
    they would put a percent into the comparison before any physics happened.
    """
    silicon = {
        "Permittivity": P.EPS_R_SI * P.EPS_0,
        "ElectronCharge": P.Q,
        "n_i": P.N_I,
        "T": P.T,
        "kT": P.K_B * P.T,
        "V_t": P.V_T,
    }
    for name, value in silicon.items():
        set_parameter(device=device, region=SILICON, name=name, value=value)

    oxide = {
        "Permittivity": P.EPS_R_OX * P.EPS_0,
        "ElectronCharge": P.Q,
    }
    for name, value in oxide.items():
        set_parameter(device=device, region=OXIDE, name=name, value=value)


def build_physics(benchmark: P.MosBenchmark, device: str) -> None:
    """Poisson with Boltzmann carriers in the silicon, Laplace in the oxide."""
    node_model(
        device=device,
        region=SILICON,
        name="NetDoping",
        equation=f"{benchmark.substrate_doping:.16e}",
    )

    for region in (SILICON, OXIDE):
        CreateSolution(device, region, "Potential")

    CreateSiliconPotentialOnly(device, SILICON)
    CreateOxidePotentialOnly(device, OXIDE, "log_damp")

    for contact in (BODY, GATE):
        set_parameter(device=device, name=f"{contact}_bias", value=0.0)
    CreateSiliconPotentialOnlyContact(device, SILICON, BODY)

    # No work function of its own: devsim's CreateOxideContact pins Potential
    # at the contact bias, so the work function is folded into the number
    # handed over. See gate_potential below.
    CreateOxideContact(device, OXIDE, GATE)

    CreateSiliconOxideInterface(device, "si_ox")


def gate_potential(benchmark: P.MosBenchmark, v_gate: float) -> float:
    """The potential to pin the gate at [V], measured from the intrinsic level.

    docs/01-physics.md writes the gate condition as psi_gate = V_gate - Phi_MS,
    which needs the doping under the gate. Since psi is measured from the
    intrinsic level in both codes, and the work function of intrinsic silicon
    is exactly chi + Eg/2, the same statement is

        psi_gate = V_gate + (chi + Eg/2 - Phi_M)

    and the doping has cancelled. ddsim uses the second form for the same
    reason: a contact has no business reading the substrate under it.
    """
    return v_gate + (P.PHI_M_MIDGAP - benchmark.work_function)


def sweep(benchmark: P.MosBenchmark, refine: float = 1.0) -> list[dict[str, Any]]:
    """Solve one benchmark at every requested gate bias and return the curve.

    Walked in ascending order and continued from the point before it, since the
    stack at one bias is a good guess for the next. The sweep starts at the
    most negative bias, which is deep accumulation and the easiest end.
    """
    device = f"{benchmark.name}_r{refine:g}".replace(".", "_")
    with quiet():
        build_mesh(benchmark, device, refine=refine)
    set_material_parameters(device)
    build_physics(benchmark, device)

    rows: list[dict[str, Any]] = []
    for v_gate in benchmark.voltages:
        set_parameter(
            device=device,
            name=f"{GATE}_bias",
            value=gate_potential(benchmark, v_gate),
        )
        with quiet():
            solve(
                type="dc",
                absolute_error=1e-10,
                relative_error=1e-12,
                maximum_iterations=100,
            )
        charge = get_contact_charge(
            device=device, contact=GATE, equation="PotentialEquation"
        )
        rows.append({"voltage": v_gate, "charge": float(charge)})
    return rows


def relative_difference(
    coarse: list[dict[str, Any]], fine: list[dict[str, Any]]
) -> tuple[float, float]:
    """Worst relative charge difference between two meshes, and where [1, V].

    Skips the points where the charge passes through zero near flatband, since
    a relative comparison between two numbers that are both nearly nothing says
    nothing about the mesh. The floor is a thousandth of the largest charge on
    the curve, which is four decades below anything the comparison cares about.
    """
    largest = max(abs(row["charge"]) for row in coarse)
    floor = 1e-3 * largest

    worst = 0.0
    where = 0.0
    for a, b in zip(coarse, fine, strict=True):
        if abs(a["charge"]) < floor and abs(b["charge"]) < floor:
            continue
        denominator = max(abs(a["charge"]), abs(b["charge"]))
        difference = abs(a["charge"] - b["charge"]) / denominator
        if difference > worst:
            worst, where = difference, a["voltage"]
    return worst, where


def write_csv(
    benchmark: P.MosBenchmark,
    rows: list[dict[str, Any]],
    mesh_check: tuple[float, float] | None,
    path: str,
) -> None:
    """Write one golden curve, header and all."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    lines: list[str] = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_mos_cv.py",
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {stamp}",
        f"# tolerance: {benchmark.tolerance}",
        f"# substrate doping: {benchmark.substrate_doping:.6e}",
        f"# t_ox: {benchmark.t_ox:.6e}",
        f"# t_si: {benchmark.t_si:.6e}",
        f"# work function: {benchmark.work_function:.6e}",
        f"# devsim h_surface: {benchmark.devsim_h_surface:.6e}",
        f"# devsim h_bulk: {benchmark.devsim_h_bulk:.6e}",
        f"# devsim oxide cells: {benchmark.devsim_oxide_cells}",
    ]
    if mesh_check is not None:
        worst, where = mesh_check
        lines.append(
            f"# mesh convergence: {worst:.3e} worst relative change in gate "
            f"charge when every spacing is halved, at {where:+g} V"
        )
    lines.append("# notes: " + benchmark.notes)
    lines.append("# models:")
    lines.extend("#   " + line for line in P.MOS_MODEL_SUMMARY)
    lines.append("# columns: gate bias [V], gate charge [C/cm^2]")
    lines.append("gate_voltage,charge")
    for row in rows:
        lines.append(f"{row['voltage']:.10g},{row['charge']:.12e}")

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    """Generate every benchmark named on the command line, or all of them."""
    parser = argparse.ArgumentParser(description="Generate MOS C-V golden data")
    parser.add_argument(
        "names",
        nargs="*",
        default=None,
        help="benchmark names to generate. Default is all of them.",
    )
    parser.add_argument(
        "--no-mesh-check",
        action="store_true",
        help="skip the halved mesh run, which doubles the runtime.",
    )
    args = parser.parse_args()

    wanted = P.MOS_BENCHMARKS
    if args.names:
        by_name = {b.name: b for b in P.MOS_BENCHMARKS}
        missing = sorted(set(args.names) - set(by_name))
        if missing:
            print(f"no such benchmark: {missing}", file=sys.stderr)
            print(f"known: {sorted(by_name)}", file=sys.stderr)
            return 1
        wanted = tuple(by_name[name] for name in args.names)

    os.makedirs(GOLDEN, exist_ok=True)
    for benchmark in wanted:
        print(f"{benchmark.name}: solving {len(benchmark.voltages)} biases")
        rows = sweep(benchmark)

        mesh_check = None
        if not args.no_mesh_check:
            print(f"{benchmark.name}: repeating on a halved mesh")
            mesh_check = relative_difference(rows, sweep(benchmark, refine=2.0))
            print(
                f"{benchmark.name}: worst relative change {mesh_check[0]:.3e} "
                f"at {mesh_check[1]:+g} V"
            )

        path = os.path.join(GOLDEN, f"{benchmark.name}.csv")
        write_csv(benchmark, rows, mesh_check, path)
        print(f"{benchmark.name}: wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
