"""Generate the tier 4 MOSFET golden data with devsim.

Run under the devsim interpreter, not the project one:

    .venv-devsim/Scripts/python.exe -m tests.regression.devsim_gen.generate_mosfet

It writes one CSV per benchmark into `data/golden/`, each holding the two
transfer curves of one device and a header that records the devsim version, the
geometry, the model choices and the mesh convergence check. See the README in
this directory.

Both codes solve this one in two dimensions
-------------------------------------------
Unlike the diodes and the MOS capacitors, a MOSFET is not a 1D problem in any
useful sense: the whole subject of phases/PHASE-5.md is what happens when the
two junctions are close enough together to see each other through the body. So
devsim solves the same 2D structure ddsim does, on its own mesh, which is still
the point of the tier. The geometry, the doping and every constant are shared
through `parameters.py`; the grid is not.

Three things about devsim 2D that are not in any error message
---------------------------------------------------------------
1. A contact on the outer boundary of the mesh is dropped without a word. The
   body sits on the bottom edge and the gate on the top, so both need a region
   on the far side of them to be a boundary between two regions rather than the
   edge of the world. `air_bot` and `air_top` are that, they carry no equations
   and they do not enter the matrix. Source and drain need none of it: they sit
   on the Si/SiO2 line, which is already internal.

2. `solve` returns after one damped step and then reports RelError and AbsError
   as exactly zero. That zero is not a converged state. `settle` below calls it
   until the potential stops moving and checks the answer, which is the only
   reason anything downstream of the equilibrium is trustworthy.

3. A failed solve leaves the solution corrupted, and putting the bias back is
   not enough to recover. Every continuation step snapshots the three solution
   arrays first and restores them on a failure, which is what ddsim's own
   continuation does for the same reason.

The potential reference is the same in both codes
-------------------------------------------------
devsim's `IntrinsicElectrons = n_i*exp(Potential/V_t)` puts its zero at the
intrinsic level, which is where ddsim's psi has always been. That is what lets
the gate bias be handed over as `V_gate + (PHI_M_MIDGAP - Phi_M)` with no
further translation.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import math
import os
import sys
from typing import Any

from devsim import (  # noqa: E402
    add_2d_contact,
    add_2d_interface,
    add_2d_mesh_line,
    add_2d_region,
    create_2d_mesh,
    create_device,
    finalize_mesh,
    get_contact_current,
    get_node_model_values,
    get_parameter,
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
    CreateOxideContact,
    CreateOxidePotentialOnly,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    ece_name,
    hce_name,
)

from tests.regression.devsim_gen import parameters as P  # noqa: E402

BULK = "bulk"
OXIDE = "oxide"
SOURCE = "source"
DRAIN = "drain"
GATE = "gate"
BODY = "body"

AIR = 1e-6
"""How far the sacrificial regions extend past the device [cm].

They exist only so the body and gate contacts sit on a boundary between two
regions. Nothing is solved in them and their thickness does not enter any
answer, so it is a round number rather than a measured one.
"""

GOLDEN = os.path.join("data", "golden")

set_parameter(name="threads_available", value=1)
"""Assemble on one thread, so that a run is reproducible.

devsim splits its assembly across threads by default, and a floating point sum
whose order is set by a thread pool is a different sum from one run to the
next. On this device the difference is not cosmetic: the same mesh at the same
bias converged in one process and raised a convergence failure in another,
under nothing but a different machine load. Golden data that depends on how
busy the machine was is not golden, so the thread pool goes.
"""

SOLUTIONS: dict[str, tuple[str, ...]] = {
    BULK: ("Potential", "Electrons", "Holes"),
    OXIDE: ("Potential",),
}
"""Every solution array on the device, which is what a snapshot has to hold."""


def devsim_version() -> str:
    """The devsim version string, for the golden file header."""
    import devsim

    return str(getattr(devsim, "__version__", "unknown"))


@contextlib.contextmanager
def quiet():
    """Swallow devsim's per iteration convergence report.

    It writes to stdout, and the generator's stdout is a progress log a person
    reads. Nothing is lost: the report is per iteration and the only thing it
    says that matters, that a solve failed, arrives as an exception.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ------------------------------------------------------------------- the mesh


def build_mesh(
    benchmark: P.MosfetBenchmark, device: str, refine: float = 1.0
) -> None:
    """Create and finalise the 2D structure for one benchmark.

    Args:
        benchmark: the device definition.
        device: devsim device name.
        refine: divide every spacing by this. 1 is the golden mesh.

    Columns land on every boundary anything is measured against: the two
    contact edges, the two gate mask edges and the centre. Rows land on the
    body contact, the implant depth, the silicon surface and the gate.
    """
    mesh = device
    process = P.MOSFET_PROCESS
    sd_length = process["sd_length"]
    contact = process["contact_length"]
    t_si = process["t_si"]
    t_ox = process["t_ox"]
    x_j = process["x_j"]
    L = benchmark.L_gate
    width = 2.0 * sd_length + L

    h_contact = benchmark.devsim_h_contact / refine
    h_junction = benchmark.devsim_h_junction / refine
    h_channel = benchmark.devsim_h_channel / refine
    h_surface = benchmark.devsim_h_surface / refine
    h_depth = benchmark.devsim_h_depth / refine
    h_oxide = t_ox / (benchmark.devsim_oxide_cells * refine)

    create_2d_mesh(mesh=mesh)
    for pos, ns, ps in (
        (0.0, h_contact, h_contact),
        (contact, h_contact, h_contact),
        (sd_length, h_junction, h_junction),
        (sd_length + 0.5 * L, h_channel, h_channel),
        (sd_length + L, h_junction, h_junction),
        (width - contact, h_contact, h_contact),
        (width, h_contact, h_contact),
    ):
        add_2d_mesh_line(mesh=mesh, dir="x", pos=pos, ns=ns, ps=ps)

    # ns is the spacing walking in -y from this line and ps the spacing walking
    # in +y. The silicon is below the interface and the oxide above it, so the
    # surface spacing is the ns of the interface line and not its ps. Getting
    # the two the wrong way round is silent: it buys a mesh that looks refined,
    # refines the oxide instead of the inversion layer, and converges slowly to
    # the right answer.
    for pos, ns, ps in (
        (-AIR, AIR, AIR),
        (0.0, AIR, 0.2 * t_si),
        (t_si - x_j, h_depth, h_depth),
        (t_si, h_surface, h_oxide),
        (t_si + t_ox, h_oxide, AIR),
        (t_si + t_ox + AIR, AIR, AIR),
    ):
        add_2d_mesh_line(mesh=mesh, dir="y", pos=pos, ns=ns, ps=ps)

    add_2d_region(mesh=mesh, region=BULK, material="Silicon", yl=0.0, yh=t_si)
    add_2d_region(
        mesh=mesh, region=OXIDE, material="Oxide", yl=t_si, yh=t_si + t_ox
    )
    add_2d_region(mesh=mesh, region="air_bot", material="metal", yl=-AIR, yh=0.0)
    add_2d_region(
        mesh=mesh,
        region="air_top",
        material="metal",
        yl=t_si + t_ox,
        yh=t_si + t_ox + AIR,
    )

    add_2d_contact(
        mesh=mesh, name=BODY, material="metal", region=BULK, yl=0.0, yh=0.0
    )
    add_2d_contact(
        mesh=mesh,
        name=SOURCE,
        material="metal",
        region=BULK,
        xl=0.0,
        xh=contact,
        yl=t_si,
        yh=t_si,
    )
    add_2d_contact(
        mesh=mesh,
        name=DRAIN,
        material="metal",
        region=BULK,
        xl=width - contact,
        xh=width,
        yl=t_si,
        yh=t_si,
    )
    add_2d_contact(
        mesh=mesh,
        name=GATE,
        material="metal",
        region=OXIDE,
        xl=sd_length,
        xh=sd_length + L,
        yl=t_si + t_ox,
        yh=t_si + t_ox,
    )
    add_2d_interface(
        mesh=mesh,
        name="si_ox",
        region0=BULK,
        region1=OXIDE,
        yl=t_si,
        yh=t_si,
    )

    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=device)


def node_count(device: str) -> int:
    """How many nodes the silicon carries, for the golden file header."""
    return len(get_node_model_values(device=device, region=BULK, name="x"))


# ---------------------------------------------------------------- the physics


def set_material_parameters(device: str) -> None:
    """Push every ddsim constant into devsim, overriding its own defaults.

    devsim's own `simple_physics` carries eps_r(Si) = 11.1, q = 1.6e-19,
    mu_n = 400 and mu_p = 200, none of which are ddsim's values. Nothing here
    is left at a devsim default: a parameter that is not written down in
    `parameters.py` is a parameter the two codes are free to disagree about.
    """
    silicon = {
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
    for name, value in silicon.items():
        set_parameter(device=device, region=BULK, name=name, value=value)
    for name, value in (
        ("Permittivity", P.EPS_R_OX * P.EPS_0),
        ("ElectronCharge", P.Q),
    ):
        set_parameter(device=device, region=OXIDE, name=name, value=value)


def set_doping(benchmark: P.MosfetBenchmark, device: str) -> None:
    """The separable implant ddsim's `nmos` builds, written as one expression.

        N(x, y) = sub + peak * lateral(x) * depth(y) + the same mirrored

    with `lateral` an erfc edge at the gate mask and `depth` a gaussian peaked
    at the surface. sigma comes from the junction depth and the two
    concentrations, the erfc length from the lateral encroachment and the same
    two, so both junctions land exactly where they were asked to in both codes.
    """
    process = P.MOSFET_PROCESS
    sigma, edge = P.implant_shape(process)
    width = 2.0 * process["sd_length"] + benchmark.L_gate

    def implant(x: str) -> str:
        return (
            f"{process['sd_peak']:.16e} * 0.5 * "
            f"erfc(({x} - {process['sd_length']:.16e})/{edge:.16e}) * "
            f"exp(-pow({process['t_si']:.16e} - y, 2)/(2*pow({sigma:.16e}, 2)))"
        )

    node_model(
        device=device,
        region=BULK,
        name="NetDoping",
        equation=(
            f"{process['substrate_doping']:.16e} + "
            + implant("x")
            + " + "
            + implant(f"({width:.16e} - x)")
        ),
    )


def set_lifetimes(device: str) -> None:
    """Scharfetter doping dependent lifetimes as node models, not parameters.

    devsim's SRH expression names `taun` and `taup`. Creating node models under
    those names shadows the scalar parameters its helpers would otherwise set,
    which is how the doping dependence gets in without touching the helper. The
    lifetimes do not depend on the carrier densities, so the SRH derivatives
    devsim differentiates symbolically are unaffected. Same choice, and the
    same `abs(NetDoping)` argument, as the diode generator.
    """
    for name, tau_max, tau_min in (
        ("taun", P.TAU_N_MAX, P.TAU_N_MIN),
        ("taup", P.TAU_P_MAX, P.TAU_P_MIN),
    ):
        CreateNodeModel(
            device,
            BULK,
            name,
            f"{tau_min:.16e} + ({tau_max:.16e} - {tau_min:.16e}) / "
            f"(1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})",
        )


def gate_potential(v_gate: float) -> float:
    """The potential to pin the gate at [V], measured from the intrinsic level.

    docs/01-physics.md writes the gate condition as psi_gate = V_gate - Phi_MS,
    which needs the doping under the gate. Since psi is measured from the
    intrinsic level in both codes, and the work function of intrinsic silicon
    is exactly chi + Eg/2, the same statement is

        psi_gate = V_gate + (chi + Eg/2 - Phi_M)

    and the doping has cancelled. ddsim uses the second form for the same
    reason: a contact has no business reading the substrate under it.
    """
    return v_gate + (P.PHI_M_MIDGAP - P.PHI_M_N_POLY)


def build_physics(device: str, v_gate: float, v_drain: float) -> None:
    """Equilibrium Poisson, then the full drift diffusion system.

    The gate bias is set before the equilibrium solve and left there. It draws
    no current, so the coupled system built on top of it is still an
    equilibrium problem, and a Poisson only solve reaches any gate bias without
    complaint because it carries no carriers to push around.

    The drain is different and is ramped on afterwards. Handing the first
    coupled solve a device that already has 50 mV across its drain junction is
    a jump into a basin nothing chose: the same mesh converges from it in one
    process and fails in another, depending only on what devsim solved before.
    Starting the coupled system at zero drain, where it is a true equilibrium,
    and walking the bias in with `ramp_to` removes the luck. It is what ddsim's
    own cold solves do, for the same reason.
    """
    for region in (BULK, OXIDE):
        CreateSolution(device, region, "Potential")
    CreateSiliconPotentialOnly(device, BULK)
    CreateOxidePotentialOnly(device, OXIDE, "log_damp")

    for contact in (BODY, SOURCE, DRAIN):
        set_parameter(device=device, name=f"{contact}_bias", value=0.0)
        CreateSiliconPotentialOnlyContact(device, BULK, contact)
    # devsim's CreateOxideContact pins Potential at the contact bias, so the
    # work function is folded into the number handed over rather than being a
    # parameter of its own. See gate_potential.
    set_parameter(device=device, name=f"{GATE}_bias", value=gate_potential(v_gate))
    CreateOxideContact(device, OXIDE, GATE)
    CreateSiliconOxideInterface(device, "si_ox")
    settle(device, poisson_only=True)

    for name, source in (
        ("Electrons", "IntrinsicElectrons"),
        ("Holes", "IntrinsicHoles"),
    ):
        CreateSolution(device, BULK, name)
        set_node_values(device=device, region=BULK, name=name, init_from=source)
    set_lifetimes(device)
    CreateSiliconDriftDiffusion(device, BULK, mu_n="mu_n", mu_p="mu_p")
    for contact in (BODY, SOURCE, DRAIN):
        CreateSiliconDriftDiffusionAtContact(device, BULK, contact)
    settle(device)
    ramp_to(device, DRAIN, v_drain)


# ------------------------------------------------------------- the solve loop


def snapshot(device: str, poisson_only: bool = False) -> dict[tuple[str, str], list]:
    """Every solution array on the device, by region and name."""
    wanted = (
        {BULK: ("Potential",), OXIDE: ("Potential",)} if poisson_only else SOLUTIONS
    )
    return {
        (region, name): list(
            get_node_model_values(device=device, region=region, name=name)
        )
        for region, names in wanted.items()
        for name in names
    }


def restore(device: str, saved: dict[tuple[str, str], list]) -> None:
    """Put a snapshot back, which is the only way to recover a failed solve."""
    for (region, name), values in saved.items():
        set_node_values(device=device, region=region, name=name, values=values)


def potential_move(
    before: dict[tuple[str, str], list], after: dict[tuple[str, str], list]
) -> float:
    """The largest change in potential at any node between two states [V].

    The potential is the right thing to watch rather than the densities: it is
    the same size everywhere on the device, so one tolerance means the same
    thing at every node, while a density spans forty decades and a relative
    move in the tail of it says nothing about whether the solve has arrived.
    """
    worst = 0.0
    for key, old in before.items():
        if key[1] != "Potential":
            continue
        for a, b in zip(old, after[key], strict=True):
            worst = max(worst, abs(a - b))
    return worst


def sane(device: str) -> bool:
    """Whether the state is physically possible at all.

    devsim's convergence report is a statement about the size of its last
    update, so an iterate that has blown up and then stopped moving is reported
    as converged. On this device that shows up as a potential of several volts
    and densities above 1e22, neither of which any bias in the sweep can
    produce, so the cheapest guard is to look at the numbers.
    """
    psi = get_node_model_values(device=device, region=BULK, name="Potential")
    if max(abs(value) for value in psi) > 5.0:
        return False
    for name in ("Electrons", "Holes"):
        values = get_node_model_values(device=device, region=BULK, name=name)
        if min(values) < 0.0 or max(values) > 1e22:
            return False
    return True


SOLVE_TOLERANCES: tuple[float, ...] = (1e-8, 1e-6, 1e-4)
"""Relative update tolerances to hand devsim, tried in order until one returns.

devsim converges a solve when its relative update falls below this and its
absolute update falls below `absolute_error`. Only the relative one binds here,
because the absolute one is a norm mixing volts with cm^-3 and no value of it
means the same thing for both.

The relative one has a floor. On the 1 um device at an inverted gate the
electron update sticks at 2.17340e-09, the identical figure at iterations 2
through 6, while the absolute update wanders between 1e4 and 4e5 cm^-3 on a
density of 1e20. That is a converged solve cycling in its own roundoff, and
asking for 1e-10 refuses it as a convergence failure. It is the same thing
docs/07-decisions.md records on the ddsim side of the MOS capacitor, where an
iterate below the residual threshold was refused on the update criterion alone.

So the tolerance handed over is a step size control and not an accuracy
control: `settle` owns accuracy, by watching the solution. Backing the
tolerance off a decade at a time until devsim stops refusing costs extra
passes, and a failed solve is rolled back by devsim and damages nothing, so
the ladder is free. It stops at 1e-4 because below that devsim returns from
the first Newton iteration before its own damping has engaged, and on this
device that hands back a transfer curve with the drain current going negative.
"""


def solve_once() -> None:
    """One devsim solve, on the tightest tolerance it will accept."""
    for index, tolerance in enumerate(SOLVE_TOLERANCES):
        try:
            solve(
                type="dc",
                absolute_error=1e30,
                relative_error=tolerance,
                maximum_iterations=100,
                maximum_error=1e40,
            )
            return
        except Exception:
            if index + 1 == len(SOLVE_TOLERANCES):
                raise


def settle(
    device: str, poisson_only: bool = False, passes: int = 400, tol: float = 1e-9
) -> int:
    """Call solve until the potential stops moving, and return the pass count.

    One devsim solve returns after a single damped step and reports RelError
    and AbsError as exactly zero, which is not a converged state: on the 1 um
    device the equilibrium potential is still 0.3 V short after the first call
    and moves another 0.12 V on the second. So convergence is decided here, by
    watching the solution rather than by reading the report.

    That is not a small point. devsim's own `python_packages/ramp.py` calls
    solve once per bias step and trusts what it says, and run that way this
    device hands back a transfer curve reading -2.209939 A/cm at 0.1 V of gate
    and +9.942419 A/cm at 0.4, with every step reported as a success.
    """
    before = snapshot(device, poisson_only)
    moved = float("inf")
    for index in range(passes):
        solve_once()
        after = snapshot(device, poisson_only)
        moved = potential_move(before, after)
        before = after
        if moved < tol:
            if not poisson_only and not sane(device):
                raise RuntimeError("settled on a state no bias can produce")
            return index + 1
    # The budget is what a failed bias step costs before ramp_to halves and
    # tries again, so it is tempting to cut it. Do not: a settle that is
    # refused early is a bias step that is refused, and cutting this to 60
    # moved the equilibrium drain current on the 1 um device by 0.4 percent
    # and left the gate walk grinding at the knee it now crosses in 35 s.
    raise RuntimeError(
        f"did not settle in {passes} passes, last move {moved:.3e} V"
    )


GROWTH_STREAK = 4
"""Consecutive successful bias steps before the step size is allowed to grow.

Growing after every success is what devsim's own `python_packages/ramp.py`
refuses to do, and the reason shows up at the inversion knee. There the step
that converges is a millivolt and the one above it does not, so a ramp that
doubles back up after each success spends every other step on a failure, and
the halving that follows eventually takes the step under `min_step` and aborts
a walk that a constant millivolt would have finished. Requiring a run of
successes first means the knee is crossed at the pace the knee needs and the
flat parts are not.
"""


def ramp_to(
    device: str,
    contact: str,
    target: float,
    step: float = 0.1,
    min_step: float = 1e-4,
) -> None:
    """Walk one contact bias to target, halving the step on a failure.

    The step grows back, but only after `GROWTH_STREAK` steps in a row have
    converged, because the hard part of a transfer curve is the knee and there
    is no reason to crawl the rest of it at the pace the knee needed.
    """
    present = get_parameter(device=device, name=f"{contact}_bias")
    streak = 0
    while abs(target - present) > 1e-12:
        saved = snapshot(device)
        move = min(step, abs(target - present))
        nxt = present + math.copysign(move, target - present)
        set_parameter(device=device, name=f"{contact}_bias", value=nxt)
        try:
            with quiet():
                settle(device)
        except Exception:
            restore(device, saved)
            set_parameter(device=device, name=f"{contact}_bias", value=present)
            step *= 0.5
            streak = 0
            if step < min_step:
                raise RuntimeError(
                    f"{contact} stuck at {present:g} V heading to {target:g} V "
                    f"on {device}: no step above {min_step:g} V converges"
                ) from None
            continue
        present = nxt
        streak += 1
        if streak >= GROWTH_STREAK:
            step = min(2.0 * step, 0.1)
            streak = 0


def terminal_current(device: str, contact: str) -> float:
    """Current into one terminal [A/cm].

    devsim reports the electron and hole contact currents separately and their
    sum already carries ddsim's convention: positive means conventional current
    flowing from the contact into the device. A 2D devsim device has unit
    depth, so the number is per centimetre of channel width and needs no area
    division, which is the same normalisation ddsim reports.
    """
    return get_contact_current(
        device=device, contact=contact, equation=ece_name
    ) + get_contact_current(device=device, contact=contact, equation=hce_name)


# ----------------------------------------------------------------- the sweeps


def device_name(
    benchmark: P.MosfetBenchmark, drain: float, refine: float = 1.0
) -> str:
    """The devsim device name for one curve. Dots are not legal in one."""
    return f"{benchmark.name}_d{drain:g}_r{refine:g}".replace(".", "_")


def transfer_curve(
    benchmark: P.MosfetBenchmark, drain: float, refine: float = 1.0
) -> list[dict[str, Any]]:
    """One Id-Vg curve at a fixed drain bias.

    A device is built per curve rather than per point. The gate is walked from
    the first bias to the last with continuation, since the state at one bias
    is the best guess available for the next, and the drain is already on the
    device before any carriers exist because it went in with the equilibrium
    Poisson solve.
    """
    device = device_name(benchmark, drain, refine)
    with quiet():
        build_mesh(benchmark, device, refine=refine)
    set_material_parameters(device)
    set_doping(benchmark, device)
    with quiet():
        build_physics(device, benchmark.gate_voltages[0], drain)

    rows: list[dict[str, Any]] = []
    for v_gate in benchmark.gate_voltages:
        ramp_to(device, GATE, gate_potential(v_gate))
        rows.append(
            {
                "gate": v_gate,
                "drain": terminal_current(device, DRAIN),
                "source": terminal_current(device, SOURCE),
                "body": terminal_current(device, BODY),
            }
        )
        print(
            f"    Vg={v_gate:+.3f} V  Id={rows[-1]['drain']:+.6e} A/cm",
            flush=True,
        )
    return rows


def sweep(
    benchmark: P.MosfetBenchmark, refine: float = 1.0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Both transfer curves of one benchmark, low drain first."""
    print(f"  {benchmark.name}: Vd = {benchmark.drain_low} V", flush=True)
    low = transfer_curve(benchmark, benchmark.drain_low, refine=refine)
    print(f"  {benchmark.name}: Vd = {benchmark.drain_high} V", flush=True)
    high = transfer_curve(benchmark, benchmark.drain_high, refine=refine)
    return low, high


def relative_difference(
    coarse: list[dict[str, Any]], fine: list[dict[str, Any]]
) -> tuple[float, float]:
    """Worst relative drain current difference between two meshes [1, V].

    Points where both curves sit under the current floor are skipped, since a
    relative comparison between two numbers that are both nearly nothing says
    nothing about the mesh.
    """
    worst = 0.0
    where = 0.0
    for a, b in zip(coarse, fine, strict=True):
        if (
            abs(a["drain"]) < P.CURRENT_FLOOR_MOSFET
            and abs(b["drain"]) < P.CURRENT_FLOOR_MOSFET
        ):
            continue
        scale = max(abs(a["drain"]), abs(b["drain"]))
        difference = abs(a["drain"] - b["drain"]) / scale
        if difference > worst:
            worst, where = difference, a["gate"]
    return worst, where


def write_csv(
    benchmark: P.MosfetBenchmark,
    low: list[dict[str, Any]],
    high: list[dict[str, Any]],
    nodes: int,
    mesh_check: tuple[float, float] | None,
    path: str,
) -> None:
    """Write one golden transfer pair, header and all."""
    process = P.MOSFET_PROCESS
    sigma, edge = P.implant_shape(process)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    lines: list[str] = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        "# generated by: tests/regression/devsim_gen/generate_mosfet.py",
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {stamp}",
        f"# tolerance: {benchmark.tolerance}",
        f"# L_gate: {benchmark.L_gate:.6e}",
        f"# drain low: {benchmark.drain_low:.6e}",
        f"# drain high: {benchmark.drain_high:.6e}",
    ]
    lines.extend(f"# {key}: {value:.6e}" for key, value in process.items())
    lines.extend(
        [
            f"# implant sigma: {sigma:.6e}",
            f"# implant edge: {edge:.6e}",
            f"# devsim silicon nodes: {nodes}",
            f"# devsim h_junction: {benchmark.devsim_h_junction:.6e}",
            f"# devsim h_channel: {benchmark.devsim_h_channel:.6e}",
            f"# devsim h_contact: {benchmark.devsim_h_contact:.6e}",
            f"# devsim h_surface: {benchmark.devsim_h_surface:.6e}",
            f"# devsim h_depth: {benchmark.devsim_h_depth:.6e}",
            f"# devsim oxide cells: {benchmark.devsim_oxide_cells}",
        ]
    )
    if mesh_check is not None:
        worst, where = mesh_check
        lines.append(
            f"# mesh convergence: {worst:.3e} worst relative change in drain "
            f"current when every spacing is halved, at {where:+g} V of gate"
        )
    lines.append("# notes: " + benchmark.notes)
    lines.append("# models:")
    lines.extend("#   " + line for line in P.MOSFET_MODEL_SUMMARY)
    lines.append(
        "# columns: gate bias [V], then drain and source current [A/cm] at "
        "the low drain bias and then at the high one"
    )
    lines.append("gate_voltage,drain_low,source_low,drain_high,source_high")
    for a, b in zip(low, high, strict=True):
        lines.append(
            f"{a['gate']:.10g},{a['drain']:.12e},{a['source']:.12e},"
            f"{b['drain']:.12e},{b['source']:.12e}"
        )

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    """Generate every benchmark named on the command line, or all of them."""
    parser = argparse.ArgumentParser(description="Generate MOSFET golden data")
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
    parser.add_argument(
        "--out",
        default=GOLDEN,
        help="directory to write into. Default is data/golden.",
    )
    args = parser.parse_args()

    wanted = P.MOSFET_BENCHMARKS
    if args.names:
        missing = sorted(set(args.names) - set(P.MOSFET_BY_NAME))
        if missing:
            print(f"no such benchmark: {missing}", file=sys.stderr)
            print(f"known: {sorted(P.MOSFET_BY_NAME)}", file=sys.stderr)
            return 1
        wanted = tuple(P.MOSFET_BY_NAME[name] for name in args.names)

    os.makedirs(args.out, exist_ok=True)
    for benchmark in wanted:
        print(
            f"{benchmark.name}: L_gate = {benchmark.L_gate * 1e7:g} nm, "
            f"{len(benchmark.gate_voltages)} gate biases on two curves",
            flush=True,
        )
        low, high = sweep(benchmark)
        nodes = node_count(device_name(benchmark, benchmark.drain_low))

        mesh_check = None
        if not args.no_mesh_check:
            print(f"{benchmark.name}: repeating on a halved mesh", flush=True)
            fine, _ = sweep(benchmark, refine=2.0)
            mesh_check = relative_difference(low, fine)
            print(
                f"{benchmark.name}: worst relative change {mesh_check[0]:.3e} "
                f"at {mesh_check[1]:+g} V",
                flush=True,
            )

        path = os.path.join(args.out, f"{benchmark.name}.csv")
        write_csv(benchmark, low, high, nodes, mesh_check, path)
        print(f"{benchmark.name}: wrote {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
