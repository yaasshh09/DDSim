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
    contact_equation,
    create_2d_mesh,
    create_device,
    delete_device,
    delete_mesh,
    edge_average_model,
    edge_from_node_model,
    finalize_mesh,
    get_contact_current,
    get_contact_list,
    get_interface_list,
    get_node_model_values,
    get_parameter,
    node_model,
    node_solution,
    set_node_values,
    set_parameter,
    solve,
    vector_gradient,
)
from devsim.python_packages.model_create import (  # noqa: E402
    CreateContactNodeModel,
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateNodeModel,
    CreateNodeModelDerivative,
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
    GetContactBiasName,
    GetContactNodeModelName,
    celec_model,
    chole_model,
    ece_name,
    hce_name,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

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
"""Assemble on one thread, which is as far as pinning threads is worth taking.

devsim splits its assembly across threads by default, and a floating point sum
whose order is set by a thread pool is a different sum from one run to the
next. That much is free to remove and it costs no measurable time.

The factorisation is the other half and it is not free. devsim's direct_solver
is mkl_pardiso, which is also multithreaded, and `MKL_NUM_THREADS=1` set before
the devsim import does make a run bit reproducible. It also turns
`build_physics` on the 1 um device from about four seconds into more than three
minutes, because the arithmetic it lands on walks a different and much worse
path through the drain ramp. So the pin is not taken, and what absorbs a
different path instead is `solve_once`, which retries a refused solve at a
looser tolerance rather than treating it as divergence. Measured across runs,
what is left of the spread is about 3e-5 of relative drain current at zero
gate, four decades under the tolerance any benchmark asks for.
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


def note(message: str) -> None:
    """Write one diagnostic line that `quiet` cannot swallow.

    The recorded bias point is settled inside `quiet`, which redirects stdout
    so devsim's per iteration report does not drown the progress log. Anything
    `settle` wants to say about how it got there goes to stderr instead, or it
    is written only on the passes nobody is watching and not on the one state
    that gets written down. Both streams land in the same log when the
    generator is run with its output redirected, which is how it is run.
    """
    print(f"    {message}", file=sys.stderr, flush=True)


def build_mesh(
    benchmark: P.MosfetBenchmark,
    device: str,
    refine: float = 1.0,
    refine_y: float | None = None,
    process: dict[str, float] | None = None,
) -> None:
    """Create and finalise the 2D structure for one benchmark.

    Args:
        benchmark: the device definition.
        device: devsim device name.
        refine: divide every spacing by this. 1 is the golden mesh.
        process: the vertical process, P.MOSFET_PROCESS if None. The
            scoreboard's robustness cases draw their own.
        refine_y: divide the vertical spacings by this instead of `refine`,
            leaving the lateral columns on `refine`. None halves both together,
            which is what a convergence check wants.

    Splitting the two axes is what turns a convergence number into a diagnosis.
    A single knob says the answer moved and nothing about where, and on the
    1 um device the answer was that the lateral columns carried 0.036 percent
    of a 2.933 percent move and the rows through the implant carried the rest.
    Refining on the strength of the total would have spent the effort on the
    axis holding a thirtieth of the error. See the 2026-09-12 rows in
    docs/07-decisions.md.

    Columns land on every boundary anything is measured against: the two
    contact edges, the two gate mask edges and the centre. Rows land on the
    body contact, the implant depth, the silicon surface and the gate.
    """
    mesh = device
    process = P.MOSFET_PROCESS if process is None else process
    sd_length = process["sd_length"]
    contact = process["contact_length"]
    t_si = process["t_si"]
    t_ox = process["t_ox"]
    x_j = process["x_j"]
    L = benchmark.L_gate
    width = 2.0 * sd_length + L

    vertical = refine if refine_y is None else refine_y
    h_contact = benchmark.devsim_h_contact / refine
    h_junction = benchmark.devsim_h_junction / refine
    h_channel = benchmark.devsim_h_channel / refine
    h_surface = benchmark.devsim_h_surface / vertical
    h_depth = benchmark.devsim_h_depth / vertical
    h_oxide = t_ox / (benchmark.devsim_oxide_cells * vertical)

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
    check_mesh_landed(device, t_si)


def check_mesh_landed(device: str, t_si: float) -> None:
    """Refuse a mesh that is not the structure `build_mesh` asked for [cm].

    devsim drops a requested mesh line when it falls close enough to a node it
    already has, and keeps the node rather than the line. Asking for a surface
    spacing of 1.5625e-9 cm under rows graded from the implant depth lands the
    top silicon row at 9.99998817e-5 instead of 1e-4, a gap of 1.18e-10 cm, and
    the silicon then has no node on the interface at all. Everything defined at
    y = t_si matches nothing: both surface contacts and the si_ox interface
    vanish, and devsim says so only later and only about the first one, as
    `Contact "source" on Device ... does not exist`.

    A mesh that quietly lost the source contact is not a coarser mesh, it is a
    different structure, and a convergence ladder that walks into one is
    comparing two unrelated devices. So the structure is checked here, where
    the answer is cheap and unambiguous, rather than inferred from a failure
    three calls later.
    """
    rows = get_node_model_values(device=device, region=BULK, name="y")
    top = max(rows)
    if abs(top - t_si) > 1e-14 * t_si:
        raise RuntimeError(
            f"{device}: the top silicon row is at {top:.17e} cm and the "
            f"interface is at {t_si:.17e}, a gap of {t_si - top:.3e} cm. "
            "devsim merged the interface mesh line into its neighbour, so the "
            "surface contacts and the si_ox interface have no nodes. Choose a "
            "surface spacing the rows below it can grade onto."
        )
    contacts = set(get_contact_list(device=device))
    missing = {BODY, SOURCE, DRAIN, GATE} - contacts
    if missing:
        raise RuntimeError(
            f"{device}: contacts {sorted(missing)} were asked for and not "
            f"created. devsim has {sorted(contacts)}."
        )
    interfaces = set(get_interface_list(device=device))
    if "si_ox" not in interfaces:
        raise RuntimeError(
            f"{device}: the si_ox interface was not created. devsim has "
            f"{sorted(interfaces)}."
        )


def node_count(device: str) -> int:
    """How many nodes the silicon carries, for the golden file header."""
    return len(get_node_model_values(device=device, region=BULK, name="x"))


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


def set_doping(
    benchmark: P.MosfetBenchmark,
    device: str,
    process: dict[str, float] | None = None,
) -> None:
    """The separable implant ddsim's `nmos` builds, written as one expression.

        N(x, y) = sub + peak * lateral(x) * depth(y) + the same mirrored

    with `lateral` an erfc edge at the gate mask and `depth` a gaussian peaked
    at the surface. sigma comes from the junction depth and the two
    concentrations, the erfc length from the lateral encroachment and the same
    two, so both junctions land exactly where they were asked to in both codes.
    """
    process = P.MOSFET_PROCESS if process is None else process
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


def joyce_dixon(u: str) -> str:
    """The Joyce-Dixon degeneracy correction, as a devsim expression in u.

        eta = ln(u) + A1 u + A2 u^2 + A3 u^3 + A4 u^4

    and this is everything after the logarithm, which is the whole of what
    Fermi-Dirac changes about a Scharfetter-Gummel flux. The argument is capped
    where the series turns over, at the same place ddsim caps it. Above the cap
    the correction is held constant and its derivative is zero, which is what
    `min` gives and what keeps the density monotone in the potential.
    """
    a1, a2, a3, a4 = P.JOYCE_DIXON
    v = f"(min({u}, {P.JOYCE_DIXON_MAX_U:.16e}))"
    return (
        f"({a1:.16e}*{v} + {a2:.16e}*pow({v},2) + "
        f"{a3:.16e}*pow({v},3) + {a4:.16e}*pow({v},4))"
    )


def set_full_stack_parameters(device: str) -> None:
    """Every Phase 5 model parameter, into devsim as a named parameter.

    Named rather than written into the expressions, so the model text below
    reads like the model it is and so `get_parameter` can be asked afterwards
    what a run actually used.
    """
    values: dict[str, float] = {
        "Nc": P.NC_300,
        "Nv": P.NV_300,
        "v_sat_n": P.V_SAT_N,
        "v_sat_p": P.V_SAT_P,
        "beta_n": P.BETA_N,
        "beta_p": P.BETA_P,
        "E_perp_floor": P.E_PERP_FLOOR,
    }
    for index, name in enumerate(("mu_min", "mu_d", "N_ref", "arora_A")):
        values[f"{name}_n"] = P.ARORA_N[index]
        values[f"{name}_p"] = P.ARORA_P[index]
    for key, value in P.LOMBARDI_N.items():
        values[f"lom_{key}_n"] = value
    for key, value in P.LOMBARDI_P.items():
        values[f"lom_{key}_p"] = value
    for name, value in values.items():
        set_parameter(device=device, region=BULK, name=name, value=value)


def create_bulk_mobility(device: str) -> None:
    """Arora doping dependent mobility, one value per node.

        mu = mu_min + mu_d / (1 + (N / N_ref)^A)

    A function of the doping alone, so it is built once and never refreshed.
    ddsim evaluates it on the same absolute net doping, which is the same
    approximation on both sides: only the net doping exists on either device
    and Arora wants the total. See the module docstring of
    ddsim/physics/mobility.py.
    """
    for carrier in ("n", "p"):
        CreateNodeModel(
            device,
            BULK,
            f"mu_arora_{carrier}",
            f"mu_min_{carrier} + mu_d_{carrier}/(1 + "
            f"pow(abs(NetDoping)/N_ref_{carrier}, arora_A_{carrier}))",
        )


def create_surface_mobility(device: str) -> None:
    """Lombardi surface scattering, frozen at the state of the last pass.

        1/mu = 1/mu_bulk + 1/mu_ac + 1/mu_sr
        mu_ac = B/E_perp + C N^tau E_perp^(-1/3) / (T/300)^kappa
        mu_sr = delta E_perp^(-gamma),  gamma = A + alpha (n + p) N^(-eta)

    Written as node models, which is where ddsim evaluates it too, and not as
    the element models devsim's own Klaassen.py uses. The reason is the one
    ddsim gives: the field normal to the interface on a horizontal channel edge
    lives on the vertical edges above and below its endpoints and not on the
    edge itself, so a node is the only place it has one value.

    **The correction is frozen and an outer fixed point is what makes it
    exact.** `E_perp` and the two mobilities are `node_solution` arrays, which
    hold values and carry no derivatives, so within one devsim solve the
    mobility is a constant and the Jacobian devsim differentiates is the
    Jacobian of the residual it assembled. `refresh_surface_mobility`
    re-evaluates them from the state that solve reached, and `settle` keeps
    calling both until the potential stops moving, which it cannot do until the
    mobility has stopped moving too. At that fixed point the frozen correction
    is the one the answer implies. Exactly what ddsim's `SurfaceScattering`
    does, and given up for the same reason: the rate, not the answer.

    The reciprocal sum is written in the product form devsim's Klaassen.py
    uses rather than as three reciprocals added. The two agree wherever both
    are finite, and where the roughness term underflows to zero the product
    form returns zero, which is the limit, while the reciprocal form returns a
    nan.
    """
    node_solution(device=device, region=BULK, name="E_perp")
    CreateNodeModel(device, BULK, "E_perp_used", "max(E_perp, E_perp_floor)")
    CreateNodeModel(
        device, BULK, "N_surface", f"max(abs(NetDoping), {P.N_I:.16e})"
    )

    for carrier in ("n", "p"):
        acoustic = (
            f"lom_B_{carrier}/E_perp_used + "
            f"lom_C_{carrier}*pow(N_surface, lom_tau_{carrier})*"
            f"pow(E_perp_used, -1.0/3.0)/pow(T/300, lom_kappa_{carrier})"
        )
        gamma = (
            f"lom_A_{carrier} + lom_alpha_{carrier}*(Electrons + Holes)*"
            f"pow(N_surface, -lom_eta_{carrier})"
        )
        roughness = f"lom_delta_{carrier}*pow(E_perp_used, -({gamma}))"
        bulk = f"mu_arora_{carrier}"
        CreateNodeModel(device, BULK, f"mu_ac_{carrier}", acoustic)
        CreateNodeModel(device, BULK, f"mu_sr_{carrier}", roughness)
        CreateNodeModel(
            device,
            BULK,
            f"mu_low_{carrier}_model",
            f"{bulk}*mu_ac_{carrier}*mu_sr_{carrier} / "
            f"({bulk}*mu_ac_{carrier} + {bulk}*mu_sr_{carrier} + "
            f"mu_ac_{carrier}*mu_sr_{carrier})",
        )
        node_solution(device=device, region=BULK, name=f"mu_low_{carrier}")

    refresh_surface_mobility(device)


SURFACE_RTOL = 1e-8
"""How still the frozen surface mobility has to be to count as arrived [1].

ddsim's `_surface_fixed_point` uses the same number for the same quantity, and
the number is not the interesting part. What matters is that there is one at
all: refreshing on every pass forever is not a fixed point, it is two
quantities chasing each other, and the potential can then never stop moving
because the mobility under it never stops moving either. Measured before this
existed: the 1 um device at 1 V of drain walked its gate to +0.50 V and then
refused +0.55 V at every step size down to 0.1 mV, which is the signature of a
failure the step size does not control.
"""

SURFACE_SWEEPS = 20
"""Refreshes one `settle` is allowed before it holds the mobility still.

ddsim's budget, and spent the same way: past it the correction is held at what
it last was and the potential is solved against that, rather than the solve
being failed. A mobility still moving after twenty refreshes is a diagnostic
and not a reason to throw away a converged potential.
"""


def refresh_surface_mobility(device: str) -> float:
    """Re-evaluate the frozen surface mobility at the state now on the device.

    `vector_gradient` is devsim's nodal gradient, and its own documentation
    says not to use what it produces in a simulation because devsim cannot
    differentiate it. That is exactly the use here: the quantity is frozen on
    purpose and never enters a Jacobian. ddsim reaches the same number by
    averaging the two vertical edges either side of a node, which is a
    different nodal gradient of the same field, and the two meshes were never
    matched to begin with.

    The magnitude is taken after the gradient and not before, for the reason
    ddsim's `normal_field` gives: where the vertical field reverses across a
    node the field there really is near zero, and a magnitude taken first would
    report the average of the two large ones instead.

    Returns:
        The largest relative change in either mobility [1], which is what the
        fixed point is watching. See `SURFACE_RTOL`.
    """
    vector_gradient(
        device=device, region=BULK, node_model="Potential", calc_type="default"
    )
    gradient = get_node_model_values(
        device=device, region=BULK, name="Potential_grady"
    )
    set_node_values(
        device=device,
        region=BULK,
        name="E_perp",
        values=[abs(value) for value in gradient],
    )
    moved = 0.0
    for carrier in ("n", "p"):
        name = f"mu_low_{carrier}"
        before = get_node_model_values(device=device, region=BULK, name=name)
        set_node_values(
            device=device, region=BULK, name=name, init_from=f"{name}_model"
        )
        after = get_node_model_values(device=device, region=BULK, name=name)
        for old_value, new_value in zip(before, after, strict=True):
            if new_value != 0.0:
                moved = max(moved, abs(new_value - old_value) / abs(new_value))
    return moved


def create_edge_mobility(device: str) -> None:
    """Onto the edges, then Caughey-Thomas around what arrives.

        mu(E) = mu_0 / (1 + (mu_0 |E| / v_sat)^beta)^(1/beta)

    The wrapping order is ddsim's and it is the only one that makes sense: the
    low field mobility is nodal, so it is corrected for the surface and
    averaged onto the edge first, and the saturation factor is applied
    afterwards with that edge's own parallel drop, because the parallel field
    is an edge quantity with no value at a node.

    **E is the component along the edge, not the magnitude of the field
    vector.** devsim's `ElectricField` is the potential drop over the edge
    length, which is that component, and phases/PHASE-5.md is specific that
    using the magnitude is the common shortcut and is wrong on a graded mesh.

    The average onto the edge is arithmetic, matching ddsim. devsim's own
    helpers reach for a geometric mean, which is a different edge mobility on
    any edge whose two ends disagree, and across an inversion layer they
    disagree by decades.

    This is the one part of the stack that is not frozen. It is a function of
    the potential and devsim differentiates it symbolically, so velocity
    saturation sits inside the Newton step on both sides.
    """
    for carrier in ("n", "p"):
        edge_average_model(
            device=device,
            region=BULK,
            node_model=f"mu_low_{carrier}",
            edge_model=f"mu_lf_{carrier}",
            average_type="arithmetic",
        )
        squared = (
            f"pow(mu_lf_{carrier}*ElectricField/v_sat_{carrier}, 2) + 1e-300"
        )
        mobility = (
            f"mu_lf_{carrier}*pow(1 + pow({squared}, 0.5*beta_{carrier}), "
            f"-1.0/beta_{carrier})"
        )
        name = f"mu_ct_{carrier}"
        CreateEdgeModel(device, BULK, name, mobility)
        CreateEdgeModelDerivatives(device, BULK, name, mobility, "Potential")


def create_degenerate_currents(device: str) -> None:
    """Replace both Scharfetter-Gummel currents with their degenerate form.

    Fermi-Dirac does not change the discretisation at all, which is the whole
    reason Joyce-Dixon is worth having. Each carrier is still exponential in a
    potential,

        n = exp((psi_eff_n - phi_n)/V_t),  psi_eff_n = psi - V_t gamma_n(n/Nc)

    so the exponential fit the Scharfetter-Gummel flux is built on stays exact
    and the only edit is which potential the Bernoulli argument is a difference
    of. The correction is below psi for electrons and above it for holes,
    because filling a band means a given density needs a higher Fermi level
    than Boltzmann says.

    This one is not lagged. devsim differentiates the correction with respect
    to Electrons and Holes symbolically, so the degeneracy is inside the Newton
    step, which is what ddsim's coupled path does too.

    The two currents keep the names devsim's own helpers gave them, because the
    contact equations reference those names and there is nothing to gain from
    rewiring them.
    """
    for carrier, density, states, sign in (
        ("n", "Electrons", "Nc", "-"),
        ("p", "Holes", "Nv", "+"),
    ):
        correction = joyce_dixon(f"{density}/{states}")
        effective = f"Potential {sign} V_t*{correction}"
        CreateNodeModel(device, BULK, f"Potential_{carrier}", effective)
        for variable in ("Potential", density):
            CreateNodeModelDerivative(
                device, BULK, f"Potential_{carrier}", effective, variable
            )
        edge_from_node_model(
            device=device, region=BULK, node_model=f"Potential_{carrier}"
        )
        for variable in ("Potential", density):
            edge_from_node_model(
                device=device,
                region=BULK,
                node_model=f"Potential_{carrier}:{variable}",
            )

        drop = f"(Potential_{carrier}@n0 - Potential_{carrier}@n1)/V_t"
        CreateEdgeModel(device, BULK, f"vdiff_{carrier}", drop)
        for variable in ("Potential", density):
            for node, sign in (("@n0", ""), ("@n1", "-")):
                CreateEdgeModel(
                    device,
                    BULK,
                    f"vdiff_{carrier}:{variable}{node}",
                    f"{sign}Potential_{carrier}:{variable}{node}/V_t",
                )
        CreateEdgeModel(device, BULK, f"Bern01_{carrier}", f"B(vdiff_{carrier})")
        for variable in ("Potential", density):
            for node in ("@n0", "@n1"):
                CreateEdgeModel(
                    device,
                    BULK,
                    f"Bern01_{carrier}:{variable}{node}",
                    f"dBdx(vdiff_{carrier}) * vdiff_{carrier}:{variable}{node}",
                )

    electrons = (
        "ElectronCharge*mu_ct_n*EdgeInverseLength*V_t*kahan3("
        "Electrons@n1*Bern01_n, Electrons@n1*vdiff_n, -Electrons@n0*Bern01_n)"
    )
    holes = (
        "-ElectronCharge*mu_ct_p*EdgeInverseLength*V_t*kahan3("
        "Holes@n1*Bern01_p, -Holes@n0*Bern01_p, -Holes@n0*vdiff_p)"
    )
    for name, current in (
        ("ElectronCurrent", electrons),
        ("HoleCurrent", holes),
    ):
        CreateEdgeModel(device, BULK, name, current)
        for variable in ("Electrons", "Holes", "Potential"):
            CreateEdgeModelDerivatives(device, BULK, name, current, variable)


def degenerate_contact_expressions() -> tuple[str, str, str]:
    """The ohmic contact, degenerate: electrons, holes, and the psi offset.

    Neutrality is untouched by the statistics, n - p = N, but mass action is
    not. With each density carrying its own degeneracy factor the product is

        n p = n_i^2 gamma_n(n/Nc) gamma_p(p/Nv)

    and the majority carrier is still the quadratic root while the minority
    still comes from the product, which is what keeps mass action exact rather
    than merely close. The product is evaluated at the Boltzmann majority
    density, which is the same number to forty digits at any doping these
    contacts carry: the correction to the root is 4 n_i^2 against N^2, and at
    1e20 that is 1e20 against 1e40.

    The potential is where degeneracy actually shows on this device. Inverting
    n = n_i exp((psi - V_t gamma_n)/V_t) gives

        psi = V_t ( ln(n/n_i) + gamma_n(n/Nc) )

    and at 1e20 that correction is 1.18, which is 30.6 mV of built in potential
    at the source and the drain. Dropping it because the minority carrier it
    also moves is irrelevant would move every threshold on the sweep.
    """
    gamma_n = f"exp(-{joyce_dixon(f'{celec_model}/Nc')})"
    gamma_p = f"exp(-{joyce_dixon(f'{chole_model}/Nv')})"
    product = f"(n_i^2*{gamma_n}*{gamma_p})"
    electrons = f"ifelse(NetDoping > 0, {celec_model}, {product}/{chole_model})"
    holes = f"ifelse(NetDoping < 0, {chole_model}, {product}/{celec_model})"
    potential = (
        f"ifelse(NetDoping > 0, "
        f"-V_t*(log({celec_model}/n_i) + {joyce_dixon(f'{celec_model}/Nc')}), "
        f"+V_t*(log({chole_model}/n_i) + {joyce_dixon(f'{chole_model}/Nv')}))"
    )
    return electrons, holes, potential


def create_degenerate_contact(device: str, contact: str) -> None:
    """Both continuity contact equations and the potential one, degenerate.

    devsim's `CreateSiliconPotentialOnlyContact` already built the potential
    condition during the equilibrium phase with the Boltzmann offset written
    into it. Re-creating the contact node model under the same name replaces
    what that equation evaluates, which is cheaper and clearer than tearing the
    equation down and building it again.
    """
    electrons, holes, potential = degenerate_contact_expressions()

    contact_model = f"Potential -{GetContactBiasName(contact)} + {potential}"
    CreateContactNodeModel(
        device, contact, GetContactNodeModelName(contact), contact_model
    )
    CreateContactNodeModel(
        device, contact, f"{GetContactNodeModelName(contact)}:Potential", "1"
    )

    for name, model, variable in (
        (f"{contact}nodeelectrons", f"Electrons - ({electrons})", "Electrons"),
        (f"{contact}nodeholes", f"Holes - ({holes})", "Holes"),
    ):
        CreateContactNodeModel(device, contact, name, model)
        CreateContactNodeModel(device, contact, f"{name}:{variable}", "1")

    for equation_name, node_model_name, current in (
        (ece_name, f"{contact}nodeelectrons", "ElectronCurrent"),
        (hce_name, f"{contact}nodeholes", "HoleCurrent"),
    ):
        contact_equation(
            device=device,
            contact=contact,
            name=equation_name,
            node_model=node_model_name,
            edge_current_model=current,
        )


def gate_potential(v_gate: float, work_function: float = P.PHI_M_N_POLY) -> float:
    """The potential to pin the gate at [V], measured from the intrinsic level.

    docs/01-physics.md writes the gate condition as psi_gate = V_gate - Phi_MS,
    which needs the doping under the gate. Since psi is measured from the
    intrinsic level in both codes, and the work function of intrinsic silicon
    is exactly chi + Eg/2, the same statement is

        psi_gate = V_gate + (chi + Eg/2 - Phi_M)

    and the doping has cancelled. ddsim uses the second form for the same
    reason: a contact has no business reading the substrate under it.
    """
    return v_gate + (P.PHI_M_MIDGAP - work_function)


def seed_potential(device: str) -> None:
    """Start the equilibrium Poisson solve from charge neutrality, not from zero.

    devsim creates a solution array full of zeros and its own examples solve
    from there, which on this device means starting 0.6 V away in the source
    and drain and 0.48 V away in the body. The solve gets there, but it gets
    there on devsim's log damping and the path is long enough to be luck: the
    identical Poisson problem, same mesh and same biases, converged on the
    first device of a process and raised a convergence failure on the second.

    The charge neutral potential is exact everywhere except in a depletion
    region, so seeding it leaves the solve a small and well conditioned
    correction rather than a long walk. It is the same guess ddsim's own ohmic
    contact makes, written without asinh because devsim's parser has none:

        psi_0 = V_t asinh(N / 2 n_i)
              = V_t sgn(N) ln( (|N| + sqrt(N^2 + 4 n_i^2)) / (2 n_i) )

    The sign has to come out front. Written as ln(N + sqrt(N^2 + 4 n_i^2)), a
    p-type side past about 1e18 cancels to exactly ln(0), and devsim stops
    with a divide by zero before it solves anything. That took out every
    scoreboard MOSFET with a heavy substrate (m003, m005, m006, m009) in 0.2 s
    on all three drivers, which is my bug and not a devsim robustness loss.

    An initial guess moves no converged answer. It only decides whether one is
    reached.
    """
    node_model(
        device=device,
        region=BULK,
        name="PotentialNeutral",
        equation=(
            f"{P.V_T:.16e} * sgn(NetDoping) * log((abs(NetDoping) + "
            f"pow(NetDoping*NetDoping + 4*{P.N_I:.16e}*{P.N_I:.16e}, 0.5)) "
            f"/ (2*{P.N_I:.16e}))"
        ),
    )
    set_node_values(
        device=device, region=BULK, name="Potential", init_from="PotentialNeutral"
    )


def build_physics(
    device: str,
    v_gate: float,
    v_drain: float,
    models: str = P.REDUCED_MODELS,
    work_function: float = P.PHI_M_N_POLY,
) -> None:
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
    seed_potential(device)
    CreateOxidePotentialOnly(device, OXIDE, "log_damp")

    for contact in (BODY, SOURCE, DRAIN):
        set_parameter(device=device, name=f"{contact}_bias", value=0.0)
        CreateSiliconPotentialOnlyContact(device, BULK, contact)
    set_parameter(
        device=device,
        name=f"{GATE}_bias",
        value=gate_potential(v_gate, work_function),
    )
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
    if models == P.REDUCED_MODELS:
        CreateSiliconDriftDiffusion(device, BULK, mu_n="mu_n", mu_p="mu_p")
        for contact in (BODY, SOURCE, DRAIN):
            CreateSiliconDriftDiffusionAtContact(device, BULK, contact)
    else:
        set_full_stack_parameters(device)
        create_bulk_mobility(device)
        create_surface_mobility(device)
        create_edge_mobility(device)
        CreateSiliconDriftDiffusion(device, BULK, mu_n="mu_ct_n", mu_p="mu_ct_p")
        create_degenerate_currents(device)
        for contact in (BODY, SOURCE, DRAIN):
            create_degenerate_contact(device, contact)
        global _surface_is_live
        _surface_is_live = True
    settle(device)
    ramp_to(device, DRAIN, v_drain)


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


_surface_is_live = False
"""Whether `settle` should refresh the frozen surface mobility each pass.

Module state rather than an argument because `settle` is reached from
`build_physics` and from `ramp_to` as well as directly, and threading a flag
through all three to say something that is true of a whole curve would be
noise. `transfer_curve` clears it before every device it builds.
"""


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
    device: str,
    poisson_only: bool = False,
    passes: int = 400,
    tol: float = 1e-9,
    balance_tol: float | None = None,
    stall: int = 25,
) -> int:
    """Call solve until the solution stops moving, and return the pass count.

    One devsim solve returns after a single damped step and reports RelError
    and AbsError as exactly zero, which is not a converged state: on the 1 um
    device the equilibrium potential is still 0.3 V short after the first call
    and moves another 0.12 V on the second. So convergence is decided here, by
    watching the solution rather than by reading the report.

    That is not a small point. devsim's own `python_packages/ramp.py` calls
    solve once per bias step and trusts what it says, and run that way this
    device hands back a transfer curve reading -2.209939 A/cm at 0.1 V of gate
    and +9.942419 A/cm at 0.4, with every step reported as a success.

    `stall` is how a settle that is not going to arrive gives up cheaply. A
    converging one takes 6 to 20 passes and keeps setting new records for the
    largest move; a stuck one bounces around a floor forever. Since a failed
    settle is a failed bias step, and `ramp_to` answers that by halving and
    trying again up to nine times, the cost of noticing late is nine times the
    full pass budget. Watching for a new best is what separates the two cases,
    and cutting the budget instead is not: it refuses settles that would have
    converged.

    `balance_tol` adds the second half of the answer. The potential is the
    Poisson residual and it arrives well before the continuity one, so a settle
    that watches only the potential returns while the current is still moving.
    See `BALANCE_TOL`. Passing a tolerance keeps the solve going until drain
    and source cancel to it, or until that stops improving, which is a floor
    and not a failure. The default is None, which is the behaviour every
    intermediate ramp step wants: those states are never reported, so paying
    twenty extra passes for each of them buys nothing.
    """
    before = snapshot(device, poisson_only)
    refreshing = _surface_is_live and not poisson_only
    surface_sweeps = 0
    moved = float("inf")
    best = float("inf")
    stalled = 0
    since_best = 0.0
    arrived = False
    imbalance = float("inf")
    best_balance = float("inf")
    balance_stalled = 0
    for index in range(passes):
        if refreshing:
            surface_sweeps += 1
            surface_moved = refresh_surface_mobility(device)
            if (
                surface_moved < SURFACE_RTOL
                or surface_sweeps >= SURFACE_SWEEPS
            ):
                refreshing = False
                best = float("inf")
                stalled = 0
                since_best = 0.0
                if surface_moved >= SURFACE_RTOL:
                    note(
                        f"surface mobility still moving at "
                        f"{surface_moved:.3e} after {surface_sweeps} "
                        "refreshes, frozen there"
                    )
        solve_once()
        after = snapshot(device, poisson_only)
        moved = potential_move(before, after)
        before = after

        if not arrived and moved >= tol:
            if moved < best:
                best, stalled, since_best = moved, 0, 0.0
                continue
            stalled += 1
            since_best = max(since_best, moved)
            if stalled < stall:
                continue
            if since_best >= SETTLE_FLOOR:
                raise RuntimeError(
                    f"stalled at {best:.3e} V for {stall} passes, worst "
                    f"{since_best:.3e} V, last move {moved:.3e} V"
                )
            note(
                f"settled on the solver floor, {since_best:.3e} V over "
                f"{stall} passes"
            )

        arrived = True
        if not poisson_only and not sane(device):
            raise RuntimeError("settled on a state no bias can produce")
        if poisson_only or balance_tol is None:
            return index + 1

        imbalance = terminal_imbalance(device)
        if imbalance < balance_tol:
            return index + 1
        if imbalance < BALANCE_PROGRESS * best_balance:
            best_balance, balance_stalled = imbalance, 0
            continue
        best_balance = min(best_balance, imbalance)
        balance_stalled += 1
        if balance_stalled >= stall:
            return index + 1
    if arrived:
        return passes
    raise RuntimeError(
        f"did not settle in {passes} passes, last move {moved:.3e} V, "
        f"imbalance {imbalance:.3e}"
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
    max_step: float = 0.1,
) -> None:
    """Walk one contact bias to target, halving the step on a failure.

    The step grows back, but only after `GROWTH_STREAK` steps in a row have
    converged, because the hard part of a transfer curve is the knee and there
    is no reason to crawl the rest of it at the pace the knee needed. It never
    grows past `max_step` [V], which the scoreboard's fine reference sets small.
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
            step = min(2.0 * step, max_step)
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


SETTLE_FLOOR = 1e-5
"""Largest potential move `settle` will accept as the solver's own floor [V].

A settle that is not going to arrive and a settle that has arrived and is
bouncing in devsim's own roundoff look identical to a counter that only knows
no pass beat the record. What separates them is how far from zero the bouncing
is, and the two cases are two decades apart. Measured on the 1 um device at
1 V of drain and a gate of +0.55 V, which is the only bias on any of the five
that needs this: the genuine failure, a frozen surface mobility chasing the
potential it was frozen from, bounced at 1.730e-4 V and never came down, while
the full stack with that fixed bounces and never improves again. The 1 um
curve accepts three floors, at 2.794e-7, 3.752e-7 and 1.418e-6 V, so the value
here has seven times the headroom it needs on the run that produces the data.
A cold solve dropped straight onto the knee, rather than walked up to it from
-0.4 V, reaches 6.2e-6, which is the widest bouncing seen anywhere and still
inside. The other four gate lengths reach `tol` outright and so does the whole
reduced model set, which is why benchmarks 6 to 9 never needed this.

Why it bounces at all is written down in `SOLVE_TOLERANCES`: the tolerance
handed to devsim is a step size control rather than an accuracy control, and
on this device at an inverted gate devsim refuses the tightest rung and the
looser one it accepts returns a looser step. `settle` owns accuracy instead,
by watching the solution.

Two things bound what the floor can let through. The first is the model: the
most sensitive quantity any benchmark reads is the subthreshold drain current,
exponential in the surface potential at the ideal 1/V_t, so 1e-5 V bounds the
current it can move to exp(1e-5/0.02585) - 1, which is 3.9e-4 relative, a
decade under the smallest mesh error any of these files records and two under
the 10 percent the comparison spends. The second is that accepting the floor
does not end the solve. It marks the potential arrived and hands the rest to
the terminal balance criterion, which is the one that certifies the current
the benchmark reads, and whatever that reaches is written into the golden file
per point and spent as tolerance. Anything above the floor still raises.
"""


BALANCE_PROGRESS = 0.9
"""How much of the record an imbalance has to beat to count as progress [1].

The potential either converges or bounces, so any improvement is progress. The
imbalance has a third behaviour: it crawls, gaining a little on most passes
while going nowhere, and a plain new best test reads that as convergence and
never gives up. Requiring a tenth off the record keeps every genuine descent,
which gains 12 to 22 percent a pass early on, and ends a crawl in `stall`
passes. See `BALANCE_TOL`.
"""

BALANCE_TOL = 1e-3
"""Terminal imbalance a recorded bias point is solved down to [1].

Nothing in this model set generates carriers in the bulk faster than SRH
removes them, so in steady state drain and source sum to zero and the body
carries decades less than either. Whatever is left over is the continuity
residual, expressed in the units of the answer.

`settle` watches the potential, which is the right thing to watch while the
solve is still moving, but the potential is the Poisson residual and it arrives
one to two decades earlier than the continuity one. Measured on the 1 um device
at zero gate and 50 mV of drain: on the pass `settle` returns, the largest
potential move is already inside 1e-9 V while drain and source disagree by 12.7
percent, and it takes about twenty further passes for that to reach its floor.
The drain current moves 0.70 percent across those passes, from 1.453019e-6 to
1.442909e-6 A/cm, so the early return was not free.

1e-3 sits above the floor and below anything the comparison cares about. The
floor is not zero: past about the twenty fourth pass the imbalance bounces
between 2e-4 and 1.1e-3 while the drain current holds to seven figures, which
is a difference of much larger fluxes finding its own roundoff. So this is a
target and not a requirement, and `settle` accepts a stalled balance rather
than failing on one.
"""


def terminal_imbalance(device: str) -> float:
    """How badly drain and source fail to cancel on the live device [1].

    The same quantity `MosfetGoldenCurve.imbalance` reports per point, measured
    during the solve so that `settle` can keep going until it arrives.
    """
    drain = terminal_current(device, DRAIN)
    source = terminal_current(device, SOURCE)
    scale = max(abs(drain), abs(source))
    return 0.0 if scale == 0.0 else abs(drain + source) / scale


def device_name(
    benchmark: P.MosfetBenchmark, drain: float, refine: float = 1.0
) -> str:
    """The devsim device name for one curve. Dots are not legal in one."""
    return f"{benchmark.name}_d{drain:g}_r{refine:g}".replace(".", "_")


def transfer_curve(
    benchmark: P.MosfetBenchmark,
    drain: float,
    refine: float = 1.0,
    refine_y: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """One Id-Vg curve at a fixed drain bias, and its silicon node count.

    A device is built per curve rather than per point. The gate is walked from
    the first bias to the last with continuation, since the state at one bias
    is the best guess available for the next, and the drain went on before any
    of it, ramped onto the coupled system once the carriers existed.

    The device is deleted on the way out, and that is not tidiness. devsim's
    `solve` takes no device argument: it solves every device in the simulation
    at once. A generator that builds a device per curve and leaves them lying
    around is therefore asking devsim, on the second curve, to converge two
    devices simultaneously, one of which is already converged and contributes
    nothing but its own roundoff to the shared error norm. Measured: building
    the same 1 um device four times in one process fails on the third, and
    deleting each one as its curve finishes makes all four succeed.
    """
    global _surface_is_live
    _surface_is_live = False
    device = device_name(benchmark, drain, refine)
    with quiet():
        build_mesh(benchmark, device, refine=refine, refine_y=refine_y)
    set_material_parameters(device)
    set_doping(benchmark, device)
    with quiet():
        build_physics(
            device, benchmark.gate_voltages[0], drain, benchmark.models
        )

    rows: list[dict[str, Any]] = []
    for v_gate in benchmark.gate_voltages:
        ramp_to(device, GATE, gate_potential(v_gate))
        with quiet():
            settle(device, balance_tol=BALANCE_TOL)
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
    nodes = node_count(device)
    _surface_is_live = False
    delete_device(device=device)
    delete_mesh(mesh=device)
    return rows, nodes


def sweep(
    benchmark: P.MosfetBenchmark,
    refine: float = 1.0,
    refine_y: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Both transfer curves of one benchmark, low drain first, and the nodes."""
    print(f"  {benchmark.name}: Vd = {benchmark.drain_low} V", flush=True)
    low, nodes = transfer_curve(
        benchmark, benchmark.drain_low, refine=refine, refine_y=refine_y
    )
    print(f"  {benchmark.name}: Vd = {benchmark.drain_high} V", flush=True)
    high, _ = transfer_curve(
        benchmark, benchmark.drain_high, refine=refine, refine_y=refine_y
    )
    return low, high, nodes


MESH_NOISE_FACTOR = 3.0
"""How much of a point's own imbalance a mesh move has to clear to count [1].

The same factor and the same reasoning as the comparison tests: a point whose
drain and source disagree by one percent is a point known to one percent, and a
change smaller than that is not a measurement of anything.

It is needed here because the imbalance floor is a property of the mesh and it
gets worse as the mesh gets finer, which is the opposite of what a convergence
check assumes. Measured on the 1 um device at zero gate and 50 mV: the shipping
mesh balances to 1.65e-4, and the halved mesh it is checked against settles to
a largest potential move of 1.1e-16 V, holds its drain current to ten figures
across 120 further passes, and still leaves drain and source 0.99 percent
apart. That is not an unconverged solve. It is a converged discrete solution
whose terminal currents do not cancel, because the off state current is a small
difference of much larger fluxes and the finer mesh has more edges to lose
digits across.

So the two meshes differ by 0.53 percent at that point while the finer of them
knows its own answer to 0.99 percent, and attributing the gap to
discretisation would be reading noise as signal. Points like that are skipped
and counted, and the count goes in the header, because a skipped point is a
point whose mesh error is unmeasurable rather than small.
"""


def relative_difference(
    coarse: list[dict[str, Any]], fine: list[dict[str, Any]]
) -> tuple[float, float, int]:
    """Worst relative drain current difference between two meshes [1, V, 1].

    Returns the worst change, the gate bias it happened at, and how many points
    were skipped.

    A point is skipped when both curves sit under the current floor, since a
    relative comparison between two numbers that are both nearly nothing says
    nothing about the mesh, or when the change is inside what the two points
    know about themselves. See `MESH_NOISE_FACTOR`.
    """
    worst = 0.0
    where = 0.0
    skipped = 0
    for a, b in zip(coarse, fine, strict=True):
        if (
            abs(a["drain"]) < P.CURRENT_FLOOR_MOSFET
            and abs(b["drain"]) < P.CURRENT_FLOOR_MOSFET
        ):
            skipped += 1
            continue
        scale = max(abs(a["drain"]), abs(b["drain"]))
        difference = abs(a["drain"] - b["drain"]) / scale
        noise = MESH_NOISE_FACTOR * max(row_imbalance(a), row_imbalance(b))
        if difference <= noise:
            skipped += 1
            continue
        if difference > worst:
            worst, where = difference, a["gate"]
    return worst, where, skipped


def row_imbalance(row: dict[str, Any]) -> float:
    """How badly drain and source fail to cancel on one recorded point [1]."""
    scale = max(abs(row["drain"]), abs(row["source"]))
    return 0.0 if scale == 0.0 else abs(row["drain"] + row["source"]) / scale


def write_csv(
    benchmark: P.MosfetBenchmark,
    low: list[dict[str, Any]],
    high: list[dict[str, Any]],
    nodes: int,
    mesh_check: tuple[float, float, int] | None,
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
        worst, where, skipped = mesh_check
        lines.append(
            f"# mesh convergence: {worst:.3e} worst relative change in drain "
            f"current when every spacing is halved, at {where:+g} V of gate, "
            f"with {skipped} of {len(low)} points skipped as unmeasurable"
        )
    lines.append("# notes: " + benchmark.notes)
    lines.append("# models:")
    lines.extend(
        "#   " + line for line in P.mosfet_model_summary(benchmark.models)
    )
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
        low, high, nodes = sweep(benchmark)

        path = os.path.join(args.out, f"{benchmark.name}.csv")
        write_csv(benchmark, low, high, nodes, None, path)

        mesh_check = None
        if not args.no_mesh_check:
            print(f"{benchmark.name}: repeating on a halved mesh", flush=True)
            fine_low, fine_high, _ = sweep(benchmark, refine=2.0)
            mesh_check = max(
                relative_difference(low, fine_low),
                relative_difference(high, fine_high),
                key=lambda check: check[0],
            )
            print(
                f"{benchmark.name}: worst relative change {mesh_check[0]:.3e} "
                f"at {mesh_check[1]:+g} V, {mesh_check[2]} points skipped",
                flush=True,
            )

            write_csv(benchmark, low, high, nodes, mesh_check, path)
        print(f"{benchmark.name}: wrote {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
