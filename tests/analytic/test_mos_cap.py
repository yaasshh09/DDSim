"""The MOS capacitor at DC: Stage 5 of the Phase 4 plan.

tests/analytic/test_mos_electrostatics.py proved the physics with the system
assembled by hand. This file asks the same questions of the device API, and
then asks the harder ones the phase document gates on: flatband and threshold
against the textbook expressions, to under 20 mV.

Why the capacitor is an equilibrium problem
-------------------------------------------
No current flows through an ideal insulator, so the semiconductor stays in
equilibrium with its body contact at every gate bias. phi_n and phi_p are flat
at the body potential and Poisson closes on psi alone. That is the whole DC
solve, and it is why this comes before extract/cv.py rather than after: the
small signal capacitance is a derivative of a curve this file already computes.

What is checked against what
----------------------------
The textbook expressions live here rather than in ddsim/, deliberately. They
are the depletion approximation, they are what the solver is being measured
against, and a project whose success criterion is that short channel effects
emerge from physics rather than from fitting should not carry them in the
solver where something could later reach for one.

Flatband is the sharpest check in the file and the one worth reading first. It
needs the gate work function, the body contact potential and the intrinsic
reference to agree exactly, and those are three pieces of code that never
otherwise meet. A millivolt of disagreement slides the whole C-V curve sideways
while every regime still looks correct.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import mos_cap

NA = 1e16
"""Substrate acceptor concentration [cm^-3]. Net doping is -NA, p-type."""

T_OX = 1e-6
"""Oxide thickness [cm], 10 nm."""

T_SI = 2e-4
"""Silicon thickness [cm], 2 um.

Well over the 304 nm maximum depletion width at this doping, so the body
contact sits in neutral material and does not hold the depletion region open.
"""

MILLIVOLT = 1e-3
GATE_TOLERANCE = 20 * MILLIVOLT
"""What phases/PHASE-4.md gates flatband and threshold at."""


# ------------------------------------------------------- textbook expressions


def phi_F(net_doping: float) -> float:
    """Bulk Fermi potential [V], positive for p-type as the textbooks write it.

    The asinh form, matching constants.semiconductor_work_function. At 1e16 it
    agrees with V_T*ln(Na/n_i) to twelve digits; the two only part company near
    intrinsic, where the log form has no answer at all.
    """
    return float(-C.V_T() * np.arcsinh(net_doping / (2.0 * C.n_i())))


def flatband_voltage(net_doping: float, work_function: float) -> float:
    """V_FB = Phi_MS [V]. Nothing else, because this is the ideal capacitor.

    Fixed oxide charge would add -Q_f/C_ox. There is none here.
    """
    return float(C.work_function_difference(work_function, net_doping))


def oxide_capacitance(t_ox: float) -> float:
    """C_ox = eps_ox / t_ox [F/cm^2], a parallel plate per unit area."""
    return C.eps_ox() / t_ox


def max_depletion_width(net_doping: float) -> float:
    """W_max [cm], the depletion width once the surface reaches 2 phi_F.

    W = sqrt(2 eps_Si (2 phi_F) / (q Na)). Past threshold the inversion layer
    screens the gate and the depletion region stops growing, which is what
    makes the low frequency C-V minimum a fixed number.
    """
    Na = abs(net_doping)
    return float(
        np.sqrt(2.0 * C.eps_Si() * 2.0 * phi_F(net_doping) / (C.q * Na))
    )


def threshold_voltage(
    net_doping: float, t_ox: float, work_function: float
) -> float:
    """V_TH [V], the textbook expression.

    V_TH = V_FB + 2 phi_F + Q_dep / C_ox, with Q_dep = q Na W_max the charge
    the depletion region holds when the surface has bent by 2 phi_F. The three
    terms are the flatband offset, the band bending, and the drop across the
    oxide that the depletion charge demands.
    """
    Na = abs(net_doping)
    q_dep = C.q * Na * max_depletion_width(net_doping)
    return (
        flatband_voltage(net_doping, work_function)
        + 2.0 * phi_F(net_doping)
        + q_dep / oxide_capacitance(t_ox)
    )


# --------------------------------------------------------------- the solve


def solved(net_doping=-NA, work_function=C.PHI_M_N_POLY, **kwargs):
    """A solved capacitor: (device, state)."""
    device = mos_cap(
        substrate_doping=net_doping,
        t_ox=T_OX,
        t_si=T_SI,
        work_function=work_function,
        **kwargs,
    )
    return device, solve_equilibrium(device)


def surface_potential(device, state) -> float:
    """psi_s [V], the band bending from the neutral bulk to the surface.

    Measured as the difference between the interface node and the node at the
    bottom of the same column, which is the body contact and therefore neutral
    by construction. Reading psi at the surface alone would measure it from
    the intrinsic level instead, which is where this code keeps its zero and
    is not where a textbook keeps its.
    """
    mesh = device.mesh
    assert device.regions is not None
    surface = device.regions.interface_nodes(mesh)
    column = mesh.nx // 2
    bulk = mesh.node_at(column, 0)
    node = int(surface[column])
    return float((state.psi.data[node] - state.psi.data[bulk]) * device.scale.psi_0)


def gate_bias_for(target_psi_s: float, bracket=(-3.0, 3.0), **kwargs) -> float:
    """The gate bias that produces a given surface potential [V].

    A root find over solved states, not a fit to anything. psi_s(V_G) is
    monotone, so brentq on the bracket is safe.
    """

    def residual(v_gate: float) -> float:
        device, state = solved(gate_voltage=v_gate, **kwargs)
        return surface_potential(device, state) - target_psi_s

    return float(brentq(residual, *bracket, xtol=1e-9))


# --------------------------------------------------------------- flatband


@pytest.mark.parametrize(
    "work_function",
    [C.PHI_M_N_POLY, C.PHI_M_P_POLY, C.PHI_M_MIDGAP],
    ids=["n+poly", "p+poly", "midgap"],
)
@pytest.mark.parametrize("net_doping", [-1e15, -1e16, -1e17], ids=str)
def test_biasing_the_gate_at_phi_ms_leaves_the_stack_flat(
    work_function, net_doping
):
    """Flatband means exactly that: one potential everywhere, oxide included.

    Across three gate materials and three dopings, because the substrate
    doping only cancels out of psi_gate if the algebra is right, and a version
    that carried it would still pass at a single doping.
    """
    v_fb = flatband_voltage(net_doping, work_function)
    device, state = solved(
        net_doping=net_doping, work_function=work_function, gate_voltage=v_fb
    )
    psi = state.psi.data
    assert np.ptp(psi) < 1e-11, f"psi spread {np.ptp(psi):g} in scaled units"


def test_the_flat_profile_is_already_the_answer():
    """The charge neutral guess goes in flat and comes out flat, so Newton has
    nothing to do. Anything above a step or two means the guess and the
    boundary conditions disagree, which is the millivolt error this file is
    really hunting."""
    _, state = solved(gate_voltage=flatband_voltage(-NA, C.PHI_M_N_POLY))
    assert state.newton.iterations == 0


@pytest.mark.parametrize(
    "work_function",
    [C.PHI_M_N_POLY, C.PHI_M_P_POLY, C.PHI_M_MIDGAP],
    ids=["n+poly", "p+poly", "midgap"],
)
def test_the_bias_that_removes_the_band_bending_is_the_flatband_voltage(
    work_function,
):
    """The other direction, and the one that is a measurement rather than a
    construction: search for the bias at which the surface stops bending, and
    it has to be Phi_MS."""
    found = gate_bias_for(0.0, work_function=work_function)
    expected = flatband_voltage(-NA, work_function)
    assert found == pytest.approx(expected, abs=GATE_TOLERANCE)
    # And far tighter than the gate, because nothing here is approximated:
    # flatband is an exact statement about three constants agreeing.
    assert found == pytest.approx(expected, abs=1e-5)


# --------------------------------------------------------------- threshold


@pytest.mark.parametrize("net_doping", [-1e15, -1e16, -1e17], ids=str)
def test_the_threshold_bias_matches_the_textbook_expression(net_doping):
    """The gate bias that bends the surface by 2 phi_F, against V_FB + 2 phi_F
    + Q_dep/C_ox.

    Not circular, though the condition and the formula share a definition. The
    condition fixes only the band bending; what is being tested is everything
    else in the expression, the flatband offset and the oxide drop that the
    depletion charge demands, and those come from the gate work function, the
    oxide permittivity and the solved charge distribution.

    The agreement is far tighter than the depletion approximation deserves,
    and that is not an accident: at exactly this surface potential the
    inversion charge and the Debye tail cancel. See
    test_the_depletion_approximation_is_exact_at_threshold_by_cancellation.
    """
    target = 2.0 * phi_F(net_doping)
    found = gate_bias_for(target, net_doping=net_doping)
    expected = threshold_voltage(net_doping, T_OX, C.PHI_M_N_POLY)
    assert found == pytest.approx(expected, abs=GATE_TOLERANCE)
    # And far inside it. The measured worst case over these three dopings is
    # 0.13 mV, at 1e17. See the cancellation test below for why the agreement
    # is this good when the approximation it rests on is several percent wrong
    # a hundred millivolts either side of threshold.
    assert found == pytest.approx(expected, abs=MILLIVOLT)


def test_the_threshold_moves_with_the_oxide_thickness_as_one_over_c_ox():
    """Doubling t_ox doubles the oxide drop, so the threshold moves by exactly
    the Q_dep/C_ox term again. This is the term the flatband test cannot see,
    since at flatband there is no charge to drop a voltage across."""
    target = 2.0 * phi_F(-NA)

    def residual(v_gate):
        device = mos_cap(
            substrate_doping=-NA, t_ox=2 * T_OX, t_si=T_SI, gate_voltage=v_gate
        )
        return surface_potential(device, solve_equilibrium(device)) - target

    found = float(brentq(residual, -3.0, 3.0, xtol=1e-9))
    expected = threshold_voltage(-NA, 2 * T_OX, C.PHI_M_N_POLY)
    thin = threshold_voltage(-NA, T_OX, C.PHI_M_N_POLY)
    assert expected - thin == pytest.approx(
        C.q * NA * max_depletion_width(-NA) / oxide_capacitance(2 * T_OX) / 2,
        rel=1e-12,
    )
    assert found == pytest.approx(expected, abs=GATE_TOLERANCE)


# ------------------------------------------------------- the depletion region


def surface_charge_exact(net_doping: float, psi_s: float) -> float:
    """Q_s [C/cm^2] from the exact 1D Poisson-Boltzmann solution.

        Q_s = sqrt(2 eps V_T q Na) sqrt(e^-u + u - 1 + r(e^u - u - 1))

    with u = psi_s/V_T and r = (n_i/Na)^2. The depletion approximation is the
    u term alone: it drops the e^-u + (-1) that the Debye tail at the depletion
    edge contributes, and the r e^u that the inversion layer contributes.

    Keeping both is what makes this a real reference rather than another
    approximation, and the difference is not small. At a quarter of the way to
    threshold the depletion form is 7.5 percent low.
    """
    Na = abs(net_doping)
    u = psi_s / C.V_T()
    r = (C.n_i() / Na) ** 2
    bracket = np.exp(-u) + u - 1.0 + r * (np.exp(u) - u - 1.0)
    return float(np.sqrt(2.0 * C.eps_Si() * C.V_T() * C.q * Na * bracket))


def surface_field(device, state) -> float:
    """dpsi/dy at the silicon surface [V/cm], one sided from just below it."""
    mesh = device.mesh
    assert device.regions is not None
    column = mesh.nx // 2
    surface = int(device.regions.interface_nodes(mesh)[column])
    below = surface - mesh.nx
    psi = state.psi.data * device.scale.psi_0
    return float(
        (psi[surface] - psi[below])
        / (mesh.node_y[surface] - mesh.node_y[below])
    )


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75], ids=str)
def test_the_surface_charge_matches_poisson_boltzmann(fraction):
    """Gauss at the surface, against the exact charge rather than the
    depletion approximation.

    eps_Si E_s is the total charge the silicon holds, so this reads the whole
    solved profile through one number and compares it to a closed form that
    makes no approximation at all. It comes out inside 0.2 percent at every
    bias between flatband and threshold.

    The second assertion is what stops this passing for the wrong reason. The
    depletion approximation is 4 to 8 percent away over the same range, so a
    tolerance loose enough to admit it would admit almost anything.
    """
    target = fraction * 2.0 * phi_F(-NA)
    v_gate = gate_bias_for(target)
    device, state = solved(gate_voltage=v_gate)

    psi_s = surface_potential(device, state)
    assert psi_s == pytest.approx(target, abs=1e-6)

    measured = C.eps_Si() * surface_field(device, state)
    assert measured == pytest.approx(surface_charge_exact(-NA, psi_s), rel=2e-3)

    depletion = np.sqrt(2.0 * C.eps_Si() * C.q * NA * psi_s)
    assert abs(measured / depletion - 1.0) > 0.02


def test_the_depletion_approximation_is_exact_at_threshold_by_cancellation():
    """Why V_TH agrees to a fraction of a millivolt when the approximation it
    comes from is several percent wrong on either side of it.

    At psi_s = 2 phi_F the inversion term is (n_i/Na)^2 exp(2 phi_F / V_T),
    and 2 phi_F is defined as V_T ln((Na/n_i)^2), so that term is exactly 1.
    It cancels the 1 the depletion tail subtracts, identically and at every
    doping. The textbook threshold expression is therefore not merely a good
    approximation at threshold, it is the exact answer there, which is a
    property of where the definition puts the point rather than luck.
    """
    for net_doping in (-1e15, -1e16, -1e17):
        psi_s = 2.0 * phi_F(net_doping)
        depletion = np.sqrt(2.0 * C.eps_Si() * C.q * abs(net_doping) * psi_s)
        assert surface_charge_exact(net_doping, psi_s) == pytest.approx(
            depletion, rel=1e-6
        )


def test_the_bulk_is_neutral_far_from_the_surface():
    """Whatever the gate does, the far side of a 2 um substrate does not know
    about it. If it did, the body contact would be holding the depletion region
    open and every capacitance would be wrong."""
    device, state = solved(gate_voltage=1.0)
    mesh = device.mesh
    column = mesh.nx // 2
    bottom = mesh.node_at(column, 0)
    one_up = mesh.node_at(column, 1)
    assert abs(state.psi.data[bottom] - state.psi.data[one_up]) < 1e-9


# ------------------------------------------------------------------ the oxide


def test_the_oxide_potential_is_a_straight_line():
    """No charge means Laplace, and Laplace across a uniform slab is linear.
    Any curvature means charge leaked into the insulator."""
    device, state = solved(gate_voltage=1.0)
    mesh = device.mesh
    assert device.regions is not None
    surface_row = int(device.regions.interface_nodes(mesh)[0]) // mesh.nx
    column = mesh.nx // 2

    rows = range(surface_row, mesh.ny)
    y = np.array([mesh.node_y[mesh.node_at(column, j)] for j in rows])
    psi = np.array([state.psi.data[mesh.node_at(column, j)] for j in rows])

    fit = np.polyfit(y, psi, 1)
    residual = psi - np.polyval(fit, y)
    assert np.max(np.abs(residual)) < 1e-10 * np.ptp(psi)


@pytest.mark.parametrize("v_gate", [-2.0, 0.0, 1.0], ids=["acc", "zero", "inv"])
def test_no_carrier_density_is_negative_anywhere(v_gate):
    """A phases/PHASE-4.md acceptance criterion. Free here, since equilibrium
    Poisson carries n and p as exponentials, but it stops being free in Phase 6
    and the check should already exist by then."""
    _, state = solved(gate_voltage=v_gate)
    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


# ------------------------------------------------------- the second dimension


def test_the_solution_does_not_vary_across_the_device():
    """The capacitor is uniform in x, so every column has to solve to the same
    profile. This is the 2D assembly checking itself: if the horizontal edges
    carried the wrong face area or the wrong permittivity, the columns would
    disagree."""
    device, state = solved(gate_voltage=1.0)
    mesh = device.mesh
    psi = state.psi.data.reshape(mesh.ny, mesh.nx)
    spread = np.ptp(psi, axis=1)
    assert np.max(spread) < 1e-12


def test_adding_columns_changes_nothing():
    """Refining the direction the physics does not use must not move the
    answer. It would if the dual areas and the face widths did not cancel."""
    _, coarse = solved(gate_voltage=1.0, nx=3)
    device, fine = solved(gate_voltage=1.0, nx=7)

    coarse_column = coarse.psi.data.reshape(-1, 3)[:, 0]
    fine_column = fine.psi.data.reshape(-1, 7)[:, 0]
    np.testing.assert_allclose(fine_column, coarse_column, rtol=1e-10)
