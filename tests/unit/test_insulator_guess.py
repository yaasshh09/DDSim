"""The initial guess where there is no semiconductor.

The charge neutral guess, psi = asinh(N/2), is the exact answer in uniform
material and the starting point for every solve in this project. In an
insulator it is meaningless: N is zero there, so it returns the intrinsic level
of a material that has no carriers to be intrinsic about, and the guess arrives
at zero while the gate is tens of scaled units away.

That is not a small inefficiency. psi is damped to 5 V_T per Newton step, and
the damping is one factor over the whole vector, so an oxide starting 138
scaled units from the gate throttles the entire solve. Measured on the default
capacitor, cold, the iteration count grows at 7.7 per volt of gate bias and
runs out of the 50 step budget at about 6.5 V.

The fix is to give the insulator the answer to its own equation. With no charge
in it, Poisson there is Laplace, which is linear, so one solve with everything
around it held where it is gets it exactly rather than approximately. On the
MOS stack that reproduces a straight line from the surface to the gate, which
is what the oxide potential is; the point of writing it as a solve instead of
an interpolation is that it needs to know nothing about which way the layers
stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.equilibrium import insulator_guess, solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.physics.statistics import psi_equilibrium_scaled

V_GATE = 3.0
"""A bias far enough from flatband that the oxide has real work to do."""


def neutral(device):
    """The charge neutral guess, before anything is filled in."""
    return np.asarray(
        psi_equilibrium_scaled(device.net_doping_scaled.data), dtype=np.float64
    )


@pytest.fixture(scope="module")
def cap():
    return mos_cap(gate_voltage=V_GATE)


@pytest.fixture(scope="module")
def filled(cap):
    return insulator_guess(cap, neutral(cap))


def test_the_semiconductor_is_left_exactly_as_it_was(cap, filled):
    """Bit for bit. The charge neutral guess is right where there is charge to
    be neutral, and this is not entitled to an opinion about it."""
    assert cap.regions is not None
    silicon = cap.regions.semiconductor_volume > 0.0
    np.testing.assert_array_equal(filled[silicon], neutral(cap)[silicon])


def test_the_insulator_is_not_left_as_it_was(cap, filled):
    """The guard on the test above: if nothing moved, everything here passes
    for the wrong reason."""
    assert cap.regions is not None
    oxide = cap.regions.oxide_nodes
    assert np.max(np.abs(filled[oxide] - neutral(cap)[oxide])) > 1.0


def test_the_filled_oxide_is_a_straight_line(cap, filled):
    """Laplace across a slab with no charge in it. This is the whole content of
    the fill, and on the stack it is checkable directly."""
    mesh = cap.mesh
    assert cap.regions is not None
    surface_row = int(cap.regions.interface_nodes(mesh)[0]) // mesh.nx
    column = mesh.nx // 2

    rows = range(surface_row, mesh.ny)
    y = np.array([mesh.node_y[mesh.node_at(column, j)] for j in rows])
    psi = np.array([filled[mesh.node_at(column, j)] for j in rows])

    fit = np.polyfit(y, psi, 1)
    residual = psi - np.polyval(fit, y)
    assert np.max(np.abs(residual)) < 1e-12 * np.ptp(psi)


def test_the_fill_reaches_the_gate_potential(cap, filled):
    """The top of that line is the gate, which the fill has to have read from
    the contact rather than been told."""
    from ddsim.discretize.boundary import GateContact, gate_psi_scaled

    gate = next(c for c in cap.contacts if isinstance(c, GateContact))
    expected = gate_psi_scaled(
        gate.voltage / cap.scale.psi_0, gate.work_function, cap.material.T
    )
    np.testing.assert_allclose(filled[list(gate.nodes)], expected, rtol=1e-12)


def test_a_device_with_no_insulator_is_returned_untouched():
    """Every 1D device in Phases 1 to 3 is this case. Nothing already measured
    is allowed to move, so the array comes back as itself rather than as an
    equal copy that went through a solve."""
    diode = pn_diode()
    guess = neutral(diode)
    assert insulator_guess(diode, guess) is guess


def test_an_all_silicon_2d_device_is_returned_untouched():
    """Carrying a RegionMap is not the same as having an insulator in it."""
    cap = mos_cap(t_ox=1e-6, t_si=2e-4)
    all_silicon = cap.regions.__class__(
        cell_material=np.zeros_like(cap.regions.cell_material),
        eps_r=np.ones_like(cap.regions.eps_r),
        semiconductor_volume=cap.mesh.volume,
        semiconductor_face=cap.mesh.dual_face,
        oxide_nodes=np.array([], dtype=np.int64),
    )
    import dataclasses

    device = dataclasses.replace(cap, regions=all_silicon)
    guess = neutral(device)
    assert insulator_guess(device, guess) is guess


# ------------------------------------------------------------ what it buys


@pytest.mark.parametrize("v_gate", [-8.0, -4.0, 4.0, 8.0], ids=str)
def test_a_cold_solve_far_from_flatband_converges_in_a_modest_budget(v_gate):
    """The measurement that motivated the fill. Without it these take 38, 28,
    38 and 69 Newton steps and the last two need the budget raised; with it
    they are all under 20, and the count stops growing with the bias because
    the oxide is no longer what the solve is spending its steps on.
    """
    device = mos_cap(gate_voltage=v_gate)
    state = solve_equilibrium(device, max_iterations=20)
    assert state.newton.converged


def test_the_flat_stack_is_recognised_as_already_solved():
    """At flatband the filled guess is the answer everywhere, oxide included,
    so Newton has nothing to do. It is the sharpest statement available that
    the fill agrees with the gate work function and the body contact, since
    all three have to produce the same number for the profile to be flat."""
    v_fb = float(C.work_function_difference(C.PHI_M_N_POLY, -1e16))
    state = solve_equilibrium(mos_cap(gate_voltage=v_fb))
    assert state.newton.iterations == 0
