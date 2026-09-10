"""Tests for Fermi-Dirac statistics where they meet the assemblies.

tests/unit/test_statistics.py covers the series and the object built on it.
This file covers the seam: the device flag, the contacts, the Poisson
diagonal, the two Bernoulli arguments and the Jacobian blocks that grow
because of them.

Two claims run through everything here.

**A Boltzmann device is unchanged.** Every path takes None and does nothing
with it, so a device built without the flag runs the arithmetic it ran before
Phase 5 and returns the same bits. That is asserted rather than assumed,
because the alternative is a phase that quietly moves every number the
project already validated against DEVSIM.

**A degenerate device is one state, not three.** docs/07-decisions.md, row
dated 2026-09-01, says the degenerate contacts and the degenerate Bernoulli
argument have to land in one change: the discrete equations hold the
Boltzmann relation exactly at equilibrium, so a contact carrying a
Fermi-Dirac density on a Boltzmann interior would put the whole 30.5 mV into
a boundary layer one node wide. The fixed point test at the bottom is what
says both halves landed.

On the complex step reference
-----------------------------
tests/reference/complexstep.py records that Im(B(x + ih)) loses about
2*eps/abs(x), so the harness is worse than the code it checks for edge drops
between zero and roughly 1e-5. A degenerate junction at equilibrium has
several such edges inside the 1e20 side, where psi is flat to 1e-15 but not
to zero. So the block checks here run at a flat state, where every drop is
exactly zero and the harness is exact, and at a perturbed state, where every
drop is of order 0.1 and it is exact again. Measured on the equilibrium
junction the harness disagrees with the assembled dF_n/dn by 1.4e-4 while a
real central difference agrees with it to ten digits, which is the harness
losing and not the Jacobian.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device, build_device
from ddsim.device.doping import Uniform, abrupt_junction
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    TransportModels,
    electron_block,
    hole_block,
    initial_state,
    solve_bias,
    solve_bias_newton,
)
from ddsim.discretize.boundary import (
    Carrier,
    OhmicContact,
    apply_ohmic_densities,
    impose_ohmic_densities,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.continuity import assemble_electron_continuity
from ddsim.discretize.coupled import (
    UNKNOWNS_PER_NODE,
    Unknown,
    apply_contacts_coupled,
    assemble_coupled_terms,
    coupled_jacobian,
    coupled_residual,
    effective_potentials,
    pack,
    residual_term_scales,
    unknown_index,
    unpack,
)
from ddsim.discretize.poisson import _carrier_densities
from ddsim.extract.iv import terminal_currents
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics import Degeneracy, einstein_ratio
from tests.reference.complexstep import complex_step_jacobian

N_NODES = 20
"""The mesh size phases/PHASE-3.md names for the block verification."""

HEAVY = 1e20
"""Source and drain doping [cm^-3], where n/Nc is 3.5 and Boltzmann is not on."""


# ------------------------------------------------------------------ fixtures


def junction(degenerate: bool, n_nodes: int = N_NODES):
    """A 1e17 / 1e20 abrupt junction, degenerate on the n side only.

    Deliberately under-resolved on a uniform mesh, for the reason the Phase 3
    fixture is: it puts several volts on a single edge and drives the
    Bernoulli arguments out to where the branches differ.
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=n_nodes)
    return build_device(
        mesh=mesh,
        doping=abrupt_junction(Na=1e17, Nd=HEAVY, position=0.5e-4),
        contacts=(
            OhmicContact(name="anode", node=0, voltage=0.0),
            OhmicContact(name="cathode", node=n_nodes - 1, voltage=0.0),
        ),
        degenerate=degenerate,
    )


def flat_bar(doping: float):
    """A uniformly doped bar at its exact equilibrium, every edge drop zero.

    Built from the closed form rather than solved for, so psi, n and p are
    constant to the bit and every Bernoulli argument sits on the removable
    singularity. That is the one state where the complex step harness is
    exact, and it is also where the degenerate term is at full strength.

    Returns (device, models, h, volume, x).
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=N_NODES)
    device = build_device(
        mesh=mesh,
        doping=Uniform(doping),
        contacts=(
            OhmicContact(name="anode", node=0, voltage=0.0),
            OhmicContact(name="cathode", node=N_NODES - 1, voltage=0.0),
        ),
        degenerate=True,
    )
    net = device.net_doping_scaled.data
    n, p = device.degeneracy.equilibrium_densities(net[0])
    psi = np.full(mesh.n_nodes, float(device.degeneracy.equilibrium_psi(net[0])))
    x = pack(psi, np.full(mesh.n_nodes, float(n)), np.full(mesh.n_nodes, float(p)))
    scale = device.scale
    return (
        device,
        TransportModels.for_device(device),
        mesh.h / scale.x_0,
        mesh.volume / scale.x_0,
        x,
    )


def perturbed(device):
    """Off the solution manifold in psi, n and p at once.

    Every edge drop is then of order 0.1, which is where the complex step
    harness is exact again. The densities move multiplicatively so they stay
    positive and stay within a factor of a few of a reachable state.
    """
    state = initial_state(device)
    k = np.linspace(0.0, 3.0 * np.pi, device.mesh.n_nodes)
    return pack(
        state.psi.data + 0.35 * np.cos(k),
        state.n.data * np.exp(0.20 * np.sin(k)),
        state.p.data * np.exp(-0.15 * np.cos(2.0 * k)),
    )


def dense(rows, cols, values, size):
    return coo_matrix((values, (rows, cols)), shape=(size, size)).toarray()


def blocks_agree(assembled, reference, rtol=1e-10):
    """Compare the nine blocks, each against its own largest entry.

    The blocks span many decades inside themselves, and an entry that is small
    because two large terms cancelled carries no more absolute information
    than the cancellation left in it.
    """
    for row in Unknown:
        for col in Unknown:
            got = assembled[row::UNKNOWNS_PER_NODE, col::UNKNOWNS_PER_NODE]
            want = reference[row::UNKNOWNS_PER_NODE, col::UNKNOWNS_PER_NODE]
            floor = max(float(np.max(np.abs(want))), 1e-300)
            np.testing.assert_allclose(
                got,
                want,
                rtol=rtol,
                atol=rtol * floor,
                err_msg=f"dF_{row.name}/d{col.name}",
            )


# --------------------------------------------------------------- the flag


def test_a_device_is_boltzmann_unless_it_says_otherwise() -> None:
    """The default has to be off, or Phase 5 silently moves Phases 1 to 4.

    Both defaults, because they are two: build_device names the flag in its
    own signature and passes it on, so flipping the dataclass field alone
    would leave every built device Boltzmann and every hand assembled one
    degenerate, which is the kind of split nothing else here would notice.
    """
    assert junction(degenerate=False).degeneracy is None
    assert not Device.__dataclass_fields__["degenerate"].default
    assert (
        inspect.signature(build_device).parameters["degenerate"].default is False
    )


def test_the_flag_builds_the_statistics_in_the_devices_own_scaling() -> None:
    """A bare Nc in a scaled assembly is off by ten decades and converges."""
    device = junction(degenerate=True)
    assert device.degeneracy == Degeneracy.for_silicon(device.scale.C_0)
    assert device.degeneracy.Nc == pytest.approx(C.Nc(C.T_ROOM) / device.scale.C_0)


# ------------------------------------------------------------- the contacts


def test_the_contact_values_are_boltzmann_without_the_statistics() -> None:
    """Not close to. The same call with None has to be the same arithmetic.

    Against math.asinh and not np.arcsinh. They are two implementations of the
    same function and they disagree in the last two bits on some platforms, so
    comparing across them tests the C library rather than this branch. CI read
    23.025850929940454 from one and ...57 from the other on Python 3.11 and
    3.12. math.asinh is what the Boltzmann branch actually calls.
    """
    assert ohmic_psi_scaled(1e10, 0.0) == math.asinh(1e10 / 2.0)
    assert ohmic_density_scaled(1e10, Carrier.ELECTRON) * ohmic_density_scaled(
        1e10, Carrier.HOLE
    ) == pytest.approx(1.0, rel=1e-14)


def test_the_degenerate_contact_moves_the_potential_by_thirty_millivolts() -> None:
    """Row 119 of docs/07-decisions.md, now at an actual contact node."""
    degeneracy = Degeneracy.for_silicon(C.n_i())
    doping = HEAVY / C.n_i()
    shift = (
        ohmic_psi_scaled(doping, 0.0, degeneracy) - ohmic_psi_scaled(doping, 0.0)
    ) * C.V_T() * 1e3  # [mV]
    assert shift == pytest.approx(30.5, rel=1e-2)


def test_the_degenerate_contact_carries_the_applied_bias_unchanged() -> None:
    """The statistics belong to the material and the bias to the terminal, so
    a contact at 0.4 V has to sit exactly 0.4 V above the same contact at 0.
    """
    degeneracy = Degeneracy.for_silicon(C.n_i())
    doping = HEAVY / C.n_i()
    applied = 0.4 / C.V_T()
    assert ohmic_psi_scaled(doping, applied, degeneracy) - ohmic_psi_scaled(
        doping, 0.0, degeneracy
    ) == pytest.approx(applied, rel=1e-14)


def test_the_three_contact_values_are_one_state() -> None:
    """psi, n and p pinned at one node have to satisfy the same relation the
    interior does, or the contact fights the discretization from the first
    iteration. This is the condition row 94 of the decisions log names.
    """
    degeneracy = Degeneracy.for_silicon(C.n_i())
    doping = HEAVY / C.n_i()
    psi = ohmic_psi_scaled(doping, 0.0, degeneracy)
    n = ohmic_density_scaled(doping, Carrier.ELECTRON, degeneracy)
    p = ohmic_density_scaled(doping, Carrier.HOLE, degeneracy)

    assert n - p == pytest.approx(doping, rel=1e-14)
    assert float(degeneracy.electron_potential(psi, n)) == pytest.approx(
        float(np.log(n)), rel=1e-14
    )
    assert float(degeneracy.hole_potential(psi, p)) == pytest.approx(
        float(-np.log(p)), rel=1e-14
    )


# ------------------------------------------------------------- Poisson terms


def test_the_poisson_densities_are_their_own_derivatives_under_boltzmann() -> None:
    """dn/dpsi is n, which is the term that makes the Phase 1 matrix an
    M-matrix. Returning the same array is the arithmetic that was there.
    """
    psi = np.linspace(-10.0, 10.0, 7)
    n, p, dn, dp = _carrier_densities(psi, None, None)
    np.testing.assert_array_equal(dn, n)
    np.testing.assert_array_equal(dp, p)


def test_the_degenerate_poisson_diagonal_is_divided_by_the_einstein_ratio() -> None:
    """Filling the band buys less density per volt, so the charge derivative
    falls below the charge. It stays positive, so the diagonal is still only
    strengthened by it and the matrix is still the one Phase 1 converged on.
    """
    degeneracy = Degeneracy.for_silicon(C.n_i())
    psi = np.array([0.0, 10.0, 20.0, 24.0])
    n, _, dn, _ = _carrier_densities(psi, None, None, None, degeneracy)
    np.testing.assert_allclose(
        dn, n / einstein_ratio(n / degeneracy.Nc), rtol=1e-14
    )
    assert np.all(dn > 0.0)
    assert np.all(dn <= n)


def test_an_insulator_node_holds_no_carriers_under_either_statistics() -> None:
    """An exponent of -inf has to come back as exactly zero, not as a nan out
    of inf minus inf. The nan would land on the gate row of a MOS stack, be
    overwritten by the Dirichlet condition, and surface in the charge.
    """
    degeneracy = Degeneracy.for_silicon(C.n_i())
    psi = np.array([800.0, 0.0])
    carriers = np.array([False, True])
    n, p, dn, dp = _carrier_densities(psi, None, None, carriers, degeneracy)
    assert n[0] == 0.0
    assert p[0] == 0.0
    assert dn[0] == 0.0
    assert dp[0] == 0.0


# -------------------------------------------------- the effective potentials


def test_boltzmann_returns_psi_itself_for_both_carriers() -> None:
    """The same array, not a copy of it. Nothing is computed on this path."""
    psi = np.linspace(-5.0, 5.0, 11)
    psi_n, psi_p = effective_potentials(psi, psi, psi, None)
    assert psi_n is psi
    assert psi_p is psi


def test_the_two_carriers_see_different_potentials_when_degenerate() -> None:
    """The electron one falls below psi and the hole one rises above it, which
    is the same statement twice: a filled band pushes its own carriers out.
    """
    degeneracy = Degeneracy.for_silicon(C.n_i())
    psi = np.zeros(3)
    n = np.array([1e6, 1e9, 1e10])
    psi_n, psi_p = effective_potentials(psi, n, n, degeneracy)
    assert np.all(psi_n < 0.0)
    assert np.all(psi_p > 0.0)

    # Not mirror images of each other: the two bands have different densities
    # of states, so the same carrier density is a different fraction of each.
    assert not np.allclose(psi_n, -psi_p, rtol=1e-3)


# --------------------------------------------------- the Jacobian, all nine


@pytest.mark.parametrize("doping", [HEAVY, -HEAVY], ids=["n+", "p+"])
def test_every_block_matches_complex_step_on_a_flat_degenerate_bar(doping) -> None:
    """Every edge drop exactly zero, so the harness is exact, and the majority
    carrier at n/Nc = 3.5, so the degenerate term is at full strength. Both
    signs of the doping, because the hole branch is separate code.
    """
    device, models, h, volume, x = flat_bar(doping)
    net = device.net_doping_scaled.data

    def residual(v):
        return coupled_residual(
            h=h,
            volume=volume,
            x=v,
            net_doping=net,
            Dn=models.Dn,
            Dp=models.Dp,
            recombination=models.recombination,
            degeneracy=device.degeneracy,
        )

    rows, cols, values = coupled_jacobian(
        h=h,
        volume=volume,
        x=x,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
        degeneracy=device.degeneracy,
    )
    blocks_agree(
        dense(rows, cols, values, x.size), complex_step_jacobian(residual, x)
    )


@pytest.mark.parametrize(
    "mobility,field",
    [("constant", False), ("arora", True)],
    ids=["constant", "arora+field"],
)
def test_every_block_matches_complex_step_at_a_perturbed_junction(
    mobility, field
) -> None:
    """A 1e17 / 1e20 junction pushed off its manifold in all three unknowns.

    Caughey-Thomas is carried because it is the one model whose diffusivity
    reads the potential drop, and the degenerate change had to leave that
    reading the real potential rather than the effective one. A field
    dependence that followed the electron effective potential would be a
    mobility that changes when a density does, which is not what the model
    says and is exactly what a complex step catches.
    """
    device = junction(degenerate=True)
    models = TransportModels.for_device(
        device, mobility=mobility, auger=True, field_dependent=field
    )
    scale = device.scale
    h = device.mesh.h / scale.x_0
    volume = device.mesh.volume / scale.x_0
    x = perturbed(device)
    net = device.net_doping_scaled.data

    def residual(v):
        return coupled_residual(
            h=h,
            volume=volume,
            x=v,
            net_doping=net,
            Dn=models.Dn,
            Dp=models.Dp,
            recombination=models.recombination,
            degeneracy=device.degeneracy,
        )

    rows, cols, values = coupled_jacobian(
        h=h,
        volume=volume,
        x=x,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
        degeneracy=device.degeneracy,
    )
    blocks_agree(
        dense(rows, cols, values, x.size), complex_step_jacobian(residual, x)
    )


def test_the_continuity_diagonal_picks_up_the_einstein_ratio() -> None:
    """The closed form behind the new blocks, without a harness in the way.

    On a flat bar every Bernoulli argument is zero, so B = 1 and B' = -1/2,
    and the edge coefficient of dF_n/dn works out to D/h times 1 + u*dcorr/du,
    which is the generalized Einstein ratio. At 1e20 that is 2.13, so the
    degenerate stencil is more than twice the Boltzmann one. That factor is
    the physical content of the change: degeneracy raises the diffusivity.
    """
    device, models, h, volume, x = flat_bar(HEAVY)
    _, n, _ = unpack(x)
    ratio = float(einstein_ratio(n[0] / device.degeneracy.Nc))
    assert ratio == pytest.approx(2.13, rel=1e-2)

    def off_diagonal(degeneracy):
        rows, cols, values = coupled_jacobian(
            h=h,
            volume=volume,
            x=x,
            Dn=models.Dn,
            Dp=models.Dp,
            recombination=models.recombination,
            degeneracy=degeneracy,
        )
        matrix = dense(rows, cols, values, x.size)
        block = matrix[Unknown.N::UNKNOWNS_PER_NODE, Unknown.N::UNKNOWNS_PER_NODE]
        return float(np.diag(block, 1)[N_NODES // 2])

    assert off_diagonal(device.degeneracy) == pytest.approx(
        off_diagonal(None) * ratio, rel=1e-12
    )



def test_the_solver_entry_point_assembles_what_the_public_ones_do() -> None:
    """assemble_coupled_terms is the only thing a Newton loop calls, and it
    evaluates the shared work once rather than going through the three public
    functions. That saving is where a degenerate device is easiest to get
    wrong: the two carriers now need two Bernoulli pairs, and handing the
    electron pair to both halves costs nothing, breaks nothing visibly, and
    makes the hole flux answer a different equation from the one the residual
    above defines. Nothing but a comparison against the public path catches
    it, because both halves would then be wrong together.
    """
    device = junction(degenerate=True)
    models = TransportModels.for_device(device)
    scale = device.scale
    h = device.mesh.h / scale.x_0
    volume = device.mesh.volume / scale.x_0
    x = perturbed(device)
    net = device.net_doping_scaled.data

    shared = assemble_coupled_terms(
        h,
        volume,
        x,
        net,
        models.Dn,
        models.Dp,
        models.recombination,
        degeneracy=device.degeneracy,
    )

    np.testing.assert_allclose(
        shared.assembly.residual,
        coupled_residual(
            h=h,
            volume=volume,
            x=x,
            net_doping=net,
            Dn=models.Dn,
            Dp=models.Dp,
            recombination=models.recombination,
            degeneracy=device.degeneracy,
        ),
        rtol=1e-14,
    )
    rows, cols, values = coupled_jacobian(
        h=h,
        volume=volume,
        x=x,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
        degeneracy=device.degeneracy,
    )
    np.testing.assert_allclose(
        dense(
            shared.assembly.rows,
            shared.assembly.cols,
            shared.assembly.values,
            x.size,
        ),
        dense(rows, cols, values, x.size),
        rtol=1e-14,
    )
    standalone = residual_term_scales(
        h,
        volume,
        x,
        net,
        models.Dn,
        models.Dp,
        np.asarray(models.recombination.rate(*unpack(x)[1:]), dtype=np.float64),
        degeneracy=device.degeneracy,
    )
    for from_shared, alone in zip(shared.scales, standalone, strict=True):
        np.testing.assert_array_equal(from_shared, alone)


def test_the_coupled_contacts_pin_the_degenerate_values() -> None:
    """apply_dirichlet_nodes writes x - target into the pinned row, so a state
    carrying the Boltzmann contact values has a residual there of exactly the
    difference between the two statistics. Zero would mean the contacts never
    heard about the flag, and 30.5 mV of psi is what it should be.
    """
    device = junction(degenerate=True)
    models = TransportModels.for_device(device)
    scale = device.scale
    h = device.mesh.h / scale.x_0
    volume = device.mesh.volume / scale.x_0
    net = device.net_doping_scaled.data

    doping = float(net[device.mesh.n_nodes - 1])
    boltzmann = pack(
        np.full(device.mesh.n_nodes, ohmic_psi_scaled(doping, 0.0)),
        np.full(device.mesh.n_nodes, ohmic_density_scaled(doping, Carrier.ELECTRON)),
        np.full(device.mesh.n_nodes, ohmic_density_scaled(doping, Carrier.HOLE)),
    )

    assembly = assemble_coupled_terms(
        h,
        volume,
        boltzmann,
        net,
        models.Dn,
        models.Dp,
        models.recombination,
        degeneracy=device.degeneracy,
    ).assembly
    pinned = apply_contacts_coupled(
        assembly,
        boltzmann,
        net,
        device.contacts,
        scale,
        degeneracy=device.degeneracy,
    )

    cathode = device.mesh.n_nodes - 1
    row = unknown_index(cathode, Unknown.PSI)
    assert pinned.residual[row] * C.V_T() * 1e3 == pytest.approx(-30.5, rel=1e-2)

    # And the densities are pinned at their own degenerate values, which is
    # the half of the contact that mass action changes rather than asinh.
    row = unknown_index(cathode, Unknown.N)
    assert pinned.residual[row] == pytest.approx(
        boltzmann[row]
        - ohmic_density_scaled(doping, Carrier.ELECTRON, device.degeneracy),
        rel=1e-12,
    )



# ------------------------------------------------ the quasi-Fermi levels


def test_the_quasi_fermi_level_is_read_under_the_states_own_statistics() -> None:
    """phi_n is psi_eff - ln(n), not psi - ln(n).

    A density and a potential do not by themselves say where the Fermi level
    is. Reading a degenerate state with the Boltzmann formula is wrong by the
    30.5 mV correction, and the nonlinear Poisson solve then holds that wrong
    level fixed while psi moves, so the error is not a reporting slip but a
    different equation.
    """
    device = junction(degenerate=True, n_nodes=201)
    state = solve_equilibrium(device)

    # Equilibrium, so both levels are flat at zero whatever the statistics.
    assert float(np.max(np.abs(state.phi_n.data))) < 1e-12
    assert float(np.max(np.abs(state.phi_p.data))) < 1e-12

    # And the Boltzmann reading of the same state is not, by the correction.
    boltzmann = replace(state, degeneracy=None)
    shift = float(np.max(np.abs(boltzmann.phi_n.data))) * C.V_T() * 1e3  # [mV]
    assert shift == pytest.approx(30.5, rel=1e-2)


def test_a_boltzmann_state_reads_its_levels_the_way_it_always_did() -> None:
    """psi - ln(n) exactly, with no correction path taken at all."""
    device = junction(degenerate=False, n_nodes=201)
    state = solve_equilibrium(device)
    assert state.degeneracy is None
    np.testing.assert_array_equal(
        state.phi_n.data, state.psi.data - np.log(state.n.data)
    )


def test_the_lagged_gummel_path_lands_where_the_coupled_newton_does() -> None:
    """A Gummel block cannot carry the degenerate Bernoulli argument exactly,
    because its whole premise is that continuity is linear in its own carrier.
    So the correction is lagged at the incoming density, the way a field
    dependent diffusivity already is.

    Lagging is allowed to cost convergence rate and is not allowed to move the
    answer: at the fixed point the lagged density is the solved one. The check
    is that the coupled Newton, handed the converged Gummel state, has nothing
    left to do. Measured before the quasi-Fermi levels knew about the
    statistics, it had 2.4e-4 of the current left to do.

    What it does have left is the last decade of the Gummel solve's own
    convergence. Newton starts here at a residual of 2.7e-10 and spends two
    steps taking it to 1.1e-14, which moves psi by 9.3e-10 and the terminal
    current by 1.6e-6 of itself. That is the Gummel update tolerance, not the
    lagging, and it is two decades below the discrepancy this test was written
    to catch. Before the residual was measured row by row the same solve
    reported convergence at iteration zero, so the agreement here used to be
    exact for the wrong reason. See docs/07-decisions.md.
    """
    device = replace(
        pn_diode(Na=1e17, Nd=HEAVY, n_nodes=201, anode_voltage=0.3),
        degenerate=True,
    )
    gummel = solve_bias(device)
    assert gummel.gummel.converged

    newton = solve_bias_newton(device, guess=gummel)
    assert newton.newton.converged
    np.testing.assert_allclose(
        newton.n.data, gummel.n.data, rtol=1e-8
    )
    np.testing.assert_allclose(
        newton.psi.data, gummel.psi.data, rtol=0.0, atol=1e-8
    )


@pytest.mark.parametrize(
    "doping,make_block",
    [(-HEAVY, electron_block), (HEAVY, hole_block)],
    ids=["electrons in p+", "holes in n+"],
)
def test_equilibrium_is_a_fixed_point_of_each_continuity_block(
    doping, make_block
) -> None:
    """One more step of a continuity block from the answer has to move nothing.

    This is what says the block solves against the boundary value it reports.
    A block whose pinned Dirichlet row still names the Boltzmann contact while
    its imposed value is the degenerate one converges regardless, because the
    imposed value is written over the solved one every cycle and the update at
    that node is zero either way. The interior is solved against a boundary
    condition nothing ever reports, and stepping once more from the answer is
    what notices.

    Each block is run on the bar where its own carrier is the minority one at
    1e20, because that is where the two statistics disagree: the majority
    density at a contact is the doping either way, while the minority one
    comes from the mass action product and is 3.3 times smaller under
    Fermi-Dirac. Measured with the target left Boltzmann, the electron block
    on a p+ bar moves by 6.6e-11 against 4.5e-13 when the two agree.
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=51)
    device = build_device(
        mesh=mesh,
        doping=Uniform(doping),
        contacts=(
            OhmicContact(name="left", node=0, voltage=0.0),
            OhmicContact(name="right", node=50, voltage=0.0),
        ),
        degenerate=True,
    )
    state = solve_equilibrium(device)
    _, update = make_block(device, TransportModels.for_device(device))(state)
    assert update < 1e-11


def test_the_two_contact_writers_name_the_same_density() -> None:
    """A Gummel block writes its contact twice: apply_ohmic_densities pins the
    row it solves against, and impose_ohmic_densities writes the value into
    the answer. They have to name the same number.

    If only one of them hears about the statistics the block still converges,
    because the imposed value overwrites the solved one every cycle and the
    update at that node is zero either way. What it does instead is solve the
    interior against a boundary value it never reports, which is the failure
    mode that produces a plausible answer and no complaint at all. Pinning the
    row at the value the state already carries makes the residual there
    exactly zero, and that is what this asserts.
    """
    device = replace(pn_diode(Na=1e17, Nd=HEAVY, n_nodes=51), degenerate=True)
    state = solve_equilibrium(device)
    doping = device.net_doping_scaled.data

    imposed = impose_ohmic_densities(
        state.n.data,
        doping,
        device.ohmic_contacts,
        Carrier.ELECTRON,
        device.degeneracy,
    )
    n = Field(imposed, "cm^-3", ScalingState.SCALED, Location.NODE, name="n")

    assembly = apply_ohmic_densities(
        assemble_electron_continuity(
            device.mesh_1d,
            state.psi,
            n,
            state.p,
            TransportModels.for_device(device).recombination,
            device.scale,
            TransportModels.for_device(device).Dn,
        ),
        imposed,
        doping,
        device.ohmic_contacts,
        Carrier.ELECTRON,
        device.degeneracy,
    )

    for contact in device.ohmic_contacts:
        for node in contact.nodes:
            assert assembly.residual[node] == 0.0


def test_the_gummel_path_imposes_the_degenerate_contact_densities() -> None:
    """impose_ohmic_densities writes the contact value into the solved profile
    rather than arriving at it, so the density at a contact node after a
    Gummel solve is exactly the value the boundary condition names. It has to
    be the degenerate one.

    Read at the 1e17 anode rather than at the 1e20 cathode, and on the
    minority carrier. The majority density at a contact is the doping under
    either statistics, so it says nothing; the minority one comes from the
    mass action product, which is 0.9977 there instead of 1, and that is the
    half of the contact only the degenerate branch gets right.
    """
    device = replace(
        pn_diode(Na=1e17, Nd=HEAVY, n_nodes=201, anode_voltage=0.3),
        degenerate=True,
    )
    state = solve_bias(device)
    assert state.gummel.converged

    doping = float(device.net_doping_scaled.data[0])
    degenerate = ohmic_density_scaled(doping, Carrier.ELECTRON, device.degeneracy)
    boltzmann = ohmic_density_scaled(doping, Carrier.ELECTRON)

    assert float(state.n.data[0]) == pytest.approx(degenerate, rel=1e-14)
    assert abs(degenerate / boltzmann - 1.0) > 1e-3


def test_the_degenerate_diode_carries_a_current_close_to_the_boltzmann_one() -> None:
    """The forward current of a 1e17 / 1e20 diode is set by injection into the
    lightly doped side, which is nowhere degenerate, so the statistics move it
    by a fraction of a percent rather than by the three times Boltzmann is
    wrong by at 1e20. A change that moved this by a decade would be putting
    the correction in the wrong place, not turning on new physics.
    """
    current = {}
    for degenerate in (False, True):
        device = replace(
            pn_diode(Na=1e17, Nd=HEAVY, n_nodes=201, anode_voltage=0.5),
            degenerate=degenerate,
        )
        state = solve_bias_newton(device, guess=solve_bias(device))
        assert state.newton.converged
        current[degenerate] = terminal_currents(device, state)["anode"]

    assert current[True] == pytest.approx(current[False], rel=1e-3)
    assert current[True] != current[False]


# ------------------------------------------------------------ the fixed point


def test_degenerate_equilibrium_is_a_fixed_point_of_the_coupled_system() -> None:
    """The invariant that says both halves of the change landed together.

    The equilibrium Poisson solve, the contacts and the Bernoulli arguments
    are three separate pieces of code that have to agree about what the
    degenerate relation is. If any one of them is still Boltzmann, the state
    that solves the first does not satisfy the third, and the coupled residual
    at the equilibrium answer is a boundary layer rather than roundoff.

    Measured against each row's own term scale, because the three residuals
    differ by six decades in size and a single threshold would declare the
    Poisson equation converged a million times above its floor. Row by row
    rather than family by family for the same reason one step further down:
    the electron flux terms span 11.5 decades between the two sides of this
    junction, so a family wide scale is set on the degenerate side and says
    nothing about the lightly doped one.

    Reading it row by row is what showed the fixed point is not exact, and
    what the one term missing from it is. SRH takes its equilibrium product
    from n_i squared, and under Fermi-Dirac the equilibrium product is not
    n_i squared: it is n_i squared times gamma_n gamma_p, which on the 1e20
    side of this junction is 0.307. So the recombination term at rest is not
    zero, it is a net generation of 2.6e-12 in scaled units, and it is 3.1e-5
    of the flux terms of the rows that carry it. The family wide measure
    divided that by a scale set on the degenerate side and reported 1e-15.

    So the claim here is the exact one rather than a threshold that hides the
    difference: subtract the recombination and what is left is roundoff at
    1.4e-14. Everything the change was meant to land, the Poisson solve, the
    contacts and both Bernoulli arguments, agrees to the last bit. The one
    thing that does not is named, and named in one place. Under Boltzmann the
    same subtraction changes nothing because R itself is 6.7e-22 there. See
    docs/07-decisions.md.
    """
    device = junction(degenerate=True, n_nodes=201)
    models = TransportModels.for_device(device)
    state = solve_equilibrium(device)
    x = pack(state.psi.data, state.n.data, state.p.data)
    scale = device.scale
    h = device.mesh.h / scale.x_0
    volume = device.mesh.volume / scale.x_0
    net = device.net_doping_scaled.data

    _, n, p = unpack(x)
    R = np.asarray(models.recombination.rate(n, p), dtype=np.float64)
    scales = residual_term_scales(
        h, volume, x, net, models.Dn, models.Dp, R, degeneracy=device.degeneracy
    )
    residual = coupled_residual(
        h=h,
        volume=volume,
        x=x,
        net_doping=net,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
        degeneracy=device.degeneracy,
    )

    # The equilibrium product is broken by exactly the degeneracy factors,
    # which is what makes SRH generate at rest. Asserted so this test says
    # why it carries the term below rather than only that it does.
    assert float(np.max(np.abs(n * p - 1.0))) == pytest.approx(0.693, rel=1e-2)

    psi_row, n_row, p_row = unpack(residual)
    psi_scale, n_scale, p_scale = scales
    assert float(np.max(np.abs(psi_row) / psi_scale)) < 1e-14
    assert float(np.max(np.abs(n_row - R * volume) / n_scale)) < 1e-13
    assert float(np.max(np.abs(p_row - R * volume) / p_scale)) < 1e-13


def test_a_boltzmann_equilibrium_has_no_recombination_to_subtract() -> None:
    """The other half of the test above, and the reason it is not a loosening.

    Under Boltzmann the equilibrium product is n_i squared to the last bit, so
    the SRH rate at rest is 6.7e-22 rather than 2.6e-12 and the fixed point is
    exact with nothing subtracted from it. If the degenerate contacts or the
    degenerate Bernoulli argument ever regress to Boltzmann, this is the test
    that still holds and the one above that fails.
    """
    device = junction(degenerate=False, n_nodes=201)
    models = TransportModels.for_device(device)
    state = solve_equilibrium(device)
    x = pack(state.psi.data, state.n.data, state.p.data)
    scale = device.scale
    h = device.mesh.h / scale.x_0
    volume = device.mesh.volume / scale.x_0
    net = device.net_doping_scaled.data

    _, n, p = unpack(x)
    R = np.asarray(models.recombination.rate(n, p), dtype=np.float64)
    scales = residual_term_scales(h, volume, x, net, models.Dn, models.Dp, R)
    residual = coupled_residual(
        h=h,
        volume=volume,
        x=x,
        net_doping=net,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
    )

    assert float(np.max(np.abs(n * p - 1.0))) < 1e-15
    assert float(np.max(np.abs(R * volume))) < 1e-20
    for family, size in zip(unpack(residual), scales, strict=True):
        assert float(np.max(np.abs(family) / size)) < 1e-14


def test_the_solved_state_holds_the_degenerate_relation_at_every_node() -> None:
    """ln(n) - psi_eff_n is one number across the whole device, contacts
    included. Under Boltzmann psi_eff is psi and this is the check
    docs/04-validation.md already asks for; under Fermi-Dirac it is the same
    check with the correction in, and it is what a half landed change fails.
    """
    device = junction(degenerate=True, n_nodes=201)
    state = solve_equilibrium(device)
    psi_n, psi_p = effective_potentials(
        state.psi.data, state.n.data, state.p.data, device.degeneracy
    )
    assert float(np.ptp(np.log(state.n.data) - psi_n)) < 1e-13
    assert float(np.ptp(np.log(state.p.data) + psi_p)) < 1e-13


def test_the_built_in_potential_rises_by_the_predicted_correction() -> None:
    """A given electron density needs a higher Fermi level once the band is
    filling, so the 1e20 side sits higher and the junction is 30.5 mV
    stronger. Row 119 of docs/07-decisions.md predicted the number before
    there was anything to apply it to, and this is that number at a junction.
    """
    contacts = {
        degenerate: solve_equilibrium(junction(degenerate, n_nodes=201)).psi.data
        for degenerate in (False, True)
    }
    built_in = {
        key: (psi[-1] - psi[0]) * C.V_T() for key, psi in contacts.items()
    }  # [V]
    assert (built_in[True] - built_in[False]) * 1e3 == pytest.approx(
        30.5, rel=1e-2
    )


def test_a_lightly_doped_device_barely_notices_the_statistics() -> None:
    """At 1e16 the correction to the built in potential is 0.1 mV, which is
    four decades below the 1 percent docs/04-validation.md asks of a solved
    potential. Turning the flag on there costs nothing and changes nothing,
    which is what makes it safe to leave on for a whole MOSFET rather than
    switching it on region by region.
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=101)
    built_in = {}
    for degenerate in (False, True):
        device = build_device(
            mesh=mesh,
            doping=abrupt_junction(Na=1e16, Nd=1e16, position=0.5e-4),
            contacts=(
                OhmicContact(name="anode", node=0, voltage=0.0),
                OhmicContact(name="cathode", node=100, voltage=0.0),
            ),
            degenerate=degenerate,
        )
        psi = solve_equilibrium(device).psi.data
        built_in[degenerate] = (psi[-1] - psi[0]) * C.V_T()  # [V]

    assert abs(built_in[True] - built_in[False]) * 1e3 < 0.2  # [mV]
