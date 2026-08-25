"""The full 3N coupled Jacobian, block by block.

phases/PHASE-3.md: "Every Jacobian block matches complex-step differentiation
to 1e-10 on a 20 node mesh. All nine blocks, individually tested.
Non-negotiable." This file is that criterion.

The blocks are tested individually rather than as one matrix comparison
because a single wrong block is the failure this is guarding against, and a
whole-matrix assertion reports "something is wrong" where nine assertions
report which derivative. docs/05-pitfalls.md puts checking the Jacobian second
in the debugging order for exactly this reason: a wrong derivative turns
quadratic convergence into stagnation, and stagnation looks like
ill-conditioning.

Three states are used.

Equilibrium on the diode is the one Newton actually starts from. Measured, its
Bernoulli arguments run from 2.2e-3 to 6.5 and **no edge sits at exactly
zero**, which was worth checking rather than assuming: an abrupt junction on a
20 node mesh leaves structure in psi everywhere, so the quasi neutral regions
are flat to a few parts in a thousand rather than flat exactly.

The perturbed state is off the solution manifold in psi, n and p at once, so
no term is accidentally zero and nothing cancels by symmetry.

The uniform bar is the state that does have X = 0 on every edge, exactly, and
it is the reason the complex step harness needed fixing at the origin. It is
constructed rather than solved for, because a solve would land near the flat
answer and not on it. All three psi blocks come back with exactly zero error
there; with the naive cos(y) - 1 in the complex expm1 the reference would
report B'(0) as 0.0 and the two flux-versus-potential blocks would be checked
against nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import build_device
from ddsim.device.doping import Uniform, abrupt_junction
from ddsim.device.transport import TransportModels, initial_state
from ddsim.discretize.boundary import (
    Carrier,
    OhmicContact,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.continuity import (
    electron_continuity_residual,
    hole_continuity_residual,
)
from ddsim.discretize.coupled import (
    UNKNOWNS_PER_NODE,
    Unknown,
    apply_ohmic_contacts_coupled,
    assemble_coupled,
    assemble_coupled_arrays,
    assemble_coupled_terms,
    coupled_jacobian,
    coupled_residual,
    pack,
    residual_term_scales,
    row_weights,
    scale_rows,
    unknown_index,
    unpack,
)
from ddsim.discretize.poisson import poisson_residual
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.solve.linear import SparseLU
from tests.reference.complexstep import complex_step_jacobian

N_NODES = 20
"""The mesh size phases/PHASE-3.md names for the verification."""


# ----------------------------------------------------------------- fixtures


@pytest.fixture
def device():
    """A 1e16 / 1e16 diode on a 20 node uniform mesh.

    Deliberately under-resolved: the Debye length at 1e16 is 41 nm and the
    spacing here is 53 nm. That is wrong for physics and right for a
    derivative check, because it puts several volts of potential drop on a
    single edge and drives the Bernoulli arguments out to where the branches
    differ. A well resolved mesh would leave every X near zero and test one
    branch.
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=N_NODES)
    return build_device(
        mesh=mesh,
        doping=abrupt_junction(Na=1e16, Nd=1e16, position=0.5e-4),
        contacts=(
            OhmicContact(name="anode", node=0, voltage=0.0),
            OhmicContact(name="cathode", node=N_NODES - 1, voltage=0.0),
        ),
    )


@pytest.fixture(
    params=[
        ("constant", False),
        ("arora", False),
        ("constant", True),
        ("arora", True),
    ],
    ids=["constant", "arora", "constant+auger", "arora+auger"],
)
def models(request, device):
    """Recombination and diffusivities, in scaled units.

    Parametrized over both the mobility model and the Auger flag, because the
    two change the *shape* of what the assemblies receive rather than only the
    numbers. Constant mobility makes Dn and Dp scalars, and a scalar broadcasts
    against an edge array no matter how the edges are indexed. Arora makes them
    one value per edge, where an off by one or a node/edge mixup stops being
    invisible. Verifying the nine blocks only against the scalar case leaves
    the alignment of the array case unpinned, which is the one thing the array
    case exists to get right.

    Auger is carried here for the same reason on the recombination side: it is
    the only model whose rate is not linear in a single carrier, so its
    derivative blocks are the ones a wrong linearization would show up in.
    """
    mobility, auger = request.param
    return TransportModels.for_device(device, mobility=mobility, auger=auger)


@pytest.fixture
def geometry(device):
    """(h, volume) in units of the Debye length, as every assembly wants."""
    scale = device.scale
    return device.mesh.h / scale.x_0, device.mesh.volume / scale.x_0


@pytest.fixture
def equilibrium_x(device):
    """The state Newton starts from.

    Not flat anywhere, despite the name suggesting it should be: the smallest
    edge potential difference on this mesh is 2.2e-3, not zero. See the module
    docstring. The uniform bar below covers the exactly flat case.
    """
    state = initial_state(device)
    return pack(state.psi.data, state.n.data, state.p.data)


@pytest.fixture
def perturbed_x(device):
    """Off the solution manifold in psi, n and p at once.

    The densities are scaled multiplicatively through an exponential, so they
    stay strictly positive and stay within a factor of a few of a physically
    reachable state. A perturbation large enough to make n negative would test
    a Jacobian at a point the solver can never visit.
    """
    state = initial_state(device)
    k = np.linspace(0.0, 3.0 * np.pi, device.mesh.n_nodes)

    psi = state.psi.data + 0.35 * np.cos(k)
    n = state.n.data * np.exp(0.20 * np.sin(k))
    p = state.p.data * np.exp(-0.15 * np.cos(2.0 * k))
    return pack(psi, n, p)


@pytest.fixture(params=["equilibrium", "perturbed"])
def state_x(request, equilibrium_x, perturbed_x):
    """Both states, so every block test runs against each."""
    return equilibrium_x if request.param == "equilibrium" else perturbed_x


# ------------------------------------------------------------------ helpers


def residual_at(geometry, device, models):
    """A one argument residual, which is what the complex step harness takes."""
    h, volume = geometry

    def evaluate(x):
        return coupled_residual(
            h=h,
            volume=volume,
            x=x,
            net_doping=device.net_doping_scaled.data,
            Dn=models.Dn,
            Dp=models.Dp,
            recombination=models.recombination,
        )

    return evaluate


def dense_jacobian(geometry, x, models):
    """The assembled Jacobian, densified for comparison."""
    h, volume = geometry
    rows, cols, values = coupled_jacobian(
        h=h,
        volume=volume,
        x=x,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
    )
    size = x.size
    return coo_matrix((values, (rows, cols)), shape=(size, size)).toarray()


def block(matrix, row: Unknown, col: Unknown):
    """One N x N block, gathered out of the node-interleaved ordering."""
    return matrix[row::UNKNOWNS_PER_NODE, col::UNKNOWNS_PER_NODE]


def node_field(values, unit, name):
    """A scaled node Field, which is the only kind an assembly accepts."""
    return Field(
        values.copy(), unit, ScalingState.SCALED, Location.NODE, name=name
    )


def assemble_state(device, models, x):
    """The Field level assembly at a packed state, without the boilerplate."""
    psi, n, p = unpack(x)
    return assemble_coupled(
        mesh=device.mesh,
        psi=node_field(psi, "V", "psi"),
        n=node_field(n, "cm^-3", "n"),
        p=node_field(p, "cm^-3", "p"),
        net_doping=device.net_doping_scaled,
        recombination=models.recombination,
        scale=device.scale,
        Dn=models.Dn,
        Dp=models.Dp,
    )


# ------------------------------------------------------------------ ordering


def test_pack_and_unpack_round_trip():
    """Whatever the ordering is, it has to be reversible."""
    psi = np.array([1.0, 2.0, 3.0])
    n = np.array([4.0, 5.0, 6.0])
    p = np.array([7.0, 8.0, 9.0])

    got_psi, got_n, got_p = unpack(pack(psi, n, p))

    np.testing.assert_array_equal(got_psi, psi)
    np.testing.assert_array_equal(got_n, n)
    np.testing.assert_array_equal(got_p, p)


def test_ordering_is_interleaved_by_node():
    """docs/02-numerics.md: psi_0, n_0, p_0, psi_1, ... for fill reduction.

    Blocking by variable instead would put the three unknowns of one node 2N
    apart and give the factorization a much wider band to fill.
    """
    psi = np.array([1.0, 2.0, 3.0])
    n = np.array([4.0, 5.0, 6.0])
    p = np.array([7.0, 8.0, 9.0])

    got = pack(psi, n, p)

    np.testing.assert_array_equal(
        got, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
    )


def test_unpack_returns_views_not_copies():
    """A copy per call would double the cost of every residual evaluation."""
    x = np.arange(9.0)
    psi, n, p = unpack(x)

    psi[0] = -1.0

    assert x[0] == -1.0


def test_pack_rejects_mismatched_lengths():
    """Three arrays of different length is a caller bug, not a broadcast."""
    with pytest.raises(ValueError, match="same length"):
        pack(np.zeros(3), np.zeros(4), np.zeros(3))


# ----------------------------------------------------------------- residual


def test_psi_rows_match_the_poisson_residual(device, geometry, models):
    """The psi block is the Phase 1 residual with n and p read, not derived.

    Checked on a Boltzmann consistent state, where the two agree by
    construction. Off that manifold they differ, and that difference is the
    whole point of the coupled system.
    """
    h, volume = geometry
    state = initial_state(device)
    psi = state.psi.data
    phi_n = state.phi_n.data
    phi_p = state.phi_p.data
    n = np.exp(psi - phi_n)
    p = np.exp(phi_p - psi)

    got = coupled_residual(
        h=h,
        volume=volume,
        x=pack(psi, n, p),
        net_doping=device.net_doping_scaled.data,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
    )
    expected = poisson_residual(
        h, volume, psi, device.net_doping_scaled.data, phi_n, phi_p
    )

    np.testing.assert_allclose(
        got[Unknown.PSI :: UNKNOWNS_PER_NODE], expected, rtol=1e-13, atol=0.0
    )


def test_electron_rows_match_the_uncoupled_continuity_residual(
    device, geometry, models, perturbed_x
):
    """Same equation, different unknown vector. The residual cannot move."""
    h, volume = geometry
    psi, n, p = unpack(perturbed_x)
    R = np.asarray(models.recombination.rate(n, p), dtype=np.float64)

    got = residual_at(geometry, device, models)(perturbed_x)
    expected = electron_continuity_residual(h, volume, models.Dn, psi, n, R)

    np.testing.assert_allclose(
        got[Unknown.N :: UNKNOWNS_PER_NODE], expected, rtol=1e-13, atol=0.0
    )


def test_hole_rows_match_the_uncoupled_continuity_residual(
    device, geometry, models, perturbed_x
):
    """The mirror of the electron check, with the flux asymmetry intact."""
    h, volume = geometry
    psi, n, p = unpack(perturbed_x)
    R = np.asarray(models.recombination.rate(n, p), dtype=np.float64)

    got = residual_at(geometry, device, models)(perturbed_x)
    expected = hole_continuity_residual(h, volume, models.Dp, psi, p, R)

    np.testing.assert_allclose(
        got[Unknown.P :: UNKNOWNS_PER_NODE], expected, rtol=1e-13, atol=0.0
    )


def test_residual_preserves_a_complex_dtype(device, geometry, models, perturbed_x):
    """Without this the complex step verification silently reports zeros."""
    got = residual_at(geometry, device, models)(
        perturbed_x.astype(np.complex128)
    )

    assert np.iscomplexobj(got)


@pytest.fixture
def flat_bar():
    """A uniformly doped bar at equilibrium, with X exactly zero everywhere.

    Built from the closed form rather than solved for. In scaled units the
    equilibrium of a uniform bar is psi = asinh(N/2) with n = exp(psi) and
    p = exp(-psi), constant across the mesh, so every edge potential
    difference is exactly 0.0 and every Bernoulli argument sits on the
    removable singularity.

    Returns (device, models, geometry, x).
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=N_NODES)
    device = build_device(
        mesh=mesh,
        doping=Uniform(1e16),
        contacts=(
            OhmicContact(name="anode", node=0, voltage=0.0),
            OhmicContact(name="cathode", node=N_NODES - 1, voltage=0.0),
        ),
    )
    models = TransportModels.for_device(device)
    scale = device.scale

    doping = device.net_doping_scaled.data
    psi = np.full(mesh.n_nodes, np.arcsinh(doping[0] / 2.0))
    x = pack(psi, np.exp(psi), np.exp(-psi))

    return device, models, (mesh.h / scale.x_0, mesh.volume / scale.x_0), x


# ------------------------------------------- the nine blocks, the acceptance


ALL_BLOCKS = [
    (row, col) for row in Unknown for col in Unknown
]


@pytest.mark.parametrize("row,col", ALL_BLOCKS, ids=lambda u: u.name)
def test_every_jacobian_block_matches_complex_step(
    device, geometry, models, state_x, row, col
):
    """The Phase 3 acceptance criterion. Non-negotiable, all nine blocks.

    Compared entry by entry against the block's own largest entry rather than
    against each entry's own magnitude. The blocks span many decades inside
    themselves, and an entry that is small because two large terms cancelled
    carries no more absolute information than the cancellation left in it.
    """
    reference = complex_step_jacobian(
        residual_at(geometry, device, models), state_x
    )
    assembled = dense_jacobian(geometry, state_x, models)

    got = block(assembled, row, col)
    expected = block(reference, row, col)
    floor = np.max(np.abs(expected))

    np.testing.assert_allclose(
        got, expected, rtol=1e-10, atol=1e-10 * max(floor, 1e-300)
    )


@pytest.mark.parametrize("row,col", ALL_BLOCKS, ids=lambda u: u.name)
def test_every_jacobian_block_matches_complex_step_at_a_flat_potential(
    flat_bar, row, col
):
    """The same nine blocks where every Bernoulli argument is exactly zero.

    B(0) = 1 and B'(0) = -1/2 are both removable singularities reached by a
    different branch of the implementation and a different branch of the
    reference, so this is not the diode test on easier numbers. It is the case
    that fails silently when the complex step reference drops the second order
    term in expm1, and it is the state a uniformly doped region sits in.
    """
    device, models, geometry, x = flat_bar

    psi, _, _ = unpack(x)
    assert np.all(psi[1:] - psi[:-1] == 0.0), "the fixture is not flat"

    reference = complex_step_jacobian(residual_at(geometry, device, models), x)
    assembled = dense_jacobian(geometry, x, models)

    got = block(assembled, row, col)
    expected = block(reference, row, col)
    floor = np.max(np.abs(expected))

    np.testing.assert_allclose(
        got, expected, rtol=1e-10, atol=1e-10 * max(floor, 1e-300)
    )


@pytest.fixture
def lopsided_bar():
    """A device whose Arora diffusivity genuinely differs from edge to edge.

    The shared device fixture cannot do this job. It is a 1e16 / 1e16
    junction, so abs(net doping) is 1e16 on every node, and Arora reads only
    the total doping: Dn comes back with a single unique value across all
    nineteen edges. A constant array is indistinguishable from a scalar under
    broadcasting, so running the nine blocks against it verifies the array
    code path without verifying that the array is *aligned* to the edges it
    belongs to. Measured on the 1e18 / 1e15 profile used here, Dn takes three
    distinct values with a factor of 4.7 between the ends, and is not
    symmetric under reversal, which is what makes an off by one or a reversed
    gather visible.

    Returns (device, models, geometry, x).
    """
    mesh = uniform_mesh_1d(length=1e-4, n_nodes=N_NODES)
    device = build_device(
        mesh=mesh,
        doping=abrupt_junction(Na=1e18, Nd=1e15, position=0.5e-4),
        contacts=(
            OhmicContact(name="anode", node=0, voltage=0.0),
            OhmicContact(name="cathode", node=N_NODES - 1, voltage=0.0),
        ),
    )
    models = TransportModels.for_device(device, mobility="arora")
    scale = device.scale
    geometry = (mesh.h / scale.x_0, mesh.volume / scale.x_0)

    state = initial_state(device)
    k = np.linspace(0.0, 3.0 * np.pi, mesh.n_nodes)
    x = pack(
        state.psi.data + 0.35 * np.cos(k),
        state.n.data * np.exp(0.3 * np.sin(k)),
        state.p.data * np.exp(-0.3 * np.sin(k)),
    )
    return device, models, geometry, x


@pytest.mark.parametrize("row,col", ALL_BLOCKS, ids=lambda u: u.name)
def test_every_jacobian_block_matches_complex_step_per_edge_diffusivity(
    lopsided_bar, row, col
):
    """The nine blocks again, with a diffusivity that varies along the device.

    Doping dependent mobility turns Dn and Dp from scalars into one value per
    edge. Every other block test here runs with scalars, which broadcast
    correctly no matter how the edges are indexed, so this is the only place
    that pins the per-edge alignment of the flux coefficients and their
    derivatives.
    """
    device, models, geometry, x = lopsided_bar

    Dn = np.asarray(models.Dn)
    assert Dn.size == device.mesh.n_edges, "Dn is not per edge"
    assert np.unique(Dn).size > 1, "Dn does not vary, so alignment is untested"
    assert not np.array_equal(Dn, Dn[::-1]), "Dn is reversal symmetric"

    reference = complex_step_jacobian(residual_at(geometry, device, models), x)
    assembled = dense_jacobian(geometry, x, models)

    got = block(assembled, row, col)
    expected = block(reference, row, col)
    floor = np.max(np.abs(expected))

    np.testing.assert_allclose(
        got, expected, rtol=1e-10, atol=1e-10 * max(floor, 1e-300)
    )


def test_the_psi_block_carries_no_boltzmann_charge_term(
    device, geometry, models, perturbed_x
):
    """dF_psi/dpsi is the bare Laplacian here, unlike the Phase 1 Poisson.

    In Phase 1 n and p are functions of psi and the diagonal picks up
    (n + p)*volume, which is what makes that matrix an M-matrix. In the
    coupled system they are separate unknowns, so that term moves into
    dF_psi/dn and dF_psi/dp instead. Carrying it in both places is the most
    likely way to get this wrong, because the Phase 1 Jacobian is right there
    to copy, and the result would be a Jacobian that is wrong by exactly the
    term the coupling was introduced to represent.
    """
    h, volume = geometry
    assembled = dense_jacobian(geometry, perturbed_x, models)
    got = block(assembled, Unknown.PSI, Unknown.PSI)

    expected = np.zeros((device.mesh.n_nodes, device.mesh.n_nodes))
    conductance = 1.0 / h
    for e in range(h.size):
        expected[e, e] += conductance[e]
        expected[e + 1, e + 1] += conductance[e]
        expected[e, e + 1] -= conductance[e]
        expected[e + 1, e] -= conductance[e]

    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_the_charge_blocks_are_plus_and_minus_the_cell_volume(
    device, geometry, models, perturbed_x
):
    """dF_psi/dn = +volume and dF_psi/dp = -volume, diagonal only.

    The signs come straight from -(p - n + N)*volume in the residual. Getting
    them the wrong way round flips the sign of the electrostatic feedback and
    turns Newton's correction into an amplification.
    """
    _, volume = geometry
    assembled = dense_jacobian(geometry, perturbed_x, models)

    np.testing.assert_allclose(
        block(assembled, Unknown.PSI, Unknown.N), np.diag(volume), atol=0.0
    )
    np.testing.assert_allclose(
        block(assembled, Unknown.PSI, Unknown.P), np.diag(-volume), atol=0.0
    )


def test_the_recombination_cross_blocks_are_diagonal(
    device, geometry, models, perturbed_x
):
    """R is a point function, so dF_n/dp and dF_p/dn touch one node only.

    Any off diagonal entry here means a flux term leaked into the wrong
    block, which complex step would still confirm if the residual leaked the
    same way.
    """
    assembled = dense_jacobian(geometry, perturbed_x, models)

    for row, col in ((Unknown.N, Unknown.P), (Unknown.P, Unknown.N)):
        got = block(assembled, row, col)
        assert np.count_nonzero(got - np.diag(np.diag(got))) == 0


def test_the_jacobian_sparsity_is_block_tridiagonal(geometry, models, perturbed_x):
    """No entry may couple nodes more than one edge apart in 1D.

    A stray entry would still satisfy the complex step check if the residual
    put it there too, so the structure is asserted separately from the values.
    """
    h, volume = geometry
    rows, cols, _ = coupled_jacobian(
        h=h,
        volume=volume,
        x=perturbed_x,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
    )

    row_node = rows // UNKNOWNS_PER_NODE
    col_node = cols // UNKNOWNS_PER_NODE

    assert np.all(np.abs(row_node - col_node) <= 1)


# ------------------------------------------------------------- Field layer


def test_assemble_coupled_agrees_with_the_array_level_functions(
    device, geometry, models, perturbed_x
):
    """The Field wrapper must not change a single number."""
    assembly = assemble_state(device, models, perturbed_x)

    expected = residual_at(geometry, device, models)(perturbed_x)

    assert assembly.shape == (
        UNKNOWNS_PER_NODE * device.mesh.n_nodes,
        UNKNOWNS_PER_NODE * device.mesh.n_nodes,
    )
    np.testing.assert_array_equal(assembly.residual, expected)


def test_assemble_coupled_rejects_a_physical_field(device, models, perturbed_x):
    """A physical density here is wrong by C_0 and would still converge."""
    psi, n, p = unpack(perturbed_x)

    with pytest.raises(ValueError, match="SCALED"):
        assemble_coupled(
            mesh=device.mesh,
            psi=Field(
                psi.copy(), "V", ScalingState.PHYSICAL, Location.NODE, name="psi"
            ),
            n=Field(n.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="n"),
            p=Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="p"),
            net_doping=device.net_doping_scaled,
            recombination=models.recombination,
            scale=device.scale,
            Dn=models.Dn,
            Dp=models.Dp,
        )


def test_unknown_index_agrees_with_the_packed_layout(device):
    """The index helper and pack must not drift apart.

    Two ways to say the same thing, so one test pins them together rather
    than letting a later reordering fix one and leave the other.
    """
    psi = np.arange(4.0)
    n = np.arange(4.0) + 10.0
    p = np.arange(4.0) + 20.0
    x = pack(psi, n, p)

    for node in range(4):
        assert x[unknown_index(node, Unknown.PSI)] == psi[node]
        assert x[unknown_index(node, Unknown.N)] == n[node]
        assert x[unknown_index(node, Unknown.P)] == p[node]


def test_assemble_coupled_rejects_an_edge_field(device, models, perturbed_x):
    """A density on edges would be silently one entry short of the mesh."""
    psi, n, p = unpack(perturbed_x)

    with pytest.raises(ValueError, match="NODE"):
        assemble_coupled(
            mesh=device.mesh,
            psi=Field(
                psi.copy(), "V", ScalingState.SCALED, Location.EDGE, name="psi"
            ),
            n=Field(n.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="n"),
            p=Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="p"),
            net_doping=device.net_doping_scaled,
            recombination=models.recombination,
            scale=device.scale,
            Dn=models.Dn,
            Dp=models.Dp,
        )


def test_assemble_coupled_rejects_a_field_of_the_wrong_length(
    device, models, perturbed_x
):
    """A length mismatch broadcasts into a plausible wrong answer otherwise."""
    psi, n, p = unpack(perturbed_x)

    with pytest.raises(ValueError, match="length"):
        assemble_coupled(
            mesh=device.mesh,
            psi=Field(
                psi[:-1].copy(), "V", ScalingState.SCALED, Location.NODE, name="psi"
            ),
            n=Field(n.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="n"),
            p=Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="p"),
            net_doping=device.net_doping_scaled,
            recombination=models.recombination,
            scale=device.scale,
            Dn=models.Dn,
            Dp=models.Dp,
        )


# ------------------------------------------------------------------ contacts


def test_contacts_pin_all_three_unknowns_at_the_contact_node(
    device, models, perturbed_x
):
    """A coupled ohmic contact is three Dirichlet conditions, not one."""
    assembly = assemble_state(device, models, perturbed_x)

    pinned = apply_ohmic_contacts_coupled(
        assembly,
        perturbed_x,
        device.net_doping_scaled.data,
        device.contacts,
        device.scale,
    )

    solver = SparseLU()
    solver.factorize(pinned.rows, pinned.cols, pinned.values, pinned.shape)
    delta = solver.solve(-pinned.residual)
    solved = perturbed_x + delta

    doping = device.net_doping_scaled.data
    for contact in device.contacts:
        node = contact.node
        assert solved[unknown_index(node, Unknown.PSI)] == pytest.approx(
            ohmic_psi_scaled(
                float(doping[node]), contact.voltage / device.scale.psi_0
            ),
            rel=1e-14,
        )
        assert solved[unknown_index(node, Unknown.N)] == pytest.approx(
            ohmic_density_scaled(float(doping[node]), Carrier.ELECTRON), rel=1e-14
        )
        assert solved[unknown_index(node, Unknown.P)] == pytest.approx(
            ohmic_density_scaled(float(doping[node]), Carrier.HOLE), rel=1e-14
        )


def test_the_contact_densities_agree_with_the_contact_potential(device):
    """n = exp(psi - phi_n) at a contact, with phi_n = phi_p = the bias.

    The two boundary conditions are written from different physics, one from
    neutrality plus mass action and one from asinh of the doping, so their
    agreeing is a real check rather than a restatement. If they disagreed the
    solver would be pulled between two incompatible statements at one node
    and the terminal current would come out wrong with everything converged.
    """
    biased = device.with_bias(anode=0.35, cathode=0.0)
    doping = biased.net_doping_scaled.data

    for contact in biased.contacts:
        applied = contact.voltage / biased.scale.psi_0
        psi = ohmic_psi_scaled(float(doping[contact.node]), applied)
        n = ohmic_density_scaled(float(doping[contact.node]), Carrier.ELECTRON)
        p = ohmic_density_scaled(float(doping[contact.node]), Carrier.HOLE)

        assert n == pytest.approx(np.exp(psi - applied), rel=1e-12)
        assert p == pytest.approx(np.exp(applied - psi), rel=1e-12)
        assert n * p == pytest.approx(1.0, rel=1e-12)


def test_the_applied_bias_moves_psi_and_leaves_the_densities_alone(device):
    """An ohmic contact stays in equilibrium whatever the terminal voltage."""
    doping = device.net_doping_scaled.data[0]

    unbiased = ohmic_psi_scaled(float(doping), 0.0)
    biased = ohmic_psi_scaled(float(doping), 1.0)

    assert biased - unbiased == pytest.approx(1.0, rel=1e-14)
    assert ohmic_density_scaled(
        float(doping), Carrier.ELECTRON
    ) == ohmic_density_scaled(float(doping), Carrier.ELECTRON)


def test_two_contacts_sharing_a_name_are_rejected(device, models, perturbed_x):
    """Mirrors the Poisson path. A duplicate name breaks current reporting."""
    assembly = assemble_state(device, models, perturbed_x)

    with pytest.raises(ValueError, match="unique"):
        apply_ohmic_contacts_coupled(
            assembly,
            perturbed_x,
            device.net_doping_scaled.data,
            (
                OhmicContact(name="anode", node=0, voltage=0.0),
                OhmicContact(name="anode", node=N_NODES - 1, voltage=0.0),
            ),
            device.scale,
        )


# ------------------------------------------------------------- row scaling


def test_the_poisson_term_scale_counts_the_carriers_not_only_the_doping():
    """An intrinsic bar has no doping and its Poisson terms are not zero.

    The residual carries -(p - n + N)*volume. On intrinsic material that sum
    is exactly zero, but the terms going into it are n*volume and p*volume,
    both equal to one dual cell in scaled units. A scale built from the net
    doping alone reports zero there, and dividing by it makes every row nan.

    Measured before the fix: an undoped 41 node bar came back with a residual
    of nan and the message blamed the LU factorization for being singular,
    which sends you debugging the linear algebra instead of the scale.
    """
    h = np.full(4, 0.1)
    volume = np.full(5, 0.1)
    x = pack(np.zeros(5), np.ones(5), np.ones(5))

    psi_scale, _, _ = residual_term_scales(
        h, volume, x, np.zeros(5), Dn=1.0, Dp=1.0
    )

    assert psi_scale > 0.0


def test_every_term_scale_is_strictly_positive_on_a_real_device(
    device, geometry, models, perturbed_x
):
    """Nothing downstream can divide by these safely otherwise."""
    h, volume = geometry
    scales = residual_term_scales(
        h, volume, perturbed_x, device.net_doping_scaled.data, models.Dn, models.Dp
    )

    assert all(scale > 0.0 for scale in scales)


def test_row_scaling_keeps_the_system_finite(device, geometry, models, perturbed_x):
    """The whole point of the scaling is defeated if it introduces a nan."""
    h, volume = geometry
    scales = residual_term_scales(
        h, volume, perturbed_x, device.net_doping_scaled.data, models.Dn, models.Dp
    )
    system = assemble_coupled_arrays(
        h=h,
        volume=volume,
        x=perturbed_x,
        net_doping=device.net_doping_scaled.data,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
    )

    scaled = scale_rows(system, row_weights(scales, device.mesh.n_nodes))

    assert np.all(np.isfinite(scaled.residual))
    assert np.all(np.isfinite(scaled.values))


def test_a_state_with_no_carriers_anywhere_is_refused():
    """Not a physical state, and silently producing nan hides where it came from.

    A density of exactly zero everywhere leaves every term in every equation
    at zero, so there is no scale to measure against. Refusing names the
    problem; dividing by zero renames it as a singular matrix three call
    frames later.
    """
    h = np.full(4, 0.1)
    volume = np.full(5, 0.1)
    x = pack(np.zeros(5), np.zeros(5), np.zeros(5))

    with pytest.raises(ValueError, match="no terms"):
        residual_term_scales(h, volume, x, np.zeros(5), Dn=1.0, Dp=1.0)


def test_the_shared_path_reproduces_the_standalone_functions_exactly(
    device, geometry, models, perturbed_x
):
    """assemble_coupled_terms and the two public functions must not diverge.

    This is what keeps the block verification meaningful. Those tests
    differentiate coupled_residual and compare against coupled_jacobian, but a
    solve runs assemble_coupled_terms, which shares one Bernoulli pair between
    the three. Two code paths where only one is verified is how a verification
    stops being one.

    Found by mutation rather than by inspection: after the shared path was
    introduced, replacing the exact SRH tangent with the Gummel frozen slope
    inside it left every block test passing, because no test executed it.

    Bit for bit, not to a tolerance. The two do the same operations in the
    same order on the same inputs, so anything less than exact equality means
    they have genuinely drifted apart.
    """
    h, volume = geometry
    doping = device.net_doping_scaled.data

    shared = assemble_coupled_terms(
        h, volume, perturbed_x, doping, models.Dn, models.Dp, models.recombination
    ).assembly

    residual = coupled_residual(
        h, volume, perturbed_x, doping, models.Dn, models.Dp, models.recombination
    )
    rows, cols, values = coupled_jacobian(
        h, volume, perturbed_x, models.Dn, models.Dp, models.recombination
    )

    np.testing.assert_array_equal(shared.residual, residual)
    np.testing.assert_array_equal(shared.rows, rows)
    np.testing.assert_array_equal(shared.cols, cols)
    np.testing.assert_array_equal(shared.values, values)


def test_the_shared_path_scales_match_the_standalone_scales(
    device, geometry, models, perturbed_x
):
    """The third output of the shared path needs the same guard."""
    h, volume = geometry
    doping = device.net_doping_scaled.data
    psi, n, p = unpack(perturbed_x)

    _, shared = assemble_coupled_terms(
        h, volume, perturbed_x, doping, models.Dn, models.Dp, models.recombination
    )
    standalone = residual_term_scales(
        h,
        volume,
        perturbed_x,
        doping,
        models.Dn,
        models.Dp,
        R=np.asarray(models.recombination.rate(n, p), dtype=np.float64),
    )

    assert shared == standalone
