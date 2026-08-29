"""Structured tensor product 2D mesh, box integration.

phases/PHASE-4.md says to start structured rather than Delaunay, and
docs/02-numerics.md says why: a rectangle cannot produce an obtuse triangle, so
the negative dual area failure mode that docs/05-pitfalls.md warns about cannot
arise at all. Unstructured meshing waits until geometry demands it.

The mesh is built as the tensor product of two Mesh1D axes, which is not a
shortcut. It means the grading logic and the dual grid construction are
inherited rather than written a second time, and the dual cell sum invariant
comes with them.

That invariant is a tolerance rather than an equality, which I only found by
checking. See test_the_dual_cells_sum_to_the_domain_area below.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import normal_field, tensor_mesh_2d, uniform_mesh_2d


@pytest.fixture
def mesh():
    """A 5 by 4 node rectangle, deliberately not square."""
    return uniform_mesh_2d(width=2e-4, height=1e-4, nx=5, ny=4)


def test_the_node_count_is_the_product_of_the_axes(mesh):
    assert mesh.nx == 5
    assert mesh.ny == 4
    assert mesh.n_nodes == 20


def test_the_edge_count_is_both_families_added(mesh):
    """(nx-1)*ny horizontal, nx*(ny-1) vertical. No diagonals."""
    assert mesh.n_edges == (5 - 1) * 4 + 5 * (4 - 1)
    assert mesh.n_edges == mesh.h.size == mesh.dual_face.size


def test_the_dual_cells_sum_to_the_domain_area(mesh):
    """The 2D version of the invariant Mesh1D holds itself to.

    Not asserted as exactly zero, and it was worth checking rather than
    assuming. Summing the outer product visits the terms in a different order
    from multiplying the two 1D sums, so the last bit can move: measured at
    3.3e-16 relative on this fixture. The 1D sum is not unconditionally exact
    either, which is easy to miss because it happens to be exact on the Phase 0
    acceptance case. It is exact for a 1 um domain on 200 nodes and off by
    1.4e-16 for the same domain on 5. The tolerance here matches the one the
    1D tests already use.
    """
    assert mesh.volume.sum() == pytest.approx(2e-4 * 1e-4, rel=1e-14)


def test_every_edge_joins_two_geometrically_adjacent_nodes(mesh):
    """h is the actual distance between the two nodes of the edge.

    This is the test that catches a wrong node numbering convention. If the
    row major index arithmetic is off, the edge list still looks plausible and
    the counts still come out right, but some edge will join two nodes that
    are not neighbours and its length will not match the distance between them.
    """
    left, right = mesh.edge_nodes[:, 0], mesh.edge_nodes[:, 1]
    dx = mesh.node_x[right] - mesh.node_x[left]
    dy = mesh.node_y[right] - mesh.node_y[left]

    np.testing.assert_allclose(np.hypot(dx, dy), mesh.h, rtol=1e-14)

    # Every edge is axis aligned: exactly one of the two offsets is zero.
    assert np.all((dx == 0.0) ^ (dy == 0.0))


def test_the_dual_faces_are_all_strictly_positive(mesh):
    """A negative dual face is the 2D failure mode. Structured cannot have one."""
    assert np.all(mesh.dual_face > 0.0)
    assert np.all(mesh.volume > 0.0)
    assert np.all(mesh.h > 0.0)


def test_no_edge_joins_a_node_to_itself_and_all_are_in_range(mesh):
    assert np.all(mesh.edge_nodes >= 0)
    assert np.all(mesh.edge_nodes < mesh.n_nodes)
    assert np.all(mesh.edge_nodes[:, 0] != mesh.edge_nodes[:, 1])


def test_every_node_is_touched_by_at_least_two_edges(mesh):
    """A corner has two, an edge node three, an interior node four."""
    touches = np.bincount(mesh.edge_nodes.ravel(), minlength=mesh.n_nodes)
    assert touches.min() == 2
    assert touches.max() == 4


def test_a_horizontal_edge_carries_the_vertical_dual_extent(mesh):
    """The face a horizontal flux crosses is vertical, and vice versa.

    Getting this pair swapped is the classic box integration slip. On a
    non-square mesh it changes the answer; on a square one it does not, which
    is exactly why the fixture is 5 by 4 and the domain is 2:1.
    """
    x_axis = uniform_mesh_1d(length=2e-4, n_nodes=5)
    y_axis = uniform_mesh_1d(length=1e-4, n_nodes=4)

    n_horizontal = (5 - 1) * 4
    horizontal_face = mesh.dual_face[:n_horizontal]
    vertical_face = mesh.dual_face[n_horizontal:]

    # A horizontal edge's face extent is drawn from the y dual grid.
    assert set(np.round(horizontal_face, 18)) <= set(np.round(y_axis.volume, 18))
    assert set(np.round(vertical_face, 18)) <= set(np.round(x_axis.volume, 18))


def test_the_x_structure_matches_the_1d_mesh_it_was_built_from():
    """A row of the 2D mesh has to be the 1D mesh, node for node."""
    x_axis = graded_mesh_1d(length=1e-4, n_nodes=41, refine_at=0.5e-4, h_min=1e-7)
    y_axis = uniform_mesh_1d(length=1e-5, n_nodes=3)
    mesh = tensor_mesh_2d(x_axis, y_axis)

    np.testing.assert_array_equal(mesh.node_x[: mesh.nx], x_axis.x)
    np.testing.assert_array_equal(mesh.node_x[mesh.nx : 2 * mesh.nx], x_axis.x)


def test_a_graded_axis_survives_the_tensor_product():
    """Grading is inherited, not reimplemented."""
    x_axis = graded_mesh_1d(length=1e-4, n_nodes=41, refine_at=0.5e-4, h_min=1e-7)
    mesh = tensor_mesh_2d(x_axis, uniform_mesh_1d(length=1e-5, n_nodes=3))

    assert mesh.volume.sum() == pytest.approx(1e-4 * 1e-5, rel=1e-15)
    assert mesh.h.min() == pytest.approx(x_axis.h.min(), rel=1e-15)


def test_a_mesh_needs_at_least_two_nodes_on_each_axis():
    """A one node axis has no edges and is not a 2D mesh."""
    with pytest.raises(ValueError, match="at least 2 nodes"):
        uniform_mesh_2d(width=1e-4, height=1e-4, nx=1, ny=4)
    with pytest.raises(ValueError, match="at least 2 nodes"):
        uniform_mesh_2d(width=1e-4, height=1e-4, nx=4, ny=1)


def test_node_at_agrees_with_the_row_major_numbering(mesh):
    """The accessor and the actual node positions have to tell one story."""
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            node = mesh.node_at(i, j)
            assert mesh.node_x[node] == mesh.x_axis.x[i]
            assert mesh.node_y[node] == mesh.y_axis.x[j]

    assert mesh.node_at(0, 0) == 0
    assert mesh.node_at(mesh.nx - 1, mesh.ny - 1) == mesh.n_nodes - 1


def test_the_repr_says_the_shape_and_the_size(mesh):
    text = repr(mesh)

    assert "5x4" in text
    assert "20 nodes" in text
    assert f"{mesh.n_edges} edges" in text


def test_the_geometry_it_hands_the_assemblies_is_consistent(mesh):
    """edge_geometry packages the edge list and the dual faces together."""
    geometry = mesh.edge_geometry()

    left, right = geometry.ends(mesh.n_edges)
    np.testing.assert_array_equal(left, mesh.edge_nodes[:, 0])
    np.testing.assert_array_equal(right, mesh.edge_nodes[:, 1])
    np.testing.assert_array_equal(np.asarray(geometry.dual_face), mesh.dual_face)


# ============================================ the field normal to a flat
# interface, Phase 5
#
# Lombardi surface mobility wants the magnitude of the field normal to the
# Si/SiO2 interface. On this mesh the interface is a horizontal line, so the
# normal direction is y and the quantity is abs(dpsi/dy) at each node.
#
# It is nodal rather than edge based on purpose. A horizontal channel edge's
# normal field lives on the vertical edges above and below its endpoints, not
# on the edge itself, so there is no edge quantity to read. Working it out at
# nodes and averaging onto edges afterwards is the same route the mobility
# already takes, and it keeps the answer independent of which edge asked.


def test_a_uniform_vertical_gradient_is_recovered_exactly(mesh):
    """The straightest possible check. psi = g*y everywhere makes dpsi/dy the
    constant g at every node including the two boundary rows, so anything that
    misreads a spacing or drops an edge shows up as a number that is not g."""
    gradient = 3.7e4
    psi = gradient * mesh.node_y

    E = normal_field(mesh, psi)

    np.testing.assert_allclose(E, gradient, rtol=1e-12)


def test_a_purely_horizontal_potential_has_no_normal_field(mesh):
    """psi varying only along x has no y derivative anywhere. This is the test
    that fails if the two edge families get swapped, and swapping them is the
    single most likely error here: both are arrays over edges of the same mesh
    and neither carries a label saying which it is."""
    psi = 5.0 * mesh.node_x

    np.testing.assert_allclose(normal_field(mesh, psi), 0.0, atol=1e-9)


def test_the_normal_field_is_a_magnitude(mesh):
    """Lombardi takes abs(E_perp) and refuses a signed one, so the sign has to
    be gone before it gets there. A gate above and a gate below the same
    channel produce the same scattering."""
    psi = 2.5e4 * mesh.node_y

    up = normal_field(mesh, psi)
    down = normal_field(mesh, -psi)

    assert np.all(up > 0.0)
    np.testing.assert_allclose(up, down, rtol=1e-14)


def test_an_interior_node_averages_the_edges_either_side():
    """Where the two vertical edges at a node disagree, the node takes their
    mean. A graded mesh makes them disagree on purpose: the same potential
    difference across a shorter edge is a larger field.

    Worked by hand rather than against the function, so the test knows the
    answer independently.
    """
    y_axis = graded_mesh_1d(
        length=3e-5, n_nodes=3, refine_at=0.0, h_min=1e-5, max_ratio=3.0
    )
    mesh = tensor_mesh_2d(uniform_mesh_1d(length=1e-5, n_nodes=2), y_axis)

    # A potential that is 0, 1, 3 down the three rows, so the two differences
    # are 1 and 2 across edges of length h[0] and h[1].
    psi = np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0])
    h = y_axis.h

    E = normal_field(mesh, psi)

    lower, upper = 1.0 / h[0], 2.0 / h[1]
    np.testing.assert_allclose(E[0], lower, rtol=1e-12)   # bottom row, one edge
    np.testing.assert_allclose(E[2], 0.5 * (lower + upper), rtol=1e-12)
    np.testing.assert_allclose(E[4], upper, rtol=1e-12)   # top row, one edge


def test_a_boundary_row_uses_the_single_edge_it_has(mesh):
    """The top and bottom rows have one vertical neighbour, not two. Dividing
    by a hardcoded two there would halve the field at exactly the row the gate
    sits on, which is the row the whole model is about."""
    psi = 1.1e4 * mesh.node_y

    E = normal_field(mesh, psi)
    bottom = E[: mesh.nx]
    top = E[-mesh.nx :]

    np.testing.assert_allclose(bottom, 1.1e4, rtol=1e-12)
    np.testing.assert_allclose(top, 1.1e4, rtol=1e-12)


def test_the_normal_field_is_one_value_per_node(mesh):
    psi = np.zeros(mesh.n_nodes)
    assert normal_field(mesh, psi).shape == (mesh.n_nodes,)


def test_a_potential_of_the_wrong_length_is_refused(mesh):
    """One value per node, and the failure is otherwise a broadcast that
    silently produces the wrong shape rather than an error."""
    with pytest.raises(ValueError, match="node"):
        normal_field(mesh, np.zeros(mesh.n_nodes + 1))


def test_a_reversing_field_averages_to_near_zero_before_the_magnitude():
    """The magnitude is taken after the averaging, not before, and this is the
    test that tells the two apart.

    A potential with a minimum on the middle row has a field pointing one way
    below it and the other way above. The field at the node itself is zero,
    and that is what a centred difference says. Taking abs() of each edge
    first and then averaging would report the full size of both instead, and
    would put a large normal field at exactly the places a device has a
    potential well.
    """
    y_axis = uniform_mesh_1d(length=2e-5, n_nodes=3)
    mesh = tensor_mesh_2d(uniform_mesh_1d(length=1e-5, n_nodes=2), y_axis)

    # Symmetric well: 1, 0, 1 down the three rows.
    psi = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    E = normal_field(mesh, psi)

    np.testing.assert_allclose(E[2:4], 0.0, atol=1e-9)
    assert np.all(E[:2] > 0.0), "the boundary rows still see their one edge"
