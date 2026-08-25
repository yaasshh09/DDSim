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
from ddsim.mesh.mesh2d import tensor_mesh_2d, uniform_mesh_2d


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


def test_the_geometry_it_hands_the_assemblies_is_consistent(mesh):
    """edge_geometry packages the edge list and the dual faces together."""
    geometry = mesh.edge_geometry()

    left, right = geometry.ends(mesh.n_edges)
    np.testing.assert_array_equal(left, mesh.edge_nodes[:, 0])
    np.testing.assert_array_equal(right, mesh.edge_nodes[:, 1])
    np.testing.assert_array_equal(np.asarray(geometry.dual_face), mesh.dual_face)
