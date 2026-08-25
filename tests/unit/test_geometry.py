"""Edge geometry: the topology and weights an assembly needs to be dimension free.

The 1D assemblies were written against contiguous slicing, which hardcodes "edge
e joins node e and node e+1". That is true on a 1D mesh and false on every 2D
one. EdgeGeometry is the object that carries the truth instead, and its default
reproduces the 1D case exactly so that no existing result moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.discretize.geometry import UNIFORM_1D, EdgeGeometry


def test_the_default_geometry_is_contiguous_1d():
    """No edge_nodes means edge e joins node e and node e+1."""
    left, right = UNIFORM_1D.ends(4)

    np.testing.assert_array_equal(left, [0, 1, 2, 3])
    np.testing.assert_array_equal(right, [1, 2, 3, 4])


def test_the_default_weights_are_exactly_one():
    """1.0 exactly, not approximately.

    Every assembly multiplies by eps_r and dual_face. Multiplication by exactly
    1.0 is exact in IEEE754, so the 1D path keeps its bit pattern and every
    number the test suite already pins stays put. A default of 1.0000000001
    would silently move all of them.
    """
    assert UNIFORM_1D.eps_r == 1.0
    assert UNIFORM_1D.dual_face == 1.0
    assert UNIFORM_1D.eps_r * UNIFORM_1D.dual_face == 1.0


def test_explicit_edge_nodes_are_used():
    """An arbitrary edge list is honoured, including a reversed one."""
    edge_nodes = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    left, right = EdgeGeometry(edge_nodes=edge_nodes).ends(3)

    np.testing.assert_array_equal(left, [2, 1, 0])
    np.testing.assert_array_equal(right, [0, 2, 1])


def test_spelling_out_the_1d_edge_list_matches_the_default():
    """The explicit form of the default has to agree with the default."""
    spelled_out = EdgeGeometry(
        edge_nodes=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    )

    for got, expected in zip(spelled_out.ends(3), UNIFORM_1D.ends(3), strict=True):
        np.testing.assert_array_equal(got, expected)


def test_an_edge_count_mismatch_is_rejected():
    """The edge list has to agree with the number of edges the mesh has."""
    geometry = EdgeGeometry(edge_nodes=np.array([[0, 1], [1, 2]], dtype=np.int64))

    with pytest.raises(ValueError, match="2 edges"):
        geometry.ends(5)


def test_an_edge_list_of_the_wrong_shape_is_rejected():
    """(n_edges, 2), not (2, n_edges). Transposing it is the obvious slip."""
    with pytest.raises(ValueError, match="shape"):
        EdgeGeometry(edge_nodes=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64))


def test_an_edge_joining_a_node_to_itself_is_rejected():
    """A self edge has zero length and would divide by zero downstream."""
    with pytest.raises(ValueError, match="itself"):
        EdgeGeometry(edge_nodes=np.array([[0, 1], [2, 2]], dtype=np.int64))


def test_the_default_edge_count_is_one_fewer_than_the_nodes():
    """A 1D chain of N nodes has N-1 edges. That is what the default means."""
    assert UNIFORM_1D.edge_count(5) == 4


def test_an_explicit_edge_list_reports_its_own_length():
    """In 2D the edge count has nothing to do with the node count."""
    edge_nodes = np.array([[0, 1], [1, 2], [0, 2], [0, 3]], dtype=np.int64)

    assert EdgeGeometry(edge_nodes=edge_nodes).edge_count(4) == 4


def test_ends_of_takes_a_node_count_instead_of_an_edge_count():
    """The convenience the flux kernels use, which only ever see psi."""
    for got, expected in zip(
        UNIFORM_1D.ends_of(5), UNIFORM_1D.ends(4), strict=True
    ):
        np.testing.assert_array_equal(got, expected)


def test_per_edge_weights_are_allowed():
    """dual_face and eps_r are per edge in 2D, scalar in 1D."""
    geometry = EdgeGeometry(
        edge_nodes=np.array([[0, 1], [1, 2]], dtype=np.int64),
        dual_face=np.array([2.0, 3.0]),
        eps_r=np.array([1.0, 0.333]),
    )

    np.testing.assert_allclose(
        np.asarray(geometry.eps_r) * np.asarray(geometry.dual_face), [2.0, 0.999]
    )


# --------------------------------------------- what a mesh hands an assembly


class TestScaledMesh:
    """Both meshes have to present the assemblies with the same three things.

    Every caller of an assembly currently writes `mesh.h / scale.x_0` and
    `mesh.volume / scale.x_0` by hand. That second one is wrong in 2D, where
    the volume is an area and needs x_0 squared, and it is wrong silently:
    the device just comes out the wrong size by a factor of the Debye length.

    So the mesh is asked instead. It knows its own dimension and nobody else
    has to.
    """

    def test_a_1d_mesh_reproduces_what_callers_compute_by_hand(self):
        """Bit for bit, or every existing 1D number moves."""
        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh1d import uniform_mesh_1d

        scale = ScaleFactors.for_silicon()
        mesh = uniform_mesh_1d(length=1e-4, n_nodes=11)
        scaled = mesh.scaled(scale)

        np.testing.assert_array_equal(scaled.h, mesh.h / scale.x_0)
        np.testing.assert_array_equal(scaled.volume, mesh.volume / scale.x_0)
        assert scaled.geometry is UNIFORM_1D

    def test_a_2d_mesh_scales_its_areas_by_x_0_squared(self):
        """volume / x_0^d and face / x_0^(d-1), evaluated at d = 2."""
        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh2d import uniform_mesh_2d

        scale = ScaleFactors.for_silicon()
        mesh = uniform_mesh_2d(width=2e-4, height=1e-4, nx=5, ny=4)
        scaled = mesh.scaled(scale)

        np.testing.assert_array_equal(scaled.h, mesh.h / scale.x_0)
        np.testing.assert_array_equal(
            scaled.volume, mesh.volume / scale.x_0**2
        )
        np.testing.assert_array_equal(
            np.asarray(scaled.geometry.dual_face), mesh.dual_face / scale.x_0
        )

    def test_the_2d_geometry_carries_the_edge_list(self):
        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh2d import uniform_mesh_2d

        mesh = uniform_mesh_2d(width=2e-4, height=1e-4, nx=5, ny=4)
        scaled = mesh.scaled(ScaleFactors.for_silicon())

        np.testing.assert_array_equal(
            scaled.geometry.edge_nodes, mesh.edge_nodes
        )

    def test_a_2d_mesh_can_be_given_permittivities(self):
        """A two material device hands its eps_r in here."""
        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh2d import uniform_mesh_2d

        mesh = uniform_mesh_2d(width=2e-4, height=1e-4, nx=5, ny=4)
        eps_r = np.full(mesh.n_edges, 0.3333)
        scaled = mesh.scaled(ScaleFactors.for_silicon(), eps_r=eps_r)

        np.testing.assert_array_equal(np.asarray(scaled.geometry.eps_r), eps_r)

    def test_the_bundle_reports_the_node_and_edge_counts(self):
        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh2d import uniform_mesh_2d

        mesh = uniform_mesh_2d(width=2e-4, height=1e-4, nx=5, ny=4)
        scaled = mesh.scaled(ScaleFactors.for_silicon())

        assert scaled.n_nodes == mesh.n_nodes
        assert scaled.n_edges == mesh.n_edges
