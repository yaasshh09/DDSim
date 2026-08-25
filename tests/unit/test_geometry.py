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
