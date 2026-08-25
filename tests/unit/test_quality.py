"""Mesh quality: the gate that has to exist before unstructured meshing does.

phases/PHASE-4.md requires a checker that rejects a deliberately obtuse mesh.
Nothing in Phase 4 builds one, because the structured tensor product mesh
cannot produce an obtuse triangle. That is exactly why this is written now and
tested directly: it is the guard that must already be in place the first time a
Delaunay mesh arrives, not something to write after the symptom shows up.

The symptom, from docs/05-pitfalls.md, is localised negative carrier density in
a region with no physical reason, persisting under refinement. The mechanism is
that the finite volume conductance of an edge is proportional to the sum of the
cotangents of the two angles facing it, so an angle past 90 degrees makes a
cotangent negative, a large enough one makes the conductance negative, and a
negative conductance destroys the M-matrix property that keeps densities
positive.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.mesh.quality import (
    MeshQualityError,
    check_triangulation,
    cotangent_edge_weights,
    obtuse_triangles,
    triangle_angles,
)

EQUILATERAL = (
    np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]),
    np.array([[0, 1, 2]]),
)

RIGHT = (
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    np.array([[0, 1, 2]]),
)

OBTUSE = (
    # Apex pushed far to the right, so the angle at node 0 opens past 90.
    np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.5]]),
    np.array([[0, 1, 2]]),
)


def test_an_equilateral_triangle_has_three_sixty_degree_angles():
    angles = triangle_angles(*EQUILATERAL)

    np.testing.assert_allclose(np.degrees(angles), 60.0, rtol=1e-12)
    assert angles.shape == (1, 3)


def test_the_angles_of_any_triangle_sum_to_pi():
    """Cheap invariant, and it catches an arccos fed the wrong argument."""
    for points, triangles in (EQUILATERAL, RIGHT, OBTUSE):
        angles = triangle_angles(points, triangles)
        np.testing.assert_allclose(angles.sum(axis=1), np.pi, rtol=1e-12)


def test_a_right_triangle_is_not_obtuse():
    """Exactly 90 degrees is the boundary and is allowed.

    The cotangent of a right angle is zero, so the edge facing it gets zero
    conductance, not negative. Degenerate, not broken. Refusing it would
    reject the standard split of a rectangle into two triangles, which is the
    most common legitimate mesh there is.
    """
    angles = triangle_angles(*RIGHT)

    assert np.isclose(np.degrees(angles).max(), 90.0)
    assert obtuse_triangles(*RIGHT).size == 0
    check_triangulation(*RIGHT)


def test_an_obtuse_triangle_is_detected():
    angles = triangle_angles(*OBTUSE)

    assert np.degrees(angles).max() > 90.0
    np.testing.assert_array_equal(obtuse_triangles(*OBTUSE), [0])


def test_check_triangulation_refuses_a_deliberately_obtuse_mesh():
    """The acceptance criterion, stated in phases/PHASE-4.md."""
    with pytest.raises(MeshQualityError, match="obtuse"):
        check_triangulation(*OBTUSE)


def test_the_refusal_names_the_offending_triangle_and_its_angle():
    """A quality failure has to be actionable, not just a no."""
    with pytest.raises(MeshQualityError) as failure:
        check_triangulation(*OBTUSE)

    message = str(failure.value)
    assert "triangle 0" in message
    assert "deg" in message


def test_cotangent_weights_are_positive_on_an_acute_mesh():
    edges, weights = cotangent_edge_weights(*EQUILATERAL)

    assert edges.shape == (3, 2)
    assert np.all(weights > 0.0)


def test_the_hypotenuse_of_a_right_triangle_gets_zero_weight():
    """cot(90) = 0. This is the boundary the M-matrix property sits on."""
    edges, weights = cotangent_edge_weights(*RIGHT)

    # The hypotenuse faces the right angle at node 0, so it is edge (1, 2).
    hypotenuse = np.flatnonzero(
        (edges[:, 0] == 1) & (edges[:, 1] == 2)
    )
    assert hypotenuse.size == 1
    assert weights[hypotenuse[0]] == pytest.approx(0.0, abs=1e-15)


def test_an_obtuse_triangle_produces_a_negative_weight():
    """The whole reason the angle matters. Not a separate rule, the same one."""
    _, weights = cotangent_edge_weights(*OBTUSE)

    assert weights.min() < 0.0


def test_two_triangles_sharing_an_edge_add_their_cotangents():
    """A unit square split along its diagonal.

    The shared diagonal faces a right angle in each half, so its weight is
    cot(90) + cot(90) = 0. The four outer edges each face a 45 degree angle in
    one triangle only, so each gets cot(45) = 1, halved by the 1/2 in the
    formula.
    """
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])

    edges, weights = cotangent_edge_weights(points, triangles)

    assert edges.shape == (5, 2)
    diagonal = np.flatnonzero((edges[:, 0] == 0) & (edges[:, 1] == 2))
    assert weights[diagonal[0]] == pytest.approx(0.0, abs=1e-15)
    check_triangulation(points, triangles)


def test_a_degenerate_triangle_is_refused():
    """Three collinear points have zero area and no usable dual cell."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    triangles = np.array([[0, 1, 2]])

    with pytest.raises(MeshQualityError, match="degenerate"):
        check_triangulation(points, triangles)


def test_a_triangle_list_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        triangle_angles(EQUILATERAL[0], np.array([[0, 1], [1, 2]]))


def test_the_tolerance_is_honoured():
    """A mesh barely over 90 degrees can be waved through deliberately.

    Not a way to silence the check. It exists because a right angle produced
    by floating point arithmetic can land a few ulps past 90, and refusing
    that would reject a legitimate rectangular split.
    """
    barely = (
        np.array([[0.0, 0.0], [1.0, 0.0], [-1e-9, 1.0]]),
        np.array([[0, 1, 2]]),
    )
    assert obtuse_triangles(*barely).size == 1
    assert obtuse_triangles(*barely, tolerance_deg=1e-3).size == 0
    check_triangulation(*barely, tolerance_deg=1e-3)
