"""Triangulation quality: obtuse angles and negative dual areas.

Nothing in Phase 4 builds a triangulation. The structured tensor product mesh
in mesh2d.py is made of rectangles and cannot produce an obtuse triangle, which
is exactly why docs/02-numerics.md recommends taking that path first. This
module exists anyway, and is tested directly, because it is the gate that has
to already be in place the first time an unstructured mesh arrives. Writing it
after the symptom appears means debugging the symptom instead.

Why an angle is a numerical property and not an aesthetic one
-------------------------------------------------------------
Box integration puts the finite volume conductance of an edge at

    w_ij = (1/2) * sum over the triangles sharing the edge of cot(theta)

where theta is the angle facing the edge in that triangle. That factor is the
length of the Voronoi dual face divided by the length of the edge, so it is a
geometric quantity, not a modelling choice.

    theta <  90 deg     cot > 0     positive conductance
    theta == 90 deg     cot = 0     zero conductance, degenerate but safe
    theta >  90 deg     cot < 0     negative conductance

A negative conductance breaks the M-matrix property of the assembled system,
and with it the discrete maximum principle that keeps carrier densities
positive. docs/05-pitfalls.md gives the symptom: localised negative n or p in a
region with no physical reason, which persists under refinement. That is not a
solver problem and no amount of damping fixes it.

A right angle is allowed. The standard split of a rectangle into two triangles
puts a right angle in each half and gives the shared diagonal exactly zero
conductance, which is degenerate rather than wrong, and refusing it would
reject the most ordinary legitimate mesh there is.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MeshQualityError(ValueError):
    """A mesh that box integration cannot be trusted on.

    Deliberately fatal. docs/05-pitfalls.md is explicit that the right response
    to a bad mesh is to refuse to proceed, because every downstream symptom
    looks like a solver bug instead.
    """


def _check_shape(triangles: npt.NDArray[np.int64]) -> None:
    """Triangles are (n_triangles, 3)."""
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            f"triangles must have shape (n_triangles, 3), got {triangles.shape}"
        )


def triangle_angles(
    points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """The three interior angles of every triangle [rad].

    Args:
        points: node positions, shape (n_nodes, 2) [cm].
        triangles: node indices, shape (n_triangles, 3).

    Returned in the same order as the vertices, so `angles[t, 0]` is the angle
    at `triangles[t, 0]`, which is the angle facing the edge between the other
    two vertices.

    Computed from atan2 of the cross and dot products rather than from
    arccos of a normalised dot product. The arccos form loses precision
    exactly where it matters most, near 0 and 180 degrees, because the
    derivative of arccos is unbounded there and the argument can also drift a
    few ulps outside [-1, 1] and produce nan.
    """
    _check_shape(triangles)

    corners = points[triangles]
    angles = np.empty(triangles.shape, dtype=np.float64)

    for vertex in range(3):
        here = corners[:, vertex]
        first = corners[:, (vertex + 1) % 3] - here
        second = corners[:, (vertex + 2) % 3] - here

        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        dot = first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1]
        angles[:, vertex] = np.arctan2(np.abs(cross), dot)

    return angles


def triangle_areas(
    points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Signed area of every triangle [cm^2], positive for counterclockwise."""
    _check_shape(triangles)

    corners = points[triangles]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return np.asarray(0.5 * cross)


def obtuse_triangles(
    points: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int64],
    tolerance_deg: float = 0.0,
) -> npt.NDArray[np.int64]:
    """Indices of the triangles with an angle past 90 degrees.

    Args:
        points: node positions, shape (n_nodes, 2) [cm].
        triangles: node indices, shape (n_triangles, 3).
        tolerance_deg: how far past 90 degrees is still accepted [deg].
            Exists because an angle that is a right angle in exact arithmetic
            can land a few ulps past it once the coordinates have been through
            a mesh generator, and refusing that would reject a legitimate
            rectangular split. It is not a way to wave a bad mesh through.
    """
    limit = np.pi / 2 + np.radians(tolerance_deg)
    return np.flatnonzero(triangle_angles(points, triangles).max(axis=1) > limit)


def cotangent_edge_weights(
    points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """The finite volume weight of every edge, and the edges themselves.

    Returns (edges, weights). `edges` has shape (n_edges, 2) with the smaller
    node index first, so an edge shared by two triangles appears once. The
    weight is dual face length over edge length [1], which is the `dual_face/h`
    that discretize/geometry.py wants, gathered per edge:

        w_ij = (1/2) * sum over adjacent triangles of cot(angle facing ij)

    A negative entry here is the thing that actually breaks the solve, and it
    is the same statement as an obtuse angle rather than a second rule.
    """
    angles = triangle_angles(points, triangles)

    # The angle at vertex v faces the edge joining the other two vertices.
    facing = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    pairs = []
    contributions = []
    for vertex, first, second in facing:
        pairs.append(np.sort(triangles[:, [first, second]], axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            contributions.append(0.5 / np.tan(angles[:, vertex]))

    all_pairs = np.concatenate(pairs)
    all_weights = np.concatenate(contributions)

    edges, inverse = np.unique(all_pairs, axis=0, return_inverse=True)
    weights = np.zeros(edges.shape[0], dtype=np.float64)
    np.add.at(weights, inverse.ravel(), all_weights)

    return edges.astype(np.int64), weights


def check_triangulation(
    points: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int64],
    tolerance_deg: float = 0.0,
    min_area: float = 0.0,
) -> None:
    """Refuse a triangulation box integration cannot be trusted on.

    Args:
        points: node positions, shape (n_nodes, 2) [cm].
        triangles: node indices, shape (n_triangles, 3).
        tolerance_deg: slack past 90 degrees, as in obtuse_triangles.
        min_area: triangles at or below this area are degenerate [cm^2].

    Raises MeshQualityError, naming the worst offender and its angle, because
    a quality failure has to be actionable. Called at mesh construction time,
    never during a solve: the answer to a bad mesh is a better mesh.
    """
    _check_shape(triangles)

    areas = np.abs(triangle_areas(points, triangles))
    degenerate = np.flatnonzero(areas <= min_area)
    if degenerate.size:
        raise MeshQualityError(
            f"{degenerate.size} degenerate triangle(s), the first being "
            f"triangle {degenerate[0]} with area {areas[degenerate[0]]:.3e} "
            "cm^2. Three collinear points have no dual cell to integrate over."
        )

    angles = np.degrees(triangle_angles(points, triangles))
    worst_per_triangle = angles.max(axis=1)
    offenders = obtuse_triangles(points, triangles, tolerance_deg)

    if offenders.size:
        worst = offenders[np.argmax(worst_per_triangle[offenders])]
        raise MeshQualityError(
            f"{offenders.size} obtuse triangle(s), the worst being triangle "
            f"{worst} at {worst_per_triangle[worst]:.3f} deg. An angle past 90 "
            "gives the edge facing it a negative finite volume conductance, "
            "which breaks the M-matrix property and lets carrier densities go "
            "negative for no physical reason. Refine or re-triangulate; this "
            "is not something the solver can be tuned around."
        )
