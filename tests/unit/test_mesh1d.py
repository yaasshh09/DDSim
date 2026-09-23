"""Tests for mesh/mesh1d.py.

All lengths are in cm, per the CGS-adjacent convention in
docs/06-constants.md. A 1 um device is 1e-4 cm and 1 nm spacing is 1e-7 cm.

The invariant that catches the most mistakes is that the dual cell widths sum
to the domain length. Get the boundary half cells wrong and every integrated
charge in the device is off by a sliver that looks like a physics effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.mesh.mesh1d import (
    Mesh1D,
    graded_mesh_1d,
    graded_mesh_1d_at,
    graded_mesh_1d_through,
    stacked_mesh_1d,
    uniform_mesh_1d,
)
from tests.reference.grading import solve_ratio as reference_solve_ratio

MICRON = 1e-4
"""One micron [cm]."""

NANOMETRE = 1e-7
"""One nanometre [cm]."""


def test_uniform_mesh_has_the_requested_node_count() -> None:
    mesh = uniform_mesh_1d(MICRON, 101)
    assert mesh.n_nodes == 101
    assert mesh.n_edges == 100


def test_uniform_mesh_spans_the_requested_length() -> None:
    mesh = uniform_mesh_1d(MICRON, 101)
    assert mesh.x[0] == 0.0
    assert mesh.x[-1] == pytest.approx(MICRON, rel=1e-15)
    assert mesh.length == pytest.approx(MICRON, rel=1e-15)


def test_uniform_mesh_edge_lengths_are_all_equal() -> None:
    """Equal to a few ulp, not bitwise.

    linspace guarantees both endpoints exactly, which matters more here than
    identical spacings, and the price is that consecutive differences wobble
    by about one ulp. Measured spread is 1.1e-14 relative over 100 cells.
    """
    mesh = uniform_mesh_1d(MICRON, 101)
    np.testing.assert_allclose(mesh.h, MICRON / 100.0, rtol=1e-13)


def test_uniform_mesh_rejects_fewer_than_two_nodes() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        uniform_mesh_1d(MICRON, 1)


def test_uniform_mesh_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        uniform_mesh_1d(0.0, 10)


@pytest.mark.parametrize(
    "mesh",
    [
        uniform_mesh_1d(MICRON, 101),
        uniform_mesh_1d(MICRON, 2),
        graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=NANOMETRE),
        graded_mesh_1d(MICRON, 51, refine_at=0.0, h_min=NANOMETRE),
    ],
    ids=["uniform-101", "uniform-2", "graded-centre", "graded-left"],
)
class TestMeshInvariants:
    """Properties every mesh must satisfy, whatever generated it."""

    def test_node_positions_are_strictly_increasing(self, mesh: Mesh1D) -> None:
        assert np.all(np.diff(mesh.x) > 0.0)

    def test_edge_lengths_are_the_node_differences(self, mesh: Mesh1D) -> None:
        np.testing.assert_allclose(mesh.h, np.diff(mesh.x), rtol=0.0, atol=0.0)

    def test_edge_lengths_are_positive(self, mesh: Mesh1D) -> None:
        assert np.all(mesh.h > 0.0)

    def test_cell_volumes_sum_to_the_domain_length(self, mesh: Mesh1D) -> None:
        """The single best structural check on a finite volume mesh."""
        assert mesh.volume.sum() == pytest.approx(mesh.length, rel=1e-14)

    def test_cell_volumes_are_positive(self, mesh: Mesh1D) -> None:
        assert np.all(mesh.volume > 0.0)

    def test_interior_cell_volume_is_the_half_sum_of_its_edges(
        self, mesh: Mesh1D
    ) -> None:
        for i in range(1, mesh.n_nodes - 1):
            expected = 0.5 * (mesh.h[i - 1] + mesh.h[i])
            assert mesh.volume[i] == pytest.approx(expected, rel=1e-15)

    def test_boundary_cell_volumes_are_half_edges(self, mesh: Mesh1D) -> None:
        assert mesh.volume[0] == pytest.approx(0.5 * mesh.h[0], rel=1e-15)
        assert mesh.volume[-1] == pytest.approx(0.5 * mesh.h[-1], rel=1e-15)

    def test_edge_nodes_map_each_edge_to_its_two_endpoints(
        self, mesh: Mesh1D
    ) -> None:
        assert mesh.edge_nodes.shape == (mesh.n_edges, 2)
        for edge in range(mesh.n_edges):
            left, right = mesh.edge_nodes[edge]
            assert (left, right) == (edge, edge + 1)

    def test_node_edges_is_the_inverse_of_edge_nodes(self, mesh: Mesh1D) -> None:
        for node in range(mesh.n_nodes):
            for edge in mesh.node_edges[node]:
                assert node in tuple(mesh.edge_nodes[edge])

    def test_interior_nodes_touch_two_edges_and_boundaries_touch_one(
        self, mesh: Mesh1D
    ) -> None:
        assert len(mesh.node_edges[0]) == 1
        assert len(mesh.node_edges[-1]) == 1
        for node in range(1, mesh.n_nodes - 1):
            assert len(mesh.node_edges[node]) == 2


def test_graded_mesh_has_the_requested_node_count() -> None:
    mesh = graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=NANOMETRE)
    assert mesh.n_nodes == 200


def test_graded_mesh_spans_the_requested_length_exactly() -> None:
    mesh = graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=NANOMETRE)
    assert mesh.x[0] == 0.0
    assert mesh.x[-1] == pytest.approx(MICRON, rel=1e-12)


def test_graded_mesh_achieves_the_requested_minimum_spacing() -> None:
    mesh = graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=NANOMETRE)
    assert mesh.h.min() == pytest.approx(NANOMETRE, rel=1e-9)


def test_graded_mesh_puts_the_finest_spacing_at_the_refinement_point() -> None:
    refine_at = 0.5 * MICRON
    mesh = graded_mesh_1d(MICRON, 200, refine_at=refine_at, h_min=NANOMETRE)
    finest = int(np.argmin(mesh.h))
    midpoint = 0.5 * (mesh.x[finest] + mesh.x[finest + 1])
    assert abs(midpoint - refine_at) < 2.0 * NANOMETRE


def test_graded_mesh_places_a_node_at_the_refinement_point() -> None:
    refine_at = 0.5 * MICRON
    mesh = graded_mesh_1d(MICRON, 200, refine_at=refine_at, h_min=NANOMETRE)
    assert np.min(np.abs(mesh.x - refine_at)) < 1e-16


def test_graded_mesh_spacing_is_monotonic_on_each_side() -> None:
    """Spacing falls to h_min at the refinement point and rises after it.

    Monotonic across the whole mesh is impossible for an interior refinement,
    so the requirement can only mean monotone on each side.
    """
    refine_at = 0.5 * MICRON
    mesh = graded_mesh_1d(MICRON, 200, refine_at=refine_at, h_min=NANOMETRE)
    pivot = int(np.argmin(np.abs(mesh.x - refine_at)))

    left = mesh.h[:pivot]
    right = mesh.h[pivot:]
    assert np.all(np.diff(left) < 0.0), "left spacing must shrink toward the junction"
    assert np.all(np.diff(right) > 0.0), "right spacing must grow away from it"


def test_graded_mesh_growth_ratio_is_gentle() -> None:
    """Neighbouring cells within 10 percent keeps the truncation error sane."""
    mesh = graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=NANOMETRE)
    ratios = mesh.h[1:] / mesh.h[:-1]
    assert np.all(ratios < 1.10)
    assert np.all(ratios > 1.0 / 1.10)


def test_graded_mesh_refined_at_the_left_boundary() -> None:
    mesh = graded_mesh_1d(MICRON, 51, refine_at=0.0, h_min=NANOMETRE)
    assert mesh.h[0] == pytest.approx(NANOMETRE, rel=1e-9)
    assert np.all(np.diff(mesh.h) > 0.0)


def test_graded_mesh_refined_at_the_right_boundary() -> None:
    mesh = graded_mesh_1d(MICRON, 51, refine_at=MICRON, h_min=NANOMETRE)
    assert mesh.h[-1] == pytest.approx(NANOMETRE, rel=1e-9)
    assert np.all(np.diff(mesh.h) < 0.0)


def test_graded_mesh_refined_off_centre() -> None:
    refine_at = 0.2 * MICRON
    mesh = graded_mesh_1d(MICRON, 200, refine_at=refine_at, h_min=NANOMETRE)
    assert mesh.h.min() == pytest.approx(NANOMETRE, rel=1e-9)
    assert mesh.volume.sum() == pytest.approx(MICRON, rel=1e-12)


def test_graded_mesh_supports_a_spacing_ratio_of_1000() -> None:
    """Phase 5 needs 1 nm at a junction inside a much larger device."""
    mesh = graded_mesh_1d(
        100.0 * MICRON, 400, refine_at=50.0 * MICRON, h_min=NANOMETRE
    )
    assert mesh.h.max() / mesh.h.min() > 1000.0
    assert mesh.volume.sum() == pytest.approx(100.0 * MICRON, rel=1e-12)


def test_graded_mesh_rejects_a_refinement_point_outside_the_domain() -> None:
    with pytest.raises(ValueError, match="refine_at"):
        graded_mesh_1d(MICRON, 100, refine_at=2.0 * MICRON, h_min=NANOMETRE)


def test_graded_mesh_rejects_an_infeasible_minimum_spacing() -> None:
    """200 nodes cannot cover 1 um if every cell must be at least 1 um."""
    with pytest.raises(ValueError, match="infeasible|h_min"):
        graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=MICRON)


def test_graded_mesh_reduces_to_uniform_when_h_min_is_the_uniform_spacing() -> None:
    """A consistency check on the ratio solve: r must come out as 1."""
    n_nodes = 101
    uniform_h = MICRON / (n_nodes - 1)
    mesh = graded_mesh_1d(MICRON, n_nodes, refine_at=0.5 * MICRON, h_min=uniform_h)
    np.testing.assert_allclose(mesh.h, uniform_h, rtol=1e-9)


def test_phase0_acceptance_200_nodes_1nm_at_half_a_micron() -> None:
    """The case named in phases/PHASE-0.md, asserted end to end."""
    mesh = graded_mesh_1d(MICRON, 200, refine_at=0.5 * MICRON, h_min=NANOMETRE)

    assert mesh.n_nodes == 200
    assert mesh.h.min() == pytest.approx(NANOMETRE, rel=1e-9)
    assert mesh.length == pytest.approx(MICRON, rel=1e-12)
    assert mesh.volume.sum() == pytest.approx(MICRON, rel=1e-12)

    pivot = int(np.argmin(np.abs(mesh.x - 0.5 * MICRON)))
    assert np.all(np.diff(mesh.h[:pivot]) < 0.0)
    assert np.all(np.diff(mesh.h[pivot:]) > 0.0)


def test_graded_mesh_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        graded_mesh_1d(0.0, 100, refine_at=0.0, h_min=NANOMETRE)


def test_graded_mesh_rejects_fewer_than_two_nodes() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        graded_mesh_1d(MICRON, 1, refine_at=0.0, h_min=NANOMETRE)


def test_graded_mesh_rejects_non_positive_h_min() -> None:
    with pytest.raises(ValueError, match="h_min"):
        graded_mesh_1d(MICRON, 100, refine_at=0.0, h_min=0.0)


def test_graded_mesh_rejects_a_refinement_point_that_starves_one_side() -> None:
    """Total budget fits, but no split can reach h_min on the short side."""
    with pytest.raises(ValueError, match="infeasible"):
        graded_mesh_1d(1.0, 3, refine_at=0.1, h_min=0.4)


def test_graded_mesh_rejects_a_mesh_harsher_than_max_ratio() -> None:
    """Three cells cannot go from 1e-3 to 0.5 gently, and it says so."""
    with pytest.raises(ValueError, match="max_ratio"):
        graded_mesh_1d(1.0, 4, refine_at=0.5, h_min=1e-3)


def test_max_ratio_can_be_raised_deliberately() -> None:
    mesh = graded_mesh_1d(1.0, 4, refine_at=0.5, h_min=1e-3, max_ratio=1e4)
    assert mesh.n_nodes == 4
    assert mesh.volume.sum() == pytest.approx(1.0, rel=1e-12)


def test_repr_reports_size_and_spacing_range() -> None:
    text = repr(uniform_mesh_1d(MICRON, 11))
    assert "n_nodes=11" in text
    assert "h_min" in text
    assert "h_max" in text


def test_geometric_sum_handles_a_ratio_of_exactly_one() -> None:
    """Guards the r = 1 singularity, where (r^m - 1)/(r - 1) is 0/0."""
    from ddsim.mesh.mesh1d import _geometric_sums

    total = _geometric_sums(
        2.0, np.array([1.0, 2.0]), np.array([5.0, 3.0])
    )
    assert total[0] == pytest.approx(10.0, rel=1e-15)
    assert total[1] == pytest.approx(14.0, rel=1e-15)


def test_graded_mesh_handles_many_cells_with_a_very_small_h_min() -> None:
    """The bracketing search must not overflow while looking for the ratio.

    The search starts at ratio 2 and doubles, and the geometric sum is
    h_min*(r^m - 1)/(r - 1). With 1200 cells, 2^1199 is far past the double
    range, so the probe overflows before the bisection ever starts. Phase 0
    never hit it because 200 cells and 1 nm spacing keep r^m small.
    """
    mesh = graded_mesh_1d(4.0 * MICRON, 1201, refine_at=2.0 * MICRON, h_min=2e-8)
    assert mesh.n_nodes == 1201
    assert mesh.h.min() == pytest.approx(2e-8, rel=1e-6)
    assert mesh.volume.sum() == pytest.approx(4.0 * MICRON, rel=1e-12)


def test_geometric_sum_saturates_instead_of_overflowing() -> None:
    """An enormous sum is still an answer the bisection can use."""
    from ddsim.mesh.mesh1d import _geometric_sums

    total = _geometric_sums(
        1e-8, np.array([2.0, 1.001]), np.array([1199.0, 100.0])
    )
    assert total[0] == float("inf")
    assert total[1] < 1e-5


class TestRatioSolveMatchesTheScalarReference:
    """The vectorised bisection has to be the scalar one, element for element.

    tests/reference/grading.py holds the obvious scalar version. A vectorised
    bisection goes wrong in ways that do not look wrong afterwards: an element
    that keeps iterating past its own stopping test, or a mask applied a step
    late, moves the growth ratio in the last few bits and produces a mesh that
    is still perfectly plausible. So these compare exactly rather than to a
    tolerance. The two are meant to be the same arithmetic in the same order.
    """

    @pytest.mark.parametrize(
        ("side_length", "h_min", "n_intervals"),
        [
            (0.5 * MICRON, NANOMETRE, 100),
            (0.5 * MICRON, NANOMETRE, 199),
            (MICRON, NANOMETRE, 50),
            (2.0 * MICRON, 2e-8, 600),
            (1.0, 1e-3, 4),
        ],
    )
    def test_every_interval_count_agrees_exactly(
        self, side_length: float, h_min: float, n_intervals: int
    ) -> None:
        from ddsim.mesh.mesh1d import _solve_ratios

        counts = np.arange(1, n_intervals + 1, dtype=np.int64)
        vectorised = _solve_ratios(side_length, h_min, counts)

        for index, count in enumerate(counts):
            expected = reference_solve_ratio(side_length, h_min, int(count))
            if expected is None:
                assert np.isnan(vectorised[index]), (
                    f"{count} intervals is infeasible for the reference but "
                    f"the vectorised solver returned {vectorised[index]}"
                )
            else:
                assert vectorised[index] == expected, (
                    f"{count} intervals: {vectorised[index]!r} != {expected!r}"
                )

    def test_an_infeasible_side_is_all_nan(self) -> None:
        """Even h_min repeated m times overshoots, so no ratio exists."""
        from ddsim.mesh.mesh1d import _solve_ratios

        counts = np.array([5, 10, 20], dtype=np.int64)
        assert np.all(np.isnan(_solve_ratios(NANOMETRE, MICRON, counts)))

    def test_a_zero_interval_count_is_infeasible(self) -> None:
        """A side with no cells has no ratio, which is what a boundary
        refinement point produces on the empty side."""
        from ddsim.mesh.mesh1d import _solve_ratios

        assert np.isnan(_solve_ratios(MICRON, NANOMETRE, np.array([0]))[0])


class TestStackedMesh:
    """Layers laid end to end, which is what a material stack is.

    A MOS capacitor is silicon with oxide on top of it, and the two want
    different meshes: the silicon is graded hard to the surface where the
    inversion layer sits, the oxide holds no charge at all and its potential is
    exactly linear, so a handful of uniform cells resolves it exactly. One
    graded axis over the whole height cannot express that, and it also has to
    be argued rather than guaranteed that a node lands on the interface.
    Stacking two axes guarantees it, because the join is a node by
    construction.
    """

    def test_the_layers_span_their_total_length(self) -> None:
        stack = stacked_mesh_1d(
            uniform_mesh_1d(2 * MICRON, 5), uniform_mesh_1d(MICRON, 3)
        )
        assert stack.x[0] == 0.0
        assert stack.length == pytest.approx(3 * MICRON, rel=1e-15)

    def test_the_join_is_a_node_and_is_not_duplicated(self) -> None:
        """The shared node belongs to both layers and is stored once.

        Storing it twice makes a zero width cell, whose 1/h is infinite.
        """
        stack = stacked_mesh_1d(
            uniform_mesh_1d(2 * MICRON, 5), uniform_mesh_1d(MICRON, 3)
        )
        assert stack.n_nodes == 5 + 3 - 1
        assert np.count_nonzero(stack.x == 2 * MICRON) == 1
        assert np.all(stack.h > 0.0)

    def test_the_join_lands_exactly_on_the_layer_boundary(self) -> None:
        """Exactly, not nearly. stacked_regions refuses an interface that
        misses a node line, and half a cell of oxide is a percent of t_ox."""
        stack = stacked_mesh_1d(
            graded_mesh_1d(
                length=5 * MICRON, n_nodes=41, refine_at=5 * MICRON,
                h_min=NANOMETRE,
            ),
            uniform_mesh_1d(10 * NANOMETRE, 5),
        )
        assert stack.x[40] == 5 * MICRON

    def test_each_layer_keeps_its_own_spacing(self) -> None:
        fine = uniform_mesh_1d(MICRON, 11)
        coarse = uniform_mesh_1d(MICRON, 3)
        stack = stacked_mesh_1d(fine, coarse)
        np.testing.assert_allclose(stack.h[:10], fine.h, rtol=1e-15)
        np.testing.assert_allclose(stack.h[10:], coarse.h, rtol=1e-15)

    def test_a_single_layer_is_returned_unchanged(self) -> None:
        one = graded_mesh_1d(
            length=MICRON, n_nodes=81, refine_at=0.5 * MICRON, h_min=NANOMETRE
        )
        np.testing.assert_array_equal(stacked_mesh_1d(one).x, one.x)

    def test_the_dual_cells_still_sum_to_the_total_length(self) -> None:
        """The invariant that catches boundary half cell mistakes, now across
        a join where two half cells of different sizes meet."""
        stack = stacked_mesh_1d(
            uniform_mesh_1d(2 * MICRON, 5), uniform_mesh_1d(MICRON, 9)
        )
        assert stack.volume.sum() == pytest.approx(3 * MICRON, rel=1e-14)

    def test_the_cell_across_the_join_is_not_averaged(self) -> None:
        """The two layers meet at a node, so the last cell of one and the first
        cell of the next stay their own sizes. The node between them gets a
        dual cell that is half of each, which is what the join means."""
        stack = stacked_mesh_1d(
            uniform_mesh_1d(2 * MICRON, 3), uniform_mesh_1d(MICRON, 3)
        )
        assert stack.h[1] == pytest.approx(MICRON, rel=1e-15)
        assert stack.h[2] == pytest.approx(0.5 * MICRON, rel=1e-15)
        assert stack.volume[2] == pytest.approx(0.75 * MICRON, rel=1e-14)

    def test_no_layers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            stacked_mesh_1d()

    def test_a_layer_that_does_not_start_at_zero_is_refused(self) -> None:
        """Every constructor here returns a mesh on [0, length]. A layer that
        does not is one somebody has already translated, and stacking it would
        translate it twice."""
        shifted = Mesh1D(
            x=np.array([1.0, 2.0]),
            h=np.array([1.0]),
            volume=np.array([0.5, 0.5]),
            edge_nodes=np.array([[0, 1]], dtype=np.int64),
            node_edges=((0,), (0,)),
        )
        with pytest.raises(ValueError, match="starts at"):
            stacked_mesh_1d(uniform_mesh_1d(MICRON, 3), shifted)


THIN_BASE = (10 * MICRON, 10.05 * MICRON, 10.1 * MICRON)
"""Three junctions, two of them 50 nm apart either side of a thin base, in a
20.1 um device. Grading each junction on its own and joining halfway broke
here, with neighbouring cells jumping by up to 2.7."""


def worst_ratio(mesh: Mesh1D) -> float:
    ratios = mesh.h[1:] / mesh.h[:-1]
    return float(max(ratios.max(), (1.0 / ratios).max()))


def test_one_point_is_graded_mesh_1d_bit_for_bit() -> None:
    np.testing.assert_array_equal(
        graded_mesh_1d_at(MICRON, 201, (0.5 * MICRON,), NANOMETRE).x,
        graded_mesh_1d(MICRON, 201, 0.5 * MICRON, NANOMETRE).x,
    )


@pytest.mark.parametrize("n_nodes", [201, 301, 401, 801])
def test_every_point_is_a_node_with_h_min_either_side(n_nodes) -> None:
    mesh = graded_mesh_1d_at(20.1 * MICRON, n_nodes, THIN_BASE, NANOMETRE)
    assert mesh.n_nodes == n_nodes
    assert mesh.x[-1] == 20.1 * MICRON
    for point in THIN_BASE:
        node = int(np.flatnonzero(mesh.x == point)[0])
        np.testing.assert_allclose(mesh.h[node - 1 : node + 1], NANOMETRE, rtol=1e-6)


@pytest.mark.parametrize("n_nodes", [201, 301, 401, 801])
def test_several_points_grade_as_gently_as_one(n_nodes) -> None:
    """One growth rate over the whole mesh, so nowhere is harsher than the
    limit graded_mesh_1d holds a single junction to."""
    mesh = graded_mesh_1d_at(20.1 * MICRON, n_nodes, THIN_BASE, NANOMETRE)
    assert worst_ratio(mesh) <= 1.5


def test_more_nodes_grade_more_gently() -> None:
    ratios = [
        worst_ratio(graded_mesh_1d_at(20.1 * MICRON, n, THIN_BASE, NANOMETRE))
        for n in (201, 401, 801)
    ]
    assert ratios[0] > ratios[1] > ratios[2]


def test_the_spacing_grows_away_from_every_point() -> None:
    """Coarsest halfway between two points and at the far ends, finest at
    the points, and monotone in between."""
    mesh = graded_mesh_1d_at(20.1 * MICRON, 301, THIN_BASE, NANOMETRE)
    marks = [0] + [int(np.flatnonzero(mesh.x == p)[0]) for p in THIN_BASE]
    marks.append(mesh.n_nodes - 1)
    first = mesh.h[: marks[1]]
    assert np.all(np.diff(first) <= 0.0)
    last = mesh.h[marks[-2] :]
    assert np.all(np.diff(last) >= 0.0)
    for left, right in zip(marks[1:-2], marks[2:-1], strict=True):
        between = mesh.h[left:right]
        peak = int(np.argmax(between))
        assert np.all(np.diff(between[: peak + 1]) >= 0.0)
        assert np.all(np.diff(between[peak:]) <= 0.0)


def test_the_dual_cells_still_sum_to_the_length() -> None:
    mesh = graded_mesh_1d_at(20.1 * MICRON, 301, THIN_BASE, NANOMETRE)
    assert mesh.volume.sum() == pytest.approx(20.1 * MICRON, rel=1e-14)


def test_more_nodes_than_fit_at_h_min_are_refused() -> None:
    with pytest.raises(ValueError, match="room for 101 nodes"):
        graded_mesh_1d_at(10 * NANOMETRE * 10, 102, (5e-6, 6e-6), NANOMETRE)


def test_too_few_nodes_to_grade_gently_are_refused() -> None:
    with pytest.raises(ValueError, match="max_ratio"):
        graded_mesh_1d_at(20.1 * MICRON, 31, THIN_BASE, NANOMETRE)


def test_a_single_point_must_be_inside_too() -> None:
    with pytest.raises(ValueError, match="inside"):
        graded_mesh_1d_at(MICRON, 201, (MICRON,), NANOMETRE)


def test_no_points_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        graded_mesh_1d_at(MICRON, 201, (), NANOMETRE)


def test_points_must_be_inside_and_increasing() -> None:
    with pytest.raises(ValueError, match="increasing"):
        graded_mesh_1d_at(MICRON, 201, (0.6 * MICRON, 0.4 * MICRON), NANOMETRE)
    with pytest.raises(ValueError, match="inside"):
        graded_mesh_1d_at(MICRON, 201, (0.5 * MICRON, MICRON), NANOMETRE)


DRAWN_LINES = (0.2 * MICRON, 0.4 * MICRON, 1.4 * MICRON, 1.6 * MICRON)
DRAWN_POINTS = (0.4 * MICRON, 1.4 * MICRON)


@pytest.mark.parametrize("n_nodes", [81, 121, 161])
def test_every_line_and_point_is_a_node(n_nodes) -> None:
    mesh = graded_mesh_1d_through(
        1.8 * MICRON, n_nodes, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE
    )
    assert mesh.n_nodes == n_nodes
    assert mesh.x[0] == 0.0
    assert mesh.x[-1] == 1.8 * MICRON
    for position in DRAWN_LINES + DRAWN_POINTS:
        assert position in mesh.x


def test_the_spacing_at_every_point_is_near_h_min() -> None:
    """Near, not exact: each span between lines takes a whole number of
    cells, which stretches its spacing by the rounding. Measured the other
    way, a mesh that ignored the points would sit at 1.8 um / 120, 75 times
    h_min, so ten percent is a real test."""
    mesh = graded_mesh_1d_through(
        1.8 * MICRON, 121, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE
    )
    for point in DRAWN_POINTS:
        node = int(np.flatnonzero(mesh.x == point)[0])
        np.testing.assert_allclose(mesh.h[node - 1 : node + 1], 2 * NANOMETRE, rtol=0.1)


@pytest.mark.parametrize("n_nodes", [81, 121, 161])
def test_lines_do_not_break_the_grading(n_nodes) -> None:
    """A line pinned where the grading did not want a node is where a jump
    would come from, so the whole mesh is held to the same limit as one
    junction."""
    mesh = graded_mesh_1d_through(
        1.8 * MICRON, n_nodes, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE
    )
    assert worst_ratio(mesh) <= 1.5


def test_the_spacing_grows_away_from_a_point() -> None:
    mesh = graded_mesh_1d_through(
        MICRON, 101, (0.3 * MICRON,), (0.5 * MICRON,), NANOMETRE
    )
    centre = int(np.flatnonzero(mesh.x == 0.5 * MICRON)[0])
    assert mesh.h[centre] < mesh.h[centre + 10] < mesh.h[-1]
    assert mesh.h[centre - 1] < mesh.h[centre - 10] < mesh.h[0]


def test_with_no_points_the_lines_share_the_nodes_evenly() -> None:
    """Nothing to grade towards, so the spacing is as even as whole cells
    between the lines allow."""
    mesh = graded_mesh_1d_through(MICRON, 11, (0.35 * MICRON,), (), NANOMETRE)
    assert 0.35 * MICRON in mesh.x
    np.testing.assert_allclose(mesh.h, 0.1 * MICRON, rtol=0.2)


def test_the_dual_cells_sum_to_the_length_through_lines() -> None:
    mesh = graded_mesh_1d_through(
        1.8 * MICRON, 121, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE
    )
    assert mesh.volume.sum() == pytest.approx(1.8 * MICRON, rel=1e-14)


def test_more_nodes_than_h_min_holds_are_refused_through_lines() -> None:
    with pytest.raises(ValueError, match="room for 101 nodes"):
        graded_mesh_1d_through(100 * NANOMETRE, 102, (), (50 * NANOMETRE,), NANOMETRE)


def test_fewer_nodes_than_lines_need_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 6 nodes"):
        graded_mesh_1d_through(
            1.8 * MICRON, 5, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE
        )


def test_too_few_nodes_to_grade_through_lines_are_refused() -> None:
    with pytest.raises(ValueError, match="max_ratio"):
        graded_mesh_1d_through(
            1.8 * MICRON, 13, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE
        )


def test_lines_and_points_must_lie_on_the_axis() -> None:
    with pytest.raises(ValueError, match="inside"):
        graded_mesh_1d_through(MICRON, 101, (2 * MICRON,), (), NANOMETRE)
    with pytest.raises(ValueError, match="inside"):
        graded_mesh_1d_through(MICRON, 101, (), (-MICRON,), NANOMETRE)


def test_with_no_points_h_min_limits_nothing() -> None:
    """The MOS capacitor's x axis: 0.1 um across with nothing to grade
    towards, which takes any number of columns whatever h_min says."""
    mesh = graded_mesh_1d_through(0.1 * MICRON, 63, (), (), 2 * NANOMETRE)
    np.testing.assert_allclose(mesh.h, 0.1 * MICRON / 62, rtol=1e-12)


@pytest.mark.parametrize("h_min", [0.0, -2 * NANOMETRE])
def test_a_spacing_that_is_not_positive_is_refused_by_name(h_min) -> None:
    """graded_mesh_1d refuses it by name. This one divided by zero, which
    reached the page as a bare 500 when a drawing's h_min was typed as 0, and
    a negative one reported room for a negative number of nodes."""
    with pytest.raises(ValueError, match="h_min must be positive"):
        graded_mesh_1d_through(1.8 * MICRON, 81, DRAWN_LINES, DRAWN_POINTS, h_min)
