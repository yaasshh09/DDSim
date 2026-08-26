"""Tests for mesh/mesh1d.py.

All lengths are in cm, per the CGS-adjacent convention in CLAUDE.md. A 1 um
device is 1e-4 cm and 1 nm spacing is 1e-7 cm.

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
    stacked_mesh_1d,
    uniform_mesh_1d,
)
from tests.reference.grading import solve_ratio as reference_solve_ratio

MICRON = 1e-4
"""One micron [cm]."""

NANOMETRE = 1e-7
"""One nanometre [cm]."""


# --------------------------------------------------------------- uniform mesh


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


# ------------------------------------------------------- structural invariants


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


# ---------------------------------------------------------------- graded mesh


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


# --------------------------------------------------- the Phase 0 definition of done


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


# --------------------------------------------------------- validation and repr


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


# --------------------------------------------------------------- stacked mesh


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
