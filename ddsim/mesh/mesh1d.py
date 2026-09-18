"""One dimensional meshes, uniform and graded.

All lengths are in cm, matching the CGS-adjacent convention used throughout.
A 1 um device is 1e-4 cm and 1 nm spacing is 1e-7 cm.

The mesh carries both the primal grid (nodes and the edges between them) and
the dual grid (the cell around each node). Scharfetter-Gummel fluxes live on
edges and densities live on nodes, so both are needed everywhere.

Dual cells in 1D are the intervals between edge midpoints:

    volume[i] = (x[i+1] - x[i-1]) / 2      interior
    volume[0] = (x[1] - x[0]) / 2          left boundary, a half cell
    volume[-1] = (x[-1] - x[-2]) / 2       right boundary, a half cell

Those two half cells at the ends are easy to get wrong, and getting them wrong
makes every integrated charge in the device slightly off in a way that looks
like a physical effect. The invariant to check is that the volumes sum to the
domain length exactly.

Grading requirement, from docs/02-numerics.md: the spacing at a junction must
resolve the local Debye length, h < L_D / 2. At 1e18 cm^-3 that is about 2 nm,
so a 1 um device with a 1e18 junction needs roughly a 1000 to 1 spacing range.
The generator here handles that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import UNIFORM_1D, ScaledMesh

_RATIO_TOLERANCE = 1e-14
"""Relative tolerance for the geometric ratio solve [1]."""

_DEGENERATE_TOLERANCE = 1e-12
"""Below this relative difference, a side is treated as exactly uniform [1]."""

_SCORE_SLACK = 1e-9
"""How far above the best lower bound a split may still be worth scoring [1].

The worst neighbouring cell ratio of a split is its growth ratio, up to the
rounding in building the spacing array, which is at the last bit. This is many
orders of magnitude above that rounding and many orders below the spacing
between the growth ratios of two different splits.
"""


@dataclass(frozen=True)
class Mesh1D:
    """A 1D mesh with its primal and dual grids and the index maps between them."""

    x: npt.NDArray[np.float64]
    """Node positions [cm], strictly increasing, x[0] = 0."""

    h: npt.NDArray[np.float64]
    """Edge lengths [cm], length n_nodes - 1. Always equal to diff(x)."""

    volume: npt.NDArray[np.float64]
    """Dual cell widths [cm], length n_nodes. Sums to the domain length."""

    edge_nodes: npt.NDArray[np.int64]
    """Shape (n_edges, 2). edge_nodes[e] is the (left, right) node of edge e."""

    node_edges: tuple[tuple[int, ...], ...]
    """node_edges[i] lists the edges touching node i. One entry at each
    boundary, two in the interior."""

    def scaled(self, scale: ScaleFactors) -> ScaledMesh:
        """This mesh in the units the assemblies work in.

        In 1D the dual face is the unit cross section, so it stays exactly
        1.0 and the geometry is the shared UNIFORM_1D default. That is what
        keeps every number this project has ever produced where it is.
        """
        return ScaledMesh(
            h=self.h / scale.x_0,
            volume=self.volume / scale.x_0,
            geometry=UNIFORM_1D,
        )

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.x.size)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return int(self.h.size)

    @property
    def length(self) -> float:
        """Total domain length [cm]."""
        return float(self.x[-1] - self.x[0])

    def __repr__(self) -> str:
        return (
            f"Mesh1D n_nodes={self.n_nodes} length={self.length:.4e} cm "
            f"h_min={self.h.min():.4e} h_max={self.h.max():.4e}"
        )


def _assemble(x: npt.NDArray[np.float64]) -> Mesh1D:
    """Build the dual grid and index maps from node positions [cm].

    h is recomputed from x rather than carried through, so that h == diff(x)
    holds exactly no matter how x was constructed.
    """
    h = np.diff(x)

    volume = np.empty_like(x)
    volume[1:-1] = 0.5 * (h[:-1] + h[1:])
    volume[0] = 0.5 * h[0]
    volume[-1] = 0.5 * h[-1]

    n_edges = h.size
    edge_nodes = np.empty((n_edges, 2), dtype=np.int64)
    edge_nodes[:, 0] = np.arange(n_edges)
    edge_nodes[:, 1] = np.arange(1, n_edges + 1)

    node_edges: list[tuple[int, ...]] = []
    for node in range(x.size):
        touching = []
        if node > 0:
            touching.append(node - 1)
        if node < n_edges:
            touching.append(node)
        node_edges.append(tuple(touching))

    return Mesh1D(x=x, h=h, volume=volume, edge_nodes=edge_nodes,
                  node_edges=tuple(node_edges))


def uniform_mesh_1d(length: float, n_nodes: int) -> Mesh1D:
    """A uniformly spaced mesh on [0, length] [cm]."""
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if n_nodes < 2:
        raise ValueError(f"a mesh needs at least 2 nodes, got {n_nodes}")

    return _assemble(np.linspace(0.0, length, n_nodes))


def _geometric_sums(
    h_min: float,
    ratio: npt.NDArray[np.float64],
    n_intervals: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """_geometric_sum evaluated on a whole array of (ratio, count) pairs [cm].

    The same expression, evaluated everywhere and repaired afterwards rather
    than branched around. A ratio of exactly 1 gives 0/0, and a ratio that
    overshoots the double range gives infinity on its own, which is the answer
    the scalar version reaches by its explicit log test. Both are what the
    bisection below wants, since it only ever asks whether the sum has passed
    the side length.

    Every split of the mesh needs its own ratio solved, and solving them one
    at a time spends more time in the interpreter than in the arithmetic.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        total = h_min * (ratio**n_intervals - 1.0) / (ratio - 1.0)
    return np.where(ratio == 1.0, h_min * n_intervals, total)


def _solve_ratios(
    side_length: float, h_min: float, n_intervals: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """_solve_ratio for many interval counts at once [1].

    NaN marks a count the side cannot be covered with, which is what None
    means in the scalar version. The bracketing and the bisection follow the
    same schedule as the scalar version, element by element, with each entry
    frozen as soon as it meets the same stopping test, so the answers agree
    to the last bit.
    """
    counts = np.asarray(n_intervals, dtype=np.float64)
    ratio = np.full(counts.shape, np.nan)

    uniform_total = h_min * counts
    feasible = (
        (counts > 0.0)
        & (h_min <= side_length * (1.0 + _DEGENERATE_TOLERANCE))
        & (uniform_total <= side_length * (1.0 + _DEGENERATE_TOLERANCE))
    )

    # A single cell spans the side on its own, and a side that the minimum
    # spacing already fills exactly has no room to grow.
    degenerate = feasible & (
        (counts == 1.0)
        | (np.abs(uniform_total - side_length) <= _DEGENERATE_TOLERANCE * side_length)
    )
    ratio[degenerate] = 1.0

    solving = feasible & ~degenerate
    if not solving.any():
        return ratio

    counts = counts[solving]
    low = np.ones(counts.shape)
    high = np.full(counts.shape, 2.0)

    # Bracket first. The sum grows monotonically with r, so doubling the upper
    # bound until it overshoots is enough.
    below = _geometric_sums(h_min, high, counts) < side_length
    while below.any():
        high[below] *= 2.0
        if np.any(high > 1e6):  # pragma: no cover
            # Unreachable while the feasibility check above holds, since
            # h_min * m <= side_length guarantees some r >= 1 exists. Kept so
            # a future caller that skips that check cannot spin forever.
            return ratio
        below = _geometric_sums(h_min, high, counts) < side_length

    active = np.ones(counts.shape, dtype=bool)
    for _ in range(200):
        middle = 0.5 * (low + high)
        below = _geometric_sums(h_min, middle, counts) < side_length
        low = np.where(active & below, middle, low)
        high = np.where(active & ~below, middle, high)
        active &= high - low > _RATIO_TOLERANCE * low
        if not active.any():
            break

    ratio[solving] = 0.5 * (low + high)
    return ratio


def _side_spacings(
    side_length: float, h_min: float, n_intervals: int, ratio: float
) -> npt.NDArray[np.float64]:
    """Spacings for one side, ordered outward from the refinement point [cm].

    Rescaled so the side sums to exactly side_length. The bisection above
    lands within 1e-14 relative, and rescaling by a single positive factor
    removes the remainder without disturbing the monotone ordering.
    """
    spacings = h_min * ratio ** np.arange(n_intervals, dtype=np.float64)
    return spacings * (side_length / spacings.sum())


def graded_mesh_1d(
    length: float,
    n_nodes: int,
    refine_at: float,
    h_min: float,
    max_ratio: float = 1.5,
) -> Mesh1D:
    """A mesh refined to h_min at refine_at, growing geometrically away from it.

    Args:
        length: domain length [cm].
        n_nodes: total node count, at least 2.
        refine_at: position of the finest spacing [cm], anywhere in [0, length].
        h_min: spacing at the refinement point [cm].
        max_ratio: largest allowed ratio between neighbouring cells [1].

    The node count, the total length and the minimum spacing are all honoured
    exactly. What gives is the growth ratio, which is solved for. If the
    resulting mesh would be harsher than max_ratio, that is reported rather
    than returned quietly, because a mesh with a 3x jump between neighbouring
    cells produces truncation error that looks like a physical effect.

    Spacing is monotone on each side of the refinement point. Monotone across
    the whole mesh is not possible for an interior refinement, since the
    spacing has to fall to h_min and rise again.

    The split of nodes between the two sides is chosen to minimise the worst
    neighbouring cell ratio. Every split's growth ratios are solved for, which
    is enough to know its score without building its spacing array, so only the
    splits that can actually win are built and measured. See the comment on the
    bound below for why that is the same answer as scoring all of them.
    """
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if n_nodes < 2:
        raise ValueError(f"a mesh needs at least 2 nodes, got {n_nodes}")
    if h_min <= 0.0:
        raise ValueError(f"h_min must be positive, got {h_min}")
    if not 0.0 <= refine_at <= length:
        raise ValueError(
            f"refine_at must lie in [0, {length}], got {refine_at}"
        )

    n_intervals = n_nodes - 1
    left_length = refine_at
    right_length = length - refine_at

    if h_min * n_intervals > length:
        raise ValueError(
            f"infeasible request: {n_intervals} cells of at least h_min={h_min:g} cm "
            f"need {h_min * n_intervals:g} cm but the domain is only {length:g} cm. "
            "Reduce h_min or reduce n_nodes."
        )

    # A refinement point on a boundary means one side only.
    if left_length == 0.0:
        splits = np.array([0], dtype=np.int64)
    elif right_length == 0.0:
        splits = np.array([n_intervals], dtype=np.int64)
    else:
        splits = np.arange(1, n_intervals, dtype=np.int64)

    on_left = splits
    on_right = n_intervals - splits
    ratio_left = _solve_ratios(left_length, h_min, on_left)
    ratio_right = _solve_ratios(right_length, h_min, on_right)

    feasible = ((on_left == 0) | np.isfinite(ratio_left)) & (
        (on_right == 0) | np.isfinite(ratio_right)
    )
    if not feasible.any():
        raise ValueError(
            f"infeasible request: no split of {n_intervals} cells reaches "
            f"h_min={h_min:g} cm at refine_at={refine_at:g} cm within a domain "
            f"of {length:g} cm."
        )

    # Every cell on one side grows by the same ratio, so every neighbouring
    # jump inside a side of two cells or more is exactly that ratio. The one
    # jump that is not is the pair straddling the pivot, and since each side
    # is rescaled to land on its own length, those two cells are h_min times
    # their side's rescale factor. So the whole score is known from the two
    # ratios and the two rescale factors, without building a single spacing
    # array. Rounding puts it a last bit or so away from what the arrays give,
    # which is what the slack below allows for.
    growth = np.ones(splits.size)
    inside_left = on_left >= 2
    growth[inside_left] = ratio_left[inside_left]
    inside_right = on_right >= 2
    growth[inside_right] = np.maximum(growth[inside_right], ratio_right[inside_right])

    junction = np.ones(splits.size)
    straddles = feasible & (on_left > 0) & (on_right > 0)
    across = (
        right_length
        / _geometric_sums(
            h_min,
            ratio_right[straddles],
            on_right[straddles].astype(np.float64),
        )
    ) / (
        left_length
        / _geometric_sums(
            h_min, ratio_left[straddles], on_left[straddles].astype(np.float64)
        )
    )
    junction[straddles] = np.maximum(across, 1.0 / across)

    bound = np.maximum(growth, junction)
    bound[~feasible] = np.inf

    def score_splits(
        indices: npt.NDArray[np.intp],
    ) -> tuple[npt.NDArray[np.float64] | None, float]:
        """Worst neighbouring cell ratio of each split, best one kept."""
        chosen: npt.NDArray[np.float64] | None = None
        best = np.inf
        for index in indices:
            pieces = []
            if on_left[index] > 0:
                # Ordered outward from the pivot, so reverse it to run left
                # to right.
                pieces.append(
                    _side_spacings(
                        left_length,
                        h_min,
                        int(on_left[index]),
                        float(ratio_left[index]),
                    )[::-1]
                )
            if on_right[index] > 0:
                pieces.append(
                    _side_spacings(
                        right_length,
                        h_min,
                        int(on_right[index]),
                        float(ratio_right[index]),
                    )
                )

            spacings = np.concatenate(pieces)
            neighbour_ratios = spacings[1:] / spacings[:-1]
            score = float(
                max(neighbour_ratios.max(), (1.0 / neighbour_ratios).max())
            )
            if score < best:
                best = score
                chosen = spacings
        return chosen, best

    best_bound = float(bound.min())
    best_spacings, best_score = score_splits(
        np.flatnonzero(bound <= best_bound * (1.0 + _SCORE_SLACK))
    )

    if best_score > best_bound * (1.0 + _SCORE_SLACK):  # pragma: no cover
        # The bound only holds up to the rounding in building the spacings,
        # which is at the last bit. If that ever grew past the slack the
        # pruning would no longer be justified, so score every split instead
        # of trusting it.
        best_spacings, best_score = score_splits(np.flatnonzero(feasible))

    assert best_spacings is not None

    if best_score > max_ratio:
        raise ValueError(
            f"the gentlest mesh meeting these constraints jumps by "
            f"{best_score:.3f} between neighbouring cells, above max_ratio="
            f"{max_ratio}. Add nodes, relax h_min, or raise max_ratio "
            "deliberately."
        )

    x = np.empty(n_nodes, dtype=np.float64)
    x[0] = 0.0
    x[1:] = np.cumsum(best_spacings)

    # Pin the two positions that callers depend on exactly, then let _assemble
    # recompute the spacings so that h == diff(x) holds to the last bit.
    n_left_final = int(np.argmin(np.abs(x - refine_at)))
    x[n_left_final] = refine_at
    x[-1] = length

    return _assemble(x)


def graded_mesh_1d_at(
    length: float,
    n_nodes: int,
    points: tuple[float, ...],
    h_min: float,
    max_ratio: float = 1.5,
) -> Mesh1D:
    """A mesh refined to h_min at every one of several points.

    Args:
        length: domain length [cm].
        n_nodes: total node count.
        points: where the finest spacing goes [cm], increasing, each strictly
            inside the domain. A stack of doped regions passes its junctions.
        h_min: spacing either side of every point [cm].
        max_ratio: largest allowed ratio between neighbouring cells [1].

    One point is graded_mesh_1d exactly, which is what keeps a two region
    stack the Phase 2 diode to the last bit.

    Several points are graded together, with one growth rate for the whole
    mesh. Grading each point on its own over its share of the domain and
    joining the shares halfway does not work: each share's growth is set by
    its own longest side, so two shares meet at different spacings, and a
    thin region between two junctions joined its neighbours with jumps of up
    to 2.7.

    The spacing the mesh aims for grows linearly with the distance d to the
    nearest point, h = h_min + g d, which is what geometric growth by 1 + g
    per cell looks like as a function of position. A side of length D then
    wants ln(1 + g D / h_min) / g cells, and g is solved so that the sides
    add up to the node count. Each side is then laid exactly as
    graded_mesh_1d lays one, geometric from h_min, so every point is a node
    with h_min either side of it. The two halves between neighbouring points
    are the same length and get the same count, so they are mirror images
    and meet at equal cells.
    """
    if len(points) == 1:
        if not 0.0 < points[0] < length:
            raise ValueError(
                f"the point must lie inside (0, {length:g}), got {points[0]:g}"
            )
        return graded_mesh_1d(length, n_nodes, points[0], h_min, max_ratio)
    if not points:
        raise ValueError("a graded mesh needs at least one point to grade towards")
    if any(np.diff(points) <= 0.0):
        raise ValueError(f"the points must be increasing, got {points}")
    if not (0.0 < points[0] and points[-1] < length):
        raise ValueError(
            f"every point must lie inside (0, {length:g}), got {points}"
        )

    # Each side runs outward from one point, towards a boundary or halfway to
    # the next point. Reversed means it runs right to left in the mesh.
    sides: list[tuple[float, bool]] = [(points[0], True)]
    for left, right in zip(points[:-1], points[1:], strict=True):
        half = 0.5 * (right - left)
        sides += [(half, False), (half, True)]
    sides.append((length - points[-1], False))
    spans = np.array([span for span, _ in sides])
    room = np.floor(spans / h_min * (1.0 + _DEGENERATE_TOLERANCE)).astype(np.int64)

    cells_wanted = n_nodes - 1
    if cells_wanted > room.sum():
        raise ValueError(
            f"n_nodes={n_nodes} is more than this mesh holds: at h_min={h_min:g} "
            f"cm everywhere it has room for {room.sum() + 1} nodes. Use fewer "
            "nodes or a smaller h_min."
        )

    def wanted(g: float) -> npt.NDArray[np.float64]:
        """Cells each side wants at growth rate g [1]."""
        return np.asarray(np.log1p(g * spans / h_min) / g)

    # Fewer cells the faster the spacing grows. At g -> 0 every side is at
    # h_min, which is the room just checked, so the bracket holds.
    low, high = 1e-12, 1e6
    for _ in range(200):
        middle = np.sqrt(low * high)
        if wanted(middle).sum() > cells_wanted:
            low = middle
        else:
            high = middle
    counts = np.clip(np.rint(wanted(high)), 1, room).astype(np.int64)

    # Rounding leaves the total a few cells out. The two end sides take the
    # difference, the longer first, so the mirrored halves stay mirrored.
    short = cells_wanted - int(counts.sum())
    for end in sorted((0, len(sides) - 1), key=lambda side: -spans[side]):
        moved = int(np.clip(counts[end] + short, 1, room[end])) - int(counts[end])
        counts[end] += moved
        short -= moved
    if short:  # pragma: no cover
        # Both ends full or down to one cell, which needs end sides a few
        # h_min long. Refused rather than taken out of a mirrored pair.
        raise ValueError(
            f"could not share {n_nodes} nodes between the sides of this mesh. "
            "Change n_nodes by one or two."
        )

    pieces = []
    for (span, reversed_), count in zip(sides, counts, strict=True):
        ratio = float(_solve_ratios(span, h_min, np.array([count]))[0])
        spacing = _side_spacings(span, h_min, int(count), ratio)
        pieces.append(spacing[::-1] if reversed_ else spacing)
    spacings = np.concatenate(pieces)

    neighbour_ratios = spacings[1:] / spacings[:-1]
    worst = float(max(neighbour_ratios.max(), (1.0 / neighbour_ratios).max()))
    if worst > max_ratio:
        raise ValueError(
            f"the gentlest mesh meeting these constraints jumps by {worst:.3f} "
            f"between neighbouring cells, above max_ratio={max_ratio}. Add "
            "nodes, relax h_min, or raise max_ratio deliberately."
        )

    x = np.empty(n_nodes, dtype=np.float64)
    x[0] = 0.0
    x[1:] = np.cumsum(spacings)
    # Pin every point and the far end, as graded_mesh_1d pins its one point.
    at_point = np.cumsum(counts)[0::2][: len(points)]
    x[at_point] = points
    x[-1] = length
    return _assemble(x)


def graded_mesh_1d_through(
    length: float,
    n_nodes: int,
    lines: tuple[float, ...],
    points: tuple[float, ...],
    h_min: float,
    max_ratio: float = 1.5,
) -> Mesh1D:
    """A mesh with a node on every line, refined to h_min at every point.

    Args:
        length: domain length [cm].
        n_nodes: total node count [1].
        lines: positions that must be nodes [cm], anywhere in [0, length]. A
            drawn 2D device passes the edges of everything drawn on it, since
            a cell that is half oxide has no single permittivity.
        points: positions the spacing is graded towards [cm], also nodes. A
            drawn device passes its doping edges and its Si/SiO2 interfaces.
            Empty for an axis with nothing to resolve, which then spaces its
            nodes as evenly as the lines allow.
        h_min: spacing aimed for at every point [cm].
        max_ratio: largest allowed ratio between neighbouring cells [1].

    The spacing aimed for is graded_mesh_1d_at's, h = h_min + g d with d the
    distance to the nearest point, which is geometric growth written as a
    function of position. Its node density 1/h integrates in closed form, and
    g is solved so that the integral over the axis is the cell count. Each
    span between neighbouring lines then takes a whole number of cells, the
    integral over it rounded, and lays them at equal steps of the integral.
    The density is continuous across a line, so a line lands where the
    grading already was rather than where two gradings meet, which is what
    kept Stage 4's thin base from jumping by 2.7.

    The price of pinning the lines is the rounding: a span's spacing is
    stretched by up to half a cell's worth, so h_min at a point is near
    rather than exact.
    """
    for name, positions in (("line", lines), ("point", points)):
        for position in positions:
            if not 0.0 <= position <= length:
                raise ValueError(
                    f"every {name} must lie inside [0, {length:g}], got {position:g}"
                )

    breaks = np.unique(np.concatenate([[0.0, length], lines, points]))
    spans = len(breaks) - 1
    if n_nodes - 1 < spans:
        raise ValueError(
            f"n_nodes={n_nodes} cannot put a node on every line: the lines cut "
            f"the axis into {spans} spans, so it needs at least {spans + 1} nodes"
        )
    centres = np.unique(np.asarray(points, dtype=np.float64))
    # With nothing to grade towards h_min plays no part, so it limits nothing.
    room = int(np.floor(length / h_min * (1.0 + _DEGENERATE_TOLERANCE)))
    if centres.size and n_nodes - 1 > room:
        raise ValueError(
            f"n_nodes={n_nodes} is more than this mesh holds: at h_min={h_min:g} "
            f"cm everywhere it has room for {room + 1} nodes. Use fewer nodes "
            "or a smaller h_min."
        )
    # Between two points d rises from each towards the midpoint, so the axis
    # splits at the points and the midpoints, and on each piece d is a
    # straight line of slope +1 or -1.
    knots = np.unique(
        np.concatenate([[0.0, length], centres, 0.5 * (centres[1:] + centres[:-1])])
    )

    def distance(at: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance to the nearest point [cm]."""
        return np.asarray(np.min(np.abs(at[:, None] - centres[None, :]), axis=1))

    def piece(
        d_a: npt.NDArray[np.float64], d_b: npt.NDArray[np.float64], g: float
    ) -> npt.NDArray[np.float64]:
        """Cells between two distances on one straight piece of d [1]."""
        return np.asarray(
            np.abs(np.log1p(g * d_b / h_min) - np.log1p(g * d_a / h_min)) / g
        )

    def integral(x: npt.NDArray[np.float64], g: float) -> npt.NDArray[np.float64]:
        """Cells from 0 to x at growth rate g, the integral of 1/h [1]."""
        if centres.size == 0:
            return np.asarray(x / h_min)
        whole = piece(distance(knots[:-1]), distance(knots[1:]), g)
        before = np.concatenate([[0.0], np.cumsum(whole)])
        which = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, whole.size - 1)
        return np.asarray(
            before[which] + piece(distance(knots[which]), distance(x), g)
        )

    cells_wanted = n_nodes - 1
    g = 1.0
    if centres.size:
        # Fewer cells the faster the spacing grows, and at g -> 0 every cell
        # is h_min, which is the room just checked, so the bracket holds.
        low, high = 1e-12, 1e6
        for _ in range(200):
            middle = float(np.sqrt(low * high))
            if integral(np.array([length]), middle)[0] > cells_wanted:
                low = middle
            else:
                high = middle
        g = high

    at_breaks = integral(breaks, g)
    share = np.diff(at_breaks) * cells_wanted / at_breaks[-1]

    # Whole cells per span, at least one each, the remainder going to the
    # spans rounded down furthest, so the total is exactly the node count.
    counts = np.maximum(np.floor(share), 1.0).astype(np.int64)
    while counts.sum() > cells_wanted:
        spare = np.flatnonzero(counts > 1)
        counts[spare[np.argmin((share - counts)[spare])]] -= 1
    while counts.sum() < cells_wanted:
        counts[np.argmax(share - counts)] += 1

    pieces = [np.array([0.0])]
    for a, b, start, end, count in zip(
        breaks[:-1], breaks[1:], at_breaks[:-1], at_breaks[1:], counts, strict=True
    ):
        # Equal steps of the integral across the span, each position found by
        # bisection, since the integral is monotone in x.
        targets = np.linspace(start, end, int(count) + 1)[1:-1]
        left, right = np.full_like(targets, a), np.full_like(targets, b)
        for _ in range(100):
            halfway = 0.5 * (left + right)
            below = integral(halfway, g) < targets
            left = np.where(below, halfway, left)
            right = np.where(below, right, halfway)
        pieces += [0.5 * (left + right), np.array([b])]
    nodes = np.concatenate(pieces)

    spacings = np.diff(nodes)
    neighbour_ratios = spacings[1:] / spacings[:-1]
    worst = float(max(neighbour_ratios.max(), (1.0 / neighbour_ratios).max()))
    if worst > max_ratio:
        raise ValueError(
            f"the gentlest mesh through these lines jumps by {worst:.3f} "
            f"between neighbouring cells, above max_ratio={max_ratio}. Add "
            "nodes, relax h_min, or raise max_ratio deliberately."
        )
    return _assemble(nodes)


def stacked_mesh_1d(*layers: Mesh1D) -> Mesh1D:
    """Several meshes laid end to end, sharing one node at every join.

    Args:
        layers: the meshes to stack, in order, each on [0, its own length].

    A material stack is what this is for. A MOS capacitor is silicon with
    oxide on top, and the two layers want different meshes: the silicon is
    graded hard to the surface, where the inversion layer sits inside a few
    nanometres, while the oxide holds no charge at all, so its potential is
    exactly linear and a handful of uniform cells resolves it exactly. One
    graded axis over the whole height cannot say that.

    The join is a node by construction, which is the other half of the reason.
    device/regions.py refuses an interface that does not lie on a line of mesh
    nodes, because a cell that is half oxide and half silicon has no single
    permittivity. Stacking makes that a property of how the mesh was built
    rather than something to check afterwards.

    The shared node is stored once. Storing it twice would make a cell of zero
    width, and every flux across it carries 1/h.
    """
    if not layers:
        raise ValueError("a stack needs at least one layer, got none")

    for index, layer in enumerate(layers):
        if layer.x[0] != 0.0:
            raise ValueError(
                f"layer {index} starts at x={layer.x[0]:g} cm rather than 0. "
                "Every constructor here returns a mesh on [0, length], so a "
                "layer that does not is one somebody has already translated, "
                "and stacking would translate it twice."
            )

    x = layers[0].x
    for layer in layers[1:]:
        # Drop the layer's own first node: it is the one already sitting at the
        # top of the stack so far, and it is the join.
        x = np.concatenate([x, x[-1] + layer.x[1:]])

    return _assemble(x)
