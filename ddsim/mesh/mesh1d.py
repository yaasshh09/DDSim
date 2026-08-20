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

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_RATIO_TOLERANCE = 1e-14
"""Relative tolerance for the geometric ratio solve [1]."""

_DEGENERATE_TOLERANCE = 1e-12
"""Below this relative difference, a side is treated as exactly uniform [1]."""

_LOG_MAX_DOUBLE = 700.0
"""ln of a number comfortably below the largest double, about 1.8e308 [1]."""


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


def _geometric_sum(h_min: float, ratio: float, n_intervals: int) -> float:
    """Total length of n_intervals spacings growing geometrically [cm].

    Saturates to infinity rather than overflowing. The bracketing search below
    starts at ratio 2 and doubles, and with a thousand cells 2^1000 is far
    past the double range. The bisection only ever asks whether the sum
    exceeds the side length, so infinity is a perfectly usable answer.
    """
    if ratio == 1.0:
        return h_min * n_intervals
    if n_intervals * math.log(ratio) > _LOG_MAX_DOUBLE:
        return math.inf
    return h_min * (ratio**n_intervals - 1.0) / (ratio - 1.0)


def _solve_ratio(side_length: float, h_min: float, n_intervals: int) -> float | None:
    """Growth ratio r such that h_min * (r^m - 1)/(r - 1) = side_length.

    Returns None if the side cannot be covered, which happens when even the
    minimum spacing repeated m times overshoots the available length.
    """
    if n_intervals <= 0:
        return None
    if h_min > side_length * (1.0 + _DEGENERATE_TOLERANCE):
        return None

    uniform_total = h_min * n_intervals
    if uniform_total > side_length * (1.0 + _DEGENERATE_TOLERANCE):
        return None
    if n_intervals == 1:
        return 1.0
    if abs(uniform_total - side_length) <= _DEGENERATE_TOLERANCE * side_length:
        return 1.0

    # Bracket first. The sum grows monotonically with r, so doubling the upper
    # bound until it overshoots is enough.
    low, high = 1.0, 2.0
    while _geometric_sum(h_min, high, n_intervals) < side_length:
        high *= 2.0
        if high > 1e6:  # pragma: no cover
            # Unreachable while the feasibility check above holds, since
            # h_min * m <= side_length guarantees some r >= 1 exists. Kept so
            # a future caller that skips that check cannot spin forever.
            return None

    for _ in range(200):
        middle = 0.5 * (low + high)
        if _geometric_sum(h_min, middle, n_intervals) < side_length:
            low = middle
        else:
            high = middle
        if high - low <= _RATIO_TOLERANCE * low:
            break

    return 0.5 * (low + high)


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
    neighbouring cell ratio, by trying every split. With a few hundred nodes
    that is microseconds, and it is far easier to follow than a closed form.
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

    best_spacings: npt.NDArray[np.float64] | None = None
    best_score = np.inf

    # A refinement point on a boundary means one side only.
    if left_length == 0.0:
        splits = [0]
    elif right_length == 0.0:
        splits = [n_intervals]
    else:
        splits = list(range(1, n_intervals))

    for n_left in splits:
        n_right = n_intervals - n_left

        ratio_left = _solve_ratio(left_length, h_min, n_left)
        ratio_right = _solve_ratio(right_length, h_min, n_right)
        if (n_left > 0 and ratio_left is None) or (n_right > 0 and ratio_right is None):
            continue

        pieces = []
        if n_left > 0:
            assert ratio_left is not None
            # Ordered outward from the pivot, so reverse it to run left to right.
            pieces.append(
                _side_spacings(left_length, h_min, n_left, ratio_left)[::-1]
            )
        if n_right > 0:
            assert ratio_right is not None
            pieces.append(_side_spacings(right_length, h_min, n_right, ratio_right))

        spacings = np.concatenate(pieces)
        neighbour_ratios = spacings[1:] / spacings[:-1]
        score = float(max(neighbour_ratios.max(), (1.0 / neighbour_ratios).max()))

        if score < best_score:
            best_score = score
            best_spacings = spacings

    if best_spacings is None:
        raise ValueError(
            f"infeasible request: no split of {n_intervals} cells reaches "
            f"h_min={h_min:g} cm at refine_at={refine_at:g} cm within a domain "
            f"of {length:g} cm."
        )

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
