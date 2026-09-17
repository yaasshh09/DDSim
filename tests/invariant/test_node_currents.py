"""Total current density at the nodes, for streamlines.

The solver's currents live on edges. A streamline needs a vector at a point,
so ddsim/extract/iv.py averages each node's neighbouring edges, x from the
horizontal edges and y from the vertical ones, weighted by the face each edge
offers a carrier. What has to hold is what holds for the edges: on the 2D
diode that is uniform in y the vector points along x, every row carries the
same x current, and a column of it integrates to the cut current.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ddsim.extract import iv
from ddsim.extract.iv import current_densities, node_current_density
from ddsim.physics.recombination import NoRecombination
from tests.invariant.test_current_continuity import solved as solved_1d
from tests.invariant.test_current_continuity_2d import (
    capped_diode_2d,
    cut_currents,
    diode_2d,
    solved,
)


@pytest.fixture(scope="module")
def plain():
    device = diode_2d()
    state, models = solved(device)
    return device, state, models


def test_on_a_diode_uniform_in_y_the_current_points_along_x(plain) -> None:
    device, state, models = plain
    Jx, Jy = node_current_density(device, state, models)

    assert np.max(np.abs(Jy)) < 1e-9 * np.max(np.abs(Jx))


def test_every_row_carries_the_same_x_current(plain) -> None:
    device, state, models = plain
    Jx, _ = node_current_density(device, state, models)
    rows = Jx.reshape(device.mesh.ny, device.mesh.nx)

    np.testing.assert_allclose(rows, np.broadcast_to(rows[0], rows.shape), rtol=1e-9)


def test_a_column_of_node_current_integrates_to_the_cut_current(plain) -> None:
    device, state, models = plain
    Jx, _ = node_current_density(device, state, models)
    mesh = device.mesh
    heights = np.asarray(mesh.y_axis.volume)
    column = Jx.reshape(mesh.ny, mesh.nx)[:, mesh.nx // 2]

    expected = cut_currents(device, state, models)[mesh.nx // 2]
    assert float(np.sum(column * heights)) == pytest.approx(expected, rel=1e-6)


def test_a_vertical_edge_lands_on_the_nodes_above_and_below_it(monkeypatch) -> None:
    """The diodes above carry no vertical current, so they cannot see where a
    vertical edge's current goes. Here one vertical edge carries a unit
    density and nothing else carries anything. Its current has to land in Jy
    at node (i, j) and node (i, j + 1), and nowhere else.
    """
    device = diode_2d()
    mesh = device.mesh
    i, j = 3, 2
    edge = mesh.n_horizontal + j * mesh.nx + i

    total = np.zeros(mesh.n_edges)
    total[edge] = 1.0
    fake = SimpleNamespace(data=total)
    zero = SimpleNamespace(data=np.zeros(mesh.n_edges))
    monkeypatch.setattr(iv, "current_densities", lambda *args: (fake, zero))

    Jx, Jy = iv.node_current_density(device, None)

    assert np.all(Jx == 0.0)
    assert set(np.flatnonzero(Jy)) == {mesh.node_at(i, j), mesh.node_at(i, j + 1)}


def test_on_a_1d_diode_every_node_carries_the_edge_current() -> None:
    """In 1D with recombination off, Jn + Jp is the same on every edge, so the
    mean onto any node is that same number, and there is no y to point along.
    """
    device, state, models = solved_1d(0.5, NoRecombination())
    Jn, Jp = current_densities(device, state, models)
    Jx, Jy = node_current_density(device, state, models)

    assert Jx.shape == Jy.shape == (device.mesh.n_nodes,)
    assert np.all(Jy == 0.0)
    np.testing.assert_allclose(Jx, np.mean(Jn.data + Jp.data), rtol=1e-6)


def test_the_oxide_carries_no_current() -> None:
    device = capped_diode_2d()
    state, models = solved(device)
    Jx, Jy = node_current_density(device, state, models)
    oxide = device.regions.oxide_nodes

    assert np.all(Jx[oxide] == 0.0)
    assert np.all(Jy[oxide] == 0.0)
