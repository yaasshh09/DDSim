"""Material regions on a structured mesh, and what box integration wants.

A MOS capacitor is two materials stacked. The oxide solves Poisson only, with a
different permittivity and no carriers at all, and docs/01-physics.md asks for
continuity of the normal component of D across the interface rather than of E.

The reason box integration was chosen is that the interface condition is not a
special case in it. Each edge carries its own permittivity, and the flux
balance at an interface node is the displacement continuity, written down
without anybody having to code an interface.

Materials belong to cells, not to nodes
---------------------------------------
That is the part worth getting right first. Put the interface on a line of
nodes and every edge still lies wholly in one material, but the *face* a
horizontal edge at the interface crosses does not: it spans half a cell of
oxide above and half a cell of silicon below. So the permittivity an edge
carries is an area weighted sum over the cells either side of it, and the same
argument gives an interface node a dual cell that is half semiconductor.

Get this wrong by classifying nodes instead and the whole interface row picks
up one material or the other, which shifts the oxide capacitance by the ratio
of half a mesh cell to the oxide thickness. On a coarse mesh that is percent
level, and it looks like a physics error rather than a bookkeeping one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.regions import (
    OXIDE,
    SILICON,
    RegionMap,
    stacked_regions,
)
from ddsim.mesh.mesh2d import uniform_mesh_2d

# 4 columns by 5 rows, so cells are 1e-5 wide and 5e-6 tall. The interface is
# put at y = 1e-5, which is the row j = 2 node line, so rows 0 and 1 of cells
# are silicon and rows 2 and 3 are oxide.
WIDTH = 3e-5
HEIGHT = 2e-5
NX = 4
NY = 5
INTERFACE = 1e-5


@pytest.fixture
def mesh():
    return uniform_mesh_2d(width=WIDTH, height=HEIGHT, nx=NX, ny=NY)


@pytest.fixture
def regions(mesh):
    """Silicon below the interface, oxide above it."""
    return stacked_regions(mesh, interface_y=INTERFACE)


def test_the_cells_are_split_at_the_interface(mesh, regions):
    """Two rows of silicon cells, two of oxide, on this mesh."""
    assert regions.cell_material.shape == (NY - 1, NX - 1)

    np.testing.assert_array_equal(regions.cell_material[:2], SILICON)
    np.testing.assert_array_equal(regions.cell_material[2:], OXIDE)


def test_only_nodes_strictly_inside_the_oxide_have_no_carriers(mesh, regions):
    """The interface node is a semiconductor node. It has silicon under it.

    Pinning it as an oxide node would delete the inversion layer, which is the
    entire thing a MOS capacitor is for.
    """
    interface_nodes = [mesh.node_at(i, 2) for i in range(NX)]

    for node in interface_nodes:
        assert node not in set(regions.oxide_nodes.tolist())

    # Everything on the two node rows above the interface is oxide.
    for row in (3, 4):
        for i in range(NX):
            assert mesh.node_at(i, row) in set(regions.oxide_nodes.tolist())


def test_an_oxide_node_has_no_semiconductor_volume(mesh, regions):
    """No charge and no recombination there, so the volume it integrates is 0."""
    assert np.all(regions.semiconductor_volume[regions.oxide_nodes] == 0.0)


def test_the_interface_node_keeps_exactly_half_its_dual_cell(mesh, regions):
    """Half of it is oxide and contributes no charge.

    An interior interface node's dual cell spans half a cell down into silicon
    and half a cell up into oxide, and those two halves are equal on a mesh
    uniform in y, so the semiconductor share is exactly one half.
    """
    node = mesh.node_at(1, 2)

    assert regions.semiconductor_volume[node] == pytest.approx(
        0.5 * mesh.volume[node], rel=1e-14
    )


def test_a_bulk_silicon_node_keeps_all_of_its_dual_cell(mesh, regions):
    node = mesh.node_at(1, 1)

    assert regions.semiconductor_volume[node] == pytest.approx(
        mesh.volume[node], rel=1e-14
    )


def test_the_semiconductor_volume_sums_to_the_silicon_area(mesh, regions):
    """Nothing is lost or double counted at the interface."""
    assert regions.semiconductor_volume.sum() == pytest.approx(
        WIDTH * INTERFACE, rel=1e-14
    )


def test_an_edge_wholly_in_one_material_carries_that_permittivity(mesh, regions):
    """Silicon edges come back at 1.0, because the scaling uses eps_Si."""
    eps_r = regions.eps_r

    # A horizontal edge in the middle of the silicon, row j = 1.
    silicon_edge = 1 * (NX - 1) + 0
    assert eps_r[silicon_edge] == pytest.approx(1.0, rel=1e-14)

    # A horizontal edge in the middle of the oxide, row j = 4 (the top).
    oxide_edge = 4 * (NX - 1) + 0
    assert eps_r[oxide_edge] == pytest.approx(
        C.EPS_R_OX / C.EPS_R_SI, rel=1e-14
    )


def test_an_interface_edge_carries_the_average_of_the_two_sides(mesh, regions):
    """The face it crosses is half oxide and half silicon.

    This is the number that node based classification gets wrong, and it is
    wrong in a way that still produces a plausible C-V curve.
    """
    interface_edge = 2 * (NX - 1) + 0
    expected = 0.5 * (1.0 + C.EPS_R_OX / C.EPS_R_SI)

    assert regions.eps_r[interface_edge] == pytest.approx(expected, rel=1e-14)


def test_a_single_material_device_is_all_ones(mesh):
    """The reduction that keeps every 1D result where it was."""
    regions = stacked_regions(mesh, interface_y=HEIGHT * 2.0)

    np.testing.assert_allclose(regions.eps_r, 1.0, rtol=0.0)
    assert regions.oxide_nodes.size == 0
    np.testing.assert_allclose(
        regions.semiconductor_volume, mesh.volume, rtol=1e-14
    )


def test_the_geometry_it_produces_carries_the_permittivity(mesh, regions):
    """What actually reaches the assembly."""
    geometry = regions.edge_geometry(mesh)

    np.testing.assert_allclose(
        np.asarray(geometry.eps_r), regions.eps_r, rtol=0.0
    )
    np.testing.assert_array_equal(geometry.edge_nodes, mesh.edge_nodes)


def test_an_interface_that_misses_every_node_line_is_refused(mesh):
    """The interface has to lie on a mesh line, not inside a cell.

    A cell that is half oxide and half silicon has no single permittivity, and
    silently rounding it to the nearer node line moves the oxide thickness by
    up to half a cell without saying so. t_ox is the thing the accumulation
    capacitance is measured against, so that is not an acceptable rounding.
    """
    with pytest.raises(ValueError, match="node line"):
        stacked_regions(mesh, interface_y=1.2e-5)


def test_a_region_map_reports_what_it_is(mesh, regions):
    assert "silicon" in repr(regions).lower()
    assert isinstance(regions, RegionMap)
