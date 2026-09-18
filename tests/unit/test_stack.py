"""Tests for device/stack.py, the 1D device builder of phases/PHASE-7.md Stage 4.

A stack is doped regions laid left to right with a contact at each end. It
is built on the same graded mesh and the same doping profiles as pn_diode, so
the first thing it has to do is be pn_diode when it is drawn as one. The
rest is what the phase asks of it: grading at every junction, and refusals
that name their reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.pn_diode import pn_diode
from ddsim.device.stack import DOPING_RANGE, Region, stack

MICRON = 1e-4
"""One micron [cm]."""


def net_doping(device) -> np.ndarray:
    return device.doping(device.mesh.x)


# ----------------------------------------------------------- the Phase 2 diode


def test_the_default_stack_is_the_phase_2_diode_bit_for_bit() -> None:
    """Same mesh, same doping, to the last bit. The I-V agreement the phase
    asks for follows from this, and tests/unit/test_api_devices.py checks
    that too, through the request the page sends."""
    built, diode = stack(), pn_diode()
    np.testing.assert_array_equal(built.mesh.x, diode.mesh.x)
    np.testing.assert_array_equal(net_doping(built), net_doping(diode))


def test_a_stack_has_a_contact_at_each_end() -> None:
    device = stack(left_voltage=0.3, right_voltage=-0.1)
    left, right = device.contacts
    assert (left.name, left.nodes, left.voltage) == ("left", (0,), 0.3)
    last = device.mesh.n_nodes - 1
    assert (right.name, right.nodes, right.voltage) == ("right", (last,), -0.1)


def test_each_region_holds_its_own_doping() -> None:
    regions = (
        Region("p", 0.2 * MICRON, 1e18),
        Region("n", 1.0 * MICRON, 1e14),
        Region("n", 0.2 * MICRON, 1e18),
    )
    device = stack(regions)
    x = device.mesh.x
    doping = net_doping(device)
    assert np.all(doping[x < 0.2 * MICRON] == -1e18)
    assert np.all(doping[(x >= 0.2 * MICRON) & (x < 1.2 * MICRON)] == 1e14)
    assert np.all(doping[x >= 1.2 * MICRON] == 1e18)


# ------------------------------------------------------------------- the mesh


@pytest.mark.parametrize(
    "regions",
    [
        (
            Region("p", 0.2 * MICRON, 1e18),
            Region("n", 1.0 * MICRON, 1e14),
            Region("n", 0.2 * MICRON, 1e18),
        ),
        (
            Region("n", 0.3 * MICRON, 1e18),
            Region("p", 0.2 * MICRON, 1e17),
            Region("n", 0.5 * MICRON, 1e16),
        ),
        (
            Region("p", 0.1 * MICRON, 1e17),
            Region("n", 0.1 * MICRON, 1e17),
            Region("p", 1.0 * MICRON, 1e15),
            Region("n", 0.1 * MICRON, 1e17),
        ),
    ],
    ids=["pin", "npn", "four regions"],
)
def test_every_junction_is_a_node_at_the_finest_spacing(regions) -> None:
    """Graded at every junction, as the phase asks: a node sits exactly on
    each one, the cells either side are the finest in the device to within
    the rescale graded_mesh_1d applies, and nowhere do neighbouring cells
    jump by more than graded_mesh_1d itself allows."""
    h_min = 1e-7
    device = stack(regions, h_min=h_min)
    mesh = device.mesh
    for junction in np.cumsum([r.length for r in regions])[:-1]:
        node = int(np.argmin(np.abs(mesh.x - junction)))
        assert mesh.x[node] == junction
        assert max(mesh.h[node - 1], mesh.h[node]) < 1.5 * h_min
    ratios = mesh.h[1:] / mesh.h[:-1]
    assert max(ratios.max(), (1.0 / ratios).max()) <= 1.5


def test_the_node_count_is_the_one_asked_for() -> None:
    regions = (
        Region("n", 0.3 * MICRON, 1e18),
        Region("p", 0.2 * MICRON, 1e17),
        Region("n", 0.5 * MICRON, 1e16),
    )
    assert stack(regions, n_nodes=301).mesh.n_nodes == 301


def test_more_nodes_never_breaks_a_stack_with_a_short_segment() -> None:
    """Two thin regions between two long ones. Shared by weight alone, the
    1 um segment between the two close junctions is handed more cells than
    fit in it at h_min, and a refusal to add nodes would be the answer to
    adding nodes. It takes what it can hold and the rest go elsewhere."""
    regions = (
        Region("p", 10 * MICRON, 1e15),
        Region("n", 0.01 * MICRON, 1e17),
        Region("p", 0.01 * MICRON, 1e17),
        Region("n", 10 * MICRON, 1e15),
    )
    for n_nodes in (101, 201, 401):
        assert stack(regions, n_nodes=n_nodes).mesh.n_nodes == n_nodes


def test_more_nodes_than_the_stack_holds_at_h_min_is_refused() -> None:
    """At h_min everywhere the mesh is as fine as it can be, so a node count
    past that has nowhere to go."""
    regions = (Region("p", 5e-6, 1e17), Region("n", 5e-6, 1e17))
    with pytest.raises(ValueError, match="n_nodes") as raised:
        stack(regions, n_nodes=1001, h_min=1e-7)
    assert "h_min" in str(raised.value)


def test_a_region_the_graded_mesh_leaves_too_few_nodes_in_is_refused() -> None:
    """Three h_min long, so it passes the length check, but the gradings
    from its two junctions meet inside it and leave one node."""
    regions = (
        Region("n", MICRON, 1e16),
        Region("p", 3e-7, 1e18),
        Region("n", MICRON, 1e16),
    )
    with pytest.raises(ValueError, match="region 2.*holds 1 mesh nodes"):
        stack(regions, h_min=1e-7)


def test_a_boundary_between_two_equal_regions_is_not_graded_towards() -> None:
    """No doping changes there, so there is nothing to resolve. The stack is
    the two region diode with its p side drawn in two pieces."""
    split = (
        Region("p", 0.2 * MICRON, 1e16),
        Region("p", 0.3 * MICRON, 1e16),
        Region("n", 0.5 * MICRON, 1e16),
    )
    np.testing.assert_array_equal(stack(split).mesh.x, pn_diode().mesh.x)


# ------------------------------------------------------------------- refusals


def test_a_stack_with_no_junction_is_refused() -> None:
    with pytest.raises(ValueError, match="no junction"):
        stack((Region("n", MICRON, 1e16),))


def test_a_stack_of_equal_regions_has_no_junction_either() -> None:
    with pytest.raises(ValueError, match="no junction"):
        stack((Region("n", MICRON, 1e16), Region("n", MICRON, 1e16)))


def test_an_empty_stack_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two regions"):
        stack(())


@pytest.mark.parametrize("concentration", [1e13, 1e20, 0.0])
def test_a_doping_outside_the_model_range_is_refused(concentration) -> None:
    low, high = DOPING_RANGE
    regions = (Region("p", MICRON, 1e16), Region("n", MICRON, concentration))
    with pytest.raises(ValueError, match="region 2") as raised:
        stack(regions)
    assert f"{low:g}" in str(raised.value) and f"{high:g}" in str(raised.value)
    assert "docs/01-physics.md" in str(raised.value)


def test_the_ends_of_the_doping_range_are_accepted() -> None:
    low, high = DOPING_RANGE
    stack((Region("p", MICRON, high), Region("n", MICRON, low)), n_nodes=301)


def test_a_dopant_that_is_not_n_or_p_is_refused() -> None:
    with pytest.raises(ValueError, match="region 1.*'i'"):
        stack((Region("i", MICRON, 1e16), Region("n", MICRON, 1e16)))


def test_a_region_with_no_length_is_refused() -> None:
    with pytest.raises(ValueError, match="region 2.*length"):
        stack((Region("p", MICRON, 1e16), Region("n", 0.0, 1e16)))


def test_a_region_shorter_than_the_mesh_resolves_is_refused() -> None:
    """Half a nanometre between two junctions graded to one nanometre has no
    node inside it at all. The refusal names the region and what to change."""
    regions = (
        Region("n", MICRON, 1e16),
        Region("p", 0.5e-7, 1e18),
        Region("n", MICRON, 1e16),
    )
    with pytest.raises(ValueError, match="region 2") as raised:
        stack(regions, h_min=1e-7)
    assert "h_min" in str(raised.value) and "n_nodes" in str(raised.value)
