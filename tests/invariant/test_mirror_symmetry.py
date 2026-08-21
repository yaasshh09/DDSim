"""The discretization must have no preferred direction along the mesh.

Turn a diode back to front and it is the same diode seen from the other side.
The anode is still on the p side, electrons still carry Dn and tau_n, and the
terminal current has to come out the same number. Nothing in the scheme is
allowed to care which end of the array the p side sits at.

That is worth testing because so much of the code has a left and a right in it.
The Scharfetter-Gummel flux attaches B(X) to the right node for electrons and
the left node for holes. The dual cells at the two ends are half cells built by
two separate lines. Dirichlet elimination walks the triplet arrays. The graded
mesh solves a growth ratio per side and reverses one of them. A stray index
slip or an asymmetric half cell in any of those produces a device that still
converges, still conserves current, and still gives a plausible I-V, but reads
differently depending on which way round it is built.

Mirroring is a strong test because the exact answer is known without an
analytic solution: it is whatever the unmirrored device gave, to the last bit.

One thing has to be arranged first. `Step` is right continuous, so a node
sitting exactly on the junction takes the n side value in one orientation and
the p side value in the other, and the two devices then differ by one node of
doping rather than by nothing. That is a real difference, not a rounding one:
these diodes are short base, so the saturation current goes as 1/W, and moving
the metallurgical junction by one cell out of a half micron base moves the
current by about a tenth of a percent. Both facts are pinned below, the exact
agreement when no node lands on the step and the measured cost when one does.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.builder import Device, build_device
from ddsim.device.doping import Step
from ddsim.device.transport import TransportModels, solve_bias
from ddsim.discretize.boundary import OhmicContact
from ddsim.extract.iv import current_densities, terminal_currents
from ddsim.mesh.mesh1d import graded_mesh_1d
from tests.invariant.test_current_continuity import EPS, largest_flux_term

LENGTH = 1e-4
"""Device length [cm]."""

JUNCTION = 0.5e-4
"""Junction position [cm], at the middle of the device."""

H_MIN = 1e-7
"""Mesh spacing at the junction [cm]."""

N_NODES = 201
"""Node count. Even interval count, so a node lands on the middle."""

DOPING = 1e16
"""Both sides [cm^-3], so the only asymmetry left is the one under test."""

OFF_NODE = H_MIN / 3.0
"""How far to displace the step so it falls strictly between two nodes [cm].

A third of the finest spacing. Any fraction that is not zero or one works;
this one keeps the step inside the finest cell at the junction.
"""


def mesh():
    """A fresh graded mesh. Meshes are frozen, but devices cache off them."""
    return graded_mesh_1d(LENGTH, N_NODES, refine_at=JUNCTION, h_min=H_MIN)


def p_on_the_left(offset: float) -> Device:
    """The usual orientation: anode on node 0, in p type material."""
    return build_device(
        mesh=mesh(),
        doping=Step(left=-DOPING, right=DOPING, position=JUNCTION + offset),
        contacts=(
            OhmicContact("anode", 0, 0.0),
            OhmicContact("cathode", N_NODES - 1, 0.0),
        ),
    )


def n_on_the_left(offset: float) -> Device:
    """The same device written back to front: anode on the last node.

    Every position maps to LENGTH minus itself, so the step moves to the
    mirror image of where it was and the two contacts change ends.
    """
    return build_device(
        mesh=mesh(),
        doping=Step(
            left=DOPING, right=-DOPING, position=LENGTH - (JUNCTION + offset)
        ),
        contacts=(
            OhmicContact("cathode", 0, 0.0),
            OhmicContact("anode", N_NODES - 1, 0.0),
        ),
    )


def anode_current(device: Device, voltage: float) -> float:
    """Terminal current at the anode [A/cm^2], solved at one bias."""
    biased = device.with_bias(anode=voltage)
    state = solve_bias(biased)
    assert state.gummel is not None and state.gummel.converged, (
        f"the solve at {voltage:+g} V did not converge: {state.gummel}"
    )
    return terminal_currents(biased, state)["anode"]


# --------------------------------------------------- the mesh itself mirrors


def test_the_graded_mesh_is_symmetric_about_a_central_refinement() -> None:
    """x[i] and LENGTH - x[-1-i] are the same point, or the rest is untestable.

    The two sides are solved separately and one of them is reversed, so this
    is the assumption every test below rests on.
    """
    x = mesh().x
    np.testing.assert_allclose(x, LENGTH - x[::-1], rtol=0.0, atol=1e-18)


def test_the_dual_cells_mirror_too() -> None:
    """The two boundary half cells are built by separate lines, and getting
    one of them wrong tilts every integrated charge in the device.

    The bound is not machine epsilon. Node positions come from a cumulative
    sum, so they carry an absolute error of a few eps times the domain length,
    and the spacings are differences of them. Dividing an absolute error at
    the scale of the device by a spacing a thousand times smaller is where the
    1e-13 comes from, and it is a property of building a graded mesh by
    accumulation rather than a defect in the mirroring.
    """
    volume = mesh().volume
    floor = 8.0 * EPS * LENGTH / volume.min()

    worst = float(np.max(np.abs(volume - volume[::-1]) / volume))
    assert worst < floor, f"worst {worst:.3e}, accumulation floor {floor:.3e}"


# ------------------------------------------------- the physics mirrors exactly


@pytest.mark.parametrize("voltage", [-1.0, -0.2, 0.0, 0.2, 0.4, 0.5])
def test_a_mirrored_diode_gives_the_same_terminal_current(voltage: float) -> None:
    """The headline invariant, and it holds to machine precision.

    Not to a tolerance that would let a small directional bias through. The
    two devices are the same discrete problem with the unknowns numbered
    backwards, so a direct solve on either has to reach the same numbers.
    """
    forward = anode_current(p_on_the_left(OFF_NODE), voltage)
    backward = anode_current(n_on_the_left(OFF_NODE), voltage)

    assert forward == pytest.approx(backward, rel=1e-12), (
        f"at {voltage:+g} V the p-on-the-left diode gives {forward:.9e} and the "
        f"n-on-the-left diode gives {backward:.9e}. The mesh and the doping "
        "mirror exactly, so a difference here is a left-right bias in the "
        "assembly, the boundary conditions or the flux."
    )


def test_the_built_in_potential_mirrors_and_changes_sign() -> None:
    """psi rises toward n type, so reversing the device negates the drop."""
    forward = solve_bias(p_on_the_left(OFF_NODE))
    backward = solve_bias(n_on_the_left(OFF_NODE))

    rise = float(forward.psi.data[-1] - forward.psi.data[0])
    fall = float(backward.psi.data[-1] - backward.psi.data[0])

    assert rise > 0.0
    assert fall == pytest.approx(-rise, rel=1e-12)


def test_the_solved_profiles_are_reflections_of_each_other() -> None:
    """Node by node, not just at the terminals.

    A boundary condition applied to the wrong end, or a half cell missing at
    one end only, moves interior nodes while leaving the terminal current
    almost untouched.
    """
    forward = solve_bias(p_on_the_left(OFF_NODE).with_bias(anode=0.4))
    backward = solve_bias(n_on_the_left(OFF_NODE).with_bias(anode=0.4))

    np.testing.assert_allclose(
        forward.n.data, backward.n.data[::-1], rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        forward.p.data, backward.p.data[::-1], rtol=1e-11, atol=0.0
    )


def test_the_current_densities_mirror_with_a_sign_change() -> None:
    """Jn and Jp point along +x, so reversing the axis reverses both.

    This is the one that would catch B(X) being attached to the wrong end of
    an edge for one carrier, since that error lives entirely in the edge
    fluxes and can leave the node densities looking reasonable.

    The tolerance is the same cancellation floor as in
    test_current_continuity.py, not a round number. Each edge current is the
    difference of two terms worth about 5.8e9 in scaled units against a
    current of 5.8e2, so roughly seven digits of the sixteen survive the
    subtraction and two mirrored evaluations cannot agree any closer than
    that however symmetric the scheme is.
    """
    device = p_on_the_left(OFF_NODE).with_bias(anode=0.4)
    mirrored = n_on_the_left(OFF_NODE).with_bias(anode=0.4)

    state = solve_bias(device)
    models = TransportModels.for_device(device)
    Jn, Jp = current_densities(device, state, models)
    Jn_mirror, Jp_mirror = current_densities(mirrored, solve_bias(mirrored))

    floor = 3.0 * EPS * largest_flux_term(device, state, models) / (
        np.abs(Jn.data).max() / device.scale.J_0
    )

    for measured, reflected, name in (
        (Jn.data, Jn_mirror.data, "Jn"),
        (Jp.data, Jp_mirror.data, "Jp"),
    ):
        worst = float(
            np.max(np.abs(measured + reflected[::-1]) / np.abs(measured).max())
        )
        assert worst < floor, (
            f"{name} disagrees with its reflection by {worst:.3e}, above the "
            f"cancellation floor of {floor:.3e}. A difference above the floor "
            "is a left-right bias in the flux, not arithmetic."
        )


# ------------------------------------- what a node landing on the step costs


def test_a_node_on_the_step_is_the_only_thing_that_breaks_the_mirror() -> None:
    """Step is right continuous, so the junction node picks a side.

    With the step exactly on node 100 the two orientations disagree about
    that one node, and about nothing else. This is the documented convention
    rather than a defect, but the size of it is worth pinning: it is the only
    reason the mirror test above displaces the step at all.
    """
    on_node = p_on_the_left(0.0).net_doping.data
    on_node_mirror = n_on_the_left(0.0).net_doping.data
    assert np.count_nonzero(on_node != on_node_mirror[::-1]) == 1

    off_node = p_on_the_left(OFF_NODE).net_doping.data
    off_node_mirror = n_on_the_left(OFF_NODE).net_doping.data
    np.testing.assert_array_equal(off_node, off_node_mirror[::-1])


def test_one_cell_of_base_width_is_worth_about_a_tenth_of_a_percent() -> None:
    """Why the mirror disagreement is 1e-3 and not 1e-15 when a node lands on
    the step.

    These diodes are short base: the quasi-neutral region is half a micron
    against a diffusion length of about 175 um, so the saturation current
    goes as 1/W and a one cell shift in W shows up directly in the current.
    Moving the step by one h_min on a single device reproduces the mirror
    disagreement, which is what identifies the cause rather than assuming it.
    """
    voltage = 0.5
    unshifted = anode_current(p_on_the_left(0.0), voltage)
    shifted = anode_current(p_on_the_left(H_MIN), voltage)
    mirrored = anode_current(n_on_the_left(0.0), voltage)

    from_shift = (shifted - unshifted) / unshifted
    from_mirror = (mirrored - unshifted) / unshifted

    assert from_shift < 0.0, "widening the p side base must lower the current"
    assert abs(from_shift) == pytest.approx(abs(from_mirror), rel=1e-6)
    assert 5e-4 < abs(from_mirror) < 5e-3
