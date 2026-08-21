"""A uniformly doped bar has to obey Ohm's law, and it is exact.

docs/04-validation.md lists the Debye length, the built-in potential, the
depletion width and the Shockley diode as the Tier 2 analytic cases. This one
is not in the list and should be, because it is the only case where the answer
has no approximation in it at all.

    J = sigma * V / L,    sigma = q * (mu_n * n + mu_p * p)

Every other analytic target is a limit that the simulator is allowed to miss by
a percent or three: the depletion approximation is wrong at the edges, the
built-in potential expression degrades above 1e18, the Shockley form assumes
low injection. Ohm's law for a uniform bar is not a limit. With constant
mobility and Boltzmann statistics the drift-diffusion system reduces to it
identically, so the simulator is required to reproduce it to solver tolerance
rather than to a percentage. Measured agreement below is around 1e-12, and it
does not improve with mesh refinement because there is nothing to refine: the
Scharfetter-Gummel flux is exact for a constant field.

What that buys is a check no diode test can give. It pins the whole chain end
to end with one number: the Einstein relation turning the mobility into the
diffusivity the solver actually uses, the sign of the drift term, the ohmic
contacts, the scaling, and the terminal current extraction. Get the Einstein
relation wrong by a factor and every diode number moves in a way that looks
like a lifetime being off, while this test fails by exactly that factor.

The near-intrinsic cases matter for the same reason. At 1e16 the minority
carrier contributes 1e-12 of the conductivity and the test cannot tell whether
it is there. At zero net doping the two carriers contribute q*mu_n*n_i and
q*mu_p*n_i, and the hole term is a quarter of the total.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.builder import Device, build_device
from ddsim.device.doping import Uniform
from ddsim.device.transport import solve_bias
from ddsim.discretize.boundary import OhmicContact
from ddsim.extract.iv import terminal_currents
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics import equilibrium_densities_scaled

MICRON = 1e-4
"""One micron [cm]."""


def bar(net_doping: float, length: float = MICRON, n_nodes: int = 101) -> Device:
    """A uniformly doped bar with an ohmic contact at each end."""
    return build_device(
        mesh=uniform_mesh_1d(length, n_nodes),
        doping=Uniform(net_doping),
        contacts=(
            OhmicContact("left", 0, 0.0),
            OhmicContact("right", n_nodes - 1, 0.0),
        ),
    )


def conductivity(device: Device, net_doping: float) -> float:
    """sigma = q * (mu_n * n + mu_p * p) [S/cm], both carriers.

    The densities come from neutrality plus mass action, the same pair the
    ohmic contacts are built from, so this is the equilibrium bulk and not an
    approximation of it.
    """
    n, p = equilibrium_densities_scaled(net_doping / device.scale.C_0)
    n_bulk = float(n) * device.scale.C_0
    p_bulk = float(p) * device.scale.C_0
    T = device.material.T
    return C.q * (C.mu_n(T) * n_bulk + C.mu_p(T) * p_bulk)


def measured_current(device: Device, voltage: float) -> float:
    """Terminal current into the left contact [A/cm^2] at one bias."""
    biased = device.with_bias(left=voltage)
    state = solve_bias(biased)
    assert state.gummel is not None and state.gummel.converged, (
        f"the bar did not converge at {voltage:+g} V: {state.gummel}"
    )
    return terminal_currents(biased, state)["left"]


# ------------------------------------------------------------------ the law


@pytest.mark.parametrize("net_doping", [1e18, 1e16, 1e15, -1e16])
@pytest.mark.parametrize("voltage", [1e-4, 1e-2, 0.1])
def test_current_matches_ohms_law(net_doping: float, voltage: float) -> None:
    """The headline. Not a percentage, a solver tolerance."""
    device = bar(net_doping)
    expected = conductivity(device, net_doping) * voltage / MICRON

    assert measured_current(device, voltage) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("net_doping", [1e11, 1e10, 0.0, -1e10])
def test_ohms_law_holds_where_both_carriers_conduct(net_doping: float) -> None:
    """Near intrinsic, so the hole term is a real fraction of the total.

    At zero net doping n and p are both n_i and the holes carry a quarter of
    the current. A conductivity that had quietly dropped one carrier passes
    every doped case above and fails here by that quarter.
    """
    device = bar(net_doping)
    n, p = equilibrium_densities_scaled(net_doping / device.scale.C_0)
    hole_share = C.mu_p() * float(p) / (C.mu_n() * float(n) + C.mu_p() * float(p))
    assert hole_share > 1e-3, "this case is meant to exercise the hole term"

    expected = conductivity(device, net_doping) * 1e-3 / MICRON
    assert measured_current(device, 1e-3) == pytest.approx(expected, rel=1e-7)


# --------------------------------------------------------- shape of the curve


def test_the_bar_is_linear_over_four_decades_of_bias() -> None:
    """Constant mobility means no saturation. Velocity saturation is Phase 5."""
    device = bar(1e16)
    resistance = [measured_current(device, v) / v for v in (1e-4, 1e-3, 1e-2, 0.1)]

    assert np.ptp(resistance) / np.mean(resistance) < 1e-7


def test_reversing_the_bias_reverses_the_current() -> None:
    """A resistor has no preferred direction. A diode does, so this separates
    a bar that is genuinely uniform from one with an accidental junction."""
    device = bar(1e16)

    assert measured_current(device, 0.01) == pytest.approx(
        -measured_current(device, -0.01), rel=1e-9
    )


def test_current_falls_as_one_over_the_length() -> None:
    device_short = bar(1e16, length=MICRON)
    device_long = bar(1e16, length=10.0 * MICRON)

    assert measured_current(device_short, 0.01) == pytest.approx(
        10.0 * measured_current(device_long, 0.01), rel=1e-7
    )


def test_current_rises_in_proportion_to_the_doping() -> None:
    """sigma goes as n, and n goes as the doping once it is well above n_i."""
    assert measured_current(bar(1e17), 0.01) == pytest.approx(
        10.0 * measured_current(bar(1e16), 0.01), rel=1e-4
    )


def test_the_answer_does_not_depend_on_the_mesh() -> None:
    """Scharfetter-Gummel is exact for a constant field, so eleven nodes and
    two hundred give the same number. Anything that changed with refinement
    here would be a discretization error that should not exist."""
    coarse = measured_current(bar(1e16, n_nodes=11), 0.01)
    fine = measured_current(bar(1e16, n_nodes=201), 0.01)

    assert coarse == pytest.approx(fine, rel=1e-9)
