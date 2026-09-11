"""A cold solve at a large applied bias, reached by ramping the bias in.

The gate length sweep starts every transfer curve with one cold solve at
gate zero and the drain already at its full bias, and on a MOSFET that is a
long walk against the potential step limiter rather than a Newton solve.
Measured on the 1 um device this file uses, from the Poisson guess with no
ramp, at the 2835 node vertical mesh that was the nmos default when the ramp
was written:

    Vd = 0.00 V    1 step,   0 limited
    Vd = 0.25 V   10 steps,  0 limited
    Vd = 0.50 V   12 steps,  2 limited
    Vd = 1.00 V   22 steps, 12 limited

Twenty two of a budget of thirty, twelve of them clipped, is not a basin. It
is arriving anyway, and which side of it a machine lands on is set by the
order the linear solver sums its floating point. CI landed on the other side
and reported a residual of 9.889e+03 with psi spread over [-22.2, 62.9], on
all three Python versions, while the same solve converged here.

So the bias is ramped in as a fraction of itself, which is the same
continuation the sweep already runs between its gate points, applied to the
step that got there.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.mosfet import nmos
from ddsim.device.transport import (
    TransportModels,
    solve_bias_newton,
    solve_bias_ramped,
)
from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS

DRAIN_BIAS = 1.0
"""The saturated drain bias the roll-off sweep takes its DIBL curve at [V]."""


@pytest.fixture(scope="module")
def hot_device():
    """A 1 um NMOS with the drain already at full bias and the gate off."""
    return nmos(
        L_gate=1e-4,
        gate_voltage=0.0,
        drain_voltage=DRAIN_BIAS,
        **SHORT_CHANNEL_PROCESS,
    )


@pytest.fixture(scope="module")
def models(hot_device):
    return TransportModels.for_device(
        hot_device, mobility="arora", field_dependent=True, surface=True
    )


@pytest.fixture(scope="module")
def ramped(hot_device, models):
    return solve_bias_ramped(hot_device, models=models)


@pytest.fixture(scope="module")
def direct(hot_device, models):
    return solve_bias_newton(hot_device, models=models, guess=None)


def test_the_ramp_converges(ramped):
    assert ramped.newton is not None
    assert ramped.newton.converged, ramped.newton.message


def test_the_ramp_lands_on_the_same_root(ramped, direct):
    """Same bias, same equations, so it has to be the same solution.

    The point of the ramp is the path, not the answer. If the two disagreed
    on the answer then one of them is at a bias nobody asked for, which is
    the failure mode a continuation that quietly stops short would produce.
    """
    assert direct.newton is not None and direct.newton.converged
    assert ramped.psi.data == pytest.approx(direct.psi.data, abs=1e-8)
    for ramp_side, direct_side in (
        (ramped.n.data, direct.n.data),
        (ramped.p.data, direct.p.data),
    ):
        relative = np.abs(ramp_side - direct_side) / (np.abs(direct_side) + 1.0)
        assert np.max(relative) < 1e-8


def test_the_final_step_is_a_newton_solve_and_not_a_limited_walk(ramped, direct):
    """The reason the ramp exists, stated as the thing that changed.

    The cold solve spends most of its steps clipped by max_psi_step, which
    means the quadratic tail never starts and the iterate is being dragged
    rather than converged. Arriving at the target from the neighbouring
    fraction, none of it is clipped.
    """
    assert direct.newton is not None and ramped.newton is not None
    assert direct.newton.limited_steps > 0
    assert ramped.newton.limited_steps == 0


def test_a_device_at_zero_bias_is_not_disturbed_by_the_ramp(models):
    """Every fraction of zero is zero, so this has to be the equilibrium solve."""
    off = nmos(
        L_gate=1e-4, gate_voltage=0.0, drain_voltage=0.0, **SHORT_CHANNEL_PROCESS
    )
    ramped = solve_bias_ramped(off, models=models)
    direct = solve_bias_newton(off, models=models, guess=None)

    assert ramped.newton is not None and ramped.newton.converged
    assert ramped.psi.data == pytest.approx(direct.psi.data, abs=1e-10)


def test_the_ramp_builds_its_own_models_when_given_none(hot_device):
    """The same default TransportModels.for_device gives every other solver."""
    ramped = solve_bias_ramped(hot_device)
    assert ramped.newton is not None and ramped.newton.converged
