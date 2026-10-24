from __future__ import annotations
from functools import cache
import numpy as np
import pytest
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device
from ddsim.device.mosfet import nmos
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract.error import ErrorEstimate, adapt, current_error, prolonged_state, refined
from ddsim.extract.iv import gate_sweep, terminal_currents
from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
from tests.convergence.test_error_estimate import EFFECTIVITY, richardson
from tests.regression.devsim_gen import parameters as P

FETS = ((6, "nmos_1um"), (7, 'nmos_180nm'))

UNSETTLED = (8, "nmos_65nm")


def constant(d: Device) -> TransportModels:
    return TransportModels.for_device(d, mobility="constant")


@cache
def transistor(name: str) -> tuple[Device, DeviceState, ErrorEstimate]:
    u = P.MOSFET_BY_NAME[name]
    b = nmos(L_gate=u.L_gate, drain_voltage=0.05, degenerate=False, **SHORT_CHANNEL_PROCESS)
    i = gate_sweep(b, [min(1.0, max(u.gate_voltages))], models=constant(b))
    assert i.complete, i.message
    p = i.points[-1]
    d = b.with_bias(gate=p.voltage)
    return d, p.state, current_error(d, p.state, "drain", models_for=constant)


@pytest.mark.parametrize("number,name", FETS, ids=[n for _, n in FETS])
def test_drain_current_effectivity_against_richardson(number: int, name: str) -> None:
    a, limit, _ = richardson(name)
    d, _, e = transistor(name)
    assert e.value == pytest.approx(a, rel=1e-6, abs=0)
    assert e.linear, f"the fine Newton step is {e.step:.2f} V_T, too far for a linearization"
    k = e.estimate / (limit - a)
    assert EFFECTIVITY[0] <= k <= EFFECTIVITY[1], f"benchmark {number}: estimated {e.estimate:.4e}, Richardson says {limit - a:.4e}, effectivity {k:.3f}"
    assert e.indicator.shape == ((d.mesh.nx - 1) * (d.mesh.ny - 1),)
    assert float(np.sum(e.indicator)) == pytest.approx(e.estimate, rel=1e-9, abs=0)


def test_the_65nm_transistor_has_no_richardson_limit_to_score_against() -> None:
    with pytest.raises(AssertionError, match="no Richardson limit"):
        richardson(UNSETTLED[1])


def test_the_65nm_estimate_predicts_the_mesh_halved_everywhere() -> None:
    d, s, e = transistor(UNSETTLED[1])
    f = refined(d, e.refinement)
    m = constant(f)
    t = solve_bias_newton(f, m, guess=prolonged_state(d, f, s))
    assert t.newton is not None and t.newton.converged
    moved = terminal_currents(f, t, m)["drain"] - e.value
    k = e.estimate * (1.0 - float(e.refinement) ** -CONFIG.errors.assumed_order) / moved
    assert EFFECTIVITY[0] <= k <= EFFECTIVITY[1], f"estimated {e.estimate:.4e}, halving moved the current by {moved:.4e}, effectivity {k:.3f}"


def test_adaptive_refinement_beats_the_hand_graded_180nm_transistor() -> None:
    name = FETS[1][1]
    a, limit, _ = richardson(name)
    target = abs(limit - a) / abs(a)
    u = P.MOSFET_BY_NAME[name]
    hand, _, _ = transistor(name)
    d = nmos(L_gate=u.L_gate, drain_voltage=0.05, gate_voltage=1.0, degenerate=False, n_contact=3, n_sd=8, n_channel=6, n_silicon=31, n_oxide=3, h_min_x=1e-6, h_min_y=2e-6, **SHORT_CHANNEL_PROCESS)
    r = adapt(d, "drain", rtol=target, budget=hand.mesh.n_nodes, models_for=constant)
    assert r.converged, r.message
    assert r.device.mesh.n_nodes < hand.mesh.n_nodes
    assert abs(r.estimate.value - limit) <= 2.0 * target * abs(limit)


def test_a_mesh_that_misses_the_junctions_is_flagged_nonlinear() -> None:
    u = P.MOSFET_BY_NAME[FETS[0][1]]
    d = nmos(L_gate=u.L_gate, drain_voltage=0.05, gate_voltage=1.0, degenerate=False, n_contact=3, n_sd=5, n_channel=6, n_silicon=31, n_oxide=3, h_min_x=4e-6, h_min_y=2e-6, **SHORT_CHANNEL_PROCESS)
    m = constant(d)
    s = solve_bias_ramped(d, m)
    assert s.newton is not None and s.newton.converged
    e = current_error(d, s, "drain", models_for=constant)
    assert not e.linear, f"a 20 nm cell across a 25 nm junction passed for linear, step {e.step:.2f} V_T"
