from __future__ import annotations
import numpy as np, pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.transient import backward_euler

SMALL = {"n_nodes": 61, "h_min": 5e-7}


def dc(v):
    d = pn_diode(anode_voltage=v, **SMALL)
    m = TransportModels.for_device(d)
    s = solve_bias_ramped(d, m)
    assert s.newton.converged
    return d, m, s


@pytest.mark.parametrize("v", [0.0, 0.4])
def test_rest_stays_at_rest(v):
    d, m, s = dc(v)
    t = backward_euler(d, s, np.arange(1, 1001) * 1e-11, m)
    assert t.complete
    assert len(t.points) == 1000
    last = t.points[-1].state
    assert np.max(np.abs(last.psi.data - s.psi.data)) < 1e-12
    assert np.max(np.abs(last.n.data / s.n.data - 1)) < 1e-10
    assert np.max(np.abs(last.p.data / s.p.data - 1)) < 1e-10


def test_long_after_a_bias_step_the_state_is_the_dc_state():
    d0, m, s0 = dc(0.0)
    d = d0.with_bias(anode=0.4)
    t = backward_euler(d, s0, np.geomspace(1e-12, 1e-1, 133), m)
    assert t.complete
    last = t.points[-1].state
    want = solve_bias_ramped(d, m)
    assert want.newton.converged
    assert np.max(np.abs(last.psi.data - want.psi.data)) < 1e-9
    assert np.max(np.abs(last.n.data / want.n.data - 1)) < 1e-7
    assert np.max(np.abs(last.p.data / want.p.data - 1)) < 1e-7


def test_the_first_step_after_a_bias_step_is_not_yet_the_dc_state():
    d0, m, s0 = dc(0.0)
    d = d0.with_bias(anode=0.4)
    first = backward_euler(d, s0, [1e-12], m).points[0].state
    want = solve_bias_ramped(d, m)
    assert np.max(np.abs(first.p.data / want.p.data - 1)) > 1e-2


def test_times_have_to_increase():
    d, m, s = dc(0.0)
    with pytest.raises(ValueError, match="increas"):
        backward_euler(d, s, [1e-9, 1e-9], m)
    with pytest.raises(ValueError, match="increas"):
        backward_euler(d, s, [0.0], m)


def test_the_stored_holes_rise_to_their_dc_value_without_overshoot():
    d0, m, s0 = dc(0.0)
    d = d0.with_bias(anode=0.4)
    t = backward_euler(d, s0, np.geomspace(1e-13, 1e-6, 50), m)
    n_side = d.net_doping.data > 0
    stored = np.array([np.sum((p.state.p.data - s0.p.data)[n_side]) for p in t.points])
    final = np.sum((solve_bias_ramped(d, m).p.data - s0.p.data)[n_side])
    assert stored[0] < 0.5 * final
    assert np.all(np.diff(stored) >= -1e-9 * final)
    assert stored[-1] == pytest.approx(final, rel=1e-6)
