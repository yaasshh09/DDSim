from __future__ import annotations
import numpy as np, pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.transient import backward_euler, bdf2

T = 1e-6
STEPS = [16, 32, 64, 128]


def drive(t):
    return {"anode": 0.4 + 0.05 * (1 - np.cos(np.pi * t / (2 * T)))}


@pytest.fixture(scope="module")
def diode():
    d = pn_diode(anode_voltage=0.4, n_nodes=61, h_min=5e-7)
    m = TransportModels.for_device(d)
    return d, m, solve_bias_ramped(d, m)


def at_end(diode, method, n):
    d, m, s = diode
    t = method(d, s, np.arange(1, n + 1) * T / n, m, bias=drive)
    assert t.complete
    return t.points[-1].currents["anode"], t.points[-1].state.p.data


@pytest.fixture(scope="module")
def reference(diode):
    return at_end(diode, bdf2, 2048)


def slopes(diode, reference, method):
    i0, p0 = reference
    ei, ep = [], []
    for n in STEPS:
        i, p = at_end(diode, method, n)
        ei.append(abs(i / i0 - 1))
        ep.append(np.max(np.abs(p / p0 - 1)))
    x = np.log(np.array(STEPS, dtype=float))
    return -np.polyfit(x, np.log(ei), 1)[0], -np.polyfit(x, np.log(ep), 1)[0]


def test_backward_euler_is_first_order(diode, reference):
    for k in slopes(diode, reference, backward_euler):
        assert 0.9 < k < 1.2


def test_bdf2_is_second_order(diode, reference):
    for k in slopes(diode, reference, bdf2):
        assert 1.8 < k < 2.2


def test_bdf2_stays_second_order_on_steps_that_grow(diode, reference):
    d, m, s = diode
    errors = []
    for n in STEPS:
        t = bdf2(d, s, T * (np.arange(1, n + 1) / n) ** 2, m, bias=drive)
        errors.append(abs(t.points[-1].currents["anode"] / reference[0] - 1))
    assert 1.8 < -np.polyfit(np.log(np.array(STEPS, dtype=float)), np.log(errors), 1)[0] < 2.2


def test_adaptive_bdf2_stays_inside_its_tolerance_on_a_smooth_drive(diode, reference):
    d, m, s = diode
    errors = []
    for rtol in (1e-2, 1e-3, 1e-4):
        t = bdf2(d, s, [T], m, bias=drive, rtol=rtol)
        assert t.complete
        errors.append(abs(t.points[-1].currents["anode"] / reference[0] - 1))
        assert errors[-1] < rtol
    assert errors[0] > errors[1] > errors[2]


def test_adaptive_bdf2_tightens_with_its_tolerance_across_a_bias_step():
    d0 = pn_diode(n_nodes=61, h_min=5e-7)
    m = TransportModels.for_device(d0)
    s0 = solve_bias_ramped(d0, m)
    d = d0.with_bias(anode=0.4)
    outs = np.geomspace(1e-12, 1e-6, 7)
    fine = {round(float(np.log10(p.time)), 6): p.currents["anode"] for p in bdf2(d, s0, np.union1d(np.geomspace(1e-15, 1e-6, 3000), outs), m).points}
    worst = []
    for rtol in (1e-2, 1e-3):
        t = bdf2(d, s0, outs, m, rtol=rtol)
        assert t.complete
        assert t.steps < 250
        worst.append(max(abs(p.currents["anode"] / fine[round(float(np.log10(p.time)), 6)] - 1) for p in t.points))
    assert worst[1] < worst[0] / 3
