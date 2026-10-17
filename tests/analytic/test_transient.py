from __future__ import annotations
import numpy as np, pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.transient import backward_euler, bdf2

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


def kirchhoff(t):
    big = max(abs(v) for p in t.points for v in p.currents.values())
    return max(abs(sum(p.currents.values())) for p in t.points) / big


@pytest.mark.parametrize("method", [backward_euler, bdf2])
def test_the_diode_currents_sum_to_zero_at_every_step(method):
    d0, m, s0 = dc(0.0)
    t = method(d0.with_bias(anode=0.4), s0, np.geomspace(1e-13, 1e-6, 30), m)
    assert kirchhoff(t) < 1e-10


def test_a_gate_carries_only_displacement_current_and_it_balances_the_body():
    from ddsim.api.devices import COARSE
    from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.transport import solve_bias_newton
    d0 = mos_cap(**COARSE["mos_cap"].parameters)
    m = TransportModels.for_device(d0)
    s0 = solve_bias_newton(d0, m, guess=solve_equilibrium(d0, quasi_fermi=frozen_quasi_fermi(d0)))
    t = backward_euler(d0.with_bias(gate=-0.5), s0, np.geomspace(1e-15, 1e-9, 20), m)
    assert t.complete
    assert abs(t.points[0].currents["gate"]) > 0.0
    assert kirchhoff(t) < 1e-10


def test_long_after_the_step_the_current_is_the_dc_current():
    from ddsim.extract.iv import terminal_currents
    d0, m, s0 = dc(0.0)
    d = d0.with_bias(anode=0.4)
    t = backward_euler(d, s0, np.geomspace(1e-12, 1e-1, 133), m)
    assert t.points[-1].currents["anode"] == pytest.approx(terminal_currents(d, solve_bias_ramped(d, m), m)["anode"], rel=1e-6)


def fundamental(samples, times, f):
    return 2.0 / len(times) * np.sum(np.asarray(samples) * np.exp(-2j * np.pi * f * np.asarray(times)))


@pytest.mark.parametrize("f", [3e5, 1.7e6, 1e7])
def test_a_one_millivolt_sinusoid_reproduces_the_ac_admittance(f):
    from ddsim.extract.ac import admittance
    d, m, s = dc(0.4)
    steps, periods = 1000, 3
    times = np.arange(1, periods * steps + 1) / (steps * f)

    def drive(t):
        return {"anode": 0.4 + 1e-3 * np.sin(2 * np.pi * f * t)}

    t = backward_euler(d, s, times, m, bias=drive)
    assert t.complete
    last = times[-steps:]
    i = [p.currents["anode"] for p in t.points[-steps:]]
    v = [drive(x)["anode"] for x in last]
    y = fundamental(i, last, f) / fundamental(v, last, f)
    want = admittance(d, s, f, m).y("anode", "anode")
    assert abs(y - want) < 5e-3 * abs(want)


def test_a_source_through_a_resistor_settles_where_the_load_line_meets_the_diode():
    from ddsim.extract.iv import terminal_currents
    from ddsim.extract.transient import Circuit
    d0, m, s0 = dc(0.0)
    c = Circuit(contact="anode", resistance=10.0, voltage=lambda t: 1.0)
    t = bdf2(d0, s0, np.geomspace(1e-12, 1e-1, 60), m, circuit=c)
    assert t.complete
    last = t.points[-1]
    v, i = last.voltages["anode"], last.currents["anode"]
    assert i == pytest.approx((1.0 - v) / 10.0, rel=1e-9)
    d = d0.with_bias(anode=v)
    assert terminal_currents(d, solve_bias_ramped(d, m), m)["anode"] == pytest.approx(i, rel=1e-6)
    assert 0.3 < v < 0.8


def test_a_current_source_settles_on_the_diode_voltage_that_carries_it():
    from ddsim.extract.iv import terminal_currents
    from ddsim.extract.transient import Circuit
    d0, m, s0 = dc(0.0)
    t = bdf2(d0, s0, np.geomspace(1e-12, 1e-1, 60), m, circuit=Circuit(contact="anode", current=lambda t: 0.01))
    assert t.complete
    last = t.points[-1]
    assert last.currents["anode"] == pytest.approx(0.01, rel=1e-9)
    d = d0.with_bias(anode=last.voltages["anode"])
    assert terminal_currents(d, solve_bias_ramped(d, m), m)["anode"] == pytest.approx(0.01, rel=1e-6)


def test_a_circuit_needs_exactly_one_source_on_a_contact_that_exists():
    from ddsim.extract.transient import Circuit
    d, m, s = dc(0.0)
    with pytest.raises(ValueError, match="source"):
        bdf2(d, s, [1e-9], m, circuit=Circuit(contact="anode"))
    with pytest.raises(ValueError, match="source"):
        bdf2(d, s, [1e-9], m, circuit=Circuit(contact="anode", voltage=lambda t: 0.1, current=lambda t: 0.1))
    with pytest.raises(KeyError, match="gate"):
        bdf2(d, s, [1e-9], m, circuit=Circuit(contact="gate", voltage=lambda t: 0.1))


def recovered(tau, V0, ratio, steps=150):
    import math
    from scipy.special import erfinv
    from ddsim.core import constants as C
    from ddsim.extract.iv import terminal_currents
    from ddsim.extract.transient import Circuit, storage_time
    from ddsim.physics.recombination import SRHRecombination
    VT = C.V_T(C.T_ROOM)
    Ln, Lp = math.sqrt(C.MU_N_300 * VT * tau), math.sqrt(C.MU_P_300 * VT * tau)
    d = pn_diode(Na=1e19, Nd=1e18, length=5 * Ln + 5 * Lp, junction=5 * Ln, n_nodes=500, h_min=2e-7, anode_voltage=V0)
    ni = d.material.n_i / d.scale.C_0
    m = TransportModels.for_device(d, recombination=SRHRecombination(tau_n=tau / d.scale.t_0, tau_p=tau / d.scale.t_0, ni2=ni * ni, n1=ni, p1=ni))
    s = solve_bias_ramped(d, m)
    R = (100.0 + V0) / (ratio * terminal_currents(d, s, m)["anode"])
    kingston = tau * erfinv(1 / (1 + ratio)) ** 2
    t = bdf2(d, s, np.arange(1, steps + 1) * 1.3 * kingston / steps, m, circuit=Circuit(contact="anode", resistance=R, voltage=lambda t: -100.0))
    assert t.complete
    return storage_time(t, "anode"), kingston


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_storage_time_is_kingstons_where_the_junction_capacitance_is_negligible(ratio):
    ts, kingston = recovered(1e-2, 0.836, ratio)
    assert ts == pytest.approx(kingston, rel=0.05)


def test_where_the_junction_capacitance_matters_storage_follows_the_depletion_charge():
    from tests.reference.recovery import storage_time as reference
    ts, kingston = recovered(1e-3, 0.776, 1.0)
    assert ts > 1.1 * kingston
    assert ts == pytest.approx(reference(1e19, 1e18, 1e-3, 0.776, 100.0, 1.0), rel=0.02)
