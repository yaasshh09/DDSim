from __future__ import annotations
import numpy as np, pytest
from ddsim.api.devices import COARSE
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import DRAIN, GATE, nmos
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract import cv
from ddsim.extract.ac import ACFrame, ac_sweep, admittance


@pytest.fixture(scope="module")
def cap():
    d = mos_cap(**COARSE["mos_cap"].parameters)
    frames = []
    return d, ac_sweep(d, "gate", [-1.0, 1.0], [1.0, 1e6], on_frame=frames.append), frames


def test_every_bias_point_gets_every_frequency(cap):
    _, c, _ = cap
    assert c.complete
    assert [p.voltage for p in c.points] == [-1.0, 1.0]
    assert c.frequencies == (1.0, 1e6)
    for p in c.points:
        assert [a.frequency for a in p.admittances] == [1.0, 1e6]


def test_in_accumulation_c_is_the_quasi_static_c_per_area(cap):
    d, c, _ = cap
    e = d.with_bias(gate=-1.0)
    q = frozen_quasi_fermi(e)
    want = cv.small_signal_capacitance(e, solve_equilibrium(e, quasi_fermi=q), "gate", cv.Response.LOW_FREQUENCY, quasi_fermi=q)
    a = c.points[0].admittances[1]
    g = a.contacts.index("gate")
    assert a.C[g, g] == pytest.approx(want, rel=1e-5, abs=0)


def test_the_two_orderings_agree_where_the_solve_is_well_posed(cap):
    _, c, _ = cap
    assert c.points[0].C_spread[0][1, 1] < 1e-6
    assert c.points[1].C_spread[1][1, 1] < 1e-6


def test_a_floating_inversion_layer_at_one_hertz_is_flagged_as_roundoff(cap):
    _, c, _ = cap
    assert c.points[1].C_spread[0][1, 1] > 1e-3


def test_a_conductance_that_is_only_roundoff_is_flagged_even_where_c_is_clean(cap):
    _, c, _ = cap
    a = c.points[0].admittances[0]
    b, g = a.contacts.index("body"), a.contacts.index("gate")
    assert c.points[0].C_spread[0][b, g] < 1e-6
    assert c.points[0].G_spread[0][b, g] > 1e-1


def test_a_frame_goes_out_per_bias_and_frequency(cap):
    _, c, frames = cap
    got = [f for f in frames if isinstance(f, ACFrame)]
    assert [(f.index, f.voltage, f.frequency) for f in got] == [(0, -1.0, 1.0), (0, -1.0, 1e6), (1, 1.0, 1.0), (1, 1.0, 1e6)]
    a = c.points[1].admittances[1]
    g = a.contacts.index("gate")
    assert got[-1].capacitance == a.C[g, g]
    assert got[-1].conductance == a.G[g, g]
    assert got[-1].G_spread == c.points[1].G_spread[1][g, g]
    assert got[-1].C_spread == c.points[1].C_spread[1][g, g]


def test_a_drain_bias_the_equilibrium_start_cannot_reach_is_ramped_to():
    d = nmos(**COARSE["nmos"].parameters, drain_voltage=1.0)
    m = TransportModels.for_device(d)
    c = ac_sweep(d, GATE, [1.0], [1e9], models=m)
    assert c.complete
    e = d.with_bias(gate=1.0)
    want = admittance(e, solve_bias_ramped(e, m), 1e9, m, width=d.mesh.x_axis.length)
    assert c.points[0].admittances[0].y(DRAIN, GATE) == pytest.approx(want.y(DRAIN, GATE), rel=1e-6)


def test_no_frequencies_is_refused():
    with pytest.raises(ValueError, match="frequenc"):
        ac_sweep(mos_cap(), "gate", [0.0], [])


@pytest.mark.parametrize("f", [-1.0, float("nan"), float("inf")])
def test_a_frequency_that_is_not_a_frequency_is_refused(f):
    with pytest.raises(ValueError, match="frequenc"):
        ac_sweep(mos_cap(), "gate", [0.0], [1e3, f])


def test_a_contact_the_device_does_not_have_is_refused():
    with pytest.raises(KeyError, match="drain"):
        ac_sweep(mos_cap(), "drain", [0.0], [1e3])


def test_the_orderings_really_differ():
    d = mos_cap(**COARSE["mos_cap"].parameters, gate_voltage=1.0)
    m = TransportModels.for_device(d)
    s = solve_bias_newton(d, m, guess=solve_equilibrium(d, quasi_fermi=frozen_quasi_fermi(d)))
    a = admittance(d, s, 1.0, m, ordering="COLAMD")
    b = admittance(d, s, 1.0, m, ordering="MMD_AT_PLUS_A")
    assert not np.array_equal(a.Y, b.Y)
