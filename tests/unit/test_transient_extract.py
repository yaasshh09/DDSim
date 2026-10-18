from __future__ import annotations
import pytest
from ddsim.extract.transient import Transient, TransientPoint, recovery_time, storage_time


def waveform(rows):
    return Transient(points=tuple(TransientPoint(time=t, state=None, currents={"anode": i}, voltages={"anode": v}) for t, v, i in rows), complete=True)


RECOVERY = waveform([
    (1.0, 0.7, -1.0),
    (2.0, 0.6, -1.0),
    (3.0, 0.2, -1.0),
    (4.0, -0.2, -0.6),
    (5.0, -5.0, -0.3),
    (6.0, -9.0, -0.05),
])


def test_storage_ends_where_the_voltage_crosses_zero_interpolated():
    assert storage_time(RECOVERY, "anode") == pytest.approx(3.5)


def test_recovery_ends_where_the_current_falls_to_the_fraction_interpolated():
    assert recovery_time(RECOVERY, "anode") == pytest.approx(5.0 + 0.2 / 0.25)
    assert recovery_time(RECOVERY, "anode", fraction=0.5) == pytest.approx(4.0 + 0.1 / 0.3)


def test_a_voltage_that_never_crosses_zero_has_no_storage_time():
    with pytest.raises(ValueError, match="never"):
        storage_time(waveform([(1.0, 0.7, -1.0), (2.0, 0.5, -1.0)]), "anode")


def test_a_current_that_never_falls_far_enough_has_no_recovery_time():
    with pytest.raises(ValueError, match="never"):
        recovery_time(RECOVERY, "anode", fraction=0.01)


def test_an_unknown_contact_is_named():
    with pytest.raises(KeyError, match="gate"):
        storage_time(RECOVERY, "gate")


TIMES = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]


def small():
    from ddsim.device.pn_diode import pn_diode
    return pn_diode(n_nodes=61, h_min=5e-7)


def dc_current(v):
    from ddsim.device.transport import TransportModels, solve_bias_ramped
    from ddsim.extract.iv import terminal_currents
    d = small().with_bias(anode=v)
    m = TransportModels.for_device(d)
    return terminal_currents(d, solve_bias_ramped(d, m), m)["anode"]


def test_a_step_sweep_sends_one_frame_per_accepted_step_and_lands_on_every_sample():
    from ddsim.extract.transient import TransientFrame, transient_sweep
    got = []
    c = transient_sweep(small(), "anode", [0.0, 0.3], TIMES, on_frame=got.append)
    steps = [f for f in got if isinstance(f, TransientFrame)]
    assert c.complete
    assert c.contact == "anode"
    assert len(steps) == c.steps > len(TIMES)
    assert [f.step for f in steps] == list(range(1, c.steps + 1))
    at = [f.time for f in steps]
    assert all(b > a for a, b in zip(at[:-1], at[1:], strict=True))
    assert set(TIMES) <= set(at)
    assert all(f.dt > 0.0 for f in steps)
    assert [p.time for p in c.points] == TIMES
    assert steps[-1].currents == c.points[-1].currents
    assert steps[-1].voltages == c.points[-1].voltages


def test_a_step_to_the_same_voltage_stays_at_the_dc_state_it_started_from():
    from ddsim.extract.transient import transient_sweep
    c = transient_sweep(small(), "anode", [0.3, 0.3], TIMES)
    want = dc_current(0.3)
    assert want > 0.0
    for p in c.points:
        assert p.currents["anode"] == pytest.approx(want, rel=1e-6)


def test_a_step_ends_on_the_new_voltage_and_its_dc_current():
    from ddsim.extract.transient import transient_sweep
    c = transient_sweep(small(), "anode", [0.0, 0.3], [1e-12, 1e-9, 1e-6, 1e-3])
    assert c.points[-1].voltages["anode"] == 0.3
    assert c.points[-1].currents["anode"] == pytest.approx(dc_current(0.3), rel=1e-6)


def test_a_step_through_a_resistor_settles_on_the_load_line():
    from ddsim.extract.transient import transient_sweep
    c = transient_sweep(small(), "anode", [0.0, 1.0], [1e-12, 1e-9, 1e-6, 1e-3, 1e-1], resistance=10.0, rtol=1e-2)
    last = c.points[-1]
    v, i = last.voltages["anode"], last.currents["anode"]
    assert 0.3 < v < 0.8
    assert i == pytest.approx((1.0 - v) / 10.0, rel=1e-9)


@pytest.mark.parametrize("voltages", [[0.3], [0.0, 0.3, 0.0]])
def test_a_step_sweep_takes_exactly_a_before_and_an_after_voltage(voltages):
    from ddsim.extract.transient import transient_sweep
    with pytest.raises(ValueError, match="two voltages"):
        transient_sweep(small(), "anode", voltages, TIMES)


@pytest.mark.parametrize("times", [[], [0.0, 1e-9], [1e-9, 1e-9], [1e-9, 1e-10], [1e-9, float("inf")], [float("nan")]])
def test_times_that_are_not_a_rising_list_after_zero_are_refused_before_any_solve(times):
    from ddsim.extract.transient import check_times
    with pytest.raises(ValueError, match="time"):
        check_times(times)


@pytest.mark.parametrize("rtol", [0.0, -1e-3, float("nan")])
def test_a_tolerance_that_is_not_positive_is_refused(rtol):
    from ddsim.extract.transient import transient_sweep
    with pytest.raises(ValueError, match="rtol"):
        transient_sweep(small(), "anode", [0.0, 0.3], TIMES, rtol=rtol)


def test_a_step_on_a_contact_the_device_lacks_is_named():
    from ddsim.extract.transient import transient_sweep
    with pytest.raises(KeyError, match="gate"):
        transient_sweep(small(), "gate", [0.0, 0.3], TIMES)
