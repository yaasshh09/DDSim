from __future__ import annotations
from functools import cache
import numpy as np
import pytest
from ddsim.device.bjt import npn_1d
from ddsim.device.transport import TransportModels
from ddsim.extract.bjt import GummelPlot, early_voltage, gummel_plot, ideality, peak_ft

V_BE = [round(0.2 + 0.05 * k, 2) for k in range(17)]


@cache
def plot() -> GummelPlot:
    d = npn_1d()
    g = gummel_plot(d, V_BE, models=TransportModels.for_device(d, field_dependent=True))
    assert g.complete, g.message
    return g


def test_the_gummel_plot_reads_the_currents_it_was_asked_for() -> None:
    g = plot()
    assert list(g.v_be) == V_BE
    assert np.all(g.collector > 0.0) and np.all(g.base > 0.0)
    assert np.all(np.diff(g.collector) > 0.0)


def test_beta_falls_at_both_ends() -> None:
    b = plot().beta
    k = int(np.argmax(b))
    assert 0 < k < b.size - 1
    assert b[0] < 0.9 * b[k], "beta does not fall at low current, so recombination in the emitter junction is missing"
    assert b[-1] < 0.5 * b[k], "beta does not fall at high current, so high injection is missing"


def test_at_low_current_the_base_current_is_less_ideal_than_the_collector() -> None:
    g = plot()
    w = (0.2, 0.3)
    assert ideality(g.v_be, g.collector, w) == pytest.approx(1.0, abs=0.02)
    assert ideality(g.v_be, g.base, w) > 1.15


def test_f_t_peaks_and_then_falls() -> None:
    g = plot()
    f, i = peak_ft(g)
    k = int(np.argmax(g.f_T))
    assert 0 < k < len(V_BE) - 1
    assert g.f_T[-1] < 0.5 * f
    assert i == g.collector[k]


def test_the_early_voltage_is_positive_and_finite() -> None:
    d = npn_1d()
    v = early_voltage(d, 0.6, [1.0, 1.5, 2.0])
    assert 1.0 < v < 1e3


@cache
def output() -> tuple[list[float], object]:
    from ddsim.extract.bjt import output_curve
    d = npn_1d()
    v = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5]
    return v, output_curve(d, 1e-3, v, models=TransportModels.for_device(d, field_dependent=True))


def test_an_output_curve_holds_the_base_current_it_was_given() -> None:
    _, c = output()
    assert c.complete, c.message
    assert np.allclose(c.base, 1e-3, rtol=1e-6, atol=0)


def test_in_the_active_region_the_collector_current_is_beta_times_the_base() -> None:
    _, c = output()
    g = plot()
    k = int(np.argmin(abs(c.v_ce - 1.5)))
    want = float(np.interp(np.log(c.collector[k]), np.log(g.collector), g.beta))
    assert c.collector[k] / c.base[k] == pytest.approx(want, rel=0.05)


def test_the_collector_current_collapses_in_saturation() -> None:
    _, c = output()
    assert c.collector[0] < 0.5 * c.collector[-1]
    assert np.all(np.diff(c.collector) > 0.0)
