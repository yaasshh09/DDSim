from __future__ import annotations
import numpy as np; import pytest
from ddsim.core.field import Location, ScalingState
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias, solve_bias_ramped
from ddsim.extract.iv import (IVCurve, continuity_residuals, current_densities, iv_sweep, resolved_currents, terminal_currents, total_current,)
from ddsim.physics.recombination import NoRecombination
MICRON = 1e-4


def diode(**overrides: float):
    z:dict = {
        'Na': 1e16,
        "Nd": 1e16,
        "length": 12*MICRON,
        'junction': 6 * MICRON,
        "n_nodes": 201,
        "h_min": 5e-7,
    }
    z.update(overrides)
    return pn_diode(**z)


def test_current_densities_come_back_physical_and_on_edges() -> None:
    d = diode()

    x = solve_bias(d)
    b2, val2 = current_densities(d, x)

    for u in (b2, val2):
        assert u.location is Location.EDGE
        assert u.scaling is ScalingState.PHYSICAL
        assert u.unit == 'A/cm^2'
        assert u.size == d.mesh.n_edges


def test_equilibrium_carries_no_current() -> None:
    w = diode()
    lst = solve_bias(w)
    tmp2, c = current_densities(w, lst)

    assert np.max(np.abs(tmp2.data + c.data)) < 1e-10

    assert abs(total_current(w, lst)) < 1e-10


def test_forward_bias_drives_current_into_the_anode() -> None:
    it = diode().with_bias(anode=0.4)
    i = solve_bias(it)
    assert i.gummel is not None and i.gummel.converged
    e = terminal_currents(it, i)
    assert e["anode"] > 0.0
def test_reverse_bias_gives_a_small_negative_current() -> None:
    vv = diode().with_bias(anode=-1.0)
    res2 = solve_bias(vv)

    g = terminal_currents(vv, res2)["anode"]
    cur = terminal_currents(
        diode().with_bias(anode=0.4), solve_bias(diode().with_bias(anode=0.4))
    )["anode"]

    assert g < 0.0
    assert abs(g) < 1e-3*cur


def test_terminal_currents_sum_to_zero() -> None:
    k = diode().with_bias(anode=0.4)
    j = solve_bias(k)

    res2 = terminal_currents(k, j)
    vv = max(abs(out) for out in res2.values())
    assert abs(sum(res2.values())) < 1e-8 * vv
def test_total_current_is_the_anode_current() -> None:
    h = diode().with_bias(anode=0.3)
    w = solve_bias(h)

    np.testing.assert_allclose(
        total_current(h, w),
        terminal_currents(h, w)["anode"],
        rtol=1e-12,
    )
def test_current_rises_steeply_with_forward_bias() -> None:
    g = diode().with_bias(anode=0.2)
    m2 = diode().with_bias(anode=0.3)

    y = total_current(m2, solve_bias(m2)) / total_current(
        g, solve_bias(g)
    )
    assert 10.0 < y < 100.0


def test_recombination_drops_out_of_the_terminal_current() -> None:
    x = diode().with_bias(anode=0.3)
    m = TransportModels.for_device(x, recombination=NoRecombination())
    j = solve_bias(x)

    assert (
        terminal_currents(x, j, models=m)['anode']
        == terminal_currents(x, j)["anode"]
    )

    res2 = x.contacts[0].node
    assert (
        continuity_residuals(x, j, m)[0][res2]
        == continuity_residuals(x, j)[0][res2]
    )

    e = TransportModels.for_device(x)
    d = np.asarray(e.recombination.rate(j.n.data, j.p.data))
    assert d[res2] == 0.0

def test_a_sweep_lands_on_every_requested_voltage() -> None:
    stuff = [0.0, 0.1, 0.2, 0.3]
    f = iv_sweep(diode(), 'anode', stuff)
    assert f.complete
    np.testing.assert_allclose(f.voltage, stuff, atol=0.0)

def test_a_sweep_current_increases_with_forward_bias() -> None:
    s2 = iv_sweep(diode(), "anode", [0.1, 0.2, 0.3, 0.4])

    assert np.all(np.diff(s2.current) > 0.0)


def test_a_sweep_can_run_into_reverse_bias() -> None:
    v2 = iv_sweep(diode(), "anode", [-0.5, -0.25, 0.0, 0.25])

    assert v2.complete
    assert v2.current[0] < 0.0
    assert v2.current[-1] > 0.0


def test_a_sweep_carries_the_state_at_every_point() -> None:
    t = iv_sweep(diode(), "anode", [0.0, 0.2])

    assert len(t.points) == 2
    for it, idx in zip(t.points,
         [0.0,
       0.2],
                    strict=True):
        assert it.state.gummel is not None
        assert it.state.gummel.converged
        assert it.voltage == idx


def test_a_sweep_stops_and_says_where_when_it_stalls() -> None:
    y = iv_sweep(
        diode(n_nodes=41, h_min=2e-6),
        'anode',
        [0.2, 0.4, 5.0],
        step=0.05,
        min_step=0.01,
        max_iterations=8,
    )
    assert not y.complete
    assert y.message
    assert len(y.points) >= 1

    assert y.voltage[-1] < 5.0

def test_an_unknown_contact_is_rejected_before_any_solving() -> None:
    with pytest.raises(KeyError, match='gate'):
        iv_sweep(diode(), "gate", [0.1])

def test_curve_repr_reports_the_range() -> None:
    d2 = iv_sweep(diode(), "anode", [0.0, 0.2])

    assert "anode" in repr(d2)
    assert "complete" in repr(d2)

def test_an_empty_curve_still_has_a_repr() -> None:
    dd = IVCurve(contact="anode", points=(), complete=False, message='stalled')

    assert "empty" in repr(dd)
def test_a_sweep_with_no_starting_guess_raises() -> None:
    with pytest.raises(RuntimeError,
             match="could not be started"):
        iv_sweep(diode(), "anode", [5.1], start=5.0, max_iterations=1)


def test_a_sweep_that_cannot_start_cold_ramps_its_held_bias_in() -> None:
    m = iv_sweep(pn_diode(cathode_voltage=20.0), "anode", [0.0, 0.1])

    assert m.complete
    zz, k = m.current
    assert zz != 0.0

    assert abs(k - zz)/abs(zz) < 0.05
def test_a_sweep_whose_first_point_stalls_raises() -> None:
    with pytest.raises(RuntimeError, match='did not converge'):
        iv_sweep(
            diode(), 'anode', [0.1], max_iterations=1, update_tol=1e-30
        )


def test_a_sweep_whose_cold_start_and_ramp_both_stall_raises() -> None:
    with pytest.raises(RuntimeError, match='did not converge'):
        iv_sweep(
            diode(), "anode", [0.2], start=0.2, max_iterations=1, update_tol=1e-30
        )


def test_a_floor_on_the_continuation_step_bounds_the_cost_of_a_stall() -> None:
    rr = iv_sweep(
        diode(n_nodes=41, h_min=2e-6),
        "anode",
        [0.2, 5.0],
        step=0.05,
        min_step=0.025,
        max_iterations=8,
    )
    s = iv_sweep(diode(n_nodes=41, h_min=2e-6), 'anode', [0.2, 5.0], step=0.05, min_step=0.0005, max_iterations=8,)

    assert not rr.complete and not s.complete
    assert rr.message and s.message

    assert rr.voltage[-1] <= s.voltage[-1]


def test_resolved_currents_are_the_contact_currents_wherever_those_can_be_read() -> None:
    d = diode()
    m = TransportModels.for_device(d)
    for v in (0.3, -1.0):
        b = d.with_bias(anode=v)
        st = iv_sweep(d, "anode", [v], models=m).points[-1].state
        raw = terminal_currents(b, st, m); res = resolved_currents(b, st, m)
        for c in raw:
            assert res[c] == pytest.approx(raw[c], rel=1e-6), (v, c)


def test_a_heavily_doped_reverse_diode_reads_one_current_whichever_solver_got_there() -> None:
    d = pn_diode(Na=1.329e18, Nd=3.299e18, length=3.644e-05, junction=2.291e-05).with_bias(anode=-4.236)
    m = TransportModels.for_device(d)
    newt = solve_bias_ramped(d, models=m)
    gum = solve_bias(d, models=m, guess=newt)
    a, b = terminal_currents(d, newt, m)["anode"], terminal_currents(d, gum, m)["anode"]
    assert abs(b / a) > 100.0
    x = resolved_currents(d, newt, m); y = resolved_currents(d, gum, m)
    assert x["anode"] == pytest.approx(y["anode"], rel=1e-6)
    assert abs(x["anode"] + x["cathode"]) <= 1e-12 * abs(x["anode"])
    rc = m.recombination; sc = d.scale
    g = 1.602176634e-19 * np.trapezoid(np.asarray(rc.rate(gum.n.data, gum.p.data)) * sc.C_0 / sc.t_0, np.asarray(d.mesh.x))
    assert x["anode"] == pytest.approx(g, rel=0.02)

