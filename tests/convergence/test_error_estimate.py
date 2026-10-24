from __future__ import annotations
import csv, importlib.util, math, sys
from functools import cache
from pathlib import Path
from types import ModuleType
import numpy as np
import pytest
from ddsim.device.builder import Device, build_device
from ddsim.device.doping import Uniform, abrupt_junction
from ddsim.device.pn_diode import pn_diode
from ddsim.device.state import DeviceState
from ddsim.device.transport import solve_bias_newton, solve_bias_ramped
from ddsim.discretize.boundary import OhmicContact
from ddsim.device.mos_cap import mos_cap
from ddsim.extract.error import ErrorEstimate, adapt, capacitance_error, current_error, prolonged_state, refined, split, split_cells
from ddsim.extract.iv import terminal_currents
from ddsim.mesh.mesh1d import uniform_mesh_1d
from tests.regression.devsim_gen import parameters as P

ROOT = Path(__file__).resolve().parents[2]

DIODES = ((1, "diode_1e16_1e16"), (2, 'diode_1e18_1e16'), (3, "diode_1e20_1e15"))

EFFECTIVITY = (0.5, 2.0)


def _tool() -> ModuleType:
    d = importlib.util.spec_from_file_location("ddsim_tools_scoreboard", ROOT / "tools" / "scoreboard.py")
    assert d is not None and d.loader is not None
    m = importlib.util.module_from_spec(d)
    sys.modules[d.name] = m
    d.loader.exec_module(m)
    return m


@cache
def richardson(name: str) -> tuple[float, float, float]:
    tool = _tool()
    with (ROOT / "data" / "scoreboard" / "ddsim_accuracy.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(x for x in f if not x.startswith("#")) if r["name"] == name]
    k = [(float(r["refine"]), int(r["nodes"]), float(r["value"])) for r in rows]
    fit = tool.richardson(k, 1)
    assert fit.limit is not None, f"{name} has no Richardson limit in ddsim_accuracy.csv"
    return min(k)[2], fit.limit, fit.order


def bar(n_nodes: int = 41) -> Device:
    return build_device(mesh=uniform_mesh_1d(1e-4, n_nodes), doping=Uniform(1e16), contacts=(OhmicContact("left", 0, 0.01), OhmicContact('right', n_nodes - 1, 0.0)),)


def diode(name: str, voltage: float = 0.6) -> Device:
    b = P.BY_NAME[name]
    return pn_diode(Na=b.Na, Nd=b.Nd, length=b.length, junction=b.junction, n_nodes=b.n_nodes, h_min=b.h_min, anode_voltage=voltage)


def coarse(name: str, n_nodes: int = 21) -> Device:
    b = P.BY_NAME[name]
    return build_device(mesh=uniform_mesh_1d(b.length, n_nodes), doping=abrupt_junction(Na=b.Na, Nd=b.Nd, position=b.junction), contacts=(OhmicContact("anode", 0, 0.6), OhmicContact("cathode", n_nodes - 1, 0.0)),)


@cache
def solved(name: str) -> tuple[Device, DeviceState]:
    d = diode(name)
    s = solve_bias_ramped(d)
    assert s.newton is not None and s.newton.converged
    return d, s


@cache
def estimated(name: str) -> ErrorEstimate:
    d, s = solved(name)
    return current_error(d, s, "anode")


def test_the_estimate_is_zero_where_the_mesh_already_solves_the_problem() -> None:
    d = bar()
    s = solve_bias_newton(d)
    f = refined(d, 2)
    t = solve_bias_newton(f, guess=prolonged_state(d, f, s))
    a, b = terminal_currents(d, s)["left"], terminal_currents(f, t)['left']
    assert abs(b - a) <= 1e-10 * abs(a), "the premise is gone: refining the bar moved its current"
    e = current_error(d, s, "left")
    assert e.value == pytest.approx(a, rel=1e-12, abs=0)
    assert abs(e.estimate) <= 1e-10 * abs(a)
    assert np.max(np.abs(e.indicator)) <= 1e-10 * abs(a)


def test_the_value_is_the_current_the_benchmark_scores() -> None:
    for _, name in DIODES:
        a, _, _ = richardson(name)
        assert estimated(name).value == pytest.approx(a, rel=1e-6, abs=0)


@pytest.mark.parametrize("number,name", DIODES, ids=[n for _, n in DIODES])
def test_effectivity_against_richardson(number: int, name: str) -> None:
    a, limit, _ = richardson(name)
    e = estimated(name)
    k = e.estimate / (limit - a)
    assert EFFECTIVITY[0] <= k <= EFFECTIVITY[1], f"benchmark {number}: estimated {e.estimate:.4e}, Richardson says {limit - a:.4e}, effectivity {k:.3f}"


def test_the_indicator_adds_up_to_the_estimate() -> None:
    for _, name in DIODES:
        e = estimated(name)
        assert float(np.sum(e.indicator)) == pytest.approx(e.estimate, rel=1e-9, abs=0)
        assert e.indicator.shape == (solved(name)[0].mesh.n_nodes - 1,)


def test_the_estimate_shrinks_at_the_observed_order() -> None:
    name = DIODES[0][1]
    d, s = solved(name)
    f = refined(d, 2)
    t = solve_bias_newton(f, guess=prolonged_state(d, f, s))
    assert t.newton is not None and t.newton.converged
    a, b = estimated(name).estimate, current_error(f, t, "anode").estimate
    _, _, p = richardson(name)
    assert math.log2(a / b) == pytest.approx(p, abs=0.25)


def test_refining_the_flagged_cell_moves_the_goal_the_way_the_indicator_says() -> None:
    name = DIODES[0][1]
    d, s = solved(name)
    e = estimated(name)
    w = int(np.argmax(np.abs(e.indicator)))
    f = split_cells(d, [w])
    t = solve_bias_newton(f, guess=prolonged_state(d, f, s))
    assert t.newton is not None and t.newton.converged
    moved = terminal_currents(f, t)["anode"] - e.value
    assert moved != 0.0, "refining the cell the indicator flags most left the current exactly where it was"
    assert np.sign(moved) == np.sign(e.indicator[w])
    assert abs(moved) >= 0.25 * abs(e.indicator[w])


def test_refining_a_cell_the_indicator_calls_clean_moves_nothing() -> None:
    d = bar()
    s = solve_bias_newton(d)
    e = current_error(d, s, "left")
    f = split_cells(d, [d.mesh.n_nodes // 2])
    t = solve_bias_newton(f, guess=prolonged_state(d, f, s))
    assert abs(terminal_currents(f, t)['left'] - e.value) <= 1e-10 * abs(e.value)


def test_adaptive_refinement_beats_the_hand_graded_mesh() -> None:
    name = DIODES[0][1]
    a, limit, _ = richardson(name)
    target = abs(limit - a) / abs(a)
    b = P.BY_NAME[name]
    r = adapt(coarse(name), "anode", rtol=target, budget=b.n_nodes)
    assert r.converged, r.message
    assert r.device.mesh.n_nodes < b.n_nodes
    assert abs(r.estimate.estimate) <= target * abs(r.estimate.value)
    assert abs(r.estimate.value - limit) <= 2.0 * target * abs(limit), "the estimate says the adaptive mesh is there and Richardson says it isn't"


def test_the_adaptive_loop_says_when_it_runs_out_of_nodes() -> None:
    r = adapt(coarse(DIODES[0][1]), "anode", rtol=1e-9, budget=40)
    assert not r.converged
    assert r.device.mesh.n_nodes <= 40
    assert "budget" in r.message
    assert abs(r.estimate.estimate) > 1e-9 * abs(r.estimate.value)


MOS = ((4, "mos_cap_5nm"), (5, 'mos_cap_20nm'))


@cache
def capacitor(name: str) -> ErrorEstimate:
    z = {x.name: x for x in P.MOS_BENCHMARKS}[name]
    d = mos_cap(substrate_doping=z.substrate_doping, t_ox=z.t_ox, t_si=z.t_si, n_silicon=z.n_silicon, n_oxide=z.n_oxide, h_min=z.h_min, work_function=z.work_function)
    return capacitance_error(d, "gate")


@pytest.mark.parametrize("number,name", MOS, ids=[n for _, n in MOS])
def test_capacitance_effectivity_against_richardson(number: int, name: str) -> None:
    a, limit, _ = richardson(name)
    e = capacitor(name)
    assert e.value == pytest.approx(a, rel=1e-9, abs=0)
    k = e.estimate / (limit - a)
    assert EFFECTIVITY[0] <= k <= EFFECTIVITY[1], f"benchmark {number}: estimated {e.estimate:.4e}, Richardson says {limit - a:.4e}, effectivity {k:.3f}"
    assert float(np.sum(e.indicator)) == pytest.approx(e.estimate, rel=1e-9, abs=0)


def coarse_capacitor() -> Device:
    z = {x.name: x for x in P.MOS_BENCHMARKS}[MOS[0][1]]
    return mos_cap(substrate_doping=z.substrate_doping, t_ox=z.t_ox, t_si=z.t_si, nx=2, n_silicon=11, n_oxide=3, h_min=1e-5, work_function=z.work_function)


def test_on_the_capacitor_splitting_the_flagged_row_moves_the_capacitance() -> None:
    d = coarse_capacitor()
    e = capacitance_error(d, "gate", k=1, ky=2)
    for _ in range(5):
        w = int(np.argmax(np.abs(e.indicator)))
        assert d.mesh.y_axis.x[w + 1] <= P.MOS_BENCHMARKS[0].t_si, "the flagged row is oxide, where the potential is already a straight line"
        d = split(d, [], [w])
        after = capacitance_error(d, "gate", k=1, ky=2)
        assert after.value != e.value, "splitting the row the indicator flags most left the capacitance exactly where it was"
        assert np.sign(after.value - e.value) == np.sign(e.indicator[w])
        e = after


def test_splitting_an_oxide_row_moves_nothing_and_its_indicator_says_so() -> None:
    d = coarse_capacitor()
    e = capacitance_error(d, "gate", k=1, ky=2)
    for w in range(10, d.mesh.y_axis.x.size - 1):
        assert d.mesh.y_axis.x[w] >= P.MOS_BENCHMARKS[0].t_si
        assert abs(e.indicator[w]) <= 1e-9 * abs(e.value)
        assert capacitance_error(split(d, [], [w]), "gate", k=1, ky=2).value == pytest.approx(e.value, rel=1e-12, abs=0)
