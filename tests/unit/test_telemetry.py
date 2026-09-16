"""Telemetry from the public sweeps down to the solvers, phases/PHASE-7.md.

Three hooks already exist one level down: newton_solve reports an iteration,
continue_to reports an attempt, gummel_solve reports a cycle. Nothing reached
them, because the browser does not call a solver. It calls iv_sweep,
gate_sweep or cv_sweep and everything below that is private.

So one optional argument crosses the layers between: `on_frame`. It is a
single pipe rather than three, because the three frame types are already
distinguishable by their own type and a reader that has to be handed three
callbacks has to implement three. Every frame carries scalars only, which is
the inertness argument NewtonIteration states and the reason IVPoint is not
itself a frame: a point carries the state the next point continues from, and
handing a callback a reference to that is handing it the next solve.

What is asserted here is the wiring. That the wiring changes no number is the
separate claim in tests/invariant/test_telemetry_inert.py, and it is the one
phases/PHASE-7.md lists as an acceptance criterion.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from ddsim.api.jobs import JobRegistry, JobStatus
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    _reported_by_family,
    solve_bias,
    solve_bias_hybrid,
    solve_bias_newton,
    solve_bias_ramped,
)
from ddsim.discretize.assembly import SparseAssembly
from ddsim.extract.cv import CVFrame, cv_sweep
from ddsim.extract.iv import IVFrame, gate_sweep, iv_sweep
from ddsim.solve.continuation import ContinuationEvent
from ddsim.solve.gummel import GummelIteration
from ddsim.solve.newton import NewtonIteration, newton_solve

MICRON = 1e-4
"""One micron [cm]."""

COARSE_FET = {
    "n_contact": 4,
    "n_sd": 10,
    "n_channel": 12,
    "n_silicon": 29,
    "n_oxide": 4,
    "h_min_x": 5e-7,
    "h_min_y": 1e-7,
    "drain_voltage": 0.05,
}
"""The same MOSFET tests/unit/test_surface_mobility.py uses, small enough to
solve several times in a unit test."""


class ReaderStoppedError(Exception):
    """Whatever a caller raises inside its own callback."""


def collect() -> tuple[list[Any], Callable[[object], None]]:
    """A frame list and the callback that fills it."""
    frames: list[Any] = []
    return frames, frames.append


def diode(**overrides: float):
    """A coarse version of the Phase 2 test diode."""
    settings: dict = {
        "Na": 1e16,
        "Nd": 1e16,
        "length": 12 * MICRON,
        "junction": 6 * MICRON,
        "n_nodes": 61,
        "h_min": 5e-7,
    }
    settings.update(overrides)
    return pn_diode(**settings)


def of_type(frames: list[Any], kind: type) -> list[Any]:
    return [frame for frame in frames if isinstance(frame, kind)]


# --------------------------------------------------------------- one solver


def test_a_gummel_solve_reports_its_cycles() -> None:
    """solve_bias is the Gummel path, so the frames from it are cycles."""
    frames, send = collect()

    state = solve_bias(diode(), on_frame=send)

    assert state.gummel is not None
    cycles = of_type(frames, GummelIteration)
    assert [frame.update for frame in cycles] == state.gummel.update_history


def test_a_newton_solve_reports_its_iterations() -> None:
    """Bit for bit against the history the solve was judged on. A telemetry
    number that disagrees with that one is worse than no telemetry."""
    frames, send = collect()

    state = solve_bias_newton(diode(), on_frame=send)

    assert state.newton is not None
    iterations = of_type(frames, NewtonIteration)
    assert [frame.residual for frame in iterations] == state.newton.residual_history


def test_a_coupled_newton_iteration_names_each_equation_family() -> None:
    """phases/PHASE-7.md item 5: residual and update per equation family, so
    a stalled solve can say whether Poisson or a continuity equation stalled.
    The split has to be the scalar the solve was judged on, taken apart: its
    largest family is that scalar exactly, not approximately."""
    frames, send = collect()

    solve_bias_newton(diode(anode_voltage=0.4), on_frame=send)

    iterations = of_type(frames, NewtonIteration)
    assert len(iterations) > 2
    for frame in iterations:
        assert frame.residual_by_family is not None
        assert list(frame.residual_by_family) == ["psi", "n", "p"]
        assert max(frame.residual_by_family.values()) == frame.residual
        if frame.update is None:
            assert frame.update_by_family is None
        else:
            assert frame.update_by_family is not None
            assert list(frame.update_by_family) == ["psi", "n", "p"]
            assert max(frame.update_by_family.values()) == frame.update

    # Three copies of one number would pass every check above.
    assert any(
        len(set(frame.residual_by_family.values())) == 3 for frame in iterations
    )


def test_a_diverged_iterate_is_not_paired_with_an_older_split() -> None:
    """newton_solve reports a diverged iterate as infinite without measuring
    it. The split from the evaluation before would otherwise ride along, and
    the browser would name a stalled family from a residual that was finite."""
    frames, send = collect()
    residual_norm, _, report = _reported_by_family(
        lambda residual, x: {"psi": 1e-3, "n": 2e-3, "p": 5e-4}, send
    )
    assert report is not None

    residual_norm(np.zeros(3), np.zeros(3))
    report(NewtonIteration(1, 2e-3, 0.1, 1.0, False))
    report(NewtonIteration(2, float("inf"), 0.1, 1.0, False))

    assert frames[0].residual_by_family == {"psi": 1e-3, "n": 2e-3, "p": 5e-4}
    assert frames[1].residual_by_family is None


def test_a_newton_solve_that_knows_no_families_reports_none() -> None:
    """newton_solve knows nothing about semiconductors. The split is the
    coupled transport solve's to give, and a bare solve does not invent one."""
    frames, send = collect()

    def assemble(x):
        return SparseAssembly(
            residual=x - 1.0,
            rows=np.array([0]),
            cols=np.array([0]),
            values=np.array([1.0]),
            shape=(1, 1),
        )

    newton_solve(assemble, np.array([3.0]), on_iteration=send)

    assert frames
    assert all(frame.residual_by_family is None for frame in frames)
    assert all(frame.update_by_family is None for frame in frames)


def test_an_equilibrium_solve_reports_its_iterations() -> None:
    """The C-V path is equilibrium Poisson at every point, so without this
    hook a capacitance sweep has nothing to show between points."""
    frames, send = collect()

    solve_equilibrium(mos_cap(gate_voltage=-1.0), on_frame=send)

    assert of_type(frames, NewtonIteration)


def test_a_ramped_solve_reports_the_fractions_it_stepped_through() -> None:
    """The ramp is continuation in the bias, so its attempts are the same
    ContinuationEvent a sweep between points reports."""
    frames, send = collect()

    solve_bias_ramped(nmos(gate_voltage=0.4, **COARSE_FET), on_frame=send)

    events = of_type(frames, ContinuationEvent)
    assert events
    assert events[-1].parameter == pytest.approx(1.0)


def test_a_hybrid_solve_reports_both_halves() -> None:
    """Gummel to reach the basin, then Newton. A stream with only one of them
    in it would show a gap where the prelude ran."""
    frames, send = collect()

    solve_bias_hybrid(diode(), on_frame=send)

    assert of_type(frames, GummelIteration)
    assert of_type(frames, NewtonIteration)


def test_the_guess_solve_inside_a_bias_solve_stays_quiet() -> None:
    """A cold solve_bias_newton builds its guess with equilibrium Poisson,
    whose residual is a one unknown per node quantity that shares no scale
    with the coupled residual plotted beside it. Deliberately not reported:
    the frames a caller gets are the frames of the solve it asked for."""
    frames, send = collect()

    state = solve_bias_newton(diode(), guess=None, on_frame=send)

    assert state.newton is not None
    assert len(of_type(frames, NewtonIteration)) == len(
        state.newton.residual_history
    )


# ---------------------------------------------------------------- the sweeps


def test_a_diode_sweep_reports_solver_frames_and_finished_points() -> None:
    """All three streams phases/PHASE-7.md asks for, on the path the browser
    takes for a diode."""
    frames, send = collect()

    curve = iv_sweep(diode(), "anode", [0.1, 0.2], on_frame=send)

    assert curve.complete
    assert of_type(frames, GummelIteration)
    assert of_type(frames, ContinuationEvent)
    assert len(of_type(frames, IVFrame)) == len(curve.points)


def test_a_point_frame_carries_the_numbers_that_land_on_the_curve() -> None:
    """Bit for bit. The curve the browser draws from the frames has to be the
    curve pytest gets from the return value, or the plot is of nothing."""
    frames, send = collect()

    curve = iv_sweep(diode(), "anode", [0.0, 0.15], on_frame=send)
    points = of_type(frames, IVFrame)

    assert [frame.index for frame in points] == [0, 1]
    assert [frame.voltage for frame in points] == list(curve.voltage)
    assert [frame.current for frame in points] == list(curve.current)


def test_a_point_frame_arrives_after_the_solve_that_produced_it() -> None:
    """Otherwise the residual plot for a point would draw after the point is
    already on the curve, which is backwards from what happened."""
    frames, send = collect()

    iv_sweep(diode(), "anode", [0.1], on_frame=send)

    first_point = next(
        index for index, frame in enumerate(frames) if isinstance(frame, IVFrame)
    )
    assert of_type(frames[:first_point], GummelIteration)


def test_a_stalled_sweep_reports_the_points_it_did_reach() -> None:
    """A sweep that gives up is a measurement. The frames stop where the
    curve stops rather than reporting a point that was never solved."""
    frames, send = collect()

    curve = iv_sweep(
        diode(n_nodes=41),
        "anode",
        [0.2, 0.4, 5.0],
        step=0.2,
        min_step=0.05,
        max_iterations=8,
        on_frame=send,
    )

    assert not curve.complete
    assert len(of_type(frames, IVFrame)) == len(curve.points)
    assert any(not event.converged for event in of_type(frames, ContinuationEvent))


def test_a_gate_sweep_reports_newton_iterations_not_gummel_cycles() -> None:
    """A MOSFET runs the coupled path, which has no Gummel in it at all. A
    stream carrying cycles here would mean the browser is watching a solve
    the device cannot have run."""
    frames, send = collect()

    curve = gate_sweep(nmos(**COARSE_FET), [0.2, 0.4], step=0.2, on_frame=send)

    assert curve.complete
    assert of_type(frames, NewtonIteration)
    assert not of_type(frames, GummelIteration)
    assert len(of_type(frames, IVFrame)) == len(curve.points)


def test_a_capacitance_sweep_reports_its_points() -> None:
    """The third device class, and the one whose points are a capacitance
    rather than a current."""
    frames, send = collect()

    curve = cv_sweep(mos_cap(), "gate", [-1.0, 0.0], on_frame=send)

    assert curve.complete
    points = of_type(frames, CVFrame)
    assert [frame.gate_voltage for frame in points] == list(curve.gate_voltage)
    assert [frame.capacitance for frame in points] == list(curve.capacitance)
    assert [frame.charge for frame in points] == list(curve.charge)
    assert of_type(frames, NewtonIteration)


# ---------------------------------------------------------- stopping a solve


@pytest.mark.parametrize(
    "sweep",
    [
        lambda send: iv_sweep(diode(), "anode", [0.1, 0.2], on_frame=send),
        lambda send: gate_sweep(nmos(**COARSE_FET), [0.2], on_frame=send),
        lambda send: cv_sweep(mos_cap(), "gate", [-1.0, 0.0], on_frame=send),
    ],
    ids=["diode", "mosfet", "capacitor"],
)
def test_a_callback_that_raises_unwinds_the_whole_sweep(sweep) -> None:
    """This is what cancellation rides on. api/jobs.py raises inside send and
    expects the stack to come apart; a sweep that swallowed it would leave a
    cancelled job running to the end of a MOSFET ladder."""

    def send(frame: object) -> None:
        raise ReaderStoppedError("stop here")

    with pytest.raises(ReaderStoppedError):
        sweep(send)


def test_cancelling_a_submitted_sweep_stops_it() -> None:
    """The acceptance criterion, end to end: a real registry, a real sweep,
    and a cancel that lands while the solver is inside a bias point."""
    registry = JobRegistry()
    device = diode()

    job = registry.submit(
        lambda send: iv_sweep(device, "anode", [0.1, 0.2, 0.3, 0.4], on_frame=send)
    )

    read = registry.frames(job.id, timeout=120.0)
    for _ in range(3):
        next(read)
    assert registry.cancel(job.id)

    assert registry.wait(job.id, timeout=120.0) is JobStatus.CANCELLED
    assert not registry.cancel(job.id)


# ------------------------------------------------------------- staying quiet


@pytest.mark.parametrize(
    "call",
    [
        lambda: solve_bias(diode()),
        lambda: solve_bias_newton(diode()),
        lambda: iv_sweep(diode(), "anode", [0.1]),
        lambda: cv_sweep(mos_cap(), "gate", [-1.0]),
    ],
    ids=["gummel", "newton", "iv", "cv"],
)
def test_no_callback_is_the_default(call) -> None:
    """Every hook is optional and off. A solver that needed a callback to run
    would have made the CLI depend on the browser."""
    assert call() is not None


def test_a_watched_sweep_and_a_quiet_one_draw_the_same_curve() -> None:
    """The cheap version of the inertness claim, on one device. The bit for
    bit version across all three classes is in tests/invariant."""
    quiet = iv_sweep(diode(), "anode", [0.1])
    frames, send = collect()
    watched = iv_sweep(diode(), "anode", [0.1], on_frame=send)

    assert frames
    np.testing.assert_array_equal(quiet.current, watched.current)
