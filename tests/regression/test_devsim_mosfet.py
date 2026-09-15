"""Tier 4: ddsim's MOSFET transfer curves against DEVSIM golden data.

Benchmarks 6 to 8 of the table in docs/04-validation.md, the rows
`test_devsim_diodes.py` and `test_devsim_mos.py` both had to leave out. Every
curve comes from `devsim_gen/generate_mosfet.py`, which solves the same 2D
structure in DEVSIM 2.11 with every constant overridden to ddsim's value.

This is the first benchmark where both codes are genuinely two dimensional
-------------------------------------------------------------------------
The diodes are 1D and the MOS capacitors are a 1D stack. A MOSFET is neither:
the whole subject of phases/PHASE-5.md is what happens when two junctions are
close enough together to see each other through the body. So what these rows
compare that no earlier row could is the 2D Poisson solve, the 2D transport,
the Si/SiO2 interface under bias and the plate contacts, on two meshes that
share nothing but the geometry they discretize.

What is compared, and what is deliberately not
----------------------------------------------
Both codes run Boltzmann statistics and constant mobility, not the Phase 5
stack. That is a decision with a dated row in docs/07-decisions.md and it has
a consequence worth stating twice: the drain currents here are not the drain
currents the README roll-off table reports, which are taken with Arora,
Caughey-Thomas and Lombardi on. What agrees here is the electrostatics and the
transport, which is what threshold voltage, subthreshold slope and DIBL are
made of. Mobility is not being compared at all.

Each benchmark carries two curves, one at 50 mV of drain and one at 1 V, and
both are compared. One curve would not be enough: a threshold error moves both
together, and only the gap between them is DIBL.

The tolerance policy
--------------------
The same rule the diodes use, for the same reason. In the off state the drain
current is reverse junction leakage, which both codes compute as a difference
of much larger fluxes, and no solver tolerance recovers digits floating point
has already thrown away. The golden datum measures its own uncertainty: with
no impact ionisation the body current is decades below either surface
terminal, so drain and source must sum to zero in steady state and whatever
fails to cancel is the noise on that point. The test demands the benchmark's
tolerance or three times that noise, whichever is looser.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from ddsim.core import constants as C
from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels
from ddsim.extract.iv import IVCurve, gate_sweep
from ddsim.extract.params import threshold_constant_current
from ddsim.extract.rolloff import REFERENCE_CURRENT, SHORT_CHANNEL_PROCESS
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"
"""Where the generator writes, and where this reads."""

NOISE_FACTOR = 3.0
"""How much of the golden point's own imbalance to allow [1].

The same factor test_devsim_diodes.py uses. A point whose drain and source
currents cancel to one percent is a point known to one percent, and demanding
that ddsim match it to five parts in a thousand is demanding agreement with
noise.
"""


# ------------------------------------------------------------------ the mirror


def test_process_mirrors_ddsim() -> None:
    """The process the generator handed DEVSIM is still ddsim's own.

    `MOSFET_PROCESS` is a literal copy of `SHORT_CHANNEL_PROCESS`, because the
    generator runs under an interpreter that has no ddsim on its path. A copy
    that drifts is golden data solved for a device nobody is simulating any
    more, and nothing else in this file would notice.
    """
    assert set(P.MOSFET_PROCESS) == set(SHORT_CHANNEL_PROCESS), (
        "the generator's process and ddsim's have different keys"
    )
    for key, mirrored in P.MOSFET_PROCESS.items():
        assert mirrored == SHORT_CHANNEL_PROCESS[key], (
            f"{key} is {mirrored} in the generator and "
            f"{SHORT_CHANNEL_PROCESS[key]} in ddsim. The golden data was "
            "solved with the generator's value, so it is stale. Regenerate "
            "it, do not edit the mirror."
        )


def test_the_surface_spacing_mirrors_ddsim() -> None:
    """Both codes resolve the inversion layer at the same spacing.

    This is the one mesh number the drain current is really sensitive to, and
    the two meshes are otherwise unrelated: ddsim grades its own rows and the
    generator hands devsim a list of spacings. Holding them equal is what makes
    benchmark 6 a comparison of two discretisations of the same physics rather
    than of two different devices, and it is also what
    tests/convergence/test_mosfet_mesh_convergence.py licenses, since that
    ladder only ever refines ddsim's side. If one of them is refined and the
    other is not, the golden data is stale and nothing else here would say so.
    """
    ddsim_spacing = inspect.signature(nmos).parameters["h_min_y"].default
    for benchmark in P.MOSFET_BENCHMARKS:
        assert benchmark.devsim_h_surface == pytest.approx(
            ddsim_spacing, rel=1e-12
        ), (
            f"{benchmark.name} meshes the silicon surface at "
            f"{benchmark.devsim_h_surface:.3e} cm for devsim and "
            f"{ddsim_spacing:.3e} for ddsim. Regenerate the golden data after "
            "matching them, do not edit one to suit the other."
        )
        oxide_cell = float(P.MOSFET_PROCESS["t_ox"]) / benchmark.devsim_oxide_cells
        assert oxide_cell == pytest.approx(ddsim_spacing, rel=1e-12), (
            f"{benchmark.name} puts a {oxide_cell:.3e} cm oxide cell against a "
            f"{ddsim_spacing:.3e} silicon surface spacing, a seam ratio of "
            f"{oxide_cell / ddsim_spacing:.3g}. Follow devsim_h_surface with "
            "devsim_oxide_cells, see docs/05-pitfalls.md."
        )


def test_full_stack_parameters_mirror_ddsim() -> None:
    """Every Phase 5 model parameter the generator holds is still ddsim's.

    Benchmark 10 is the one benchmark run at the full model stack, so these
    are the constants its golden data was solved with. The same rule as every
    other mirror in this file: a copy that drifts is a golden curve for a model
    nobody runs any more, and the drift is silent because both codes keep
    converging.

    Lombardi is the one set that is not a copy of ddsim in the sense the others
    are. Both codes take it from devsim's own `Klaassen.py`, which is where
    ddsim's docstring says its values came from, so this asserts that the two
    independent transcriptions of that third source still agree.
    """
    from ddsim.physics.mobility import AroraMobility, LombardiSurface
    from ddsim.physics.statistics import (
        JOYCE_DIXON_COEFFICIENTS,
        JOYCE_DIXON_MAX_U,
    )

    for name, mirror, model in (
        ("electrons", P.ARORA_N, AroraMobility.electrons()),
        ("holes", P.ARORA_P, AroraMobility.holes()),
    ):
        assert mirror == (
            model.mu_min,
            model.mu_d,
            model.N_ref,
            model.exponent,
        ), f"Arora for {name} has drifted from ddsim's"

    for name, mirror, model in (
        ("electrons", P.LOMBARDI_N, LombardiSurface.electrons()),
        ("holes", P.LOMBARDI_P, LombardiSurface.holes()),
    ):
        assert mirror == {
            "B": model.B,
            "C": model.C_ac,
            "tau": model.tau,
            "delta": model.delta,
            "A": model.A,
            "alpha": model.alpha,
            "eta": model.eta,
            "kappa": model.kappa,
        }, f"Lombardi for {name} has drifted from ddsim's"
        assert P.E_PERP_FLOOR == model.E_floor

    assert P.V_SAT_N == C.V_SAT_N_300
    assert P.V_SAT_P == C.V_SAT_P_300
    assert P.BETA_N == C.BETA_N
    assert P.BETA_P == C.BETA_P
    assert P.NC_300 == C.Nc()
    assert P.NV_300 == C.Nv()
    assert P.JOYCE_DIXON == tuple(float(a) for a in JOYCE_DIXON_COEFFICIENTS)
    assert P.JOYCE_DIXON_MAX_U == JOYCE_DIXON_MAX_U


def test_ddsim_resolves_the_implant() -> None:
    """ddsim's rows sample the source and drain Gaussian finely enough.

    This is the spacing the generator got wrong. Its rows through the implant
    depth were a flat 1e-6 cm, which on this process is 1.21 sigma, and more
    than a sigma per row puts the metallurgical junction in the wrong place:
    the reference's own halved mesh check moved 4.6 percent at zero gate and
    the split showed these rows carried almost all of it. See
    `P.H_DEPTH_SIGMAS`.

    ddsim never had the bug, because it grades its rows rather than picking a
    number, and it currently samples this implant at 0.21 sigma at the junction
    depth. What this guards is that it stays that way. `n_silicon` and
    `h_min_y` set the row distribution together, so a future change to either
    could coarsen the implant while leaving the surface spacing the mirror test
    checks untouched, and nothing else here would notice.

    Half a sigma is the line. A quarter of a sigma is where the reference
    stopped moving, and 1.21 is where it was visibly wrong.
    """
    process = SHORT_CHANNEL_PROCESS
    sigma, _ = P.implant_shape(dict(process))
    t_si = float(process["t_si"])
    x_j = float(process["x_j"])

    device = nmos(L_gate=1e-4, degenerate=False, **process)
    rows = np.asarray(device.mesh.y_axis.x)
    silicon = rows[rows <= t_si * (1.0 + 1e-12)]
    spacing = np.diff(silicon)
    # Every row from two junction depths below the surface upward, which is
    # where the profile has any structure left to resolve.
    inside = spacing[silicon[:-1] > t_si - 2.0 * x_j]
    assert inside.size > 0
    worst = float(inside.max())
    assert worst < 0.5 * sigma, (
        f"ddsim samples the implant at {worst / sigma:.2f} sigma at worst "
        f"({worst:.3e} cm against a sigma of {sigma:.3e}), which is too coarse "
        "to place the metallurgical junction. See P.H_DEPTH_SIGMAS for what "
        "that did to the reference."
    )


@pytest.mark.parametrize("target", [1e-3, 0.01, 0.1, 0.5, 1.0, 1.5, 1.99])
def test_erfcinv_matches_the_library(target: float) -> None:
    """The generator's pure python erfcinv is the function it stands in for.

    ddsim places the lateral junction with `scipy.special.erfcinv` and the
    DEVSIM interpreter has no scipy, so `parameters.py` bisects `math.erfc`
    instead. If the two disagree, the two codes put the metallurgical junction
    in different places and every current after that is a comparison of two
    different transistors.
    """
    scipy_special = pytest.importorskip("scipy.special")
    assert P.erfcinv(target) == pytest.approx(
        float(scipy_special.erfcinv(target)), rel=1e-12, abs=1e-12
    )


def test_first_resolved_point_trims_only_the_floor() -> None:
    """The trim keeps a clean curve whole and drops a floor that rises out of.

    The second case is the one benchmark 10 needs and benchmark 9 did not.
    ddsim's terminal floor lands negative, so dropping leading non positive
    points caught it. DEVSIM's floor at the full model stack lands positive,
    +3.52e-11 falling to +9.56e-12 on the 100 nm at 50 mV, which a sign test
    cannot see and a slope test can.
    """
    target = 1e-2
    clean = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
    assert P.first_resolved_point(clean, target) == 0

    floored = [3.52e-11, 9.56e-12, 9.20e-10, 4.01e-9, 1.90e-8]
    assert P.first_resolved_point(floored, target) == 1

    negative = [-2.5e-10, -8.7e-11, 1.87e-12, 3.35e-10, 1.75e-9]
    assert P.first_resolved_point(negative, target) == 2


def test_first_resolved_point_refuses_to_trim_a_real_current() -> None:
    """The bound, which is the whole reason the trim is safe.

    A curve that breaks where the threshold could be read from is a solver
    problem, and trimming it would hide one. The rule is not how many points
    are dropped but how large they are: everything dropped has to sit a ten
    thousandth of the extraction target below it.
    """
    target = 1e-2
    broken = [5e-3, 1e-3, 2e-2, 4e-2, 8e-2]
    with pytest.raises(AssertionError, match="hiding a solver problem"):
        P.first_resolved_point(broken, target)


def test_implant_shape_matches_ddsim() -> None:
    """Both codes derive the same two implant lengths from the same process.

    sigma and the erfc length are what turn a junction depth and a lateral
    diffusion into a doping profile. They are closed form in both codes and
    computed separately in each, so this is the one place the profiles can
    silently diverge while every constant still matches.
    """
    sigma, edge = P.implant_shape(P.MOSFET_PROCESS)

    process = SHORT_CHANNEL_PROCESS
    Na = -float(process["substrate_doping"])
    peak = float(process["sd_peak"])
    expected_sigma = float(process["x_j"]) / math.sqrt(2.0 * math.log(peak / Na))

    assert sigma == pytest.approx(expected_sigma, rel=1e-12)
    assert edge > 0.0
    # The erfc length is defined by what it has to produce: the net doping
    # falls to zero exactly one lateral_diffusion in from the gate mask edge.
    assert peak * 0.5 * math.erfc(
        float(process["lateral_diffusion"]) / edge
    ) == pytest.approx(Na, rel=1e-9)


# ------------------------------------------------------------- the golden data


def golden_path(benchmark: P.MosfetBenchmark) -> Path:
    """Where this benchmark's golden curves live."""
    return GOLDEN_DIR / f"{benchmark.name}.csv"


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
def test_golden_data_exists(benchmark: P.MosfetBenchmark) -> None:
    """Tier 4 has MOSFET data. Without it benchmarks 6 to 8 are not running."""
    assert golden_path(benchmark).is_file(), (
        f"no golden curves for {benchmark.name}. Generate them with "
        "tests/regression/devsim_gen/generate_mosfet.py, see the README there."
    )


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(
    benchmark: P.MosfetBenchmark,
) -> None:
    """A golden file says what made it and under which models.

    Data without provenance is worse than no data, because it looks like
    evidence.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert curve.header["device"] == benchmark.name
    assert curve.header["generator"].startswith("devsim ")
    assert "generated on" in curve.header
    assert "mesh convergence" in curve.header
    assert curve.header["statistics"] == "Boltzmann"
    assert curve.header["mobility"].startswith("constant")
    assert curve.header["recombination"].startswith("SRH only")


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_matches_the_benchmark(
    benchmark: P.MosfetBenchmark,
) -> None:
    """The stored curves are the device that parameters.py now describes.

    Editing a gate length or a bias list without regenerating leaves a file
    that still loads and still compares, against the wrong transistor.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    for key, expected in (
        ("L_gate", benchmark.L_gate),
        ("drain low", benchmark.drain_low),
        ("drain high", benchmark.drain_high),
        ("tolerance", benchmark.tolerance),
    ):
        assert float(curve.header[key]) == pytest.approx(expected, rel=1e-9), (
            f"{key} in {benchmark.name}.csv is {curve.header[key]} but "
            f"parameters.py now says {expected}. The golden data is stale."
        )
    for key, expected in P.MOSFET_PROCESS.items():
        assert float(curve.header[key]) == pytest.approx(expected, rel=1e-9), (
            f"{key} in {benchmark.name}.csv is {curve.header[key]} but the "
            f"process now says {expected}. The golden data is stale."
        )
    assert curve.gate_voltage == pytest.approx(list(benchmark.gate_voltages)), (
        "the golden gate biases are not the ones parameters.py asks for"
    )


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
def test_golden_reference_is_converged(benchmark: P.MosfetBenchmark) -> None:
    """The reference's own mesh error is small against what it asserts.

    The generator halves every spacing and records the worst change it sees.
    If that number ever reaches the tolerance being demanded, the comparison
    below is measuring the reference's mesh rather than ddsim. A tenth of the
    budget is the line, the same line test_devsim_mos.py draws.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert "mesh convergence" in curve.header, (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. The curves may be fine and nothing here can tell. "
        "Re-run the generator."
    )
    line = curve.header["mesh convergence"]
    reported = line.split()[0]
    assert float(reported) < 0.1 * benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {reported}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )
    # The check skips points whose two meshes disagree by less than the points
    # know about themselves, which is right, and would be a way to report a
    # small number by measuring almost nothing, which is not. See
    # `MESH_NOISE_FACTOR` in the generator.
    skipped, total = (int(word) for word in line.split() if word.isdigit())
    assert skipped < 0.5 * total, (
        f"{benchmark.name} skipped {skipped} of {total} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {reported} says little about the mesh."
    )


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_golden_terminals_balance(
    benchmark: P.MosfetBenchmark, high: bool
) -> None:
    """The reference conserves charge, which is what licenses the noise floor.

    Nothing in this model set generates carriers in the bulk faster than SRH
    removes them, so in steady state the body current is decades below either
    surface terminal and drain and source sum to zero. The residual is the
    golden point's own uncertainty, and the comparison below spends it as
    tolerance. A reference that did not balance would be spending something
    else.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    worst = max(
        curve.imbalance(index, high) for index in range(len(curve.gate_voltage))
    )
    assert worst < 0.01, (
        f"{benchmark.name} drain and source disagree by {worst:.2%} at worst, "
        "which is too much of the reference's own budget to be noise"
    )


# ------------------------------------------------------------- the comparison

_SOLVED: dict[tuple[str, float], IVCurve] = {}
"""One ddsim transfer curve per benchmark and drain bias, reused across tests."""


def ddsim_curve(benchmark: P.MosfetBenchmark, drain: float) -> IVCurve:
    """Solve one transfer curve in ddsim, once per session.

    `degenerate=False` and `mobility="constant"` are what make this the same
    model set the generator ran. Both are departures from ddsim's own Phase 5
    defaults and both are deliberate. See docs/07-decisions.md.
    """
    key = (benchmark.name, drain)
    if key not in _SOLVED:
        device = nmos(
            L_gate=benchmark.L_gate,
            drain_voltage=drain,
            degenerate=False,
            **SHORT_CHANNEL_PROCESS,
        )
        models = TransportModels.for_device(device, mobility="constant")
        _SOLVED[key] = gate_sweep(
            device, list(benchmark.gate_voltages), models=models
        )
    return _SOLVED[key]


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_ddsim_reaches_every_golden_bias(
    benchmark: P.MosfetBenchmark, high: bool
) -> None:
    """The sweep converges everywhere DEVSIM did.

    Separate from the numerical comparison on purpose. A sweep that stalls in
    inversion and a sweep that disagrees by ten percent are different failures
    and should not arrive as the same red line.
    """
    drain = benchmark.drain_high if high else benchmark.drain_low
    curve = ddsim_curve(benchmark, drain)
    assert curve.complete, f"ddsim stopped early: {curve.message}"
    assert list(curve.voltage) == pytest.approx(list(benchmark.gate_voltages))


@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids=lambda b: b.name)
@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_ddsim_matches_devsim_drain_current(
    benchmark: P.MosfetBenchmark, high: bool
) -> None:
    """The transfer curve agrees with DEVSIM at every gate bias.

    Every point is checked before anything is reported, because the shape of a
    disagreement is the diagnosis. A residual flat across the on state is the
    oxide capacitance or the mobility. One that grows as the gate falls is the
    surface mesh, since the inversion layer thins as it empties. One that looks
    like a sideways shift of the whole curve is the work function.
    """
    golden = P.read_mosfet_golden(str(golden_path(benchmark)))
    drain = benchmark.drain_high if high else benchmark.drain_low
    curve = ddsim_curve(benchmark, drain)
    expected = golden.drain_high if high else golden.drain_low

    failures: list[str] = []
    worst = 0.0
    for index, (v_gate, want, got) in enumerate(
        zip(golden.gate_voltage, expected, list(curve.current), strict=True)
    ):
        if abs(want) < P.CURRENT_FLOOR_MOSFET:
            # Both codes are reporting a difference of much larger fluxes, so
            # the only meaningful question is whether ddsim is also negligible.
            if abs(got) >= P.CURRENT_FLOOR_MOSFET:
                failures.append(
                    f"{v_gate:+.3f} V: devsim gives {want:.3e}, below the "
                    f"{P.CURRENT_FLOOR_MOSFET:.3e} floor, but ddsim gives "
                    f"{got:.3e}"
                )
            continue
        allowed = max(
            benchmark.tolerance, NOISE_FACTOR * golden.imbalance(index, high)
        )
        relative = abs(got - want) / abs(want)
        worst = max(worst, relative / allowed)
        if relative > allowed:
            failures.append(
                f"{v_gate:+.3f} V: devsim {want:.6e}, ddsim {got:.6e}, "
                f"off by {relative:.2%}, allowed {allowed:.2%}"
            )

    assert not failures, (
        f"{benchmark.name} at Vd = {drain} V disagrees with DEVSIM at "
        f"{len(failures)} of {len(golden.gate_voltage)} points:\n  "
        + "\n  ".join(failures)
    )
    assert worst <= 1.0


# ------------------------------------------------- benchmark 9, the Lg trend


def _threshold(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    L_gate: float,
) -> float:
    """Constant current threshold of one curve [V].

    The same extractor ddsim's own sweep uses, applied to whichever curve it
    is given. That is the whole point: a comparison between two codes through
    two estimators measures the estimators. The trim below is applied to both
    codes for the same reason, even though only ddsim's curves need it.

    Leading points whose drain current is not positive are dropped before
    extracting. ddsim's drain terminal current has a floor near 2e-9 A/cm and
    at 50 mV of drain that floor lands negative on the two longest devices,
    three points on the 200 nm and two on the 100 nm. It is a floor and not a
    current: from -0.4 to -0.35 V the value moves in its fourth digit, where a
    real subthreshold current changes by a factor of four across a 50 mV step.
    Nothing the extraction reads is lost, since the constant current target is
    5e-3 A/cm at 200 nm and every dropped point is nine decades under it, and
    the monotonicity `threshold_constant_current` insists on is still checked
    over everything kept, so a genuinely non monotonic sweep still fails loudly
    rather than being trimmed away. DEVSIM resolves this region cleanly and
    needs no trim, which is what makes the floor ddsim's rather than the
    sweep's. See the 2026-09-13 row in docs/07-decisions.md.
    """
    V = np.asarray(voltage, dtype=np.float64)
    J = np.asarray(current, dtype=np.float64)
    target = REFERENCE_CURRENT / L_gate

    keep = P.first_resolved_point(J, target)
    return threshold_constant_current(V[keep:], J[keep:], target)


def _devsim_thresholds(name: str) -> tuple[float, float]:
    """DEVSIM's linear and saturated thresholds at one gate length [V]."""
    benchmark = P.MOSFET_BY_NAME[name]
    golden = P.read_mosfet_golden(str(golden_path(benchmark)))
    V = np.asarray(golden.gate_voltage, dtype=np.float64)
    return (
        _threshold(V, np.asarray(golden.drain_low, dtype=np.float64),
                   benchmark.L_gate),
        _threshold(V, np.asarray(golden.drain_high, dtype=np.float64),
                   benchmark.L_gate),
    )


def _ddsim_thresholds(name: str) -> tuple[float, float]:
    """ddsim's linear and saturated thresholds at one gate length [V].

    Through `ddsim_curve`, so the device is built with `degenerate=False` and
    constant mobility, matching the generator. Running this against ddsim's
    own Phase 5 defaults would compare two different model sets and call the
    difference a disagreement.
    """
    benchmark = P.MOSFET_BY_NAME[name]
    out = []
    for drain in (benchmark.drain_low, benchmark.drain_high):
        curve = ddsim_curve(benchmark, drain)
        assert curve.complete, f"{name} at Vd = {drain} V: {curve.message}"
        out.append(
            _threshold(curve.voltage, curve.current, benchmark.L_gate)
        )
    return out[0], out[1]


@pytest.mark.parametrize("name", P.ROLLOFF_TREND)
def test_rolloff_golden_data_exists(name: str) -> None:
    """Benchmark 9 has data at every gate length of the trend."""
    benchmark = P.MOSFET_BY_NAME[name]
    assert golden_path(benchmark).is_file(), (
        f"no golden curves for {name}, so benchmark 9 is not running. "
        "Generate them with tests/regression/devsim_gen/generate_mosfet.py, "
        "see the README there."
    )


@pytest.mark.parametrize(
    "benchmark", P.ROLLOFF_BENCHMARKS, ids=lambda b: b.name
)
def test_rolloff_reference_is_converged(benchmark: P.MosfetBenchmark) -> None:
    """Benchmark 9's references carry their mesh error, gated on what it moves.

    Same principle as `test_golden_reference_is_converged` and deliberately a
    different line, because a different quantity is being asserted. Benchmarks
    6 to 8 compare a drain current at every bias, so a drain current mesh error
    reaches their conclusion undiminished and is held to a tenth of their
    budget. Benchmark 9 compares a roll-off magnitude and a DIBL, and the drain
    current error reaches those only after an attenuation that was measured
    rather than assumed: halving every spacing on the 50 nm device moves its
    worst drain current by 1.227e-2, but its DIBL by 0.14 percent and its
    saturated threshold by 0.34 mV, which is 0.11 percent of the 318.6 mV
    roll-off the benchmark actually compares. That is a factor near nine, and
    it is not luck. The worst drain current point sits at -0.4 V, 400 mV below
    threshold and six decades under the constant current target, a bias the
    extraction never reads.

    So the gate is the tolerance itself rather than a tenth of it, which after
    the measured attenuation is the same tenth of the budget the other
    benchmarks are held to. Refining the 50 nm mesh to meet the stricter line
    would cost a 57000 node check to improve a number benchmark 9 does not
    assert. See the 2026-09-13 rows in docs/07-decisions.md.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert "mesh convergence" in curve.header, (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. Re-run the generator."
    )
    line = curve.header["mesh convergence"]
    reported = float(line.split()[0])
    assert reported < benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {reported:.3e}, "
        f"which is not inside the {benchmark.tolerance} the roll-off and DIBL "
        "are compared to even before the attenuation is counted. Refine the "
        "generator mesh and regenerate."
    )
    skipped, total = (int(word) for word in line.split() if word.isdigit())
    assert skipped < 0.5 * total, (
        f"{benchmark.name} skipped {skipped} of {total} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {reported:.3e} says little about the mesh."
    )


@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_rolloff_threshold_falls_in_both_codes(high: bool) -> None:
    """The trend half of benchmark 9's target, and it is the unambiguous half.

    Monotonic in both codes at both drain biases. A threshold that wanders as
    the gate shortens is not roll-off however close the endpoints land.
    """
    index = 1 if high else 0
    devsim = [_devsim_thresholds(name)[index] for name in P.ROLLOFF_TREND]
    ddsim = [_ddsim_thresholds(name)[index] for name in P.ROLLOFF_TREND]

    for label, values in (("devsim", devsim), ("ddsim", ddsim)):
        steps = np.diff(np.asarray(values))
        assert np.all(steps < 0.0), (
            f"{label} threshold does not fall monotonically from 1 um to "
            f"50 nm: {[f'{v:+.4f}' for v in values]}"
        )


@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_rolloff_magnitude_matches_devsim(high: bool) -> None:
    """The 10 percent half of benchmark 9's target, on a positive quantity.

    docs/04-validation.md asks for the trend and 10 percent, and PHASE-5.md for
    the Vth against Lg curve to be within 10 percent. Vth crosses zero in this
    set, DEVSIM reading -0.1181 V at 50 nm and Vd = 1 V, so a per point
    relative comparison on it is not a measurement: one absolute disagreement
    reads as two percent at one gate length and unbounded at another. The
    roll-off itself is positive and is what the deliverable plot is about, so
    that is what carries the percentage. Per point agreement is reported in
    millivolts below rather than gated as a fraction of a signed voltage. See
    the 2026-09-12 row in docs/07-decisions.md.
    """
    index = 1 if high else 0
    devsim = [_devsim_thresholds(name)[index] for name in P.ROLLOFF_TREND]
    ddsim = [_ddsim_thresholds(name)[index] for name in P.ROLLOFF_TREND]

    devsim_rolloff = devsim[0] - devsim[-1]
    ddsim_rolloff = ddsim[0] - ddsim[-1]
    relative = abs(ddsim_rolloff - devsim_rolloff) / abs(devsim_rolloff)

    per_point = "\n  ".join(
        f"{name}: devsim {d:+.4f} V, ddsim {s:+.4f} V, "
        f"{(s - d) * 1000:+.1f} mV"
        for name, d, s in zip(P.ROLLOFF_TREND, devsim, ddsim, strict=True)
    )
    assert relative <= 0.10, (
        f"roll-off from 1 um to 50 nm disagrees by {relative:.1%}: devsim "
        f"{devsim_rolloff * 1000:.1f} mV, ddsim {ddsim_rolloff * 1000:.1f} mV."
        f"\n  {per_point}"
    )


@pytest.mark.parametrize("name", P.ROLLOFF_TREND)
def test_rolloff_dibl_matches_devsim(name: str) -> None:
    """DIBL at each gate length, the other positive emergent number.

    Nearly mobility independent, which is what licenses comparing the two
    codes here at a model set neither of them ships by default: the barrier
    the drain lowers is electrostatics. Measured at 50 nm before this test
    existed, devsim gave 131.2 mV/V against the 126.8 the README reports from
    the full Phase 5 stack, 3.5 percent apart across two different mobility
    models.
    """
    benchmark = P.MOSFET_BY_NAME[name]
    span = benchmark.drain_high - benchmark.drain_low

    d_lin, d_sat = _devsim_thresholds(name)
    s_lin, s_sat = _ddsim_thresholds(name)
    devsim_dibl = (d_lin - d_sat) / span
    ddsim_dibl = (s_lin - s_sat) / span

    assert devsim_dibl > 0.0, (
        f"{name}: devsim puts the saturated threshold above the linear one, "
        f"{d_sat:+.4f} V against {d_lin:+.4f}, which is not DIBL"
    )
    relative = abs(ddsim_dibl - devsim_dibl) / devsim_dibl
    assert relative <= 0.10, (
        f"{name} DIBL disagrees by {relative:.1%}: devsim "
        f"{devsim_dibl * 1000:.1f} mV/V, ddsim {ddsim_dibl * 1000:.1f} mV/V"
    )


# ----------------------------------- benchmark 10, the Lg trend, full stack


_SOLVED_FULL: dict[tuple[str, float], IVCurve] = {}
"""One ddsim full stack transfer curve per benchmark and drain bias."""


def ddsim_full_curve(benchmark: P.MosfetBenchmark, drain: float) -> IVCurve:
    """Solve one transfer curve in ddsim at the Phase 5 model stack.

    Every argument is ddsim's own default, which is the point: benchmark 10
    compares the two codes running the models the README roll-off figure runs
    and phases/PHASE-5.md names as not optional. `ddsim_curve` above is the
    same device with `degenerate=False` and constant mobility, matching the
    reduced set benchmarks 6 to 9 use, and the two together are what separate
    a disagreement about mobility from a disagreement about geometry.
    """
    key = (benchmark.name, drain)
    if key not in _SOLVED_FULL:
        device = nmos(
            L_gate=benchmark.L_gate,
            drain_voltage=drain,
            **SHORT_CHANNEL_PROCESS,
        )
        models = TransportModels.for_device(
            device, mobility="arora", field_dependent=True, surface=True
        )
        _SOLVED_FULL[key] = gate_sweep(
            device, list(benchmark.gate_voltages), models=models
        )
    return _SOLVED_FULL[key]


def _ddsim_full_thresholds(name: str) -> tuple[float, float]:
    """ddsim's linear and saturated thresholds at the full stack [V]."""
    benchmark = P.MOSFET_BY_NAME[name]
    out = []
    for drain in (benchmark.drain_low, benchmark.drain_high):
        curve = ddsim_full_curve(benchmark, drain)
        assert curve.complete, f"{name} at Vd = {drain} V: {curve.message}"
        out.append(_threshold(curve.voltage, curve.current, benchmark.L_gate))
    return out[0], out[1]


@pytest.mark.parametrize("name", P.FULL_STACK_TREND)
def test_full_stack_golden_data_exists(name: str) -> None:
    """Benchmark 10 has data at every gate length of the trend."""
    benchmark = P.MOSFET_BY_NAME[name]
    assert golden_path(benchmark).is_file(), (
        f"no golden curves for {name}, so benchmark 10 is not running and "
        "nothing in tier 4 compares either mobility model or the statistics "
        "against an outside code. Generate them with "
        "tests/regression/devsim_gen/generate_mosfet.py, see the README there."
    )


@pytest.mark.parametrize("name", P.FULL_STACK_TREND)
def test_full_stack_header_records_the_models(name: str) -> None:
    """The file says it was solved at the stack benchmark 10 compares against.

    The whole of what separates benchmark 10 from benchmark 9 is which models
    devsim ran, and the only record of that is the header. `ddsim_full_curve`
    asks for the full stack whatever the benchmark says, so a file generated
    once at the reduced set would sit here comparing a constant mobility curve
    against Arora and reporting the model difference as a disagreement. The
    header is what makes that loud instead of silent.
    """
    curve = P.read_mosfet_golden(str(golden_path(P.MOSFET_BY_NAME[name])))
    assert curve.header["statistics"].startswith("Fermi-Dirac")
    assert curve.header["mobility"].startswith("Arora")
    assert "Lombardi" in curve.header["mobility"]
    assert "Caughey-Thomas" in curve.header["mobility"]


@pytest.mark.parametrize(
    "benchmark", P.FULL_STACK_BENCHMARKS, ids=lambda b: b.name
)
def test_full_stack_reference_is_converged(
    benchmark: P.MosfetBenchmark,
) -> None:
    """Benchmark 10's references carry their mesh error, on benchmark 9's gate.

    The same line and the same reasoning as
    `test_rolloff_reference_is_converged`: these are the same five devices on
    the same meshes comparing the same extracted quantities, so what a drain
    current mesh error does to a roll-off and to a DIBL is what it did there.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert "mesh convergence" in curve.header, (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. Re-run the generator."
    )
    line = curve.header["mesh convergence"]
    reported = float(line.split()[0])
    assert reported < benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {reported:.3e}, "
        f"which is not inside the {benchmark.tolerance} the roll-off and DIBL "
        "are compared to even before the attenuation is counted. Refine the "
        "generator mesh and regenerate."
    )
    skipped, total = (int(word) for word in line.split() if word.isdigit())
    assert skipped < 0.5 * total, (
        f"{benchmark.name} skipped {skipped} of {total} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {reported:.3e} says little about the mesh."
    )


FULL_STACK_FLOOR = 1e-5
"""How far under the extraction target a point is still worth comparing [1].

`CURRENT_FLOOR_MOSFET` is the same idea at an absolute 1e-12 A/cm, and it is
what benchmarks 6 to 8 use. It does not separate anything here, because Arora
inside Lombardi gives a channel mobility well under the constant 1417 those
benchmarks run, so benchmark 10's whole curve sits lower and its tail lands
between 1e-12 and the current where either code means anything.

Measured across all 210 points of the five devices, the worst surviving
disagreement against the floor it is measured above:

    no floor        210 points   845.94%
    1e-7 * target   195 points   227.69%
    1e-6 * target   178 points    67.31%
    1e-5 * target   165 points     6.51%
    1e-4 * target   150 points     2.67%
    1e-3 * target   136 points     2.67%

The cliff between 1e-6 and 1e-5 is where the reference stops resolving its own
current rather than where ddsim starts agreeing with it: the point that sets
67.31% has a terminal imbalance of 0.710, so devsim's own drain and source
disagree by 71 percent there, while the one that sets 6.51% has 0.065. Below
1e-4 nothing further is bought, and what is left is the real disagreement, at
+0.60 V of gate on the 1 um with an imbalance of 1.3e-7.

1e-5 is the choice because of what it does to the comparison rather than to
the failures. It keeps 165 of 210 points, and its worst is inside the flat 10
percent tolerance on its own, so no point needs the `NOISE_FACTOR` allowance
to pass and that allowance goes back to being a backstop. Nothing under the
floor is discarded: both codes still have to put it under the floor.
"""


LOAD_BEARING = 1e-4
"""How far under the extraction target a point still has to balance [1].

Benchmark 10's curves run four decades further down than benchmarks 6 to 8's
do, into the tail where the terminal current is a ten decade cancellation
between much larger fluxes and neither code resolves it. Asking those points
to balance is asking a cancellation to cancel, which is the same thing
`MosfetGoldenCurve.imbalance` says about a diode at reverse bias, and it is why
the whole curve form of this test fails on four of these five devices while
saying nothing about whether the reference is any good.

So the test is scoped, and this is the width of the scope. Measured across the
four devices at both drain biases, the worst imbalance is 3.1e-6 within two
decades of the target, 1.0e-4 within three, 4.6e-4 within four, and 1.4e-2
within five. Four decades is where it is still saying something, and it is
already generous: the threshold is read between the two points that bracket
the target itself, so everything the extraction touches is within one 50 mV
step of it, four decades above this line.

Nothing is swept under the line either. The points below it are exactly the
ones `first_resolved_point` refuses to extract from, and the drain current
comparison does not ignore them, it spends each one's own imbalance as that
point's tolerance through `NOISE_FACTOR`. What this test adds is the guarantee
that where the reference is load bearing, that allowance stays small.
"""


@pytest.mark.parametrize(
    "benchmark", P.FULL_STACK_BENCHMARKS, ids=lambda b: b.name
)
@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_full_stack_golden_terminals_balance(
    benchmark: P.MosfetBenchmark, high: bool
) -> None:
    """The reference conserves charge where anything is read from it.

    The same claim `test_golden_terminals_balance` makes for benchmarks 6 to 8,
    and benchmark 10 needs it more than they do rather than less. Its settles
    are allowed to stop on the solver's own floor instead of reaching the
    potential tolerance, so what certifies one of its recorded points is not
    the potential at all, it is that drain and source cancel. Every floor this
    generator accepted was at high gate, well above the target current, which
    is inside the region this test bounds.
    """
    curve = P.read_mosfet_golden(str(golden_path(benchmark)))
    current = curve.drain_high if high else curve.drain_low
    floor = LOAD_BEARING * REFERENCE_CURRENT / benchmark.L_gate

    worst, worst_at = 0.0, 0.0
    for index, value in enumerate(current):
        if abs(value) < floor:
            continue
        out = curve.imbalance(index, high)
        if out > worst:
            worst, worst_at = out, curve.gate_voltage[index]

    assert worst < 0.01, (
        f"{benchmark.name} drain and source disagree by {worst:.2%} at "
        f"{worst_at:+.2f} V of gate, where the current is within four decades "
        "of the one the threshold is read at. That is too much of the "
        "reference's own budget to be noise"
    )


@pytest.mark.parametrize(
    "benchmark", P.FULL_STACK_BENCHMARKS, ids=lambda b: b.name
)
@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_full_stack_ddsim_reaches_every_golden_bias(
    benchmark: P.MosfetBenchmark, high: bool
) -> None:
    """The Phase 5 sweep converges everywhere DEVSIM did.

    Separate from the numerical comparison for the reason the reduced set
    keeps them separate: a sweep that stalls in inversion and a sweep that
    disagrees by ten percent are different failures.
    """
    drain = benchmark.drain_high if high else benchmark.drain_low
    curve = ddsim_full_curve(benchmark, drain)
    assert curve.complete, f"ddsim stopped early: {curve.message}"
    assert list(curve.voltage) == pytest.approx(list(benchmark.gate_voltages))


@pytest.mark.parametrize(
    "benchmark", P.FULL_STACK_BENCHMARKS, ids=lambda b: b.name
)
@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_full_stack_drain_current_matches_devsim(
    benchmark: P.MosfetBenchmark, high: bool
) -> None:
    """The whole curve, at the model set neither code ships as a shortcut.

    This is the only place in the project that puts Arora, Lombardi,
    Caughey-Thomas or Fermi-Dirac in front of an implementation that is not
    ddsim's. The two codes share the parameter values, pinned by
    `test_full_stack_parameters_mirror_ddsim`, and share nothing else: devsim
    evaluates its own expressions on its own mesh, and where ddsim freezes the
    surface term and iterates to a fixed point, so does devsim, because that
    is a property of the model rather than of either implementation.

    Points under `FULL_STACK_FLOOR` are checked against the floor rather than
    against each other, which is `CURRENT_FLOOR_MOSFET`'s rule with a floor
    that scales.
    """
    golden = P.read_mosfet_golden(str(golden_path(benchmark)))
    drain = benchmark.drain_high if high else benchmark.drain_low
    curve = ddsim_full_curve(benchmark, drain)
    expected = golden.drain_high if high else golden.drain_low
    floor = FULL_STACK_FLOOR * REFERENCE_CURRENT / benchmark.L_gate

    failures: list[str] = []
    worst = 0.0
    for index, (v_gate, want, got) in enumerate(
        zip(golden.gate_voltage, expected, list(curve.current), strict=True)
    ):
        if abs(want) < floor:
            if abs(got) >= floor:
                failures.append(
                    f"{v_gate:+.3f} V: devsim gives {want:.3e}, below the "
                    f"{floor:.3e} floor, but ddsim gives {got:.3e}"
                )
            continue
        allowed = max(
            benchmark.tolerance, NOISE_FACTOR * golden.imbalance(index, high)
        )
        relative = abs(got - want) / abs(want)
        worst = max(worst, relative)
        if relative > allowed:
            failures.append(
                f"{v_gate:+.3f} V: devsim {want:.6e}, ddsim {got:.6e}, "
                f"{relative:.2%} against {allowed:.2%} allowed"
            )

    assert not failures, (
        f"{benchmark.name} at Vd = {drain} V, worst {worst:.2%}:\n  "
        + "\n  ".join(failures)
    )


@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_full_stack_threshold_falls_in_both_codes(high: bool) -> None:
    """Roll-off is still monotonic once the real mobility models are on.

    Benchmark 9 already says this at constant mobility. Saying it again here
    is not repetition: Lombardi and Caughey-Thomas both depend on the state,
    so both are stronger at the short gate lengths than at the long one, and a
    trend that survives a fixed mobility need not survive one that moves with
    the device.
    """
    index = 1 if high else 0
    devsim = [_devsim_thresholds(name)[index] for name in P.FULL_STACK_TREND]
    ddsim = [
        _ddsim_full_thresholds(name)[index] for name in P.FULL_STACK_TREND
    ]

    for label, values in (("devsim", devsim), ("ddsim", ddsim)):
        steps = np.diff(np.asarray(values))
        assert np.all(steps < 0.0), (
            f"{label} threshold does not fall monotonically from 1 um to "
            f"50 nm: {[f'{v:+.4f}' for v in values]}"
        )


@pytest.mark.parametrize("high", [False, True], ids=["Vd_low", "Vd_high"])
def test_full_stack_rolloff_matches_devsim(high: bool) -> None:
    """The deliverable plot of phases/PHASE-5.md, as a number.

    Vth against Lg, ddsim against DEVSIM, at the model stack the figure draws.
    The magnitude carries the percentage rather than the per point value for
    the reason `test_rolloff_magnitude_matches_devsim` gives: Vth crosses zero
    in this set, so a relative comparison on it measures where the zero is.
    """
    index = 1 if high else 0
    devsim = [_devsim_thresholds(name)[index] for name in P.FULL_STACK_TREND]
    ddsim = [
        _ddsim_full_thresholds(name)[index] for name in P.FULL_STACK_TREND
    ]

    devsim_rolloff = devsim[0] - devsim[-1]
    ddsim_rolloff = ddsim[0] - ddsim[-1]
    relative = abs(ddsim_rolloff - devsim_rolloff) / abs(devsim_rolloff)

    per_point = "\n  ".join(
        f"{name}: devsim {d:+.4f} V, ddsim {s:+.4f} V, "
        f"{(s - d) * 1000:+.1f} mV"
        for name, d, s in zip(P.FULL_STACK_TREND, devsim, ddsim, strict=True)
    )
    assert relative <= 0.10, (
        f"roll-off from 1 um to 50 nm disagrees by {relative:.1%}: devsim "
        f"{devsim_rolloff * 1000:.1f} mV, ddsim {ddsim_rolloff * 1000:.1f} mV."
        f"\n  {per_point}"
    )


@pytest.mark.parametrize("name", P.FULL_STACK_TREND)
def test_full_stack_dibl_matches_devsim(name: str) -> None:
    """DIBL at each gate length, with the mobility models on.

    Benchmark 9 argues that DIBL is nearly mobility independent, which is what
    licensed comparing it at a reduced model set. This is the test that can
    check that argument rather than assert it: the same devices at the full
    stack should give nearly the same DIBL, and both codes should agree.
    """
    benchmark = P.MOSFET_BY_NAME[name]
    span = benchmark.drain_high - benchmark.drain_low

    d_lin, d_sat = _devsim_thresholds(name)
    s_lin, s_sat = _ddsim_full_thresholds(name)
    devsim_dibl = (d_lin - d_sat) / span
    ddsim_dibl = (s_lin - s_sat) / span

    assert devsim_dibl > 0.0, (
        f"{name}: devsim puts the saturated threshold above the linear one, "
        f"{d_sat:+.4f} V against {d_lin:+.4f}, which is not DIBL"
    )
    relative = abs(ddsim_dibl - devsim_dibl) / devsim_dibl
    assert relative <= 0.10, (
        f"{name}: DIBL disagrees by {relative:.1%}, devsim "
        f"{devsim_dibl * 1000:.1f} mV/V against ddsim "
        f"{ddsim_dibl * 1000:.1f} mV/V"
    )
