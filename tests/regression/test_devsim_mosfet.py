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

import pytest

from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels
from ddsim.extract.iv import IVCurve, gate_sweep
from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
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
    reported = curve.header["mesh convergence"].split()[0]
    assert float(reported) < 0.1 * benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {reported}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
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
