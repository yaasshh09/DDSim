"""Tier 4: ddsim's MOS capacitor against DEVSIM golden data.

Benchmarks 4 and 5 of the table in docs/04-validation.md, the two rows that
tests/regression/test_devsim_diodes.py had to leave out because ddsim could
not solve them yet. Both curves come from
`devsim_gen/generate_mos_cv.py`, which solves the same stack in 1D with every
constant overridden to ddsim's value.

What is compared is the gate charge, not the capacitance
--------------------------------------------------------
The charge is what each code actually computes, as the flux of D over the
contact's own cell. The capacitance is a derivative, and the two codes take it
differently: ddsim differentiates its solved system exactly, DEVSIM has no such
path here. So the charge is compared directly, and the capacitance is compared
after one central difference applied identically to both, which keeps the
question about the physics rather than about the differentiation. Whether
ddsim's exact derivative agrees with its own central difference is a different
question, asked in tests/analytic/test_mos_cv.py.

Why there is a mesh convergence test and not just a tolerance
-------------------------------------------------------------
The doc asks for 2 percent and ddsim clears that by a factor of four and a
half, which makes a bare 2 percent assertion a weak guard: it would sit green
through a one percent error in the permittivity or the work function.

The stronger statement is about the *shape* of the residual. A discretization
error shrinks when the mesh is refined; a wrong constant does not. So the
comparison is also run down a refinement ladder and required to converge. That
distinction is the whole point of the tier, and it is not something a single
tolerance number can express.

The floor near flatband
-----------------------
The gate charge passes through zero between accumulation and depletion, and a
relative comparison between two numbers that are both nearly nothing says
nothing about either code. Points below a thousandth of the largest charge on
the curve are checked for smallness instead, the same rule the generator uses
for its own mesh check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ddsim.core import constants as C
from ddsim.device.mos_cap import mos_cap
from ddsim.extract.cv import CVCurve, cv_sweep
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"
"""Where the generator writes, and where this reads."""

FLOOR_FRACTION = 1e-3
"""Charges below this fraction of the curve's largest are treated as zero [1].

Three decades below anything the comparison cares about, and the same rule the
generator applies to its own mesh check, so the two agree on which points are
meaningful.
"""


# ------------------------------------------------------------------ the mirror


@pytest.mark.parametrize(
    ("name", "mirror", "actual"),
    [
        ("eps_r_ox", P.EPS_R_OX, C.EPS_R_OX),
        ("chi_Si", P.CHI_SI, C.CHI_SI),
        ("Eg", P.EG, C.Eg()),
        ("Phi_M n+ poly", P.PHI_M_N_POLY, C.PHI_M_N_POLY),
        ("Phi_M midgap", P.PHI_M_MIDGAP, C.PHI_M_MIDGAP),
    ],
)
def test_mos_constants_mirror_ddsim(name: str, mirror: float, actual: float) -> None:
    """Every constant the MOS generator handed DEVSIM is still ddsim's value.

    The same guard test_devsim_diodes.py puts on the transport constants, for
    the four the MOS stack adds. The band gap is the one that matters most:
    it enters only through the midgap work function, so an error in it moves
    the whole C-V curve sideways while leaving its shape perfect, which is the
    hardest kind of disagreement to read off a plot.
    """
    assert mirror == actual, (
        f"{name} is {mirror} in the generator and {actual} in ddsim. The "
        "golden data was solved with the generator's value, so it is stale. "
        "Regenerate it, do not edit the mirror."
    )


# ------------------------------------------------------------- the golden data


def golden_path(benchmark: P.MosBenchmark) -> Path:
    """Where this benchmark's golden curve lives."""
    return GOLDEN_DIR / f"{benchmark.name}.csv"


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_golden_data_exists(benchmark: P.MosBenchmark) -> None:
    """Tier 4 has MOS data. Without it benchmarks 4 and 5 are not running."""
    assert golden_path(benchmark).is_file(), (
        f"no golden curve for {benchmark.name}. Generate it with "
        "tests/regression/devsim_gen/generate_mos_cv.py, see the README there."
    )


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(benchmark: P.MosBenchmark) -> None:
    """A golden file says what made it and under which models.

    Data without provenance is worse than no data, because it looks like
    evidence.
    """
    curve = P.read_mos_golden(str(golden_path(benchmark)))
    assert curve.header["device"] == benchmark.name
    assert curve.header["generator"].startswith("devsim ")
    assert "generated on" in curve.header
    assert "mesh convergence" in curve.header
    assert curve.header["statistics"] == "Boltzmann"
    assert curve.header["oxide"].startswith("Poisson only")


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_golden_reference_is_converged(benchmark: P.MosBenchmark) -> None:
    """The reference's own mesh error is small against what it is used to assert.

    The generator halves every spacing and records the worst change it sees. If
    that number ever creeps up to the tolerance being demanded, the test above
    is measuring the reference's mesh rather than ddsim, and it stops meaning
    what its name says. A tenth of the budget is the line.
    """
    curve = P.read_mos_golden(str(golden_path(benchmark)))
    reported = curve.header["mesh convergence"].split()[0]
    assert float(reported) < 0.1 * benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {reported}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_matches_the_benchmark(benchmark: P.MosBenchmark) -> None:
    """The stored curve is the device that parameters.py now describes.

    Editing an oxide thickness or a bias list without regenerating leaves a
    file that still loads and still compares, against the wrong device.
    """
    curve = P.read_mos_golden(str(golden_path(benchmark)))
    for key, expected in (
        ("substrate doping", benchmark.substrate_doping),
        ("t_ox", benchmark.t_ox),
        ("t_si", benchmark.t_si),
        ("work function", benchmark.work_function),
        ("tolerance", benchmark.tolerance),
    ):
        assert float(curve.header[key]) == pytest.approx(expected, rel=1e-6), (
            f"{key} in {benchmark.name}.csv is {curve.header[key]} but "
            f"parameters.py now says {expected}. The golden data is stale."
        )

    assert curve.gate_voltage == pytest.approx(list(benchmark.voltages)), (
        "the golden bias points are not the ones parameters.py asks for"
    )


# ------------------------------------------------------------- the comparison

_SOLVED: dict[str, CVCurve] = {}
"""One ddsim sweep per benchmark, reused across the tests that need it."""


def ddsim_curve(benchmark: P.MosBenchmark) -> CVCurve:
    """Solve the benchmark in ddsim, once per session."""
    if benchmark.name not in _SOLVED:
        device = mos_cap(
            substrate_doping=benchmark.substrate_doping,
            t_ox=benchmark.t_ox,
            t_si=benchmark.t_si,
            n_silicon=benchmark.n_silicon,
            n_oxide=benchmark.n_oxide,
            h_min=benchmark.h_min,
            work_function=benchmark.work_function,
        )
        _SOLVED[benchmark.name] = cv_sweep(
            device, "gate", list(benchmark.voltages)
        )
    return _SOLVED[benchmark.name]


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_ddsim_reaches_every_golden_bias(benchmark: P.MosBenchmark) -> None:
    """The sweep converges everywhere DEVSIM did.

    Separate from the numerical comparison on purpose. A sweep that stalls in
    inversion and a sweep that disagrees by ten percent are different failures
    and should not arrive as the same red line.
    """
    curve = ddsim_curve(benchmark)
    assert curve.complete, f"ddsim stopped early: {curve.message}"
    assert list(curve.gate_voltage) == pytest.approx(list(benchmark.voltages))


def compare(
    voltage: list[float],
    expected: list[float],
    actual: list[float],
    tolerance: float,
) -> tuple[list[str], float]:
    """Every point of two curves against a tolerance [1].

    Returns the failures and the worst relative disagreement, the second so a
    passing test can still report how much of its budget it used.
    """
    scale = max(abs(value) for value in expected)
    floor = FLOOR_FRACTION * scale

    failures: list[str] = []
    worst = 0.0
    for v, want, got in zip(voltage, expected, actual, strict=True):
        if abs(want) < floor:
            # Flatband. Both codes are reporting a difference of large
            # numbers, so the only meaningful question is whether ddsim is
            # also negligible.
            if abs(got) >= floor:
                failures.append(
                    f"{v:+.3f} V: devsim gives {want:.3e}, below the "
                    f"{floor:.3e} floor, but ddsim gives {got:.3e}"
                )
            continue
        relative = abs(got - want) / abs(want)
        worst = max(worst, relative)
        if relative > tolerance:
            failures.append(
                f"{v:+.3f} V: devsim {want:.6e}, ddsim {got:.6e}, "
                f"off by {relative:.3%}, allowed {tolerance:.3%}"
            )
    return failures, worst


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_ddsim_matches_devsim_charge(benchmark: P.MosBenchmark) -> None:
    """The gate charge agrees with DEVSIM at every bias.

    Every point is checked before anything is reported, because the shape of a
    disagreement is the diagnosis. A residual that grows with the charge in
    both accumulation and inversion, and vanishes at flatband, is the surface
    mesh. One that is flat across the whole sweep is the oxide capacitance.
    One that looks like a sideways shift is the work function.
    """
    golden = P.read_mos_golden(str(golden_path(benchmark)))
    curve = ddsim_curve(benchmark)

    failures, worst = compare(
        golden.gate_voltage,
        golden.charge,
        list(curve.charge),
        benchmark.tolerance,
    )
    assert not failures, (
        f"{benchmark.name} gate charge disagrees with DEVSIM at "
        f"{len(failures)} of {len(golden.gate_voltage)} points:\n  "
        + "\n  ".join(failures)
    )
    assert worst <= benchmark.tolerance


@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids=lambda b: b.name)
def test_ddsim_matches_devsim_capacitance(benchmark: P.MosBenchmark) -> None:
    """The C-V curve agrees with DEVSIM through all three regimes.

    The same central difference on both codes, so its truncation error cancels
    out of the comparison instead of being one more thing to argue about. This
    is the benchmark the doc actually names: it is a C-V comparison, and the
    charge test above is how it is made well posed.
    """
    golden = P.read_mos_golden(str(golden_path(benchmark)))
    curve = ddsim_curve(benchmark)

    bias, expected = P.central_difference(golden.gate_voltage, golden.charge)
    _, actual = P.central_difference(
        list(curve.gate_voltage), list(curve.charge)
    )

    failures, worst = compare(bias, expected, actual, benchmark.tolerance)
    assert not failures, (
        f"{benchmark.name} capacitance disagrees with DEVSIM at "
        f"{len(failures)} of {len(bias)} points:\n  " + "\n  ".join(failures)
    )
    assert worst <= benchmark.tolerance


# ------------------------------------------------- discretization, not physics

CONVERGENCE_LADDER: tuple[tuple[int, float], ...] = (
    (121, 5e-8),
    (161, 2e-8),
    (201, 1e-8),
)
"""ddsim (n_silicon, h_min) meshes, each finer at the surface than the last.

Stops at 1e-8 cm deliberately. Below that the disagreement reaches the golden
reference's own mesh error, recorded in its header as 5.5e-5, and stops being
a measurement of ddsim at all: refining further makes the number go back up.
"""

LADDER_BIASES: tuple[float, ...] = (-2.0, 1.0, 2.0)
"""Deep accumulation and two points into inversion.

The three biases carrying the most surface charge, which is where a surface
mesh error is largest and therefore where it is measurable.
"""

MINIMUM_RATIO = 2.0
"""How much each refinement must at least improve the agreement [1].

The observed factor is close to four, which is the second order convergence
the box scheme should give. Two is half of that, loose enough not to be a
tripwire on the solver tolerance and tight enough that a residual which does
not move at all fails.
"""


def test_disagreement_is_ddsim_discretization() -> None:
    """Refining ddsim's mesh drives it onto DEVSIM, so the residual is the mesh.

    This is the test that a bare tolerance cannot replace. Nothing in the model
    is refined away by adding nodes: a wrong permittivity, a wrong work
    function, a missing interface term or a sign error all produce a
    disagreement that sits exactly where it is under refinement. Only a
    discretization error shrinks, and it has to shrink at a rate the scheme
    predicts.

    Run on the 5 nm device, which has four times the surface charge of the
    20 nm one at the same bias and so the largest signal to measure.
    """
    benchmark = P.MOS_BENCHMARKS[0]
    golden = P.read_mos_golden(str(golden_path(benchmark)))
    reference = dict(zip(golden.gate_voltage, golden.charge, strict=True))
    expected = [reference[v] for v in LADDER_BIASES]

    worst_per_mesh: list[float] = []
    for n_silicon, h_min in CONVERGENCE_LADDER:
        device = mos_cap(
            substrate_doping=benchmark.substrate_doping,
            t_ox=benchmark.t_ox,
            t_si=benchmark.t_si,
            n_silicon=n_silicon,
            n_oxide=benchmark.n_oxide,
            h_min=h_min,
            work_function=benchmark.work_function,
        )
        curve = cv_sweep(device, "gate", list(LADDER_BIASES))
        assert curve.complete, (
            f"ddsim did not converge on the {n_silicon} node mesh: "
            f"{curve.message}"
        )
        _, worst = compare(
            list(LADDER_BIASES), expected, list(curve.charge), tolerance=1.0
        )
        worst_per_mesh.append(worst)

    report = ", ".join(
        f"{n} nodes at h_min {h:.0e}: {w:.4%}"
        for (n, h), w in zip(CONVERGENCE_LADDER, worst_per_mesh, strict=True)
    )

    for index in range(1, len(worst_per_mesh)):
        coarse = worst_per_mesh[index - 1]
        fine = worst_per_mesh[index]
        assert fine * MINIMUM_RATIO <= coarse, (
            "refining ddsim's surface mesh did not improve its agreement with "
            f"DEVSIM by {MINIMUM_RATIO:g}x. The residual is not "
            f"discretization, so something in the model differs. {report}"
        )

    finest = worst_per_mesh[-1]
    assert finest < 1e-3, (
        "ddsim converges under refinement but not onto DEVSIM: it settles at "
        f"{finest:.4%}, an order of magnitude above where the two meshes "
        f"should stop being the limit. {report}"
    )
