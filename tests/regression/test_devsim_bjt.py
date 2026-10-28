from __future__ import annotations
from dataclasses import fields
from functools import cache
from pathlib import Path
import pytest
from ddsim.core.config import CONFIG
from ddsim.device.bjt import npn_2d
from ddsim.device.transport import TransportModels
from ddsim.extract.bjt import GummelPlot, OutputCurve, gummel_plot, output_curve
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"


def golden(b: P.BjtBenchmark) -> P.AcGoldenCurve:
    return P.read_ac_golden(str(GOLDEN_DIR / f"{b.name}.csv"))


def rows(b: P.BjtBenchmark, curve: int) -> list[dict[str, float]]:
    return [r for r in golden(b).rows if r["curve"] == curve]


@cache
def ours(name: str) -> tuple[GummelPlot, OutputCurve]:
    b = P.BJT_BY_NAME[name]
    d = npn_2d()
    m = TransportModels.for_device(d)
    return gummel_plot(d, list(b.v_be), models=m, f_t=False), output_curve(d, b.i_b, list(b.v_ce), models=m)


def test_the_generator_builds_the_device_ddsim_builds() -> None:
    k = {f.name for f in fields(CONFIG.npn_2d)}
    for name, value in P.NPN_2D.items():
        assert name in k
        assert value == getattr(CONFIG.npn_2d, name), f"{name} is {value} in the generator and {getattr(CONFIG.npn_2d, name)} in ddsim, so the golden curve is of another device"


@pytest.mark.parametrize("b", P.BJT_BENCHMARKS, ids=lambda b: b.name)
def test_golden_data_exists(b: P.BjtBenchmark) -> None:
    assert (GOLDEN_DIR / f"{b.name}.csv").is_file(), f"no golden curve for {b.name}. Generate it with tests/regression/devsim_gen/generate_bjt.py"


@pytest.mark.parametrize("b", P.BJT_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(b: P.BjtBenchmark) -> None:
    h = golden(b).header
    assert h["device"] == b.name
    assert h["generator"].startswith("devsim ")
    assert "generated on" in h and "mesh convergence" in h
    assert float(h["tolerance"]) == b.tolerance
    assert float(h["base current"]) == b.i_b
    for name, value in P.NPN_2D.items():
        assert float(h[name]) == pytest.approx(value, rel=1e-6)
    assert [r["v_be"] for r in rows(b, 0)] == pytest.approx(list(b.v_be), rel=1e-12, abs=0)
    assert [r["v_ce"] for r in rows(b, 1)] == pytest.approx(list(b.v_ce), rel=1e-12, abs=0)


@pytest.mark.parametrize("b", P.BJT_BENCHMARKS, ids=lambda b: b.name)
def test_the_golden_curve_is_converged(b: P.BjtBenchmark) -> None:
    h = golden(b).header
    assert "mesh convergence" in h, f"{b.name} golden data has no mesh convergence line, so the generator wrote its curves and never finished the refined mesh rerun. Run it again."
    x = float(h.get("mesh error estimate", h["mesh convergence"]).split()[0])
    assert x < 0.1 * b.tolerance, f"{b.name} golden data is converged only to {x:.3e}, not comfortably inside the {b.tolerance} it is used to assert. Refine the generator mesh and regenerate."


@pytest.mark.parametrize("b", P.BJT_BENCHMARKS, ids=lambda b: b.name)
def test_the_gummel_plot_matches_devsim(b: P.BjtBenchmark) -> None:
    g, _ = ours(b.name)
    assert g.complete, g.message
    bad = []
    for p, r in zip(g.points, rows(b, 0), strict=True):
        for k, got in (("collector", p.collector), ("base", p.base)):
            e = abs(got - r[k]) / abs(r[k])
            if e > b.tolerance:
                bad.append(f"V_BE {p.v_be:.2f} V: {k} devsim {r[k]:.6e}, ddsim {got:.6e} A/cm, off by {e:.2%}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("b", P.BJT_BENCHMARKS, ids=lambda b: b.name)
def test_the_output_curve_matches_devsim(b: P.BjtBenchmark) -> None:
    _, c = ours(b.name)
    assert c.complete, c.message
    bad = []
    for p, r in zip(c.points, rows(b, 1), strict=True):
        e = abs(p.collector - r["collector"]) / abs(r["collector"])
        if e > b.tolerance:
            bad.append(f"V_CE {p.v_ce:.2f} V: collector devsim {r['collector']:.6e}, ddsim {p.collector:.6e} A/cm, off by {e:.2%}")
    assert not bad, "\n".join(bad)
