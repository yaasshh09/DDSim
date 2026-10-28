from __future__ import annotations
import math
from pathlib import Path
import pytest
from ddsim.core import constants as C
from ddsim.device.schottky import schottky_diode
from ddsim.discretize.boundary import barrier_density
from ddsim.extract.iv import iv_sweep
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"


@pytest.mark.parametrize(("name", "mirror", "actual"), [("Nc", P.NC, C.Nc()), ("Nv", P.NV, C.Nv()), ("A*_n", P.A_STAR_N, C.A_STAR_N), ("A*_p", P.A_STAR_P, C.A_STAR_P)])
def test_generator_constants_mirror_ddsim(name: str, mirror: float, actual: float) -> None:
    assert mirror == actual, f"{name} is {mirror} in the generator and {actual} in ddsim, so the golden data is stale. Regenerate it."


@pytest.mark.parametrize("b", P.SCHOTTKY_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(b: P.SchottkyBenchmark) -> None:
    g = P.read_golden(str(GOLDEN_DIR / f"{b.name}.csv"))
    assert g.header["device"] == b.name
    assert g.header["generator"].startswith("devsim ")
    assert float(g.header["tolerance"]) == b.tolerance
    assert float(g.header["n0b"]) == pytest.approx(barrier_density(b.barrier), rel=1e-11)
    assert [float(v) for v in g.voltage] == list(b.voltages)


@pytest.mark.parametrize("b", P.SCHOTTKY_BENCHMARKS, ids=lambda b: b.name)
def test_the_golden_curve_is_converged(b: P.SchottkyBenchmark) -> None:
    g = P.read_golden(str(GOLDEN_DIR / f"{b.name}.csv"))
    assert float(g.header["mesh convergence"].split()[0]) < 0.1 * b.tolerance


@pytest.mark.parametrize("b", P.SCHOTTKY_BENCHMARKS, ids=lambda b: b.name)
def test_the_schottky_diode_matches_devsim(b: P.SchottkyBenchmark) -> None:
    g = P.read_golden(str(GOLDEN_DIR / f"{b.name}.csv"))
    d = schottky_diode(Nd=b.Nd, length=b.length, barrier=b.barrier, n_nodes=b.n_nodes, h_min=b.h_min)
    bad = []
    for side in (sorted((v for v in b.voltages if v < 0.0), reverse=True), sorted(v for v in b.voltages if v > 0.0)):
        c = iv_sweep(d, "anode", [0.0, *side])
        assert c.complete, c.message
        for p in c.points[1:]:
            want = g.current[g.voltage.index(p.voltage)]
            e = abs(math.log(abs(p.current) / abs(want)))
            if e > b.tolerance or math.copysign(1.0, p.current) != math.copysign(1.0, want):
                bad.append(f"{p.voltage:+.2f} V: devsim {want:.6e}, ddsim {p.current:.6e} A/cm^2")
    assert not bad, "\n".join(bad)
