from __future__ import annotations
import math
from pathlib import Path
import pytest
from ddsim.core import constants as C
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.extract.iv import gate_sweep
from ddsim.physics.recombination import SRHRecombination
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"


@pytest.mark.parametrize(("name", "mirror", "actual"), [("a_n", P.II_A_N, C.II_A_N), ("b_n", P.II_B_N, C.II_B_N), ("a_p low", P.II_A_P_LOW, C.II_A_P_LOW), ("b_p low", P.II_B_P_LOW, C.II_B_P_LOW), ("a_p high", P.II_A_P_HIGH, C.II_A_P_HIGH), ("b_p high", P.II_B_P_HIGH, C.II_B_P_HIGH), ("switch", P.II_E_P_SWITCH, C.II_E_P_SWITCH)])
def test_generator_constants_mirror_ddsim(name: str, mirror: float, actual: float) -> None:
    assert mirror == actual, f"{name} is {mirror} in the generator and {actual} in ddsim, so the golden data is stale. Regenerate it."


@pytest.mark.parametrize("b", P.BREAKDOWN_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(b: P.BreakdownBenchmark) -> None:
    g = P.read_golden(str(GOLDEN_DIR / f"{b.name}.csv"))
    assert g.header["device"] == b.name
    assert g.header["generator"].startswith("devsim ")
    assert float(g.header["tolerance"]) == b.tolerance
    assert float(g.header["tau"]) == b.tau
    assert [float(v) for v in g.voltage] == sorted(b.voltages)


@pytest.mark.parametrize("b", P.BREAKDOWN_BENCHMARKS, ids=lambda b: b.name)
def test_the_golden_curve_is_converged(b: P.BreakdownBenchmark) -> None:
    g = P.read_golden(str(GOLDEN_DIR / f"{b.name}.csv"))
    assert float(g.header["mesh convergence"].split()[0]) < 0.1 * b.tolerance


@pytest.mark.parametrize("b", P.BREAKDOWN_BENCHMARKS, ids=lambda b: b.name)
def test_the_breakdown_curve_matches_devsim(b: P.BreakdownBenchmark) -> None:
    g = P.read_golden(str(GOLDEN_DIR / f"{b.name}.csv"))
    d = pn_diode(Na=b.Na, Nd=b.Nd, length=b.length, junction=b.junction, n_nodes=b.n_nodes, h_min=b.h_min)
    s = d.scale
    ni = d.material.n_i / s.C_0
    R = SRHRecombination(tau_n=b.tau / s.t_0, tau_p=b.tau / s.t_0, ni2=ni * ni, n1=ni, p1=ni)
    c = gate_sweep(d, [0.0, *sorted(b.voltages, reverse=True)], contact="anode", measure_at="anode", models=TransportModels.for_device(d, recombination=R, impact=True))
    assert c.complete, c.message
    bad = []
    for p in c.points[1:]:
        want = g.current[g.voltage.index(p.voltage)]
        if abs(math.log(p.current / want)) > b.tolerance:
            bad.append(f"{p.voltage:+.3f} V: devsim {want:.6e}, ddsim {p.current:.6e} A/cm^2")
    assert not bad, "\n".join(bad)
