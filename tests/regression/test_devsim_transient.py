from __future__ import annotations
from functools import cache
from pathlib import Path
import pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.iv import terminal_currents
from ddsim.extract.transient import Circuit, Transient, bdf2, storage_time
from ddsim.physics.recombination import SRHRecombination
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"

NOISE_FACTOR = 3.0


def golden(b: P.TransientBenchmark) -> P.AcGoldenCurve:
    return P.read_ac_golden(str(GOLDEN_DIR / f"{b.name}.csv"))


@cache
def recovered(name: str, scale: float = 1.0) -> tuple[float, Transient]:
    b = P.TRANSIENT_BY_NAME[name]
    d = pn_diode(Na=P.AC_DIODE_NA, Nd=P.AC_DIODE_ND, length=P.AC_DIODE_LENGTH, junction=P.AC_DIODE_JUNCTION, n_nodes=500, h_min=2e-7, anode_voltage=b.forward)
    ni = d.material.n_i / d.scale.C_0
    m = TransportModels.for_device(d, recombination=SRHRecombination(tau_n=b.tau / d.scale.t_0, tau_p=b.tau / d.scale.t_0, ni2=ni * ni, n1=ni, p1=ni))
    s = solve_bias_ramped(d, m)
    assert s.newton.converged
    R = scale * float(golden(b).header["resistance"])
    t = bdf2(d, s, b.times, m, circuit=Circuit(contact="anode", resistance=R, voltage=lambda t: b.reverse))
    assert t.complete, t.message
    return terminal_currents(d, s, m)["anode"], t


@pytest.mark.parametrize("b", P.TRANSIENT_BENCHMARKS, ids=lambda b: b.name)
def test_golden_data_exists(b):
    assert (GOLDEN_DIR / f"{b.name}.csv").is_file(), f"no golden curve for {b.name}. Generate it with tests/regression/devsim_gen/generate_transient.py, see the README there."


@pytest.mark.parametrize("b", P.TRANSIENT_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(b):
    h = golden(b).header
    assert h["device"] == b.name
    assert h["generator"].startswith("devsim ")
    assert "generated on" in h
    assert "mesh convergence" in h
    assert "step convergence" in h
    assert "devsim spread" in h
    assert h["statistics"] == "Boltzmann"


@pytest.mark.parametrize("b", P.TRANSIENT_BENCHMARKS, ids=lambda b: b.name)
def test_golden_matches_the_benchmark(b):
    g = golden(b)
    assert float(g.header["tolerance"]) == b.tolerance
    assert float(g.header["tau"]) == b.tau
    assert float(g.header["forward voltage"]) == b.forward
    assert float(g.header["reverse voltage"]) == b.reverse
    assert [r["time"] for r in g.rows] == pytest.approx(b.times, rel=1e-9, abs=0), "the golden samples are not the ones parameters.py asks for, the golden data is stale"


@pytest.mark.parametrize("b", P.TRANSIENT_BENCHMARKS, ids=lambda b: b.name)
def test_both_tools_start_from_the_same_forward_current(b):
    forward, _ = recovered(b.name)
    assert forward == pytest.approx(float(golden(b).header["forward current"]), rel=1e-3, abs=0)


@pytest.mark.parametrize("b", P.TRANSIENT_BENCHMARKS, ids=lambda b: b.name)
def test_storage_time_matches_devsim(b):
    _, t = recovered(b.name)
    ref = float(golden(b).header["storage time"])
    assert storage_time(t, "anode") == pytest.approx(ref, rel=b.tolerance, abs=0)


@pytest.mark.parametrize("b", P.TRANSIENT_BENCHMARKS, ids=lambda b: b.name)
def test_the_current_matches_devsim_at_every_sample(b):
    rows = golden(b).rows
    _, t = recovered(b.name)
    assert len(t.points) == len(rows)
    bad: list[str] = []
    for p, r in zip(t.points, rows, strict=True):
        allowed = max(b.tolerance, NOISE_FACTOR * r["spread"])
        e = abs(p.currents["anode"] - r["current"]) / abs(r["current"])
        if e > allowed:
            bad.append(f"{r['time']:.4e} s: devsim {r['current']:.6e}, ddsim {p.currents['anode']:.6e} A/cm^2, off by {e:.2%}, allowed {allowed:.2%}")
    assert not bad, f"{b.name} disagrees with DEVSIM at {len(bad)} of {len(rows)} samples:\n  " + "\n  ".join(bad)
