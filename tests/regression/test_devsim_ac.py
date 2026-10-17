from __future__ import annotations
from pathlib import Path
import pytest
from ddsim.device.builder import build_device
from ddsim.device.doping import Uniform
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.discretize.boundary import OhmicContact
from ddsim.extract.ac import admittance, noise
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.recombination import SRHRecombination
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"

NOISE_FACTOR = 3.0


def golden(b: P.AcBenchmark) -> P.AcGoldenCurve:
    return P.read_ac_golden(str(GOLDEN_DIR / f"{b.name}.csv"))


def models(d, tau):
    if tau is None:
        return TransportModels.for_device(d)
    s = d.scale
    ni = d.material.n_i / s.C_0
    return TransportModels.for_device(d, recombination=SRHRecombination(tau_n=tau / s.t_0, tau_p=tau / s.t_0, ni2=ni * ni, n1=ni, p1=ni))


def mos_points(b, v):
    m = P.MOS_BENCHMARKS[0]
    d = mos_cap(substrate_doping=m.substrate_doping, t_ox=m.t_ox, t_si=m.t_si, n_silicon=m.n_silicon, n_oxide=m.n_oxide, h_min=m.h_min, work_function=m.work_function, gate_voltage=v)
    md = models(d, b.tau)
    st = solve_bias_newton(d, md, guess=solve_equilibrium(d, quasi_fermi=frozen_quasi_fermi(d)))
    assert st.newton.converged
    for f in b.frequencies:
        yield f, admittance(d, st, f, md, width=d.mesh.x_axis.length).y("gate", "gate"), None


def diode_points(b, v):
    d = pn_diode(Na=P.AC_DIODE_NA, Nd=P.AC_DIODE_ND, length=P.AC_DIODE_LENGTH, junction=P.AC_DIODE_JUNCTION, n_nodes=1500, h_min=2e-7, anode_voltage=v)
    md = models(d, b.tau)
    st = solve_bias_ramped(d, md)
    assert st.newton.converged
    for f in b.frequencies:
        yield f, admittance(d, st, f, md).y("anode", "anode"), None


def bar_points(b, v):
    n = 401
    d = build_device(mesh=uniform_mesh_1d(P.BAR_LENGTH, n), doping=Uniform(P.BAR_DOPING), contacts=(OhmicContact(name="top", node=0, voltage=v), OhmicContact(name="bot", node=n - 1, voltage=0.0)))
    md = models(d, b.tau)
    st = solve_bias_newton(d, md, guess=solve_equilibrium(d))
    for f in b.frequencies:
        nz = noise(d, st, f, md)
        k = nz.contacts.index("top")
        yield f, complex(nz.Y[k, k]), float(nz.open_circuit("bot")[k, k].real)


POINTS = {11: mos_points, 12: diode_points, 13: bar_points}


@pytest.mark.parametrize("b", P.AC_BENCHMARKS, ids=lambda b: b.name)
def test_golden_data_exists(b):
    assert (GOLDEN_DIR / f"{b.name}.csv").is_file(), f"no golden curve for {b.name}. Generate it with tests/regression/devsim_gen/generate_ac.py, see the README there."


@pytest.mark.parametrize("b", P.AC_BENCHMARKS, ids=lambda b: b.name)
def test_golden_header_records_its_provenance(b):
    h = golden(b).header
    assert h["device"] == b.name
    assert h["generator"].startswith("devsim ")
    assert "generated on" in h
    assert "mesh convergence" in h
    assert "devsim spread" in h
    assert h["statistics"] == "Boltzmann"


@pytest.mark.parametrize("b", P.AC_BENCHMARKS, ids=lambda b: b.name)
def test_golden_matches_the_benchmark(b):
    g = golden(b)
    assert float(g.header["tolerance"]) == b.tolerance
    assert float(g.header["devsim refine"]) == b.devsim_refine
    want = [x for v in b.biases for f in b.frequencies for x in (v, f)]
    got = [x for r in g.rows for x in (r["bias"], r["frequency"])]
    assert got == pytest.approx(want, rel=1e-9, abs=0), "the golden points are not the ones parameters.py asks for, the golden data is stale"


@pytest.mark.parametrize("b", P.AC_BENCHMARKS, ids=lambda b: b.name)
def test_ddsim_matches_devsim(b):
    rows = golden(b).rows
    bad: list[str] = []
    i = 0
    for v in b.biases:
        for f, y, sv in POINTS[b.number](b, v):
            r = rows[i]
            i += 1
            allowed = max(b.tolerance, NOISE_FACTOR * r["spread"])
            ref = complex(r["re_y"], r["im_y"])
            e = abs(y - ref) / abs(ref)
            if sv is not None:
                e = max(e, abs(sv - r["s_v"]) / abs(r["s_v"]))
            if e > allowed:
                bad.append(f"{v:+g} V, {f:.3e} Hz: devsim {ref:.6e}, ddsim {y:.6e}, off by {e:.2%}, allowed {allowed:.2%}")
    assert i == len(rows)
    assert not bad, f"{b.name} disagrees with DEVSIM at {len(bad)} of {len(rows)} points:\n  " + "\n  ".join(bad)
