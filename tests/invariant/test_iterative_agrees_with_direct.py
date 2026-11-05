from __future__ import annotations
import numpy as np
import pytest
from ddsim.device.bjt import npn_2d
from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.extract.iv import gate_sweep, terminal_currents
from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
from ddsim.solve.linear import GmresILU
from tests.regression.test_devsim_gmsh import built


def bench6():
    d = nmos(L_gate=1e-4, drain_voltage=1.0, degenerate=False, **SHORT_CHANNEL_PROCESS)
    return d, TransportModels.for_device(d, mobility="constant"), "gate", [0.0, 0.4, 0.8], "drain"


def bench10():
    d = nmos(L_gate=6.5e-6, drain_voltage=1.0, **SHORT_CHANNEL_PROCESS)
    return d, TransportModels.for_device(d, mobility="arora", field_dependent=True, surface=True), "gate", [0.0, 0.4, 0.8], "drain"


def bench15():
    d = npn_2d(collector_voltage=1.0)
    return d, TransportModels.for_device(d), "base", [0.0, 0.4, 0.7], "collector"


def bench18():
    d = built(1.0)
    return d, TransportModels.for_device(d, mobility="constant"), "gate", [0.0, 0.4, 0.8], "drain"


@pytest.mark.parametrize("case", [bench6, bench10, bench15, bench18], ids=["nmos_1um", "fullstack_65nm", "npn_2d", "nmos_gmsh"])
def test_gmres_ilu_lands_on_the_direct_state(case) -> None:
    d, models, contact, biases, out = case()
    curve = gate_sweep(d, biases[:-1], contact=contact, measure_at=out, models=models)
    assert curve.complete
    target = d.with_bias(**{contact: biases[-1]})
    a = solve_bias_newton(target, models=models, guess=curve.points[-1].state, iterative=False)
    b = solve_bias_newton(target, models=models, guess=curve.points[-1].state, iterative=True)
    assert a.newton is not None and a.newton.converged
    assert b.newton is not None and b.newton.converged
    assert np.max(np.abs(b.psi.data - a.psi.data)) < 1e-8
    for x, y in ((a.n.data, b.n.data), (a.p.data, b.p.data)):
        k = x > 0.0
        assert np.array_equal(k, y > 0.0)
        assert np.max(np.abs(y[k] - x[k]) / x[k]) < 1e-8
    i, j = terminal_currents(target, a, models)[out], terminal_currents(target, b, models)[out]
    assert j == pytest.approx(i, rel=1e-6, abs=0)


def test_the_forcing_term_follows_eisenstat_walker() -> None:
    s = GmresILU(eta_max=0.5)
    assert s._next_eta(1.0) == 0.5
    s._last_norm, s._eta = 1.0, 0.3
    assert s._next_eta(0.1) == pytest.approx(0.9 * 0.01)
    s._last_norm, s._eta = 1.0, 0.45
    assert s._next_eta(0.01) == pytest.approx(0.9 * 0.45**2)
    s._last_norm, s._eta = 1.0, 1e-3
    assert s._next_eta(1e-9) == s.eta_min


def test_gmres_ilu_solves_a_badly_scaled_system() -> None:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    rng = np.random.default_rng(1)
    n = 400
    base = (sp.random(n, n, density=0.01, random_state=rng, format="csr") + sp.diags(4.0 + rng.random(n))).tocsc()
    r, c = 10.0 ** rng.uniform(-12, 12, n), 10.0 ** rng.uniform(-6, 6, n)
    a = (sp.diags(r) @ base @ sp.diags(c)).tocoo()
    b = rng.standard_normal(n)
    s = GmresILU(eta_max=1e-12)
    s.factorize(a.row, a.col, a.data, a.shape)
    np.testing.assert_allclose(s.solve(b), spsolve(base, b / r) / c, rtol=1e-6)
