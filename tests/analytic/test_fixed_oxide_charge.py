from __future__ import annotations
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.core.config import CONFIG
from ddsim.device.builder import build_device
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.extract.cv import Response, cv_sweep, terminal_charge
from ddsim.extract.iv import gate_sweep
from ddsim.extract.params import threshold_constant_current

QF = 5e11


def silicon_psi(d):
    psi = solve_equilibrium(d).psi.to_physical(d.scale).data
    return psi[d.regions.semiconductor_volume > 0.0]


@pytest.mark.parametrize('vg', [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("qf", [QF, -QF])
def test_fixed_charge_moves_flatband_by_minus_q_qf_over_c_ox(vg: float, qf: float) -> None:
    shift = -C.q * qf * CONFIG.mos_cap.t_ox / C.eps_ox()
    a = silicon_psi(mos_cap(gate_voltage=vg))
    b = silicon_psi(mos_cap(gate_voltage=vg + shift, Q_f=qf))
    assert np.max(np.abs(b - a)) < 1e-7


def test_fixed_charge_is_not_a_no_op() -> None:
    a = silicon_psi(mos_cap())
    b = silicon_psi(mos_cap(Q_f=QF))
    assert np.max(np.abs(b - a)) > 1e-2


def test_fixed_charge_needs_an_interface() -> None:
    d = pn_diode()
    with pytest.raises(ValueError, match="interface"):
        build_device(d.mesh, d.doping, d.contacts, Q_f=QF)


def test_the_coupled_solve_sees_the_same_fixed_charge() -> None:
    d = mos_cap(gate_voltage=1.0, n_silicon=41, n_oxide=3, Q_f=QF)
    a = solve_equilibrium(d)
    b = solve_bias_newton(d)
    assert b.newton is not None and b.newton.converged, b.newton
    np.testing.assert_allclose(b.psi.data, a.psi.data, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("qf", [QF, -QF])
def test_the_gate_carries_the_fixed_charge_s_image(qf: float) -> None:
    shift = -C.q * qf * CONFIG.mos_cap.t_ox / C.eps_ox()
    a, b = mos_cap(gate_voltage=0.5), mos_cap(gate_voltage=0.5 + shift, Q_f=qf)
    qa = terminal_charge(a, solve_equilibrium(a), 'gate')
    qb = terminal_charge(b, solve_equilibrium(b), 'gate')
    assert qa - qb == pytest.approx(C.q * qf, rel=1e-6)


@pytest.mark.parametrize("response", list(Response))
@pytest.mark.parametrize("qf", [QF, -QF])
def test_the_cv_curve_shifts_by_minus_q_qf_over_c_ox(response: Response, qf: float) -> None:
    shift = -C.q * qf * CONFIG.mos_cap.t_ox / C.eps_ox()
    vg = [-1.5, -0.5, 0.0, 0.3, 0.6, 1.0, 1.5]
    a = cv_sweep(mos_cap(), "gate", vg, response=response)
    b = cv_sweep(mos_cap(Q_f=qf), "gate", [v + shift for v in vg], response=response)
    assert a.complete and b.complete
    ca = np.array([x.capacitance for x in a.points])
    cb = np.array([x.capacitance for x in b.points])
    np.testing.assert_allclose(cb, ca, rtol=1e-4)


@pytest.mark.parametrize("qf", [1e11, -1e11])
def test_the_mosfet_threshold_shifts_by_minus_q_qf_over_c_ox(qf: float) -> None:
    shift = -C.q * qf * CONFIG.nmos.t_ox / C.eps_ox()
    vg = [0.2, 0.4, 0.6, 0.8, 1.0]
    a, b = nmos(drain_voltage=0.05), nmos(drain_voltage=0.05, Q_f=qf)
    ia = gate_sweep(a, vg, models=TransportModels.for_device(a), step=0.1)
    ib = gate_sweep(b, [v + shift for v in vg], models=TransportModels.for_device(b), step=0.1, start=shift)
    assert ia.complete and ib.complete
    target = float(ia.current[2])
    va = threshold_constant_current(ia.voltage, ia.current, target)
    vb = threshold_constant_current(ib.voltage, ib.current, target)
    assert abs((vb - va) - shift) < 1e-3
