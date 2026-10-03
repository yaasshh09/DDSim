from __future__ import annotations
import dataclasses
import inspect
from collections.abc import Callable
from typing import Any
import pytest
from ddsim.core.config import CONFIG
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import solve_bias, solve_bias_newton
from ddsim.solve.gummel import gummel_solve
from ddsim.solve.newton import newton_solve


def test_config_cannot_be_changed_at_runtime() -> None:
    name = "max_iterations"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(CONFIG.newton, name, 1)


@pytest.mark.parametrize(
    ("fn", "param", "value"),
    [
        (newton_solve, "update_tol", CONFIG.newton.update_tol),
        (newton_solve, "residual_atol", CONFIG.newton.residual_atol),
        (gummel_solve, "update_tol", CONFIG.gummel.update_tol),
        (solve_equilibrium, "residual_rtol", CONFIG.newton.residual_rtol),
        (solve_bias_newton, "max_psi_step", CONFIG.bias.max_psi_step),
        (solve_bias, "update_tol", CONFIG.gummel.update_tol),
        (pn_diode, "h_min", CONFIG.pn_diode.h_min),
        (nmos, "L_gate", CONFIG.nmos.L_gate),
    ],
)
def test_defaults_come_from_config(fn: Callable[..., Any], param: str, value: float) -> None:
    assert inspect.signature(fn).parameters[param].default is value
