from __future__ import annotations
from functools import cache
import numpy as np
from ddsim.device.mosfet import narrow_nmos, nmos
from ddsim.device.transport import TransportModels
from ddsim.extract.iv import gate_sweep
from ddsim.extract.params import threshold_constant_current
from ddsim.extract.rolloff import REFERENCE_CURRENT

PROCESS = dict(L_gate=1e-4, degenerate=False, n_contact=3, n_sd=8, n_channel=10, n_silicon=16, n_oxide=3, h_min_x=1e-6, h_min_y=2e-7)
GATE = [round(0.1 * k, 3) for k in range(13)]


def threshold(d, width: float, halves: int) -> float:
    c = gate_sweep(d, GATE, models=TransportModels.for_device(d, mobility="constant"))
    assert c.complete, c.message
    return threshold_constant_current(np.array([p.voltage for p in c.points]), np.array([halves * p.current for p in c.points]), REFERENCE_CURRENT, width=width)


@cache
def flat() -> float:
    return threshold(nmos(drain_voltage=0.05, **PROCESS), 1.0, 1)


@cache
def narrow(width: float) -> float:
    return threshold(narrow_nmos(width, drain_voltage=0.05, n_width=8 if width > 1e-4 else 6, **PROCESS), width, 2)


def test_a_wide_device_has_the_2d_threshold() -> None:
    assert abs(narrow(2e-4) - flat()) < 0.01, f"at 2 um the field oxide edge should be a small share of the width: 3D {narrow(2e-4):.4f} V against 2D {flat():.4f} V"


def test_the_threshold_rises_as_the_channel_narrows() -> None:
    v = [narrow(w) for w in (2e-4, 5e-5, 2.5e-5)]
    assert v[0] < v[1] < v[2], (
        f"thresholds at 2, 0.5 and 0.25 um are {v}. The gate runs on over the thick field oxide beside the channel, "
        "and its fringe field there depletes silicon outside the channel width. That charge has to be imaged on the "
        "gate before inversion, but no current flows through it, so per unit width the gate needs more voltage, and "
        "the extra charge is a fixed amount per edge, a larger share the narrower the channel"
    )
    a, b = v[1] - flat(), v[2] - flat()
    assert 1.5 < b / a < 3.5, f"a fixed charge per edge makes the shift go as 1/W, so halving W should about double it: {a * 1e3:.1f} mV at 0.5 um, {b * 1e3:.1f} mV at 0.25 um"
