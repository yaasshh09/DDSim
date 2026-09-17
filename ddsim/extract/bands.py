"""Band edges and quasi-Fermi levels from a solved state [eV].

The band diagram a student reads. Energies are referenced to the equilibrium
Fermi level at zero, so with psi measured from the intrinsic level (the sign
convention in docs/01-physics.md) the intrinsic level is E_i = -psi in eV.

The edges follow from the Boltzmann relations the solver uses:

    E_c = E_i + kT ln(Nc / n_i)        E_v = E_i - kT ln(Nv / n_i)

so the gap they imply is kT ln(Nc Nv / n_i^2). That is not Eg(300) exactly,
because n_i is anchored at 1e10 rather than derived; the difference is a few
meV and recorded in docs/07-decisions.md. Using Eg instead would draw bands
inconsistent with the carrier densities on the same screen.

The quasi-Fermi levels are E_fn = -phi_n and E_fp = -phi_p.

Oxide nodes have no silicon band edges and report NaN, rather than a number
that would draw a band through an insulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState


@dataclass(frozen=True)
class BandEdges:
    """Per node energies of one solved state [eV]."""

    Ec: npt.NDArray[np.float64]
    Ev: npt.NDArray[np.float64]
    Efn: npt.NDArray[np.float64]
    Efp: npt.NDArray[np.float64]


def band_edges(device: Device, state: DeviceState) -> BandEdges:
    """The band diagram of a solved state [eV].

    Args:
        device: the device the state was solved on.
        state: a converged solution.
    """
    T = device.material.T
    VT = C.V_T(T)
    n_i = device.material.n_i
    psi = np.asarray(state.psi.to_physical(device.scale).data, dtype=np.float64)

    Ei = -psi
    Ec = Ei + VT * math.log(C.Nc(T) / n_i)
    Ev = Ei - VT * math.log(C.Nv(T) / n_i)
    Efn = -np.asarray(state.phi_n.to_physical(device.scale).data, dtype=np.float64)
    Efp = -np.asarray(state.phi_p.to_physical(device.scale).data, dtype=np.float64)

    if device.regions is not None:
        oxide = device.regions.oxide_nodes
        for array in (Ec, Ev, Efn, Efp):
            array[oxide] = np.nan

    return BandEdges(Ec=Ec, Ev=Ev, Efn=Efn, Efp=Efp)
