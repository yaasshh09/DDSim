from __future__ import annotations
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
import numpy as np
import numpy.typing as npt
from ddsim.core import constants as C

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Semiconductor:
    name: str

    eps_r: float

    chi: float

    Eg: float

    Nc: float

    Nv: float

    @classmethod
    def silicon(cls, T: float = C.T_ROOM) -> Semiconductor:
        return cls(name="silicon", eps_r=C.EPS_R_SI, chi=C.CHI_SI, Eg=C.Eg(T), Nc=C.Nc(T), Nv=C.Nv(T))

    @classmethod
    def germanium(cls, T: float = C.T_ROOM) -> Semiconductor:
        s = (T / C.T_ROOM) ** 1.5
        return cls(name="germanium", eps_r=C.GE_EPS_R, chi=C.GE_CHI, Eg=C.varshni(C.GE_EG_0, C.GE_EG_ALPHA, C.GE_EG_BETA, T),
                   Nc=C.GE_NC_300 * s, Nv=C.GE_NV_300 * s)

    @classmethod
    def gaas(cls, T: float = C.T_ROOM) -> Semiconductor:
        s = (T / C.T_ROOM) ** 1.5
        return cls(name="gaas", eps_r=C.GAAS_EPS_R, chi=C.GAAS_CHI, Eg=C.varshni(C.GAAS_EG_0, C.GAAS_EG_ALPHA, C.GAAS_EG_BETA, T),
                   Nc=C.GAAS_NC_300 * s, Nv=C.GAAS_NV_300 * s)

    @classmethod
    def sige(cls, x: float, T: float = C.T_ROOM) -> Semiconductor:
        if not 0.0 <= x < C.SIGE_X_MAX:
            raise ValueError(f"Si(1-x)Ge(x) needs 0 <= x < {C.SIGE_X_MAX}, the Si-like range its formulas hold in; got {x}")
        si = cls.silicon(T)
        return replace(si, name="sige", eps_r=si.eps_r + C.SIGE_EPS_LINEAR * x,
                       Eg=si.Eg + C.SIGE_EG_LINEAR * x + C.SIGE_EG_QUADRATIC * x * x)


MATERIALS: dict[str, Callable[[float], Semiconductor]] = {
    "silicon": Semiconductor.silicon,
    "germanium": Semiconductor.germanium,
    "gaas": Semiconductor.gaas,
}


@dataclass(frozen=True)
class Bands:
    d_n: Array

    d_p: Array

    eps_r: Array

    @property
    def n_local(self) -> Array:
        return np.asarray(np.exp(0.5 * (self.d_n + self.d_p)))

    def edge_offsets(self, left: npt.NDArray[np.int64], right: npt.NDArray[np.int64]) -> tuple[Array, Array]:
        return np.asarray(self.d_n[right] - self.d_n[left]), np.asarray(-(self.d_p[right] - self.d_p[left]))


def layered_bands(x: Array, boundaries: tuple[float, ...], layers: tuple[Semiconductor, ...], T: float = C.T_ROOM) -> Bands:
    if len(layers) != len(boundaries) + 1:
        raise ValueError(f"{len(boundaries)} boundaries make {len(boundaries) + 1} layers, got {len(layers)} materials")
    k = np.searchsorted(np.asarray(boundaries), x, side="right")
    j = np.searchsorted(np.asarray(boundaries), 0.5 * (x[:-1] + x[1:]), side="right")
    return band_offsets([layers[i] for i in k], [layers[i] for i in j], Semiconductor.silicon(T), T)


def band_offsets(nodes: list[Semiconductor], edges: list[Semiconductor], reference: Semiconductor, T: float = C.T_ROOM) -> Bands:
    V_T = C.k_B * T / C.q
    d_n = np.array([(m.chi - reference.chi) / V_T + math.log(m.Nc / reference.Nc) for m in nodes])
    d_p = np.array([-((m.chi + m.Eg) - (reference.chi + reference.Eg)) / V_T + math.log(m.Nv / reference.Nv) for m in nodes])
    return Bands(d_n=d_n, d_p=d_p, eps_r=np.array([m.eps_r / C.EPS_R_SI for m in edges]))
