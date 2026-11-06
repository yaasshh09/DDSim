from __future__ import annotations
import math
from dataclasses import dataclass
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


def band_offsets(nodes: list[Semiconductor], edges: list[Semiconductor], reference: Semiconductor, T: float = C.T_ROOM) -> Bands:
    V_T = C.k_B * T / C.q
    d_n = np.array([(m.chi - reference.chi) / V_T + math.log(m.Nc / reference.Nc) for m in nodes])
    d_p = np.array([-((m.chi + m.Eg) - (reference.chi + reference.Eg)) / V_T + math.log(m.Nv / reference.Nv) for m in nodes])
    return Bands(d_n=d_n, d_p=d_p, eps_r=np.array([m.eps_r / C.EPS_R_SI for m in edges]))
