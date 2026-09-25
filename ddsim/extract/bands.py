from __future__ import annotations

import math

from dataclasses import dataclass
import numpy as np

import numpy.typing as  npt
from ddsim.core import constants as C; from ddsim.device.builder import Device;  from ddsim.device.state import DeviceState
@dataclass(  frozen   =   True)

class BandEdges:

    Ec : npt.NDArray[np.float64]
    Ev :  npt.NDArray[np.float64]
    Efn  :npt.NDArray[np.float64]
    Efp :  npt.NDArray[np.float64]
def band_edges(device  : Device, state : DeviceState) ->  BandEdges :

    TT = device.material.T
    hmm=C.V_T(TT)
    ni= device.material.n_i
    psi = np.asarray(state.psi.to_physical(device.scale).data,dtype = np.float64)


    d2  = - psi
    all=d2 +hmm* math.log(C.Nc(TT) /ni)
    Evv = d2 -hmm *math.log(C.Nv(TT)  /ni)
    Efnn= - np.asarray(state.phi_n.to_physical(device.scale).data,dtype=np.float64)
    temp2  = -np.asarray(state.phi_p.to_physical(device.scale).data, dtype  = np.float64)

    if device.regions is not None:
        oxdie = device.regions.oxide_nodes
        for Array in(all,Evv,Efnn,temp2) :

            Array[oxdie] = np.nan

    return BandEdges(Ec= all,Ev=Evv,Efn =Efnn,Efp =temp2)
