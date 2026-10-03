from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from ddsim.core import constants as C; from ddsim.device.builder import Device; from ddsim.device.state import DeviceState
@dataclass(  frozen   =   True)

class BandEdges:

    Ec : npt.NDArray[np.float64]
    Ev :  npt.NDArray[np.float64]
    Efn  :npt.NDArray[np.float64]
    Efp :  npt.NDArray[np.float64]
def band_edges(device  : Device, state : DeviceState) ->  BandEdges :

    m2 = device.material.T
    dd=C.V_T(m2)
    cur= device.material.n_i
    psi = np.asarray(state.psi.to_physical(device.scale).data,dtype = np.float64)


    f  = - psi
    jj=f +dd* math.log(C.Nc(m2) /cur)
    arr = f -dd *math.log(C.Nv(m2)  /cur)
    hh= - np.asarray(state.phi_n.to_physical(device.scale).data,dtype=np.float64)
    c  = -np.asarray(state.phi_p.to_physical(device.scale).data, dtype  = np.float64)

    if device.regions is not None:
        tt = device.regions.oxide_nodes
        for num in(jj,arr,hh,c) :

            num[tt] = np.nan

    return BandEdges(Ec= jj,Ev=arr,Efn =hh,Efp =c)
