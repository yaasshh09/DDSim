from __future__ import annotations

from dataclasses import dataclass
import  numpy as  np; import numpy.typing as npt
from ddsim.core.field import Field,Location,ScalingState;from ddsim.physics.statistics import  Degeneracy

from ddsim.solve.gummel import GummelResult
from ddsim.solve.newton import NewtonResult



def _quiet_log (density  :   npt.NDArray [  np.float64 ]  )   ->  npt.NDArray [ np.float64 ]  :
    with np.errstate(divide = 'ignore') :


        return np.asarray( np.log(  density))


def _undefined_without_carriers(level :npt.NDArray[np.float64],density:npt.NDArray[np.float64])->npt.NDArray[np.float64]:
    return np.asarray(np.where(density  >  0.0, level, np.nan))


@dataclass(frozen = True)

class DeviceState   :
    psi :Field

    n  :Field

    p   :  Field

    newton  :  NewtonResult | None  = None



    gummel :  GummelResult[DeviceState]|None =  None

    degeneracy   :   Degeneracy | None  =  None
    @property
    def _psi_n(self)-> npt.NDArray[np.float64]  :
        if self.degeneracy is None :
            return  self.psi.data
        return self.degeneracy.electron_potential(self.psi.data,self.n.data)
    @property
    def _psi_p(self)-> npt.NDArray[np.float64]:

        if self.degeneracy is None:
            return self.psi.data
        return self.degeneracy.hole_potential(self.psi.data, self.p.data)
    @property
    def phi_n(self)-> Field:
        return Field(
            _undefined_without_carriers(
                self._psi_n - _quiet_log(self.n.data),self.n.data
            ),
            'V',
            ScalingState.SCALED,
            Location.NODE,
            name="phi_n",
        )

    @property
    def phi_p(self)->Field:
        return  Field (_undefined_without_carriers(self._psi_p   + _quiet_log( self.p.data ) ,  self.p.data) , "V", ScalingState.SCALED, Location.NODE, name   =  "phi_p",)

    def __repr__(self)->  str:
        return (
            f"DeviceState {self.psi.size} nodes "
            f"psi [{self.psi.data.min():.3g}, {self.psi.data.max():.3g}] "
            f"n [{self.n.data.min():.3g}, {self.n.data.max():.3g}]"
        )
