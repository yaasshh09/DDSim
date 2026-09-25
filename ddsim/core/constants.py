from __future__ import annotations
import  math
from typing import  Protocol

import numpy as np
import  numpy.typing  as npt
q :  float =1.602176634e-19

k_B :float = 1.380649e-23


eps_0:float=8.8541878128e-14



h: float  =  6.62607015e-34


m_0: float =9.1093837015e-31

T_ROOM :float = 300.0

def V_T(T  : float=  T_ROOM) ->float:
    return k_B *  T/ q


def SS_min(T  :float = T_ROOM) ->  float :
    return V_T(T )  *  math.log(  10.0 )

_EG_0:float =1.1696

_EG_ALPHA  : float  =  4.73e-4


_EG_BETA   :  float  = 636.0




def Eg(T:float =  T_ROOM)  -> float :
    return _EG_0-_EG_ALPHA* T * T / (T + _EG_BETA)
class BandDensityModel(Protocol):
    def  Nc(self ,   T  :   float )  -> float :


        ...

    def Nv(self,T :float) -> float :
        ...


class  TabulatedBandDensity :


    NC_300:float=2.86e19
    NV_300  :float = 3.10e19


    def Nc(self, T :float) -> float  :

        return float( self.NC_300  *   (T /  T_ROOM)   ** 1.5)
    def  Nv ( self,  T  :   float ) ->  float  :
        return  float(self.NV_300  *  (T  /   T_ROOM )  **  1.5 )

class EffectiveMassBandDensity  :
    def  __init__( self ,   m_e   :  float, m_h  :  float,  M_c  : int  =   6)   ->  None  :

        self.m_e = m_e
        self.m_h= m_h
        self.M_c  =  M_c

    def  _density(  self,   m_star   :   float, T   :   float )  -> float   :
        M=  m_star * m_0

        return float(  2.0   *  (  2.0   *  math.pi  *  M * k_B *  T   / (  h *   h) ) **  1.5  * 1e-6  )
    def Nc ( self, T   :   float  )  ->  float :
        return self._density(self.m_e, T) * self.M_c
    def Nv(self, T :float) -> float:
        return self._density(self.m_h, T)
BAND_DENSITY  : BandDensityModel= TabulatedBandDensity()

def Nc(T: float= T_ROOM)->float:

    return BAND_DENSITY.Nc(T)



def Nv(T :  float= T_ROOM)-> float :


    return BAND_DENSITY.Nv(T)

N_I_300 : float  =1.0e10

def n_i(T :float=  T_ROOM) -> float  :
    gt= Eg(T_ROOM)/ (2.0*V_T(T_ROOM))-Eg(T) / (2.0* V_T(T)) ; return float( N_I_300   *   (T  /  T_ROOM  ) **   1.5   *  math.exp(  gt  ) )

EPS_R_SI :  float =11.7
EPS_R_OX:float  =3.9



def eps_Si()  -> float :
    return EPS_R_SI  *  eps_0




def eps_ox ( )  -> float :
    return EPS_R_OX*eps_0

AUGER_C_N   :   float  =   2.8e-31


AUGER_C_P:  float =9.9e-32

MU_N_300  :  float   =  1417.0


MU_P_300:float = 470.0

V_SAT_N_300: float=1.07e7


V_SAT_P_300  :  float   =  8.3e6


BETA_N:float=2.0


BETA_P: float=1.0
def mu_n(T : float  = T_ROOM) ->  float :
    return MU_N_300




def mu_p(T:  float =  T_ROOM)  ->float :
    return MU_P_300


def v_sat_n( T  :  float  =  T_ROOM)  ->  float  :
    return V_SAT_N_300


def v_sat_p(T  :  float  =  T_ROOM )  ->  float :
    return V_SAT_P_300

def D_n(T: float = T_ROOM)  ->float :
    return V_T(  T  )  *   mu_n( T)



def D_p(T  : float =T_ROOM) ->float :
    return V_T(T)* mu_p(T)
TAU_N_MAX: float = 1e-5

TAU_P_MAX: float=3e-6

TAU_N_MIN  :  float  = 0.0

TAU_P_MIN  :  float = 0.0
N_REF_SRH  :  float  =  5e16
GAMMA_SRH:float=1.0

CHI_SI  : float =  4.05

PHI_M_N_POLY :float=CHI_SI

PHI_M_P_POLY  :float =  CHI_SI+Eg()


PHI_M_MIDGAP : float =CHI_SI + Eg() / 2.0

def semiconductor_work_function(
    net_doping: float  |npt.NDArray[np.float64], T:  float  = T_ROOM
)  ->  npt.NDArray[np.float64]  :
    phiF=V_T(T)*np.arcsinh(
        np.asarray(net_doping,dtype=np.float64)/(2.0* n_i(T))
    )

    return np.asarray(CHI_SI+Eg(T)/ 2.0-phiF)



def work_function_difference(metal : float, net_doping :float | npt.NDArray[np.float64], T: float =  T_ROOM,)  -> npt.NDArray[np.float64] :

    return np.asarray(metal  - semiconductor_work_function(net_doping, T))
