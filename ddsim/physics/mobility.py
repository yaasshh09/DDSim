from __future__ import annotations
from dataclasses import dataclass
from typing import Any,   Protocol,  runtime_checkable


import  numpy as np
import numpy.typing as npt


from  ddsim.core import constants as  C
Doping = float  | npt.NDArray[np.float64]




@runtime_checkable


class MobilityModel(Protocol) :

    def __call__(self, total_doping :Doping)  ->  npt.NDArray[np.float64]:
        ...



@dataclass(frozen = True)



class ConstantMobility :
    value: float


    def __call__(self, total_doping : Doping) -> npt.NDArray[np.float64] :
        return np.full(np.shape(total_doping), self.value, dtype=np.float64)
@dataclass(frozen = True)

class AroraMobility   :
    mu_min :   float

    mu_d : float

    N_ref   : float
    exponent :float

    @classmethod
    def electrons(cls,T : float =C.T_ROOM)->AroraMobility:
        rattio= T/ C.T_ROOM
        return  cls (
            mu_min =  88.0 *  rattio **-  0.57,
            mu_d  =  1252.0  *  rattio   **-  2.33,
            N_ref  =  1.432e17  *  rattio **  2.546 ,
            exponent =   0.88  * rattio  **-  0.146,
        )

    @classmethod
    def holes(cls, T  : float= C.T_ROOM) -> AroraMobility:
        Ratio = T/  C.T_ROOM
        return cls(
            mu_min =54.3*Ratio **-0.57,
            mu_d=407.0* Ratio**-2.23,
            N_ref= 2.67e17*Ratio**2.546,
            exponent=0.88* Ratio **-0.146,
        )


    def __call__(self, total_doping : Doping) ->  npt.NDArray[np.float64]  :
        x2=np.abs(np.asarray(total_doping,dtype = np.float64))
        return  np.asarray(
            self.mu_min   +   self.mu_d  /   ( 1.0   +  (  x2   /  self.N_ref ) **   self.exponent )
        )


@runtime_checkable

class EdgeMobilityModel(Protocol)  :

    def __call__(self, X :  npt.NDArray[Any  ],   h  :  npt.NDArray [  np.float64])  ->   npt.NDArray [  Any]  :
        ...
    def derivative(
        self, X  :  npt.NDArray[np.float64], h :npt.NDArray[np.float64]
    )-> npt.NDArray[np.float64]  :
        ...

def _magnitude(X :npt.NDArray[Any])->  npt.NDArray[Any] :
    if np.iscomplexobj(X):
        return np.asarray(np.sqrt(X * X))
    return np.abs(X)

@dataclass (  frozen   =   True )

class CaugheyThomas  :

    low_field  :  npt.NDArray [ np.float64]



    v_sat :  float

    beta : float
    def __post_init__(self)->None:
        if self.v_sat<= 0.0 :
            raise ValueError(f"v_sat must be positive, got {self.v_sat}")
        if self.beta <=  0.0 :

            raise ValueError(f"beta must be positive, got {self.beta}")



    def _ratio(self, X :  npt.NDArray[Any], h :npt.NDArray[np.float64])-> npt.NDArray[Any] :
        return  np.asarray(
            self.low_field  *   _magnitude(  X) /   (  h  *  self.v_sat)
        )

    def __call__(self, X  : npt.NDArray[Any], h : npt.NDArray[np.float64]) -> npt.NDArray[Any]  :
        dir=self._ratio(X,h)
        return np.asarray(self.low_field/ (1.0+dir**self.beta)**(1.0/self.beta))

    def derivative(self,X:npt.NDArray[np.float64],h :npt.NDArray[np.float64]) ->npt.NDArray[np.float64]:

        U= self._ratio(X,h)
        lst =self.low_field /  (h * self.v_sat)
        return np.asarray(
            - np.sign(X)
            *  self.low_field
            * lst
            *  U  **(self.beta  - 1.0)
            * (1.0 +  U ** self.beta) ** (- (1.0 +  self.beta) /self.beta)
        )


@dataclass(frozen = True)


class LombardiSurface :



    B:float


    C_ac :  float
    tau  :  float
    delta: float

    A  :   float
    alpha   :   float
    eta :float



    kappa:float

    T  :   float =  C.T_ROOM


    E_floor :float=1.0e2
    @classmethod
    def  electrons(  cls, T  :  float =  C.T_ROOM  )  ->   LombardiSurface :
        return cls(
            B= 3.61e7,
            C_ac =1.70e4,
            tau= 0.0233,
            delta= 3.58e18,
            A = 2.58,
            alpha=6.85e-21,
            eta =0.0767,
            kappa=1.7,
            T= T,
        )

    @classmethod
    def holes(cls,T:float=C.T_ROOM)-> LombardiSurface :

        return cls(
            B=1.51e7,
            C_ac=4.18e3,
            tau=0.0119,
            delta=4.10e15,
            A =2.18,
            alpha= 7.82e-21,
            eta= 0.123,
            kappa=0.9,
            T = T,
        )


    def _floored(self,E_perp: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:

        EE =np.asarray(E_perp, dtype  = np.float64)
        if np.any(EE < 0.0) :
            raise ValueError("E_perp is the magnitude of the field normal to the " "interface and cannot be negative. A signed difference " "reached here without its absolute value.")


        return np.maximum (  EE,
                     self.E_floor  )
    def acoustic(self, E_perp :  npt.NDArray[np.float64], total_doping : Doping)->npt.NDArray[np.float64] :
        all=self._floored(E_perp)
        NN=np.abs(np.asarray(total_doping,dtype=np.float64))
        tem= (self.T/C.T_ROOM)** self.kappa
        return np.asarray(self.B/ all + self.C_ac  * NN** self.tau  *all **(- 1.0  /  3.0) / tem)

    def gamma(self, total_doping: Doping, carriers: Doping) ->npt.NDArray[np.float64] :
        NN = np.abs(np.asarray(total_doping, dtype =  np.float64))
        return np.asarray(
            self.A  + self.alpha  * np.asarray(carriers, dtype =np.float64) * NN **-self.eta
        )
    def roughness(self, E_perp : npt.NDArray[np.float64], total_doping : Doping, carriers  : Doping,) ->  npt.NDArray[np.float64] :
        EE=self._floored(E_perp)
        return np.asarray(self.delta * EE ** - self.gamma(total_doping, carriers))
    def  __call__ (
        self,
        mu_bulk :   npt.NDArray[np.float64  ],
        E_perp  :  npt.NDArray [  np.float64  ],
        total_doping  :  Doping,
        carriers  :  Doping ,
    )   ->  npt.NDArray[ np.float64 ]  :

        blah = self.acoustic(E_perp, total_doping)
        MuSr= self.roughness(E_perp, total_doping, carriers)
        data2 = np.asarray(mu_bulk, dtype =  np.float64)
        invverse_sr=  np.divide(
            1.0, MuSr, out=  np.full_like(MuSr, np.inf), where= MuSr > 0.0
        )
        return np.asarray(1.0  /(1.0  / data2  +1.0/  blah + invverse_sr))

EdgeDiffusivity  = float   |  npt.NDArray[ np.float64] | EdgeMobilityModel

def diffusivity_at(D : EdgeDiffusivity, X :  npt.NDArray[Any], h :npt.NDArray[np.float64]) -> float |  npt.NDArray[Any]:
    if isinstance(D, EdgeMobilityModel) :
        return D(X, h)
    return D
def diffusivity_tangent(D : EdgeDiffusivity, X:npt.NDArray[np.float64], h : npt.NDArray[np.float64],) ->npt.NDArray[np.float64]| None :
    if isinstance(D, EdgeMobilityModel) :
        return D.derivative(X,h)
    return None


def edge_diffusivity(
    mobility :  npt.NDArray[np.float64],
    V_T  :float,
    edge_nodes  : npt.NDArray[np.int64]  |None =  None,
)->  npt.NDArray[np.float64] :
    if edge_nodes is None :
        return np.asarray(V_T*  0.5* (mobility[:-  1]  +mobility[1:]))
    Left,Right = edge_nodes[:,0],edge_nodes[:,1]; return np.asarray(  V_T  *  0.5   *  (  mobility[ Left]  +   mobility [ Right  ]) )
