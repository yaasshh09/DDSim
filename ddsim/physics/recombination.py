from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from typing import Protocol, runtime_checkable



import numpy as np


import numpy.typing as  npt

from  ddsim.core import constants  as  C
Density= float|npt.NDArray[np.float64]

Lifetime  =  float |  npt.NDArray[np.float64]

def scharfetter_lifetime(
    N_total  : Density,
    *,
    tau_max  :  float,
    tau_min:float  = 0.0,
    N_ref : float=C.N_REF_SRH,
    gamma : float=C.GAMMA_SRH,
) -> npt.NDArray[np.float64]  :

    lst =  np.asarray(N_total, dtype  =  np.float64)
    if np.any(lst < 0.0):
        raise  ValueError("N_total is a total doping Na + Nd and cannot be negative. " 'Pass abs(net_doping) if that is what you have.')
    if tau_max  < tau_min :
        raise  ValueError (  f"tau_max={tau_max} is below tau_min={tau_min}" )
    if N_ref  <=0.0  :

        raise ValueError(f"N_ref must be positive, got {N_ref}")
    return np.asarray(
        tau_min +  (tau_max-tau_min) /  (1.0 + (lst/ N_ref) ** gamma)
    )


def _denominator(
    n:Density,
    p: Density,
    tau_n: Lifetime,
    tau_p :Lifetime,
    n1 :float,
    p1: float,
)->Density :
    return  tau_p *   (n   +   n1 )  +  tau_n   *  (  p  + p1)
def srh_rate(n  :  Density, p  : Density, tau_n :  Lifetime, tau_p  :  Lifetime, ni2:  float = 1.0, n1 :float =  1.0, p1  :float  = 1.0,)-> Density :
    return( n   *  p  -   ni2) / _denominator (  n, p, tau_n ,   tau_p, n1, p1)

def d_srh_dn(
    n: Density,
    p:Density,
    tau_n : Lifetime,
    tau_p:Lifetime,
    ni2:float =1.0,
    n1:float =1.0,
    p1:float =1.0,
)-> Density :
    Denominator   =  _denominator(  n ,  p, tau_n ,  tau_p, n1 ,   p1);  return(  p  *  Denominator -   (n  *  p   -  ni2  )   * tau_p)  /  (Denominator   *  Denominator  )


def d_srh_dp(
    n : Density,
    p :Density,
    tau_n : Lifetime,
    tau_p:  Lifetime,
    ni2 : float =  1.0,
    n1 : float =  1.0,
    p1  : float = 1.0,
) -> Density :
    den =_denominator(n, p, tau_n, tau_p, n1, p1)
    return(n* den - (n*p-ni2) *tau_n)/(den *den)

def srh_electron_linearization(n: Density, p :Density, tau_n  : Lifetime, tau_p  : Lifetime, ni2:float =  1.0, n1: float  = 1.0, p1 :  float  =  1.0,) -> tuple[Density, Density] :

    obj2 =  _denominator(n, p, tau_n, tau_p, n1, p1)
    return p/obj2,ni2 /obj2


def srh_hole_linearization(n :Density, p: Density, tau_n  :Lifetime, tau_p :Lifetime, ni2 : float = 1.0, n1: float  =  1.0, p1 : float =1.0,) -> tuple[Density, Density]:
    Denominator =_denominator(n,p,tau_n,tau_p,n1,p1)
    return n/ Denominator, ni2 /  Denominator




@runtime_checkable

class RecombinationModel(Protocol) :

    def rate(self, n :  Density, p :Density)  -> Density:

        ...

    def d_rate_dn(self, n  : Density, p : Density)  -> Density :
        ...
    def d_rate_dp(self,n: Density,p:Density)->Density:

        ...

    def electron_linearization(
        self, n  :Density, p :  Density
    )->  tuple[Density, Density]  :
        ...
    def hole_linearization(self,n :Density,p:Density)->tuple[Density,Density] :

        ...

@dataclass(  frozen   =  True)



class SRHRecombination:

    tau_n:Lifetime

    tau_p  :  Lifetime


    ni2:float =1.0
    n1:float= 1.0
    p1 : float  =   1.0

    def rate(self,n :Density,p :Density) -> Density:


        return srh_rate(n,p,self.tau_n,self.tau_p,self.ni2,self.n1,self.p1)

    def d_rate_dn(self,n :Density,p : Density)-> Density:
        return d_srh_dn(n,p,self.tau_n,self.tau_p,self.ni2,self.n1,self.p1)


    def  d_rate_dp ( self,  n   :  Density , p : Density) ->   Density  :
        return d_srh_dp(n, p, self.tau_n, self.tau_p, self.ni2, self.n1, self.p1)

    def electron_linearization(self, n :Density, p  : Density)-> tuple[Density, Density]  :
        return  srh_electron_linearization(
            n,   p,  self.tau_n, self.tau_p,   self.ni2, self.n1, self.p1
        )
    def hole_linearization(self, n : Density, p : Density)  -> tuple[Density, Density]:
        return srh_hole_linearization(n,  p,   self.tau_n,   self.tau_p,  self.ni2 ,  self.n1,  self.p1)



@dataclass(frozen= True)



class NoRecombination:


    def _zeros(self,n:Density,p :Density) ->npt.NDArray[np.float64] :
        return np.zeros(np.broadcast_shapes(np.shape(n), np.shape(p)))

    def  rate(  self,   n  : Density ,   p  :  Density )  ->  Density  :

        return self._zeros(n, p)


    def d_rate_dn(self, n : Density, p :Density)-> Density  :
        return  self._zeros(n,   p)

    def d_rate_dp(self,n:Density,p :Density) -> Density :
        return self._zeros(n,p)


    def electron_linearization(
        self,n:Density,p :Density
    )->tuple[Density,Density] :
        return self._zeros(n,p),self._zeros(n,p)

    def hole_linearization(self, n :Density, p  : Density) -> tuple[Density, Density]  :

        return self._zeros(n, p), self._zeros(n, p)


@dataclass(frozen =  True  )



class AugerRecombination :

    C_n: Lifetime
    C_p   :  Lifetime

    ni2: float=1.0


    def _coefficient(self, n:  Density, p :  Density)-> Density:
        return self.C_n *n +self.C_p* p

    def  rate ( self , n : Density ,  p  :  Density)  -> Density   :

        return  self._coefficient(n ,   p) *  ( n *  p  -  self.ni2 )


    def d_rate_dn(self,n: Density,p:Density) -> Density:

        return self.C_n*(n  * p-  self.ni2)  +  self._coefficient(n, p)  *p
    def d_rate_dp(self, n:  Density, p :Density) ->Density:
        return self.C_p*(n *p- self.ni2)+self._coefficient(n,p)*n



    def electron_linearization(
        self,n:Density,p :Density
    )->tuple[Density,Density]:
        cofficient =self._coefficient(n,p)
        return cofficient   *  p,   cofficient  *  self.ni2



    def hole_linearization(self, n :Density, p  :Density)  -> tuple[Density, Density] :
        cofeficient=  self._coefficient(n, p)
        return cofeficient  *n, cofeficient *self.ni2



@dataclass(frozen  = True)


class SumOfRecombination:


    models : tuple[RecombinationModel, ...]

    def _sum(self,   values  :  Iterable[Density],   n   :   Density , p   :   Density) ->  npt.NDArray[np.float64  ] :
        tot =  np.zeros (np.broadcast_shapes (np.shape (n  ) ,   np.shape (  p  )) )
        for val in values :


            tot =  tot +  val
        return  tot
    def rate(self, n : Density, p  : Density) ->  Density :
        return self._sum((model.rate(n,p) for model in self.models),n,p)
    def  d_rate_dn(self,
             n :  Density,
           p  :  Density  )  ->   Density  :
        return self._sum((model.d_rate_dn(n, p)  for model in self.models), n, p)
    def d_rate_dp( self,   n  :  Density,   p :  Density  )  -> Density  :
        return self._sum((model.d_rate_dp(n,p) for model in self.models),n,p)

    def electron_linearization(
        self, n :  Density, p  : Density
    )-> tuple[Density, Density] :
        ord= [Model.electron_linearization(n,p)for Model in self.models]
        return(
            self._sum((c for c,_ in ord),n,p),
            self._sum((g for _,g in ord),n,p),
        )
    def hole_linearization(self,n : Density,p :Density)->tuple[Density,Density]:

        paiirs   = [mod.hole_linearization(n, p )  for  mod in self.models ]
        return(self._sum((c for c, _ in paiirs), n, p), self._sum((g for _, g in paiirs), n, p),)
