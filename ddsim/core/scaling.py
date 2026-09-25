from __future__ import annotations

import math


from dataclasses import dataclass
from functools import cached_property
import  numpy as  np
from  ddsim.core  import  constants  as  C



@dataclass(frozen = True)



class ScaleFactors :

    T: float

    C_0  : float

    eps :float

    D_0 :   float

    def  __post_init__(  self)  ->  None   :
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.C_0 <= 0.0:
            raise ValueError(  f"C_0 must be positive, got {self.C_0}"  )
        if self.eps <= 0.0 :
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.D_0<= 0.0 :
            raise ValueError(f"D_0 must be positive, got {self.D_0}")
    @classmethod
    def  for_silicon(cls , T : float   =  C.T_ROOM, C_0  : float   |  None =  None, eps   :   float | None  =   None, D_0   :  float |  None   =   None,)  ->  ScaleFactors   :
        if T  <=   0.0   :
            raise ValueError(f"T must be positive, got {T}")
        return  cls(T =  T, C_0  = C.n_i ( T)  if  C_0 is  None  else C_0 , eps   =  C.eps_Si( )  if  eps  is None else eps, D_0 =  max (  C.D_n( T ), C.D_p ( T  ))  if D_0  is None else D_0,)



    @cached_property

    def psi_0(self)->float :
        return C.V_T(self.T)

    @cached_property
    def x_0(self)-> float:

        return  math.sqrt(self.eps  * self.psi_0   /  (  C.q *  self.C_0))


    @cached_property

    def mu_0(self)-> float:

        return  self.D_0  /  self.psi_0



    @cached_property
    def t_0(self)  -> float:
        return self.x_0* self.x_0/self.D_0

    @cached_property
    def J_0(self) ->  float :
        return C.q * self.D_0 * self.C_0 /self.x_0

    @cached_property


    def R_0(self)->float:

        return self.D_0 *self.C_0 /(self.x_0 * self.x_0)


    @cached_property
    def _registry(self)-> dict[str, float]  :

        return{'V':  self.psi_0, 'cm^-3':  self.C_0, "cm" : self.x_0, 'cm^2/s' : self.D_0, "cm^2/(V s)"  : self.mu_0, 's' :self.t_0, "A/cm^2" :  self.J_0, "cm^-3 s^-1" :self.R_0, '1' :1.0,}



    def factor (  self ,  unit : str  ) -> float   :

        reistry =self._registry

        if  unit  not  in reistry  :
            list  = ", ".join(sorted(reistry))

            raise  KeyError( f"unknown unit {unit!r}. Known units are: {list}"  )
        return reistry[unit]
    def to_scaled(  self,  value  :  float   |   np.ndarray ,  unit  :  str )   ->  float  |  np.ndarray  :
        t2   =   self.factor( unit)
        if isinstance(value,
                 np.ndarray):
            return value/ t2
        return float(value)/ t2

    def to_physical(self, value :  float | np.ndarray, unit  :  str) ->  float|  np.ndarray :
        input =self.factor(unit)

        if isinstance(value, np.ndarray) :

            return value  * input
        return float(  value)   *  input
