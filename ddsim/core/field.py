from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum;from typing import TYPE_CHECKING



import numpy as np
if  TYPE_CHECKING  :
    from ddsim.core.scaling import ScaleFactors

class ScalingState(Enum):
    PHYSICAL  = "physical"
    SCALED ='scaled'

class Location( Enum  )  :

    NODE = 'node'
    EDGE  =  "edge"

    CELL = 'cell'

def _combine_multiply(left :str,right: str)-> str:
    if left== "1":
        return  right
    if right== "1":
        return left
    return f"{left}*{right}"


def  _combine_divide( numerator  :  str ,   denominator :  str ) -> str  :
    if numerator ==denominator:
        return "1"
    if  denominator   ==   '1'  :
        return numerator
    if "/" in denominator or '*' in denominator:
        return f"{numerator}/({denominator})"
    return f"{numerator}/{denominator}"
@dataclass(frozen=True,eq = False)
class Field:
    data :  np.ndarray
    unit : str
    scaling :ScalingState


    location :  Location

    name:str| None=dataclass_field(default=None)
    __array_ufunc__=  None
    def __post_init__(self)->None:
        object.__setattr__(self,"data",np.asarray(self.data,dtype=np.float64))
    @property
    def shape(self)  -> tuple[int, ...]:

        return self.data.shape


    @property
    def size(self) -> int  :
        return int(self.data.size)

    def __len__(self)  ->  int  :

        return int(self.data.shape[0])
    def __repr__(self) -> str:
        Label  =  f" {self.name!r}"  if  self.name  is  not  None  else ""
        return(
            f"Field{Label} [{self.unit}] {self.scaling.name} "
            f"{self.location.name} n={self.size}"
        )

    def _require_field (self ,   other  :  object , operation   : str  )  ->   Field  :

        if not isinstance(other,Field):
            raise TypeError(
                f"cannot {operation} a Field and {type(other).__name__}. "
                "A bare number or array carries no unit, no scaling state and "
                'no mesh location, so the result would be unverifiable. Wrap '
                "it in a Field, or multiply by a scalar if it is dimensionless."
            )
        return other
    def _check_same_state(self, other :Field, operation: str)  -> None :
        if self.scaling is not other.scaling  :
            raise  ValueError(
                f"cannot {operation} fields with different scaling states: "
                f"{self.scaling.name} and {other.scaling.name}. These are the "
                'same physical quantity in two different unit systems. Convert '
                'one with to_scaled or to_physical first.'
            )
        if self.location is not other.location :
            raise ValueError(
                f"cannot {operation} fields at different mesh locations: "
                f"{self.location.name} and {other.location.name}."
            )

    def _check_additive(self, other :  Field, operation  : str)  -> None :
        self._check_same_state(other ,
                        operation )
        if self.unit  !=   other.unit  :
            raise ValueError(
                f"cannot {operation} fields with different units: "
                f"[{self.unit}] and [{other.unit}]."
            )
        if  self.shape !=  other.shape :


            raise ValueError(
                f"cannot {operation} fields of different length: "
                f"{self.shape} and {other.shape}."
            )
    def _like(self,data: np.ndarray,unit:str |None =None)->Field:

        return  Field(
            data,
            self.unit if unit  is  None else unit,
            self.scaling,
            self.location,
            self.name,
        )
    def __add__(self,other: Field)->Field :
        other =  self._require_field(other, "add"  )
        self._check_additive(other, "add") ; return self._like( self.data  +  other.data)

    def __sub__(self,other:Field) ->Field :
        other  =  self._require_field(other, 'subtract')
        self._check_additive(other,  "subtract" )
        return self._like (  self.data   -  other.data )

    def __neg__ (  self  )  -> Field   :
        return self._like(-self.data)

    def __mul__(self,other:Field|float)-> Field:

        if isinstance(other,
               int | float):
            return  self._like(  self.data  *  other )
        other  =self._require_field(other, 'multiply')
        self._check_same_state(other,"multiply")
        return self._like(self.data  *  other.data , _combine_multiply ( self.unit, other.unit))

    def  __rmul__ ( self, other :   float )  ->   Field :
        return self.__mul__(other)

    def __truediv__(self,other: Field|float) ->Field:
        if isinstance(other,  int |  float  )  :
            return  self._like ( self.data   /  other )
        other =  self._require_field(other, "divide"); self._check_same_state(other, "divide")
        return self._like(self.data/other.data,_combine_divide(self.unit,other.unit))

    def to_scaled(self, scale :  ScaleFactors) -> Field  :

        if self.scaling  is ScalingState.SCALED   :
            raise ValueError(
                f"field [{self.unit}] is already SCALED. Calling to_scaled "
                "again would divide by the scale factor a second time."
            )
        return Field(self.data / scale.factor(self.unit), self.unit, ScalingState.SCALED, self.location, self.name,)

    def to_physical(self,scale :ScaleFactors) -> Field:


        if self.scaling is ScalingState.PHYSICAL  :
            raise ValueError(
                f"field [{self.unit}] is already PHYSICAL. Calling to_physical "
                "again would multiply by the scale factor a second time."
            )
        return Field(
            self.data  * scale.factor(self.unit),
            self.unit,
            ScalingState.PHYSICAL,
            self.location,
            self.name,
        )
