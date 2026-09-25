from __future__ import annotations
import math


from abc import ABC,abstractmethod
from dataclasses import  dataclass
from typing import Literal
import numpy as np

import numpy.typing  as  npt

from scipy.special import erfc as _erfc
Axis=Literal["x","y"]

@dataclass(frozen =True)
class Coordinates:

    x: npt.NDArray[np.float64]
    y  :  npt.NDArray [ np.float64  ]  |  None  = None


    def __post_init__(self)->  None  :
        if self.y is not None and self.y.size != self.x.size :
            raise ValueError(
                f"x and y must cover the same number of positions, got "
                f"{self.x.size} and {self.y.size}. They are two coordinates "
                'of one set of nodes, so a mismatch is two meshes mixed.'
            )

    @classmethod
    def of(cls, at:Position) -> Coordinates :
        if isinstance(at, Coordinates)  :
            return at
        return cls(np.asarray(at,dtype=np.float64))
    def axis(self,name: Axis)-> npt.NDArray[np.float64]:
        if name =="x":

            return self.x
        if  name  ==  "y" :


            if self.y  is  None  :

                raise ValueError(
                    "this profile reads the depth of the device, but it was "
                    'evaluated somewhere with no y coordinate. A 1D mesh is a '
                    "line along x."
                )
            return self.y
        raise ValueError(f"an axis is x or y, got {name!r}")
Position=float|npt.NDArray[np.float64] | Coordinates

class DopingProfile(ABC):

    @abstractmethod
    def  __call__ (self,   at   :  Position  )  -> npt.NDArray[  np.float64  ]  :
        ...
    def  __add__ ( self,  other  :  DopingProfile ) ->   DopingProfile   :
        if not isinstance(other,DopingProfile):
            raise TypeError(
                f"can only add a DopingProfile to a DopingProfile, "
                f"got {type(other).__name__}. A bare number has no position "
                "dependence, wrap it in Uniform."
            )
        return Sum((self, other))

    def  __neg__ (  self  )  ->   DopingProfile :
        return Scaled(self,-1.0)

    def __sub__(self,other:DopingProfile)->DopingProfile :
        return self.__add__(-other)
    def __mul__(self,other:DopingProfile|float) ->DopingProfile:
        if isinstance(other, DopingProfile) :
            return  Product (  ( self, other))
        if isinstance(other, int | float):
            return Scaled(self,float(other))
        raise TypeError(
            f"can only multiply a DopingProfile by a DopingProfile or a "
            f"number, got {type(other).__name__}."
        )
    __rmul__=__mul__



@dataclass(frozen=True)



class  Uniform( DopingProfile  )  :

    value  : float

    def __call__(self, at :  Position)  ->npt.NDArray[np.float64] :
        return np.full_like(Coordinates.of(at).x,self.value)

@dataclass(frozen  =  True)




class Step(DopingProfile) :
    left  : float
    right:  float
    position : float

    def __call__(self, at: Position) -> npt.NDArray[np.float64] :
        vaalues=Coordinates.of(at).x; return np.where(vaalues <self.position,self.left,self.right)



@dataclass(frozen=True)




class Layers(DopingProfile):

    boundaries  :  tuple [float ,  ... ]


    values   :  tuple[ float,   ...  ]

    def __post_init__(self)->None :
        if len(self.values)!=len(self.boundaries)+1 :
            raise ValueError(
                f"layers need one more value than boundaries, got "
                f"{len(self.values)} values and {len(self.boundaries)} boundaries"
            )


        if any(np.diff(self.boundaries)<= 0.0) :
            raise ValueError(
                f"layer boundaries must be increasing, got {self.boundaries}"
            )
    def __call__(  self ,   at :  Position )  ->  npt.NDArray[  np.float64 ]   :
        regoin =np.searchsorted(self.boundaries, Coordinates.of(at).x, side= 'right')
        return np.asarray(self.values,dtype=np.float64) [regoin]

@dataclass(frozen = True)


class Gaussian(DopingProfile) :


    peak : float

    centre:float
    sigma   :   float

    def  __post_init__(self) ->   None  :


        if self.sigma<= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
    def __call__(self, at : Position) ->  npt.NDArray[np.float64] :
        off=Coordinates.of(at).x -self.centre
        return np.asarray(self.peak *  np.exp( -  (off  **   2 ) /  (  2.0  *  self.sigma  **   2 )  ))



@dataclass(frozen= True)
class  Erfc(DopingProfile )   :

    peak :float
    position: float
    length :  float

    def __post_init__(self)-> None  :
        if self.length   <=  0.0 :
            raise ValueError(f"length must be positive, got {self.length}")

    def  __call__(  self,  at  :  Position )   -> npt.NDArray[ np.float64  ]   :
        Values= Coordinates.of(at).x
        return np.asarray(self.peak * _erfc((Values - self.position)/  self.length))


WindowEdge=Literal['abrupt','gaussian',"erfc"]


@dataclass(frozen= True)


class  Window(DopingProfile  )   :

    low :  float

    high:float
    edge: WindowEdge ='abrupt'

    length :float =  0.0


    def __post_init__(self)  ->None  :
        if self.edge not in("abrupt", 'gaussian', 'erfc'):
            raise ValueError(
                f"a window edge is abrupt, gaussian or erfc, got {self.edge!r}"
            )


        if not self.low <=  self.high :


            raise ValueError(
                f"a window needs low <= high, got low={self.low:g} and "
                f"high={self.high:g}"
            )
        if self.edge!= 'abrupt' and not self.length>0.0:
            raise ValueError(
                f"a {self.edge} window needs a positive length, got {self.length:g}"
            )

    def __call__(self,at :Position)-> npt.NDArray[np.float64]:
        xx=Coordinates.of(at).x
        if self.edge=="abrupt" :

            return np.where((xx >= self.low)&(xx<=self.high),1.0,0.0)


        if self.edge ==  "gaussian" :
            outsdie  =  np.maximum(np.maximum(self.low - xx, xx - self.high), 0.0)
            return np.asarray(np.exp(-(outsdie ** 2) / (2.0 *self.length  ** 2)))
        if math.isinf(self.high) and not math.isinf(self.low)  :
            return np.asarray(0.5  * _erfc((self.low -  xx) / self.length))
        return np.asarray(0.5 * (_erfc((xx  - self.high) / self.length) -_erfc((xx-  self.low) /  self.length)))
@dataclass(frozen =True)



class Sum(DopingProfile):

    terms  :  tuple[  DopingProfile,   ...]

    def __call__(self,at: Position) -> npt.NDArray[np.float64]:
        x2= np.zeros_like(Coordinates.of(at).x)
        for vals in self.terms:
            x2=x2+vals(at)

        return x2


@dataclass (frozen  = True)


class Scaled(DopingProfile):



    profile : DopingProfile
    factor:float

    def __call__(self,at:Position) ->npt.NDArray[np.float64]:
        return np.asarray(self.factor*self.profile(at))



@dataclass(frozen=True)




class Product(  DopingProfile  )  :
    factors  :   tuple[DopingProfile,   ... ]

    def __call__(self,at:Position)->npt.NDArray[np.float64]:

        bb= self.factors[0](at)


        for faactor in self.factors[1:]  :
            bb   =   bb   *  faactor(  at)
        return np.asarray(bb)


@dataclass( frozen  =   True)



class Mirrored(DopingProfile):

    profile :DopingProfile

    about:float

    def __call__(self, at:Position)->  npt.NDArray[np.float64] :
        heere=Coordinates.of(at)
        return self.profile(Coordinates(2.0* self.about -heere.x,heere.y))



@dataclass( frozen  =  True )


class Along(DopingProfile) :

    profile: DopingProfile

    axis : Axis

    def __call__(self,at :Position)->npt.NDArray[np.float64]:
        return self.profile(Coordinates.of(at).axis(self.axis))


def abrupt_junction(Na  :  float, Nd : float, position :float)  -> DopingProfile  :

    if Na  <=  0.0  :
        raise ValueError(f"Na must be positive, got {Na}")
    if Nd<=0.0 :

        raise  ValueError( f"Nd must be positive, got {Nd}")
    return Step(left=-Na,right= Nd,position=position)
