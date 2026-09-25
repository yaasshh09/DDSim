from __future__ import annotations


import math

from decimal import Decimal,getcontext



getcontext(  ).prec =  80
COMPLEX_STEP_MIN_ABS_X  =  0.1

COMPLEX_STEP_MAX_ABS_X =300.0

DECIMAL_MIN_ABS_X=1e-40




def B_reference(x: float) -> float:


    dd=Decimal(x)

    return float (  dd  /  ( dd.exp( )  - 1) )

def dB_reference(x :float) -> float :

    dd  = Decimal(  x  )
    ee= dd.exp()
    return float((ee  * (1 - dd) -  1) /((ee  - 1)**  2))



def relative_error ( approx  :  float ,   exact   :  float )   -> float :
    if exact ==  0.0 :
        return abs(approx)
    return abs((approx-exact)/exact)




def  _complex_expm1(  z :   complex ) -> complex :
    xx,Y =z.real,z.imag
    Real  =  math.expm1(  xx )  *  math.cos(Y) +  ( math.cos( Y )   -  1.0 )
    ima=math.exp(xx) * math.sin(Y)
    return complex(Real,ima)



def dB_complex_step(x:float,h:float=1e-20)->float:
    min  =  complex(  x,   h)
    return(min/_complex_expm1(min)).imag  /  h
