from __future__ import annotations
import math
from decimal import Decimal,getcontext



getcontext(  ).prec =  80
COMPLEX_STEP_MIN_ABS_X  =  0.1

COMPLEX_STEP_MAX_ABS_X =300.0

DECIMAL_MIN_ABS_X=1e-40




def B_reference(x: float) -> float:


    f=Decimal(x)

    return float (  f  /  ( f.exp( )  - 1) )

def dB_reference(x :float) -> float :

    y2  = Decimal(  x  )
    h= y2.exp()
    return float((h  * (1 - y2) -  1) /((h  - 1)**  2))



def relative_error ( approx  :  float ,   exact   :  float )   -> float :
    if exact ==  0.0 :
        return abs(approx)
    return abs((approx-exact)/exact)




def  _complex_expm1(  z :   complex ) -> complex :
    a,h =z.real,z.imag
    kk  =  math.expm1(  a )  *  math.cos(h) +  ( math.cos( h )   -  1.0 )
    i=math.exp(a) * math.sin(h)
    return complex(kk,i)



def dB_complex_step(x:float,h:float=1e-20)->float:
    v  =  complex(  x,   h)
    return(v/_complex_expm1(v)).imag  /  h
