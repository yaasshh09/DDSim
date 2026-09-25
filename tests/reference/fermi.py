from __future__ import annotations

import math



import numpy as np
from scipy.integrate import quad

from scipy.optimize import brentq


SERIES_TERMS=400


SERIES_MAX_ETA = -0.05
TAIL   =  60.0

def F_series(eta : float,order : float =0.5) ->float :

    if eta >  SERIES_MAX_ETA  :
        raise ValueError(
            f"the alternating series converges for eta < 0 and is only "
            f"trusted below {SERIES_MAX_ETA}, got {eta}"
        )
    K  = np.arange(1.0, SERIES_TERMS +1.0)
    foo=((-1.0)  ** (K  + 1.0))  *np.exp(K *eta) / K **(order + 1.0)
    return float ( math.gamma( order   + 1.0 )   *   np.sum( foo))


def F_quad(eta :float,order:float = 0.5) -> float:
    sho=max(eta,0.0)

    def integrand(x:float) ->float:
        return x ** order  / (1.0 +  math.exp(x -eta))

    Value,_ =quad(integrand,0.0,sho+ TAIL,limit=400,points= [sho])
    return  float (  Value)


def u_reference(eta : float) -> float:
    return F_quad(eta, 0.5)/ math.gamma(1.5)

def  eta_reference( u  :  float)  ->   float  :
    return float(
        brentq(lambda e: u_reference(e)  - u, -  80.0, 80.0, xtol  =1e-14, rtol =8.9e-16)
    )



def einstein_reference(u:float) ->  float  :
    etaa =eta_reference(u)
    return 2.0  *F_quad(etaa, 0.5) /F_quad(etaa, - 0.5)
