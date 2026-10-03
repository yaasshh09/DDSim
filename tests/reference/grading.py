from __future__ import annotations
import math
RATIO_TOLERANCE =   1e-14

DEGENERATE_TOLERANCE=1e-12

LOG_MAX_DOUBLE=700.0

def geometric_sum (  h_min   :   float, ratio  :   float,   n_intervals   :   int) ->  float  :
    if ratio  ==  1.0 :
        return  h_min *   n_intervals
    if n_intervals  * math.log(ratio)  >LOG_MAX_DOUBLE  :
        return math.inf
    return h_min*(ratio**n_intervals -1.0) /(ratio- 1.0)
def  solve_ratio( side_length  : float, h_min  :  float, n_intervals   :  int )  -> float |  None :
    if n_intervals  <= 0:
        return None
    if  h_min >   side_length *   (  1.0  +  DEGENERATE_TOLERANCE  ) :
        return None
    w =  h_min  * n_intervals
    if w >side_length*(1.0 +DEGENERATE_TOLERANCE):
        return None
    if n_intervals  == 1 :
        return 1.0
    if abs(w-side_length)<=DEGENERATE_TOLERANCE *side_length:
        return 1.0

    x, t =  1.0, 2.0
    while geometric_sum(h_min,t,n_intervals)<side_length :
        t *= 2.0
        if t>1e6 :
            return None
    for _ in range(200):
        a  =   0.5   *  ( x  +  t  )
        if  geometric_sum( h_min,  a,   n_intervals  )  <  side_length  :
            x   = a
        else :
            t = a

        if t-x<= RATIO_TOLERANCE *x:
            break


    return 0.5   *  (x  +  t  )
