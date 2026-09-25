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
    UniformTotal =  h_min  * n_intervals
    if UniformTotal >side_length*(1.0 +DEGENERATE_TOLERANCE):
        return None
    if n_intervals  == 1 :
        return 1.0
    if abs(UniformTotal-side_length)<=DEGENERATE_TOLERANCE *side_length:
        return 1.0

    Low, High =  1.0, 2.0
    while geometric_sum(h_min,High,n_intervals)<side_length :
        High *= 2.0
        if High>1e6 :
            return None
    for _ in range(200):
        mid  =   0.5   *  ( Low  +  High  )
        if  geometric_sum( h_min,  mid,   n_intervals  )  <  side_length  :
            Low   = mid
        else :
            High = mid

        if High-Low<= RATIO_TOLERANCE *Low:
            break


    return 0.5   *  (Low  +  High  )
