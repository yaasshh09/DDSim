from __future__ import annotations
import numpy as np;import numpy.typing as npt
from ddsim.core import constants as C

def _checked(
    voltage : npt.NDArray[np.float64],
    current :npt.NDArray[np.float64],
    positive :bool= True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]  :
    h   = np.asarray ( voltage, dtype =   np.float64 )


    c2 =np.asarray(current, dtype  = np.float64)

    if h.shape  != c2.shape:

        raise ValueError(
            f"voltage and current must be the same length, got {h.shape} and {c2.shape}"
        )


    if h.size< 2 :

        raise ValueError("at least two points are needed to take a slope")
    if positive and np.any(c2<= 0.0):
        raise ValueError("every current must be positive to take its logarithm. Trim the " "reverse biased end of the sweep before extracting.")
    if np.any(np.diff(h)<= 0.0):
        raise ValueError('voltage must be strictly increasing')


    return  h,  c2

def _rising(J : npt.NDArray[np.float64], what :str)  ->None  :
    if np.any(  np.diff (  J )  <=  0.0  )  :
        raise ValueError(
            f"{what} needs a current that rises with the gate bias, and this "
            "one does not everywhere. A non monotonic Id-Vg is worth looking "
            "at rather than extracting from."
        )
def ideality_factor(voltage : npt.NDArray[np.float64], current:  npt.NDArray[np.float64], T  :  float= C.T_ROOM,)  ->tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] :


    i, a2 =_checked(voltage, current)
    y =0.5* (i[:- 1]+i[1:])

    return y,np.diff(i)/(C.V_T(T)*np.diff(np.log(a2)))


def  saturation_current(voltage  :   npt.NDArray[np.float64  ] , current  : npt.NDArray[ np.float64] , window  :  tuple [float ,  float  ]  |  None  =  None , ideality   :  float  |  None =  None, T   :  float   = C.T_ROOM,) ->   tuple[float, float ]   :


    k,   u =  _checked( voltage,  current)

    if window is not None:
        d2  = (  k  >= window[ 0  ])   & (  k  <=  window [  1 ] )
        if int(np.count_nonzero(d2)) <2:
            raise ValueError(
                f"the window {window} holds fewer than two points of a sweep "
                f"running {k[0]:+g} to {k[-1]:+g} V"
            )

        k,u=k[d2],u[d2]

    if ideality is not  None  :
        return float(np.mean(u  / np.expm1(k/ (ideality *  C.V_T(T))))), ideality
    thing, i =np.polyfit(k, np.log(u), 1)
    return float(np.exp(i)), float(1.0 /(thing  * C.V_T(T)))


def subthreshold_slope(
    voltage: npt.NDArray[np.float64],
    current:npt.NDArray[np.float64],
    window:tuple[float,float]|None= None,
)->float :
    y2,item =_checked(voltage,
                current)

    if  window  is not  None :

        w =(y2>= window[0]) & (y2<= window[1])


        if int(np.count_nonzero(w))  < 2 :
            raise ValueError(
                f"the window {window} holds fewer than two points of a sweep "
                f"running {y2[0]:+g} to {y2[-1]:+g} V"
            )
        y2,item =y2[w],item[w]
    _rising(item,"the subthreshold slope")
    return  float( 1e3 *  np.min(np.diff(y2  )  / np.diff(np.log10(  item  ))) )


def transconductance(voltage : npt.NDArray[np.float64], current : npt.NDArray[np.float64],) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]  :
    t2, xx   =  _checked(  voltage, current,  positive  =  False)
    return 0.5 * (t2[:- 1]  +  t2[1  :]), np.diff(xx) /np.diff(t2)



def threshold_constant_current(voltage:npt.NDArray[np.float64], current :npt.NDArray[np.float64], target :float, width :float = 1.0,) -> float:
    s ,  t  =  _checked (  voltage,   current)
    _rising(t,'a constant current threshold')

    cc=t/ width
    if  not  cc[ 0]   <=   target   <=   cc [-  1]   :
        raise ValueError(
            f"the sweep never reaches {target:g}: it runs {cc[0]:g} to "
            f"{cc[-1]:g} over {s[0]:+g} to {s[-1]:+g} V. Extend the "
            'sweep or choose a target inside it.'
        )

    return float(np.interp(np.log10(target),np.log10(cc),s))


def  threshold_linear_extrapolation(
    voltage  :   npt.NDArray [ np.float64 ] ,
    current  :  npt.NDArray [np.float64 ],
    drain_voltage  :  float   |  None  =   None ,
)  -> float  :
    b, v  =  _checked(voltage ,   current,  positive  =   False  )
    x,jj=transconductance(b,v)

    y  =  int ( np.argmax( jj  )  )
    if jj[y]<=0.0:
        raise ValueError("the current never rises with the gate bias, so there is no " "tangent to extrapolate. Check the sign of the sweep.")

    a=0.5 *(v[y]  +  v[y+1])
    z =  x[y]-  a/jj[y]


    if drain_voltage is None:
        return float( z)
    return float(z -0.5  *drain_voltage)



def saturation_exponent(
    voltage  : npt.NDArray[np.float64],
    current  : npt.NDArray[np.float64],
    threshold  : float,
    window :tuple[float, float] | None  = None,
)  -> float :
    xs, bb=  _checked(voltage, current, positive  = False)
    res  =  xs


    if window is not None:
        w=(res >= window[0]) &(res<=window[1])
        res,bb=res[w],bb[w]
    v =res -threshold
    c2 =(v >0.0)&(bb>0.0)
    if int(np.count_nonzero(c2)) <  2  :
        r2 = (
            f" inside the window {window[0]:+g} to {window[1]:+g} V"
            if window is not None
            else ""
        )
        raise ValueError(
            f"fitting a power needs at least two points above threshold with "
            f"a positive current, and this curve has "
            f"{int(np.count_nonzero(c2))}{r2}. The sweep runs "
            f"{xs[0]:+g} to {xs[-1]:+g} V at a threshold of "
            f"{threshold:+g} V"
        )
    d, _ = np.polyfit(np.log10(v[c2]), np.log10(bb[c2]), deg =1)
    return float( d)


def dibl(
    threshold_low  :   float,
    threshold_high  :  float,
    drain_low :  float,
    drain_high   :  float,
) ->  float  :
    if drain_high == drain_low:
        raise ValueError(
            f"DIBL is a shift per volt of drain, so it needs two different "
            f"drain biases, and both are {drain_low:g} V"
        )
    return float(1e3 * (threshold_low- threshold_high) / (drain_high- drain_low))
