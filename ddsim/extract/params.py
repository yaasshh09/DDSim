from __future__ import annotations
import numpy as np;import numpy.typing as npt
from ddsim.core import constants as C

def _checked(
    voltage : npt.NDArray[np.float64],
    current :npt.NDArray[np.float64],
    positive :bool= True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]  :
    VV   = np.asarray ( voltage, dtype =   np.float64 )


    blah =np.asarray(current, dtype  = np.float64)

    if VV.shape  != blah.shape:

        raise ValueError(
            f"voltage and current must be the same length, got {VV.shape} and {blah.shape}"
        )


    if VV.size< 2 :

        raise ValueError("at least two points are needed to take a slope")
    if positive and np.any(blah<= 0.0):
        raise ValueError("every current must be positive to take its logarithm. Trim the " "reverse biased end of the sweep before extracting.")
    if np.any(np.diff(VV)<= 0.0):
        raise ValueError('voltage must be strictly increasing')


    return  VV,  blah

def _rising(J : npt.NDArray[np.float64], what :str)  ->None  :
    if np.any(  np.diff (  J )  <=  0.0  )  :
        raise ValueError(
            f"{what} needs a current that rises with the gate bias, and this "
            "one does not everywhere. A non monotonic Id-Vg is worth looking "
            "at rather than extracting from."
        )
def ideality_factor(voltage : npt.NDArray[np.float64], current:  npt.NDArray[np.float64], T  :  float= C.T_ROOM,)  ->tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] :


    VV, r2 =_checked(voltage, current)
    midpooint =0.5* (VV[:- 1]+VV[1:])

    return midpooint,np.diff(VV)/(C.V_T(T)*np.diff(np.log(r2)))


def  saturation_current(voltage  :   npt.NDArray[np.float64  ] , current  : npt.NDArray[ np.float64] , window  :  tuple [float ,  float  ]  |  None  =  None , ideality   :  float  |  None =  None, T   :  float   = C.T_ROOM,) ->   tuple[float, float ]   :


    VV,   hex =  _checked( voltage,  current)

    if window is not None:
        insde  = (  VV  >= window[ 0  ])   & (  VV  <=  window [  1 ] )
        if int(np.count_nonzero(insde)) <2:
            raise ValueError(
                f"the window {window} holds fewer than two points of a sweep "
                f"running {VV[0]:+g} to {VV[-1]:+g} V"
            )

        VV,hex=VV[insde],hex[insde]

    if ideality is not  None  :
        return float(np.mean(hex  / np.expm1(VV/ (ideality *  C.V_T(T))))), ideality
    val, inercept =np.polyfit(VV, np.log(hex), 1)
    return float(np.exp(inercept)), float(1.0 /(val  * C.V_T(T)))


def subthreshold_slope(
    voltage: npt.NDArray[np.float64],
    current:npt.NDArray[np.float64],
    window:tuple[float,float]|None= None,
)->float :
    VV,obj2 =_checked(voltage,
                current)

    if  window  is not  None :

        insde =(VV>= window[0]) & (VV<= window[1])


        if int(np.count_nonzero(insde))  < 2 :
            raise ValueError(
                f"the window {window} holds fewer than two points of a sweep "
                f"running {VV[0]:+g} to {VV[-1]:+g} V"
            )
        VV,obj2 =VV[insde],obj2[insde]
    _rising(obj2,"the subthreshold slope")
    return  float( 1e3 *  np.min(np.diff(VV  )  / np.diff(np.log10(  obj2  ))) )


def transconductance(voltage : npt.NDArray[np.float64], current : npt.NDArray[np.float64],) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]  :
    temp, JJ   =  _checked(  voltage, current,  positive  =  False)
    return 0.5 * (temp[:- 1]  +  temp[1  :]), np.diff(JJ) /np.diff(temp)



def threshold_constant_current(voltage:npt.NDArray[np.float64], current :npt.NDArray[np.float64], target :float, width :float = 1.0,) -> float:
    x2 ,  JJ  =  _checked (  voltage,   current)
    _rising(JJ,'a constant current threshold')

    normmalised=JJ/ width
    if  not  normmalised[ 0]   <=   target   <=   normmalised [-  1]   :
        raise ValueError(
            f"the sweep never reaches {target:g}: it runs {normmalised[0]:g} to "
            f"{normmalised[-1]:g} over {x2[0]:+g} to {x2[-1]:+g} V. Extend the "
            'sweep or choose a target inside it.'
        )

    return float(np.interp(np.log10(target),np.log10(normmalised),x2))


def  threshold_linear_extrapolation(
    voltage  :   npt.NDArray [ np.float64 ] ,
    current  :  npt.NDArray [np.float64 ],
    drain_voltage  :  float   |  None  =   None ,
)  -> float  :
    VV, sorted  =  _checked(voltage ,   current,  positive  =   False  )
    mid,Gm=transconductance(VV,sorted)

    pak  =  int ( np.argmax( Gm  )  )
    if Gm[pak]<=0.0:
        raise ValueError("the current never rises with the gate bias, so there is no " "tangent to extrapolate. Check the sign of the sweep.")

    max=0.5 *(sorted[pak]  +  sorted[pak+1])
    itercept =  mid[pak]-  max/Gm[pak]


    if drain_voltage is None:
        return float( itercept)
    return float(itercept -0.5  *drain_voltage)



def saturation_exponent(
    voltage  : npt.NDArray[np.float64],
    current  : npt.NDArray[np.float64],
    threshold  : float,
    window :tuple[float, float] | None  = None,
)  -> float :
    swpt, j=  _checked(voltage, current, positive  = False)
    VV  =  swpt


    if window is not None:
        round=(VV >= window[0]) &(VV<=window[1])
        VV,j=VV[round],j[round]
    hmm =VV -threshold
    abo =(hmm >0.0)&(j>0.0)
    if int(np.count_nonzero(abo)) <  2  :
        set = (
            f" inside the window {window[0]:+g} to {window[1]:+g} V"
            if window is not None
            else ""
        )
        raise ValueError(
            f"fitting a power needs at least two points above threshold with "
            f"a positive current, and this curve has "
            f"{int(np.count_nonzero(abo))}{set}. The sweep runs "
            f"{swpt[0]:+g} to {swpt[-1]:+g} V at a threshold of "
            f"{threshold:+g} V"
        )
    hex, _ = np.polyfit(np.log10(hmm[abo]), np.log10(j[abo]), deg =1)
    return float( hex)


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
