from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from  typing  import Any
import numpy as np
import numpy.typing as npt
from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels
from ddsim.extract.iv  import IVCurve,  gate_sweep
from ddsim.extract.params import(
    dibl,
    saturation_exponent,
    subthreshold_slope,
    threshold_constant_current,
    threshold_linear_extrapolation,
    transconductance,
)

SHORT_CHANNEL_PROCESS :   Mapping[  str,   Any] =   {"substrate_doping"   : -  1e18, "sd_peak" : 1e20 , "x_j"  :  2.5e-6, 'lateral_diffusion'  :   1.0e-6, 't_ox'   :  2e-7 , 'sd_length'  :   4e-5, "contact_length"  :  2e-5 , "t_si" :   1e-4,}


REFERENCE_CURRENT = 1e-7

@dataclass(frozen=True)


class  RollOffPoint  :

    L_gate  : float
    threshold_linear: float
    threshold_saturated : float
    threshold_extrapolated:  float

    subthreshold_slope:float


    dibl  :float

    saturation_exponent  : float

    peak_transconductance :float
    linear:IVCurve

    saturated : IVCurve



def usable_span(
    voltage : npt.NDArray[np.float64], current  : npt.NDArray[np.float64]
)  ->  tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]  :
    vals , buf  = np.asarray(voltage,   dtype  =  np.float64  ), np.asarray(current,   dtype = np.float64)
    y   = 0
    for s2 in range(len(buf)):
        if buf[s2] <= 0.0:
            y  =   s2 + 1
        elif s2> 0 and buf[s2]<= buf[s2-1]:
            y =s2

    if len(buf) -y< 3:
        raise ValueError(
            f"only {len(buf) - y} points of this transfer curve are on: it "
            f"runs {vals[0]:+g} to {vals[-1]:+g} V and never becomes both positive "
            "and rising for long enough to extract from. Extend the gate "
            'sweep upward.'
        )
    return vals[y:],buf[y:]
def gate_length_sweep(
    gate_lengths:list[float],
    gate_voltages :list[float],
    process :Mapping[str,Any]| None=None,
    drain_low:float =0.05,
    drain_high:float =1.0,
    reference_current: float=REFERENCE_CURRENT,
    overdrive_window : tuple[float,float] =(0.4,1.0),
    slope_decades : float =2.0,
    mobility:str='arora',
    field_dependent:bool= True,
    surface: bool= True,
    step:float=0.05,
) ->tuple[RollOffPoint,...]:


    if  not  gate_lengths  :
        raise ValueError( "a gate length sweep needs at least one gate length"  )

    if drain_high== drain_low :
        raise  ValueError(
            f"the sweep needs two different drain biases to report DIBL, and "
            f"both are {drain_low:g} V"
        )
    b  =  dict (SHORT_CHANNEL_PROCESS if  process  is None  else process)

    it = []
    for ret in gate_lengths:
        m   =  { }
        for thing,s in(('linear',drain_low),('saturated',drain_high)) :
            r=nmos(L_gate=ret,drain_voltage=s,**b)
            m[thing] = gate_sweep(r , voltages  =  list (  gate_voltages ), models  =   TransportModels.for_device(r, mobility  = mobility , field_dependent  =   field_dependent, surface   =   surface ,) , step  =  step,)

        u,w=usable_span(
            m["linear"].voltage,m["linear"].current
        )


        j, e  =  usable_span(m['saturated'].voltage, m["saturated"].current)
        mm  =   reference_current  /   ret
        ok=  threshold_constant_current(u, w, mm)
        d  = threshold_constant_current(j, e, mm)
        out2  = threshold_linear_extrapolation(
            u ,  w , drain_voltage  =  drain_low
        )
        _, x2  =transconductance(u, w)


        i=(threshold_constant_current(u, w, mm/10.0 **  slope_decades), ok,)
        it.append(
            RollOffPoint(
                L_gate = ret,
                threshold_linear= ok,
                threshold_saturated = d,
                threshold_extrapolated =out2,
                subthreshold_slope  =  subthreshold_slope(
                    u, w, window = i
                ),
                dibl =dibl(
                    threshold_low  =ok,
                    threshold_high =  d,
                    drain_low =  drain_low,
                    drain_high = drain_high,
                ),
                saturation_exponent =  saturation_exponent(
                    j,
                    e,
                    threshold=out2,
                    window  =(
                        out2  +  overdrive_window[0],
                        out2 + overdrive_window[1],
                    ),
                ),
                peak_transconductance=  float(np.max(x2)),
                linear = m['linear'],
                saturated = m['saturated'],
            )
        )
    return tuple(it)
