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
    VV , bar  = np.asarray(voltage,   dtype  =  np.float64  ), np.asarray(current,   dtype = np.float64)
    Start   = 0
    for K in range(len(bar)):
        if bar[K] <= 0.0:
            Start  =   K + 1
        elif K> 0 and bar[K]<= bar[K-1]:
            Start =K

    if len(bar) -Start< 3:
        raise ValueError(
            f"only {len(bar) - Start} points of this transfer curve are on: it "
            f"runs {VV[0]:+g} to {VV[-1]:+g} V and never becomes both positive "
            "and rising for long enough to extract from. Extend the gate "
            'sweep upward.'
        )
    return VV[Start:],bar[Start:]
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
    Settings  =  dict (SHORT_CHANNEL_PROCESS if  process  is None  else process)

    poi = []
    for lGate in gate_lengths:
        curevs   =  { }
        for Label,drin in(('linear',drain_low),('saturated',drain_high)) :
            devce=nmos(L_gate=lGate,drain_voltage=drin,**Settings)
            curevs[Label] = gate_sweep(devce , voltages  =  list (  gate_voltages ), models  =   TransportModels.for_device(devce, mobility  = mobility , field_dependent  =   field_dependent, surface   =   surface ,) , step  =  step,)

        V_linn,myvar=usable_span(
            curevs["linear"].voltage,curevs["linear"].current
        )


        v, J_satt  =  usable_span(curevs['saturated'].voltage, curevs["saturated"].current)
        tar  =   reference_current  /   lGate
        tl=  threshold_constant_current(V_linn, myvar, tar)
        thresholdsaturated  = threshold_constant_current(v, J_satt, tar)
        thresholdextrapolated  = threshold_linear_extrapolation(
            V_linn ,  myvar , drain_voltage  =  drain_low
        )
        _, gmm  =transconductance(V_linn, myvar)


        bar=(threshold_constant_current(V_linn, myvar, tar/10.0 **  slope_decades), tl,)
        poi.append(
            RollOffPoint(
                L_gate = lGate,
                threshold_linear= tl,
                threshold_saturated = thresholdsaturated,
                threshold_extrapolated =thresholdextrapolated,
                subthreshold_slope  =  subthreshold_slope(
                    V_linn, myvar, window = bar
                ),
                dibl =dibl(
                    threshold_low  =tl,
                    threshold_high =  thresholdsaturated,
                    drain_low =  drain_low,
                    drain_high = drain_high,
                ),
                saturation_exponent =  saturation_exponent(
                    v,
                    J_satt,
                    threshold=thresholdextrapolated,
                    window  =(
                        thresholdextrapolated  +  overdrive_window[0],
                        thresholdextrapolated + overdrive_window[1],
                    ),
                ),
                peak_transconductance=  float(np.max(gmm)),
                linear = curevs['linear'],
                saturated = curevs['saturated'],
            )
        )
    return tuple(poi)
