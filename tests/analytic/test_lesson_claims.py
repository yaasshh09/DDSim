from __future__ import annotations

import functools;  import math
import numpy as np, pytest


from ddsim.api.devices import build_from_spec

from ddsim.api.learn import load_lesson
from ddsim.api.sweeps  import  run_sweep

from ddsim.core import constants as C
from  ddsim.extract.bands  import  band_edges


from ddsim.extract.params  import(
    ideality_factor,
    subthreshold_slope,
    threshold_constant_current ,
    threshold_linear_extrapolation,
)
from  ddsim.extract.rolloff  import  REFERENCE_CURRENT,   usable_span
from tests.analytic.test_mos_cap import(
    max_depletion_width,
    oxide_capacitance,
    threshold_voltage,
)

from tests.analytic.test_mosfet_transport import depletion_approximation_slope
from tests.analytic.test_pn_equilibrium import(analytic_depletion_width, analytic_V_bi, depletion_width_from_field,)


THERMAL_LIMIT =59.5



@functools.cache
def run(lesson: str,step:str|None = None) :


    fond=load_lesson(lesson)
    Request = fond.request if step is None else fond.step(step).request
    assert  Request is not None, f"{lesson}: step {step!r} sets nothing up"

    spe,swe= Request["device"],Request['sweep']
    dev= build_from_spec(spe["kind"],
                 spe['parameters'])
    cur,_ =run_sweep(
        swe['kind'],
        dev,
        swe["contact"],
        swe["voltages"],
        settings=swe.get("settings"),
        models = swe.get("models"),
        measure_at= swe.get("measure_at"),
    )
    assert cur.complete, f"{lesson} {step}: {cur.message}"
    return dev, cur, Request


def psi_of(device, state) ->  np.ndarray :

    return np.asarray(state.psi.to_physical(device.scale).data)
JUNCTION= '01-pn-junction'


@pytest.mark.parametrize('step',[None,"Dope one side harder"])


def test_built_in_potential_matches_the_textbook_formula(  step )  ->  None :
    Device,cuurve,Request=run(JUNCTION,
            step)
    idx2  =  Request[ "device" ]   ["parameters"  ]

    Na=idx2.get("Na", 1e16)
    Nd=idx2.get('Nd',
       1e16)
    psi=psi_of(Device, cuurve.points[0].state)

    assert  psi[-   1]  - psi[0 ]   ==  pytest.approx(analytic_V_bi (Na , Nd ) , rel  =  5e-3)




def test_depletion_width_matches_the_depletion_approximation() ->None:
    Device, Curve, _ =  run(JUNCTION)
    object = depletion_width_from_field(  Device, Curve.points[  0  ].state)

    assert object  == pytest.approx(analytic_depletion_width(1e16, 1e16), rel = 3e-2)



def  test_the_fermi_level_is_flat_at_equilibrium()  -> None :
    dev ,  cur , _ = run(  JUNCTION )
    yy=band_edges(dev,cur.points[0].state)

    for Level in(yy.Efn, yy.Efp):

        assert np.ptp(Level) < 1e-6

def test_the_lightly_doped_side_takes_the_potential_drop() ->  None:
    dev,  cur,  req =   run(  JUNCTION, "Dope one side harder")
    psi=psi_of(dev,cur.points[0].state)
    junnction   = req[ "device"  ]   [ "parameters"].get (  'junction' , 0.5e-4  )
    idx2 =int(np.argmin(np.abs(dev.mesh.x-junnction)))
    ns = psi[-1]-psi[idx2]
    assert ns/(psi[-  1]-psi[0])  >0.95
BIAS =  '02-bias'

def test_forward_current_rises_a_decade_every_60_millivolts() -> None :
    _,vals,_=run(BIAS)
    input  =  vals.voltage  >=   0.1
    _,ide=ideality_factor(vals.voltage[input],vals.current[input])

    assert np.all(ide>1.0)
    assert  np.all (  ide < 1.06  )



def test_reverse_bias_widens_the_depletion_region_as_the_square_root()->None:
    dveice,cuve,_=run(BIAS,'Reverse bias');lst= analytic_V_bi(1e16,
              1e16)
    Widths  = [ depletion_width_from_field(dveice,   p.state  )  for p in cuve.points]
    assert all(a <  b for a, b in zip(Widths[:-1], Widths[1 :], strict= False))
    for Width,bas in zip(Widths,cuve.voltage,strict=True) :
        perdicted=Widths[0]*math.sqrt((lst - bas)/lst)
        assert  Width ==   pytest.approx(perdicted, rel =  3e-2  )
        assert  Width ==   pytest.approx(analytic_depletion_width(1e16 ,   1e16 , float (bas) ),   rel   =  3e-2)
MOSCAP  =  '03-mos-capacitor'


def c_over_cox(step) :

    _,   currve,  round   = run(  MOSCAP,  step  )

    acc =   round["device"  ]   [  "parameters"  ].get(  "t_ox",  1e-6 )
    return currve.gate_voltage, currve.capacitance/  oxide_capacitance(acc)


@pytest.mark.parametrize( 'step',   [  None, "High frequency" ]  )


def test_the_capacitance_never_exceeds_the_oxide(step)->None  :


    s2,raio=c_over_cox(step)
    assert  np.all( raio  <  1.0  )
    assert raio[  np.argmin(s2)  ]  > 0.9


def test_the_high_frequency_minimum_matches_the_depletion_approximation()  -> None  :
    voltgae, Ratio = c_over_cox('High frequency');  w= max_depletion_width(- 1e16)

    foo=1.0  /  (1.0 + oxide_capacitance(1e-6) * w  /C.eps_Si())

    inv= Ratio[np.argmax(voltgae)]
    assert inv <foo
    assert inv == pytest.approx(foo, rel= 0.08)



def test_only_a_slow_signal_sees_the_inversion_layer( )   ->  None :
    Voltage, dict = c_over_cox(None)
    _, Fast  = c_over_cox("High frequency")
    hmm=int(np.argmax(Voltage))
    assert dict[hmm]  > 0.95
    assert Fast[hmm]<0.1

MOSFET =  "04-mosfet"

def test_the_subthreshold_slope_sits_above_the_body_factor_estimate() -> None  :
    _,tmp,_= run(MOSFET)
    VV, j  = usable_span(tmp.voltage, tmp.current)
    slo= subthreshold_slope(VV,j,window =(0.2,0.6))

    forrmula= depletion_approximation_slope(1e17,2e-6)

    assert slo  >  THERMAL_LIMIT
    assert slo  >forrmula
    assert slo  ==   pytest.approx(forrmula, rel  = 0.15  )



def test_the_threshold_lands_near_the_textbook_formula() ->None:
    _,  Curve,   reqquest  =  run(MOSFET );  v, j =usable_span(Curve.voltage, Curve.current)

    Drain = reqquest["device"]['parameters']['drain_voltage']
    mea= threshold_linear_extrapolation(v,j,drain_voltage =Drain)

    data2 =threshold_voltage(-1e17,2e-6,C.PHI_M_N_POLY)

    assert mea  == pytest.approx(data2, abs =  0.05)
SHORT= "05-short-channel"
LENGTHS  ={
    1e-4 :("A long channel", 'Raise the drain'),
    1e-5: ("Shorten the gate to 100 nm", "Raise the drain again"),
    5e-6 : ("Shorten it to 50 nm", "And raise the drain"),
}


def threshold(step: str)  -> float  :
    _ , cuve, hmm  =   run(  SHORT ,   step )
    v, j= usable_span(cuve.voltage, cuve.current)
    bar  =  hmm['device']  ['parameters'] ["L_gate"]
    return threshold_constant_current(v,
       j,
         REFERENCE_CURRENT / bar)

def dibl(L_gate :float) ->  float :
    loww, hig  =  LENGTHS[  L_gate ]
    dra  =  run(SHORT, loww)  [2] ["device"]  ["parameters"] ["drain_voltage"] ; dat = run(  SHORT, hig ) [ 2]  [ 'device'  ]  ["parameters" ]  [  "drain_voltage" ]
    return 1e3 *(threshold(loww)- threshold(hig)) /  (dat- dra)
def test_the_threshold_rolls_off_as_the_gate_shortens() -> None :
    Thresholds=[threshold(dict)for dict,_ in LENGTHS.values()]
    assert Thresholds[0]  >  Thresholds[1]  > Thresholds[2]
    assert Thresholds[0] -Thresholds[2] > 0.1
def test_the_drain_lowers_the_barrier_only_on_a_short_channel()-> None:
    Long, temp2, sho  = (dibl(L) for L in LENGTHS)

    assert Long <  15.0
    assert Long<temp2< sho
    assert sho>5.0*Long
