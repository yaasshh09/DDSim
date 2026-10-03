from __future__ import annotations
import functools; import math
import numpy as np, pytest
from ddsim.api.devices import build_from_spec
from ddsim.api.learn import load_lesson
from ddsim.api.sweeps import run_sweep
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


    rr=load_lesson(lesson)
    t2 = rr.request if step is None else rr.step(step).request
    assert  t2 is not None, f"{lesson}: step {step!r} sets nothing up"

    s,z= t2["device"],t2['sweep']
    cnt= build_from_spec(s["kind"],
                 s['parameters'])
    t,_ =run_sweep(
        z['kind'],
        cnt,
        z["contact"],
        z["voltages"],
        settings=z.get("settings"),
        models = z.get("models"),
        measure_at= z.get("measure_at"),
    )
    assert t.complete, f"{lesson} {step}: {t.message}"
    return cnt, t, t2


def psi_of(device, state) ->  np.ndarray :

    return np.asarray(state.psi.to_physical(device.scale).data)
JUNCTION= '01-pn-junction'


@pytest.mark.parametrize('step',[None,"Dope one side harder"])


def test_built_in_potential_matches_the_textbook_formula(  step )  ->  None :
    zz,dd,d=run(JUNCTION,
            step)
    g  =  d[ "device" ]   ["parameters"  ]

    Na=g.get("Na", 1e16)
    Nd=g.get('Nd',
       1e16)
    psi=psi_of(zz, dd.points[0].state)

    assert  psi[-   1]  - psi[0 ]   ==  pytest.approx(analytic_V_bi (Na , Nd ) , rel  =  5e-3)




def test_depletion_width_matches_the_depletion_approximation() ->None:
    w, row, _ =  run(JUNCTION)
    r2 = depletion_width_from_field(  w, row.points[  0  ].state)

    assert r2  == pytest.approx(analytic_depletion_width(1e16, 1e16), rel = 3e-2)



def  test_the_fermi_level_is_flat_at_equilibrium()  -> None :
    el ,  s2 , _ = run(  JUNCTION )
    val2=band_edges(el,s2.points[0].state)

    for v in(val2.Efn, val2.Efp):

        assert np.ptp(v) < 1e-6

def test_the_lightly_doped_side_takes_the_potential_drop() ->  None:
    t,  zz,  b =   run(  JUNCTION, "Dope one side harder")
    psi=psi_of(t,zz.points[0].state)
    k   = b[ "device"  ]   [ "parameters"].get (  'junction' , 0.5e-4  )
    arr =int(np.argmin(np.abs(t.mesh.x-k)))
    u = psi[-1]-psi[arr]
    assert u/(psi[-  1]-psi[0])  >0.95
BIAS =  '02-bias'

def test_forward_current_rises_a_decade_every_60_millivolts() -> None :
    _,cnt,_=run(BIAS)
    s  =  cnt.voltage  >=   0.1
    _,h=ideality_factor(cnt.voltage[s],cnt.current[s])

    assert np.all(h>1.0)
    assert  np.all (  h < 1.06  )



def test_reverse_bias_widens_the_depletion_region_as_the_square_root()->None:
    c,aa,_=run(BIAS,'Reverse bias');i= analytic_V_bi(1e16,
              1e16)
    stuff  = [ depletion_width_from_field(c,   p.state  )  for p in aa.points]
    assert all(res2 <  a for res2, a in zip(stuff[:-1], stuff[1 :], strict= False))
    for j,g in zip(stuff,aa.voltage,strict=True) :
        h=stuff[0]*math.sqrt((i - g)/i)
        assert  j ==   pytest.approx(h, rel =  3e-2  )
        assert  j ==   pytest.approx(analytic_depletion_width(1e16 ,   1e16 , float (g) ),   rel   =  3e-2)
MOSCAP  =  '03-mos-capacitor'


def c_over_cox(step) :

    _,   k,  h   = run(  MOSCAP,  step  )

    t2 =   h["device"  ]   [  "parameters"  ].get(  "t_ox",  1e-6 )
    return k.gate_voltage, k.capacitance/  oxide_capacitance(t2)


@pytest.mark.parametrize( 'step',   [  None, "High frequency" ]  )


def test_the_capacitance_never_exceeds_the_oxide(step)->None  :


    r,y=c_over_cox(step)
    assert  np.all( y  <  1.0  )
    assert y[  np.argmin(r)  ]  > 0.9


def test_the_high_frequency_minimum_matches_the_depletion_approximation()  -> None  :
    h, v = c_over_cox('High frequency');  g= max_depletion_width(- 1e16)

    a=1.0  /  (1.0 + oxide_capacitance(1e-6) * g  /C.eps_Si())

    bb= v[np.argmax(h)]
    assert bb <a
    assert bb == pytest.approx(a, rel= 0.08)



def test_only_a_slow_signal_sees_the_inversion_layer( )   ->  None :
    t, k = c_over_cox(None)
    _, c  = c_over_cox("High frequency")
    cnt=int(np.argmax(t))
    assert k[cnt]  > 0.95
    assert c[cnt]<0.1

MOSFET =  "04-mosfet"

def test_the_subthreshold_slope_sits_above_the_body_factor_estimate() -> None  :
    _,v,_= run(MOSFET)
    d, c  = usable_span(v.voltage, v.current)
    z= subthreshold_slope(d,c,window =(0.2,0.6))

    y2= depletion_approximation_slope(1e17,2e-6)

    assert z  >  THERMAL_LIMIT
    assert z  >y2
    assert z  ==   pytest.approx(y2, rel  = 0.15  )



def test_the_threshold_lands_near_the_textbook_formula() ->None:
    _,  r,   b  =  run(MOSFET );  t, s =usable_span(r.voltage, r.current)

    f = b["device"]['parameters']['drain_voltage']
    s2= threshold_linear_extrapolation(t,s,drain_voltage =f)

    tmp =threshold_voltage(-1e17,2e-6,C.PHI_M_N_POLY)

    assert s2  == pytest.approx(tmp, abs =  0.05)
SHORT= "05-short-channel"
LENGTHS  ={
    1e-4 :("A long channel", 'Raise the drain'),
    1e-5: ("Shorten the gate to 100 nm", "Raise the drain again"),
    5e-6 : ("Shorten it to 50 nm", "And raise the drain"),
}


def threshold(step: str)  -> float  :
    _ , u, g  =   run(  SHORT ,   step )
    ss, y2= usable_span(u.voltage, u.current)
    ok  =  g['device']  ['parameters'] ["L_gate"]
    return threshold_constant_current(ss,
       y2,
         REFERENCE_CURRENT / ok)

def dibl(L_gate :float) ->  float :
    j, v2  =  LENGTHS[  L_gate ]
    d  =  run(SHORT, j)  [2] ["device"]  ["parameters"] ["drain_voltage"] ; nxt = run(  SHORT, v2 ) [ 2]  [ 'device'  ]  ["parameters" ]  [  "drain_voltage" ]
    return 1e3 *(threshold(j)- threshold(v2)) /  (nxt- d)
def test_the_threshold_rolls_off_as_the_gate_shortens() -> None :
    val=[threshold(g)for g,_ in LENGTHS.values()]
    assert val[0]  >  val[1]  > val[2]
    assert val[0] -val[2] > 0.1
def test_the_drain_lowers_the_barrier_only_on_a_short_channel()-> None:
    r, dat, s  = (dibl(j) for j in LENGTHS)

    assert r <  15.0
    assert r<dat< s
    assert s>5.0*r
