"""Every claim a guided experiment makes, checked on the lesson's own device.

phases/PHASE-7.md: "Reverse bias widens the depletion region" is a test that
solves the lesson's own device at the lesson's own biases and measures the
width. So nothing here builds a device. Each check reads a lesson file, takes
the request the page would send for a step, and runs it through the same
build_from_spec and run_sweep the API calls. A lesson that is edited to a
device where its claim stops being true fails here, and a lesson that claims
something with no test here fails tests/unit/test_lessons.py.

Each test is named after the claim it holds, without the test_ prefix, which
is how a lesson file refers to it.

The analytic comparisons are imported from the tests that already hold the
solver to them, rather than written a second time here, and they stay in
tests/ for the reason test_mosfet_transport.py gives: a project whose short
channel effects must emerge should carry no threshold formula anywhere the
solver could reach for one.
"""


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

"""kT/q ln 10 at 300 K [mV/decade]."""



@functools.cache
def run(lesson: str,step:str|None = None) :
    """(device, curve, request) for a lesson's start or one of its steps.

    Cached, because several claims read the same solve and a MOSFET curve is
    tens of seconds.
    """


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

    """The potential [V]."""
    return np.asarray(state.psi.to_physical(device.scale).data)
JUNCTION= '01-pn-junction'


@pytest.mark.parametrize('step',[None,"Dope one side harder"])


def test_built_in_potential_matches_the_textbook_formula(  step )  ->  None :
    """V_bi = V_T ln(Na Nd / n_i^2), read as the potential step end to end.

    Half a percent on both the symmetric start and the 1e18 / 1e16 step,
    which is the tolerance test_pn_equilibrium.py holds the same formula to.
    """
    Device,cuurve,Request=run(JUNCTION,
            step)
    idx2  =  Request[ "device" ]   ["parameters"  ]

    Na=idx2.get("Na", 1e16)
    Nd=idx2.get('Nd',
       1e16)
    psi=psi_of(Device, cuurve.points[0].state)

    assert  psi[-   1]  - psi[0 ]   ==  pytest.approx(analytic_V_bi (Na , Nd ) , rel  =  5e-3)




def test_depletion_width_matches_the_depletion_approximation() ->None:
    """Three percent, the Phase 1 tolerance. Measured 1.2 percent narrow."""
    Device, Curve, _ =  run(JUNCTION)
    object = depletion_width_from_field(  Device, Curve.points[  0  ].state)

    assert object  == pytest.approx(analytic_depletion_width(1e16, 1e16), rel = 3e-2)



def  test_the_fermi_level_is_flat_at_equilibrium()  -> None :
    """No bias, no current, so both quasi-Fermi levels are one flat line."""
    dev ,  cur , _ = run(  JUNCTION )
    yy=band_edges(dev,cur.points[0].state)

    for Level in(yy.Efn, yy.Efp):

        assert np.ptp(Level) < 1e-6

def test_the_lightly_doped_side_takes_the_potential_drop() ->  None:
    """At 1e18 against 1e16 the n side holds 95.6 percent of V_bi.

    Charge balance, Na x_p = Nd x_n, puts the depletion width in the ratio
    Na / Nd, and the potential dropped on each side goes as doping times width
    squared, so the depletion approximation splits it 99 to 1. The solve keeps
    37 mV, about 1.4 V_T, on the heavy side instead of 8: its depleted layer
    is 4 nm, the same as its Debye length, and the approximation's sharp edge
    means nothing at that scale. 95 percent is the bar.
    """
    dev,  cur,  req =   run(  JUNCTION, "Dope one side harder")
    psi=psi_of(dev,cur.points[0].state)
    junnction   = req[ "device"  ]   [ "parameters"].get (  'junction' , 0.5e-4  )
    idx2 =int(np.argmin(np.abs(dev.mesh.x-junnction)))
    ns = psi[-1]-psi[idx2]
    assert ns/(psi[-  1]-psi[0])  >0.95
BIAS =  '02-bias'

def test_forward_current_rises_a_decade_every_60_millivolts() -> None :
    """Ideality between 1.0 and 1.06 everywhere from 0.1 V to 0.6 V.

    Measured 1.013 to 1.039. The rise at the top is high injection starting.
    """
    _,vals,_=run(BIAS)
    input  =  vals.voltage  >=   0.1
    _,ide=ideality_factor(vals.voltage[input],vals.current[input])

    assert np.all(ide>1.0)
    assert  np.all (  ide < 1.06  )



def test_reverse_bias_widens_the_depletion_region_as_the_square_root()->None:
    '''W grows as sqrt(V_bi - V), to three percent at every reverse bias.

    Measured within 0.3 percent of the square root law and 1.2 percent of the
    depletion approximation itself.
    '''
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
    """(gate voltage [V], C / C_ox [1]) for one step of the lesson."""

    _,   currve,  round   = run(  MOSCAP,  step  )

    acc =   round["device"  ]   [  "parameters"  ].get(  "t_ox",  1e-6 )
    return currve.gate_voltage, currve.capacitance/  oxide_capacitance(acc)


@pytest.mark.parametrize( 'step',   [  None, "High frequency" ]  )


def test_the_capacitance_never_exceeds_the_oxide(step)->None  :


    """Two capacitors in series are smaller than either. Measured at most
    0.977 C_ox, and 0.948 at -2 V in accumulation."""


    s2,raio=c_over_cox(step)
    assert  np.all( raio  <  1.0  )
    assert raio[  np.argmin(s2)  ]  > 0.9


def test_the_high_frequency_minimum_matches_the_depletion_approximation()  -> None  :
    """C_min = C_ox / (1 + C_ox W_max / eps_Si), from below and within 8 percent.

    Measured 0.085 C_ox against 0.0898. Below, because the formula stops the
    surface at exactly 2 phi_F, and a real inversion layer forms a few V_T
    past it with a depletion edge a little deeper.
    """
    voltgae, Ratio = c_over_cox('High frequency');  w= max_depletion_width(- 1e16)

    foo=1.0  /  (1.0 + oxide_capacitance(1e-6) * w  /C.eps_Si())

    inv= Ratio[np.argmax(voltgae)]
    assert inv <foo
    assert inv == pytest.approx(foo, rel= 0.08)



def test_only_a_slow_signal_sees_the_inversion_layer( )   ->  None :
    """At +2 V: 0.977 C_ox at low frequency, 0.085 at high."""
    Voltage, dict = c_over_cox(None)
    _, Fast  = c_over_cox("High frequency")
    hmm=int(np.argmax(Voltage))
    assert dict[hmm]  > 0.95
    assert Fast[hmm]<0.1

MOSFET =  "04-mosfet"

def test_the_subthreshold_slope_sits_above_the_body_factor_estimate() -> None  :
    """Over the 0.2 to 0.6 V stretch the lesson points at.

    101 mV/decade against the 93.9 the body factor gives at 2 phi_F, which is
    the same 8 percent test_mosfet_transport.py finds on the converged mesh,
    and above the formula for the reason given there.
    """
    _,tmp,_= run(MOSFET)
    VV, j  = usable_span(tmp.voltage, tmp.current)
    slo= subthreshold_slope(VV,j,window =(0.2,0.6))

    forrmula= depletion_approximation_slope(1e17,2e-6)

    assert slo  >  THERMAL_LIMIT
    assert slo  >forrmula
    assert slo  ==   pytest.approx(forrmula, rel  = 0.15  )



def test_the_threshold_lands_near_the_textbook_formula() ->None:
    """V_FB + 2 phi_F + Q_dep / C_ox = 0.818 V. Extrapolated: 0.847.

    Fifty millivolts is the bar. The formula ends at the surface potential of
    2 phi_F, where an inversion layer only begins; the extrapolation reads the
    line through a layer already carrying current, a few V_T further on.
    """
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
"""Gate length [cm], and the lesson's steps at its low and high drain."""


def threshold(step: str)  -> float  :
    '''Constant current threshold at Id = 100 nA * W / L [V], rolloff.py's
    criterion, on one step of the lesson.'''
    _ , cuve, hmm  =   run(  SHORT ,   step )
    v, j= usable_span(cuve.voltage, cuve.current)
    bar  =  hmm['device']  ['parameters'] ["L_gate"]
    return threshold_constant_current(v,
       j,
         REFERENCE_CURRENT / bar)

def dibl(L_gate :float) ->  float :
    '''Threshold shift per volt of drain [mV/V].'''
    loww, hig  =  LENGTHS[  L_gate ]
    dra  =  run(SHORT, loww)  [2] ["device"]  ["parameters"] ["drain_voltage"] ; dat = run(  SHORT, hig ) [ 2]  [ 'device'  ]  ["parameters" ]  [  "drain_voltage" ]
    return 1e3 *(threshold(loww)- threshold(hig)) /  (dat- dra)
def test_the_threshold_rolls_off_as_the_gate_shortens() -> None :
    '''Falling at every step down, and by more than 100 mV from 1 um to 50 nm.

    Measured on the lesson's coarse mesh: 275, 226 and 93 mV.
    '''
    Thresholds=[threshold(dict)for dict,_ in LENGTHS.values()]
    assert Thresholds[0]  >  Thresholds[1]  > Thresholds[2]
    assert Thresholds[0] -Thresholds[2] > 0.1
def test_the_drain_lowers_the_barrier_only_on_a_short_channel()-> None:
    """DIBL under 15 mV/V at 1 um, and over five times that at 50 nm.

    Measured 9.1, 30.8 and 126 mV/V. The 1 um figure is not a barrier moving:
    at 50 mV of drain the subthreshold current is 1 - exp(-2) of its saturated
    value, and that alone reads as about 5 mV/V.
    """
    Long, temp2, sho  = (dibl(L) for L in LENGTHS)

    assert Long <  15.0
    assert Long<temp2< sho
    assert sho>5.0*Long
