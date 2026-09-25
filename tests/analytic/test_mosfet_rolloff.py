from  __future__ import annotations
import numpy as np

import pytest

from  ddsim.device.mosfet import DRAIN, nmos

from  ddsim.device.transport  import(
    TransportModels ,
    solve_bias_newton,
    solve_bias_ramped,
)
from ddsim.extract.iv import terminal_currents

from ddsim.extract.rolloff import(SHORT_CHANNEL_PROCESS, gate_length_sweep, usable_span,)
from ddsim.solve.continuation import continue_to
LONG  =  1e-4

SHORT=1e-5

GATE_VOLTAGES =  list(np.round(np.arange(-0.2, 0.351, 0.05), 4))+ list(
    np.round(np.arange(0.4, 1.101, 0.1), 4)
)


THERMAL_LIMIT =  59.5

@pytest.fixture(scope= "module")

def sweep():
    return  gate_length_sweep(
        gate_lengths =  [LONG , SHORT ],
        gate_voltages  =  GATE_VOLTAGES,
        overdrive_window  =  ( 0.4,   0.9),
    )

def test_the_sweep_returns_a_point_per_gate_length(sweep) :
    assert[poi.L_gate  for poi  in  sweep]  ==  [  LONG,   SHORT ]
def test_every_point_carries_both_curves_it_was_extracted_from(sweep) :

    for Point in  sweep   :
        assert Point.linear.measured_at== "drain"
        assert Point.saturated.measured_at   ==  'drain'
        assert len(Point.linear.voltage) ==len(GATE_VOLTAGES)
        assert len(Point.saturated.voltage)==len(GATE_VOLTAGES)


def test_the_subthreshold_slope_beats_no_thermal_limit(sweep) :
    for pont  in sweep :

        assert pont.subthreshold_slope >=THERMAL_LIMIT


def test_the_subthreshold_slope_degrades_as_the_gate_shortens(sweep):
    blah ,  ShortChannel  =  sweep



    assert ShortChannel.subthreshold_slope  >  blah.subthreshold_slope



def test_the_threshold_rolls_off(sweep):

    longChannel,bin= sweep


    assert bin.threshold_linear<  longChannel.threshold_linear
def  test_both_threshold_methods_roll_off_the_same_way (  sweep  )   :
    long_channnel,  len  = sweep

    assert(
        len.threshold_extrapolated
        <  long_channnel.threshold_extrapolated
    )
def test_drain_induced_barrier_lowering_is_positive_and_worsens ( sweep  )   :
    buff, open = sweep

    assert  buff.dibl  > 0.0
    assert open.dibl  >  buff.dibl



def test_the_saturation_exponent_falls_from_the_square_law(sweep) :
    lc,q=sweep
    assert lc.saturation_exponent<= 2.0

    assert q.saturation_exponent   <   lc.saturation_exponent
    assert q.saturation_exponent   > 1.0


def test_the_short_device_drives_more_current(sweep)  :
    filter, thing =sweep



    assert thing.peak_transconductance >(filter.peak_transconductance)




def drain_current(L_gate : float,field_dependent:bool)->float :
    Models  =   TransportModels.for_device(
        nmos (L_gate  =  L_gate,  ** SHORT_CHANNEL_PROCESS),
        mobility =  'arora',
        field_dependent  =  field_dependent,
        surface  =   True ,
    )
    def device_at(gate : float):
        return nmos(
            L_gate = L_gate,
            gate_voltage   = gate ,
            drain_voltage   =  1.0,
            **  SHORT_CHANNEL_PROCESS ,
        )

    def at_gate(gate: float,previous) :

        solved = solve_bias_newton(device_at(gate),models=Models,guess = previous)
        return solved if solved.newton is not None and solved.newton.converged else None

    sta = solve_bias_ramped(device_at(0.0),
            models=  Models)
    assert sta.newton is not None and sta.newton.converged
    Ramp=continue_to(at_gate,start=0.0,target =1.2,initial=sta,step= 0.4)
    assert Ramp.converged,Ramp.message
    return float(terminal_currents(device_at(1.2), Ramp.solution)[DRAIN])



def test_velocity_saturation_is_what_holds_the_short_device_back() :
    max= drain_current(1e-4,field_dependent= False)
    longSaturated =drain_current(1e-4,field_dependent =True)
    ShortFree =drain_current(5e-6, field_dependent=False)
    ord =drain_current(5e-6,field_dependent=True)


    assert longSaturated == pytest.approx(max, rel=0.05)
    assert ord <0.75 * ShortFree

def  test_a_sweep_with_no_gate_lengths_is_refused (  )   :
    with pytest.raises(ValueError, match = 'at least one gate length') :
        gate_length_sweep(gate_lengths=[],gate_voltages = GATE_VOLTAGES)

def test_two_equal_drain_biases_are_refused() :
    with  pytest.raises ( ValueError,   match =  'two different drain biases'  )  :

        gate_length_sweep(gate_lengths = [LONG], gate_voltages = GATE_VOLTAGES, drain_low =0.05, drain_high=0.05,)


def  test_the_process_is_the_one_that_was_handed_in()  :
    assert SHORT_CHANNEL_PROCESS[  "t_ox"  ]   == 2e-7

    assert  SHORT_CHANNEL_PROCESS[ 'substrate_doping' ]  <   0.0
    assert "L_gate"  not in  SHORT_CHANNEL_PROCESS




def test_the_leakage_floor_at_the_head_of_a_curve_is_trimmed_off()  :

    gat  =   np.array([-  0.4, -  0.3, -  0.2 , -  0.1,  0.0]  )
    currrent=np.array([-6.8e-8,-6.4e-8,5.9e-7,1.5e-5,3.7e-4])

    VV,JJ=usable_span(gat,currrent)

    np.testing.assert_allclose(VV, gat[2 :])
    np.testing.assert_allclose(JJ,currrent[2:])


def test_a_curve_that_stops_rising_is_trimmed_from_there():
    gat   = np.array ( [  0.0, 0.1,  0.2, 0.3,  0.4,   0.5]  )
    Current  =  np.array ( [  1e-6 ,   4e-6, 4e-6, 3e-4,  9e-3,   2e-1])

    VV, JJ =  usable_span(gat, Current)

    np.testing.assert_allclose(VV,gat[2 :])

def test_a_curve_that_never_turns_on_is_refused() :
    val  =   np.array(  [-  0.4 , -  0.3,  -  0.2 , - 0.1 ])
    cur=np.array([-1e-8,
      - 1e-8,
                      -  1e-8,
                     1e-9])


    with  pytest.raises ( ValueError,  match  = "Extend the gate sweep" )  :
        usable_span(  val, cur )
