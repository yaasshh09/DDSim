from __future__ import  annotations
import numpy as np, pytest



from ddsim.core import constants as C

from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import BODY,GATE,mos_cap

from ddsim.device.pn_diode  import pn_diode
from ddsim.extract.cv import(
    CVCurve,
    Response,
    cv_sweep,
    small_signal_capacitance,
    terminal_charge,
)

NA  =   1e16
T_OX= 1e-6

@pytest.fixture(scope  = "module")



def cap() :
    Device   =  mos_cap (  substrate_doping =-  NA ,  t_ox   = T_OX, gate_voltage  =- 2.0 ) ; return Device, solve_equilibrium(Device )




@pytest.mark.parametrize('call', [lambda d,s :terminal_charge(d,s,"drain"), lambda d,s: small_signal_capacitance(d,s,"drain"),], ids= ["charge",'capacitance'],)



def test_a_terminal_that_is_not_there_is_named_in_the_refusal(cap,call):
    dvice,sate = cap
    with pytest.raises(KeyError,match="drain"):
        call(dvice, sate)

def test_the_refusal_lists_the_terminals_that_are_there(cap):
    Device, input  = cap
    with pytest.raises(KeyError)as buf :
        terminal_charge(Device, input, 'drain')
    assert GATE in str(buf.value)
    assert BODY in str(buf.value)


def  test_a_sweep_checks_the_contact_before_solving_anything(cap)  :
    dict,   _ =  cap
    with pytest.raises(KeyError, match = "drain") :

        cv_sweep(dict, "drain", [0.0])

def  test_a_1d_device_needs_no_width(  )   :
    dioode   = pn_diode(anode_voltage  =-  1.0); State=solve_equilibrium(dioode)
    xx= terminal_charge(dioode, State, "anode")
    assert np.isfinite(  xx )
    assert xx  !=  0.0



def  test_a_reverse_biased_diode_has_a_junction_capacitance()  :

    Diode =pn_diode(anode_voltage =-1.0)
    capactiance=small_signal_capacitance(
        Diode,solve_equilibrium(Diode),'anode'
    )
    assert capactiance >   0.0




def test_giving_a_width_scales_the_answer_by_it(cap):
    Device,sttae =cap;natuural  = terminal_charge(Device, sttae, GATE)
    object= terminal_charge(
        Device,sttae,GATE,width=2*Device.mesh.x_axis.length
    )
    assert object== pytest.approx(0.5*natuural,rel=1e-14)



def test_the_capacitance_takes_the_same_width(cap):
    dev,  sta  =  cap
    dat=small_signal_capacitance(dev,sta,GATE)
    duobled=small_signal_capacitance(dev, sta, GATE, width  = 2* dev.mesh.x_axis.length)
    assert  duobled  ==  pytest.approx(  0.5  *  dat ,  rel   =  1e-14)


def test_a_curve_reports_what_it_is():

    Curve   = cv_sweep(mos_cap( substrate_doping   =-  NA ), GATE ,  [ -  1.0,  0.0  ]  ) ; tex  =   repr(  Curve)
    assert "2 points"  in tex and  "complete" in  tex
    assert Response.LOW_FREQUENCY.value in tex

def test_an_empty_curve_says_so() :
    emp =CVCurve(
        contact = GATE,
        response = Response.LOW_FREQUENCY,
        points=(),
        complete=False,
    )
    assert  'empty'  in  repr(emp)
    assert "stopped early" in repr(emp)

    assert emp.gate_voltage.size  ==   0

    assert emp.capacitance.size ==0
    assert emp.charge.size ==   0



def test_a_sweep_that_cannot_converge_returns_what_it_reached() :
    vf =   float( C.work_function_difference(  C.PHI_M_N_POLY,   -  NA )  )
    item2 = cv_sweep(
        mos_cap(substrate_doping=- NA),
        GATE,
        [vf,vf+2.0],
        max_iterations=1,
    )
    assert not item2.complete
    assert len(item2.points) ==1
    assert "did not converge" in item2.message
    assert f"{vf + 2.0:+g}" in item2.message

def  test_the_charge_on_the_curve_is_the_charge_at_the_point ()  :

    deice = mos_cap(substrate_doping=-NA)
    Curve=cv_sweep(deice,GATE,[-1.5])
    temp2   =  Curve.points[ 0  ]
    idx2= deice.with_bias(**{GATE: -1.5})
    assert  temp2.charge ==  pytest.approx(terminal_charge( idx2,   temp2.state,   GATE  ), rel   =  1e-14)

def  test_the_response_is_carried_onto_the_curve()  :
    d2=cv_sweep(
        mos_cap(substrate_doping=-NA),
        GATE,
        [1.0],
        response=Response.HIGH_FREQUENCY,
    )
    assert d2.response is Response.HIGH_FREQUENCY
    assert d2.capacitance[0]==pytest.approx(small_signal_capacitance(mos_cap(substrate_doping=-NA,gate_voltage=1.0), d2.points[0].state, GATE, response=Response.HIGH_FREQUENCY,), rel=1e-14,)
