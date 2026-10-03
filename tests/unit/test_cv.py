from __future__ import annotations
import numpy as np, pytest
from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import BODY,GATE,mos_cap
from ddsim.device.pn_diode import pn_diode
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
    k2   =  mos_cap (  substrate_doping =-  NA ,  t_ox   = T_OX, gate_voltage  =- 2.0 ) ; return k2, solve_equilibrium(k2 )




@pytest.mark.parametrize('call', [lambda d,s :terminal_charge(d,s,"drain"), lambda d,s: small_signal_capacitance(d,s,"drain"),], ids= ["charge",'capacitance'],)



def test_a_terminal_that_is_not_there_is_named_in_the_refusal(cap,call):
    t,yy = cap
    with pytest.raises(KeyError,match="drain"):
        call(t, yy)

def test_the_refusal_lists_the_terminals_that_are_there(cap):
    i, aa  = cap
    with pytest.raises(KeyError)as g :
        terminal_charge(i, aa, 'drain')
    assert GATE in str(g.value)
    assert BODY in str(g.value)


def  test_a_sweep_checks_the_contact_before_solving_anything(cap)  :
    j,   _ =  cap
    with pytest.raises(KeyError, match = "drain") :

        cv_sweep(j, "drain", [0.0])

def  test_a_1d_device_needs_no_width(  )   :
    bb   = pn_diode(anode_voltage  =-  1.0); u=solve_equilibrium(bb)
    z= terminal_charge(bb, u, "anode")
    assert np.isfinite(  z )
    assert z  !=  0.0



def  test_a_reverse_biased_diode_has_a_junction_capacitance()  :

    v =pn_diode(anode_voltage =-1.0)
    k=small_signal_capacitance(
        v,solve_equilibrium(v),'anode'
    )
    assert k >   0.0




def test_giving_a_width_scales_the_answer_by_it(cap):
    f,el =cap;b  = terminal_charge(f, el, GATE)
    flag= terminal_charge(
        f,el,GATE,width=2*f.mesh.x_axis.length
    )
    assert flag== pytest.approx(0.5*b,rel=1e-14)



def test_the_capacitance_takes_the_same_width(cap):
    val,  w  =  cap
    cnt=small_signal_capacitance(val,w,GATE)
    m2=small_signal_capacitance(val, w, GATE, width  = 2* val.mesh.x_axis.length)
    assert  m2  ==  pytest.approx(  0.5  *  cnt ,  rel   =  1e-14)


def test_a_curve_reports_what_it_is():

    y   = cv_sweep(mos_cap( substrate_doping   =-  NA ), GATE ,  [ -  1.0,  0.0  ]  ) ; val  =   repr(  y)
    assert "2 points"  in val and  "complete" in  val
    assert Response.LOW_FREQUENCY.value in val

def test_an_empty_curve_says_so() :
    g =CVCurve(
        contact = GATE,
        response = Response.LOW_FREQUENCY,
        points=(),
        complete=False,
    )
    assert  'empty'  in  repr(g)
    assert "stopped early" in repr(g)

    assert g.gate_voltage.size  ==   0

    assert g.capacitance.size ==0
    assert g.charge.size ==   0



def test_a_sweep_that_cannot_converge_returns_what_it_reached() :
    c =   float( C.work_function_difference(  C.PHI_M_N_POLY,   -  NA )  )
    z = cv_sweep(
        mos_cap(substrate_doping=- NA),
        GATE,
        [c,c+2.0],
        max_iterations=1,
    )
    assert not z.complete
    assert len(z.points) ==1
    assert "did not converge" in z.message
    assert f"{c + 2.0:+g}" in z.message

def  test_the_charge_on_the_curve_is_the_charge_at_the_point ()  :

    c = mos_cap(substrate_doping=-NA)
    v=cv_sweep(c,GATE,[-1.5])
    dd   =  v.points[ 0  ]
    vv= c.with_bias(**{GATE: -1.5})
    assert  dd.charge ==  pytest.approx(terminal_charge( vv,   dd.state,   GATE  ), rel   =  1e-14)

def  test_the_response_is_carried_onto_the_curve()  :
    xx=cv_sweep(
        mos_cap(substrate_doping=-NA),
        GATE,
        [1.0],
        response=Response.HIGH_FREQUENCY,
    )
    assert xx.response is Response.HIGH_FREQUENCY
    assert xx.capacitance[0]==pytest.approx(small_signal_capacitance(mos_cap(substrate_doping=-NA,gate_voltage=1.0), xx.points[0].state, GATE, response=Response.HIGH_FREQUENCY,), rel=1e-14,)
