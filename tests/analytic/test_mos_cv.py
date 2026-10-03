from __future__ import annotations
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import BODY, GATE, mos_cap
from ddsim.extract.cv import(Response, cv_sweep, small_signal_capacitance, terminal_charge,)
from tests.analytic.test_mos_cap import(
    NA,
    T_OX,
    T_SI,
    flatband_voltage,
    gate_bias_for,
    max_depletion_width,
    oxide_capacitance,
    phi_F,
    solved,
)
V_FB =flatband_voltage(-NA,C.PHI_M_N_POLY)

C_OX = oxide_capacitance(T_OX)




def debye_length(net_doping:float) ->float:
    return float(np.sqrt(C.eps_Si()*C.V_T()/(C.q* abs(net_doping))))



def  in_series (  *  capacitances   :  float )   ->  float   :
    return  1.0   /   sum (1.0   /  out2 for  out2 in capacitances )


def capacitance_at(v_gate:float,response =Response.LOW_FREQUENCY,**kwargs):
    f, t= solved(gate_voltage=  v_gate, ** kwargs)
    return small_signal_capacitance(f, t, GATE, response=response)


def  test_the_gate_holds_no_charge_at_flatband ()   :
    it,  tmp3   = solved (  gate_voltage   = V_FB)
    b =terminal_charge(it, tmp3, GATE)
    assert abs(b)<1e-20,f"{b:g} C/cm^2"


def semiconductor_charge(device, state) ->float:
    zz =  device.net_doping_scaled.data;  j=state.p.data-state.n.data +zz
    g= float(np.sum(j *device.charge_volume_scaled))
    u= device.scale
    return(
        g*  C.q*  u.C_0*u.x_0 ** 2 / device.mesh.x_axis.length
    )



@pytest.mark.parametrize("v_gate", [-  3.0, V_FB, 0.0, 2.0], ids = str)


def test_the_gate_charge_is_the_silicon_charge_turned_round(v_gate):

    u,  ss =  solved(gate_voltage  =   v_gate )
    d2 =  terminal_charge(u, ss, GATE)
    d  =  terminal_charge(u, ss, BODY)

    item =semiconductor_charge(u,ss)
    i =C.q* NA*max_depletion_width(-NA)
    a = d2 + d  +item
    assert abs( a  ) <   1e-13  *   ( abs( d2  )  + abs( item)  +   i)


def test_the_body_contact_is_not_the_other_plate(_v_gate  =-  3.0)  :


    g,k= solved(gate_voltage= _v_gate)
    foo = terminal_charge(g, k, GATE)
    d= terminal_charge(g, k, BODY)
    assert abs(d) <1e-6* abs(foo)

def test_the_gate_charge_has_the_sign_the_bias_asks_for() :
    idx,   ys  = solved( gate_voltage   =  V_FB +   1.0  )
    assert terminal_charge (idx,  ys,   GATE)   >  0.0
    idx, ys = solved(gate_voltage  =V_FB - 1.0)
    assert terminal_charge(idx,ys,GATE) <0.0


def  test_the_accumulation_charge_is_the_oxide_drop ( )   :
    d =V_FB- 3.0
    s,j= solved(gate_voltage = d)
    b =terminal_charge(s,j,GATE)
    from tests.analytic.test_mos_cap  import surface_potential
    tmp3=  surface_potential(s, j)
    v2 = C_OX   *  (  d -   V_FB - tmp3 )
    assert b==pytest.approx(v2,rel=0.01)


def test_accumulation_reaches_the_oxide_capacitance() :
    e =capacitance_at(V_FB-5.0)
    assert  e  == pytest.approx (C_OX, rel   =   0.01); assert e<C_OX, "a series capacitance cannot exceed either member"



def test_the_flatband_capacitance_is_the_debye_length_in_series():
    u   = in_series (C_OX , C.eps_Si()  / debye_length(- NA  )  )
    assert capacitance_at(V_FB) ==pytest.approx(u,
                rel =1e-3)


def test_the_high_frequency_minimum_is_the_maximum_depletion_width() :
    a  = gate_bias_for(  2.0  *  phi_F( - NA)  )
    u  =   capacitance_at(  a,   response =   Response.HIGH_FREQUENCY )
    f =  in_series (C_OX ,  C.eps_Si( )   / max_depletion_width(-  NA)  )
    assert  u   == pytest.approx ( f, rel  = 0.02)



def test_the_low_frequency_curve_comes_back_up_in_inversion():
    num=capacitance_at(V_FB+ 4.0)
    assert num==pytest.approx(C_OX,rel =0.02)




def test_the_high_frequency_curve_stays_down_in_inversion( ) :
    a2= capacitance_at(V_FB +4.0)

    c =  capacitance_at (V_FB  +   4.0,  response  =   Response.HIGH_FREQUENCY  )
    assert c<0.15 *C_OX

    assert a2>8.0*c



@pytest.mark.parametrize("v_gate",[-3.0,-2.0,V_FB,V_FB+0.3],ids =str)


def test_the_two_responses_agree_wherever_there_is_no_inversion(v_gate) :

    mm = capacitance_at(v_gate)
    v= capacitance_at(v_gate, response  = Response.HIGH_FREQUENCY);  assert v== pytest.approx(mm, rel =  1e-6)

def test_every_capacitance_is_positive_and_below_the_oxide_value() :


    d =cv_sweep(mos_cap(substrate_doping =-NA, t_ox=T_OX, t_si=  T_SI), GATE, list(np.linspace(V_FB  - 2.0, V_FB + 2.0, 9)),)


    assert np.all(d.capacitance  > 0.0)
    assert  np.all (  d.capacitance  <   C_OX  )


@pytest.mark.parametrize("v_gate",[-2.0,- 0.5,0.5],ids= str)


def test_the_exact_derivative_matches_a_central_difference(v_gate)  :
    out= 1e-2
    t =  capacitance_at(v_gate)

    y2= terminal_charge(*  solved(gate_voltage=  v_gate  +out), GATE)
    h  =  terminal_charge(* solved(gate_voltage =  v_gate - out), GATE)
    assert t   == pytest.approx( ( y2 -  h  )  / (  2 *   out) , rel   =   1e-3 )

def test_the_derivative_is_not_the_difference_it_is_checked_against():
    d2=1e-2
    y  = - 0.5
    s =  capacitance_at(y)
    cnt =terminal_charge(*solved(gate_voltage=y+d2),GATE) ; k = terminal_charge(*solved(gate_voltage  =  y- d2), GATE)
    assert s!=(cnt- k)/(2*d2)



def test_an_n_type_substrate_mirrors_the_curve() :

    v= C.PHI_M_MIDGAP

    it  = flatband_voltage(-  NA, v)
    w =flatband_voltage(+NA,v)


    for m2 in(- 1.5,-0.5,0.5,1.5):
        u  =  capacitance_at(it  + m2 , net_doping  =-  NA ,
                                work_function   =  v  )
        j=capacitance_at(w- m2,net_doping=+NA, work_function= v)
        assert u ==pytest.approx(j,rel=1e-6)


def test_a_sweep_returns_a_point_for_every_bias_asked_for()  :
    w =list(np.linspace(-2.0,1.0,7))
    yy  =   cv_sweep (mos_cap(  substrate_doping  =- NA, t_ox  = T_OX ),   GATE, w );assert yy.complete
    np.testing.assert_allclose(yy.gate_voltage, w, rtol=  1e-14)

    assert len(yy.points)==  len(w)



def test_a_sweep_agrees_with_solving_each_point_alone():
    cur=[- 2.0,-0.5,1.0]

    a  =  cv_sweep (mos_cap(substrate_doping =-  NA, t_ox  =  T_OX ),  GATE,   cur)

    for vals, b in zip(cur, a.capacitance, strict  =True) :
        assert b  ==   pytest.approx( capacitance_at(vals  ),  rel  =   1e-12 )


def test_a_thicker_oxide_lowers_the_whole_curve() :

    m2=capacitance_at(V_FB -5.0)
    u = mos_cap(substrate_doping =- NA, t_ox =  2*T_OX, t_si=  T_SI, gate_voltage =  V_FB - 5.0)
    x2 = small_signal_capacitance(u,solve_equilibrium(u),GATE)
    assert  x2 ==   pytest.approx (0.5  *   m2, rel  =  0.02)


def test_the_whole_curve_shifts_with_the_body_bias()  :
    dat = [- 2.0, - 1.0, 0.0, 1.0]
    kk   =   cv_sweep (mos_cap ( substrate_doping  =-  NA,  t_ox  =   T_OX  ), GATE, [ V_FB  +  tmp for tmp  in  dat],)
    d = 0.4
    t2 =cv_sweep(mos_cap(substrate_doping=- NA, t_ox =T_OX, body_voltage= d), GATE, [V_FB + d +tmp for tmp in dat],)
    np.testing.assert_allclose(t2.capacitance, kk.capacitance,   rtol = 1e-9)




def test_the_shift_is_a_shift_and_not_a_no_op() :
    g =  0.4
    b=cv_sweep(mos_cap(substrate_doping =- NA),GATE,[V_FB])
    xx=  cv_sweep(
        mos_cap(substrate_doping =-  NA, body_voltage = g), GATE, [V_FB]
    )
    assert xx.capacitance[0]!=pytest.approx(
        b.capacitance[0],rel=0.05
    )



def test_a_terminal_that_is_not_being_swept_does_not_move() :

    from ddsim.extract.cv import _bare_poisson,_potential_derivative

    ii, cc= solved(gate_voltage =-  2.0)

    d =_bare_poisson(ii,cc)
    x =  _potential_derivative(ii,   GATE,   d.rows ,  d.cols,  d.values)
    f  = next(m for m in ii.contacts if m.name  ==  GATE)

    xs=next(z for z in ii.contacts if z.name  == BODY)
    np.testing.assert_allclose(x[list(f.nodes)] *ii.scale.psi_0, 1.0, rtol =1e-14)
    np.testing.assert_array_equal ( x[list(  xs.nodes  ) ] , 0.0 )



THICK_T_OX =1e-5


def thick_oxide_solved(v_gate:float)  :
    dat   =   mos_cap(substrate_doping   =-  NA, t_ox = THICK_T_OX, t_si = T_SI , work_function   =   C.PHI_M_N_POLY , gate_voltage  =   v_gate ,)
    return dat, solve_equilibrium(dat)


def test_a_thick_oxide_still_reports_a_charge_at_twenty_volts() :
    z = (10.0, 15.0, 20.0, 25.0)
    kk = []

    for  h in  z  :

        out2, info =thick_oxide_solved(h)
        flag= terminal_charge(out2,info,GATE)

        assert np.isfinite( flag), (
            f"gate charge at {h:+g} V is {flag}, so the oxide's own "
            "carrier term reached the answer"
        )
        kk.append(flag)

    i=oxide_capacitance(THICK_T_OX)
    for j in range(1, len(z)) :
        m2  = z[j] - z[j-  1]
        x=(kk[j]-kk[j - 1]) /m2
        assert x== pytest.approx(i,rel =0.01),(
            f"between {z[j - 1]:+g} and {z[j]:+g} V the gate "
            f"charge grows at {x:.6e} F/cm^2, not the {i:.6e} a "
            "parallel plate gives"
        )


def test_a_thick_oxide_still_reports_a_capacitance_at_twenty_volts() :
    item =oxide_capacitance(THICK_T_OX)
    for  d in ( 10.0, 15.0,   20.0 ,   25.0 )   :

        vv,num= thick_oxide_solved(d)
        r =small_signal_capacitance(vv,num,GATE)
        assert np.isfinite (r ), f"capacitance at {d:+g} V is {r}"
        assert r==  pytest.approx(item, rel  =0.02), (
            f"deep in inversion the stack is the parallel plate, but at "
            f"{d:+g} V it reads {r:.6e} against {item:.6e}"
        )
