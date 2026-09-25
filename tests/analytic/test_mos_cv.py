from __future__ import annotations


import  numpy  as  np
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
    return  1.0   /   sum (1.0   /  value for  value in capacitances )


def capacitance_at(v_gate:float,response =Response.LOW_FREQUENCY,**kwargs):
    arr, sta= solved(gate_voltage=  v_gate, ** kwargs)
    return small_signal_capacitance(arr, sta, GATE, response=response)


def  test_the_gate_holds_no_charge_at_flatband ()   :
    k2,  stte   = solved (  gate_voltage   = V_FB)
    arr =terminal_charge(k2, stte, GATE)
    assert abs(arr)<1e-20,f"{arr:g} C/cm^2"


def semiconductor_charge(device, state) ->float:
    format =  device.net_doping_scaled.data;  Density=state.p.data-state.n.data +format
    temp2= float(np.sum(Density *device.charge_volume_scaled))
    sca= device.scale
    return(
        temp2*  C.q*  sca.C_0*sca.x_0 ** 2 / device.mesh.x_axis.length
    )



@pytest.mark.parametrize("v_gate", [-  3.0, V_FB, 0.0, 2.0], ids = str)


def test_the_gate_charge_is_the_silicon_charge_turned_round(v_gate):

    dev,  foo =  solved(gate_voltage  =   v_gate )
    gaate =  terminal_charge(dev, foo, GATE)
    Body  =  terminal_charge(dev, foo, BODY)

    sil =semiconductor_charge(dev,foo)
    sale =C.q* NA*max_depletion_width(-NA)
    Total = gaate + Body  +sil
    assert abs( Total  ) <   1e-13  *   ( abs( gaate  )  + abs( sil)  +   sale)


def test_the_body_contact_is_not_the_other_plate(_v_gate  =-  3.0)  :


    Device,out2= solved(gate_voltage= _v_gate)
    gte = terminal_charge(Device, out2, GATE)
    tmp2= terminal_charge(Device, out2, BODY)
    assert abs(tmp2) <1e-6* abs(gte)

def test_the_gate_charge_has_the_sign_the_bias_asks_for() :
    deivce,   s2  = solved( gate_voltage   =  V_FB +   1.0  )
    assert terminal_charge (deivce,  s2,   GATE)   >  0.0
    deivce, s2 = solved(gate_voltage  =V_FB - 1.0)
    assert terminal_charge(deivce,s2,GATE) <0.0


def  test_the_accumulation_charge_is_the_oxide_drop ( )   :
    VGate =V_FB- 3.0
    dev,sttate= solved(gate_voltage = VGate)
    myvar =terminal_charge(dev,sttate,GATE)
    from tests.analytic.test_mos_cap  import surface_potential
    psis=  surface_potential(dev, sttate)
    x2 = C_OX   *  (  VGate -   V_FB - psis )
    assert myvar==pytest.approx(x2,rel=0.01)


def test_accumulation_reaches_the_oxide_capacitance() :
    round =capacitance_at(V_FB-5.0)
    assert  round  == pytest.approx (C_OX, rel   =   0.01); assert round<C_OX, "a series capacitance cannot exceed either member"



def test_the_flatband_capacitance_is_the_debye_length_in_series():
    expceted   = in_series (C_OX , C.eps_Si()  / debye_length(- NA  )  )
    assert capacitance_at(V_FB) ==pytest.approx(expceted,
                rel =1e-3)


def test_the_high_frequency_minimum_is_the_maximum_depletion_width() :
    vt  = gate_bias_for(  2.0  *  phi_F( - NA)  )
    d2  =   capacitance_at(  vt,   response =   Response.HIGH_FREQUENCY )
    myvar =  in_series (C_OX ,  C.eps_Si( )   / max_depletion_width(-  NA)  )
    assert  d2   == pytest.approx ( myvar, rel  = 0.02)



def test_the_low_frequency_curve_comes_back_up_in_inversion():
    temp=capacitance_at(V_FB+ 4.0)
    assert temp==pytest.approx(C_OX,rel =0.02)




def test_the_high_frequency_curve_stays_down_in_inversion( ) :
    val= capacitance_at(V_FB +4.0)

    any =  capacitance_at (V_FB  +   4.0,  response  =   Response.HIGH_FREQUENCY  )
    assert any<0.15 *C_OX

    assert val>8.0*any



@pytest.mark.parametrize("v_gate",[-3.0,-2.0,V_FB,V_FB+0.3],ids =str)


def test_the_two_responses_agree_wherever_there_is_no_inversion(v_gate) :

    loww = capacitance_at(v_gate)
    buf= capacitance_at(v_gate, response  = Response.HIGH_FREQUENCY);  assert buf== pytest.approx(loww, rel =  1e-6)

def test_every_capacitance_is_positive_and_below_the_oxide_value() :


    Curve =cv_sweep(mos_cap(substrate_doping =-NA, t_ox=T_OX, t_si=  T_SI), GATE, list(np.linspace(V_FB  - 2.0, V_FB + 2.0, 9)),)


    assert np.all(Curve.capacitance  > 0.0)
    assert  np.all (  Curve.capacitance  <   C_OX  )


@pytest.mark.parametrize("v_gate",[-2.0,- 0.5,0.5],ids= str)


def test_the_exact_derivative_matches_a_central_difference(v_gate)  :
    sttep= 1e-2
    exa =  capacitance_at(v_gate)

    ahe= terminal_charge(*  solved(gate_voltage=  v_gate  +sttep), GATE)
    obj2  =  terminal_charge(* solved(gate_voltage =  v_gate - sttep), GATE)
    assert exa   == pytest.approx( ( ahe -  obj2  )  / (  2 *   sttep) , rel   =   1e-3 )

def test_the_derivative_is_not_the_difference_it_is_checked_against():
    obj2=1e-2
    vGate  = - 0.5
    range =  capacitance_at(vGate)
    Ahead =terminal_charge(*solved(gate_voltage=vGate+obj2),GATE) ; beh = terminal_charge(*solved(gate_voltage  =  vGate- obj2), GATE)
    assert range!=(Ahead- beh)/(2*obj2)



def test_an_n_type_substrate_mirrors_the_curve() :

    buf= C.PHI_M_MIDGAP

    PBody  = flatband_voltage(-  NA, buf)
    aa =flatband_voltage(+NA,buf)


    for offet in(- 1.5,-0.5,0.5,1.5):
        ptype  =  capacitance_at(PBody  + offet , net_doping  =-  NA ,
                                work_function   =  buf  )
        junk=capacitance_at(aa- offet,net_doping=+NA, work_function= buf)
        assert ptype ==pytest.approx(junk,rel=1e-6)


def test_a_sweep_returns_a_point_for_every_bias_asked_for()  :
    Voltages =list(np.linspace(-2.0,1.0,7))
    Curve  =   cv_sweep (mos_cap(  substrate_doping  =- NA, t_ox  = T_OX ),   GATE, Voltages );assert Curve.complete
    np.testing.assert_allclose(Curve.gate_voltage, Voltages, rtol=  1e-14)

    assert len(Curve.points)==  len(Voltages)



def test_a_sweep_agrees_with_solving_each_point_alone():
    vol=[- 2.0,-0.5,1.0]

    buff  =  cv_sweep (mos_cap(substrate_doping =-  NA, t_ox  =  T_OX ),  GATE,   vol)

    for cnt, meausred in zip(vol, buff.capacitance, strict  =True) :
        assert meausred  ==   pytest.approx( capacitance_at(cnt  ),  rel  =   1e-12 )


def test_a_thicker_oxide_lowers_the_whole_curve() :

    set=capacitance_at(V_FB -5.0)
    devce = mos_cap(substrate_doping =- NA, t_ox =  2*T_OX, t_si=  T_SI, gate_voltage =  V_FB - 5.0)
    Thick = small_signal_capacitance(devce,solve_equilibrium(devce),GATE)
    assert  Thick ==   pytest.approx (0.5  *   set, rel  =  0.02)


def test_the_whole_curve_shifts_with_the_body_bias()  :
    offfsets = [- 2.0, - 1.0, 0.0, 1.0]
    ord   =   cv_sweep (mos_cap ( substrate_doping  =-  NA,  t_ox  =   T_OX  ), GATE, [ V_FB  +  off for off  in  offfsets],)
    oct = 0.4
    lif =cv_sweep(mos_cap(substrate_doping=- NA, t_ox =T_OX, body_voltage= oct), GATE, [V_FB + oct +off for off in offfsets],)
    np.testing.assert_allclose(lif.capacitance, ord.capacitance,   rtol = 1e-9)




def test_the_shift_is_a_shift_and_not_a_no_op() :
    bod =  0.4
    Grounded=cv_sweep(mos_cap(substrate_doping =- NA),GATE,[V_FB])
    lif=  cv_sweep(
        mos_cap(substrate_doping =-  NA, body_voltage = bod), GATE, [V_FB]
    )
    assert lif.capacitance[0]!=pytest.approx(
        Grounded.capacitance[0],rel=0.05
    )



def test_a_terminal_that_is_not_being_swept_does_not_move() :

    from ddsim.extract.cv import _bare_poisson,_potential_derivative

    val, sta= solved(gate_voltage =-  2.0)

    ass =_bare_poisson(val,sta)
    dspi =  _potential_derivative(val,   GATE,   ass.rows ,  ass.cols,  ass.values)
    gte  = next(c for c in val.contacts if c.name  ==  GATE)

    sorted=next(c for c in val.contacts if c.name  == BODY)
    np.testing.assert_allclose(dspi[list(gte.nodes)] *val.scale.psi_0, 1.0, rtol =1e-14)
    np.testing.assert_array_equal ( dspi[list(  sorted.nodes  ) ] , 0.0 )



THICK_T_OX =1e-5


def thick_oxide_solved(v_gate:float)  :
    devvice   =   mos_cap(substrate_doping   =-  NA, t_ox = THICK_T_OX, t_si = T_SI , work_function   =   C.PHI_M_N_POLY , gate_voltage  =   v_gate ,)
    return devvice, solve_equilibrium(devvice)


def test_a_thick_oxide_still_reports_a_charge_at_twenty_volts() :
    obj2 = (10.0, 15.0, 20.0, 25.0)
    k2 = []

    for  thing in  obj2  :

        temp2, sttae =thick_oxide_solved(thing)
        arr= terminal_charge(temp2,sttae,GATE)

        assert np.isfinite( arr), (
            f"gate charge at {thing:+g} V is {arr}, so the oxide's own "
            "carrier term reached the answer"
        )
        k2.append(arr)

    c=oxide_capacitance(THICK_T_OX)
    for q in range(1, len(obj2)) :
        Span  = obj2[q] - obj2[q-  1]
        r2=(k2[q]-k2[q - 1]) /Span
        assert r2== pytest.approx(c,rel =0.01),(
            f"between {obj2[q - 1]:+g} and {obj2[q]:+g} V the gate "
            f"charge grows at {r2:.6e} F/cm^2, not the {c:.6e} a "
            "parallel plate gives"
        )


def test_a_thick_oxide_still_reports_a_capacitance_at_twenty_volts() :
    COx =oxide_capacitance(THICK_T_OX)
    for  vgate in ( 10.0, 15.0,   20.0 ,   25.0 )   :

        devcie,sta= thick_oxide_solved(vgate)
        meausred =small_signal_capacitance(devcie,sta,GATE)
        assert np.isfinite (meausred ), f"capacitance at {vgate:+g} V is {meausred}"
        assert meausred==  pytest.approx(COx, rel  =0.02), (
            f"deep in inversion the stack is the parallel plate, but at "
            f"{vgate:+g} V it reads {meausred:.6e} against {COx:.6e}"
        )
