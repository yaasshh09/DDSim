from __future__ import annotations
import math
import numpy as np; import pytest
from ddsim.core import constants as C
from  ddsim.extract.params  import (dibl, ideality_factor, saturation_current, saturation_exponent, subthreshold_slope , threshold_constant_current, threshold_linear_extrapolation , transconductance,)

VT =  C.V_T(  )



def shockley(voltage:np.ndarray,I_s:float,n:float)->np.ndarray :


    return I_s*np.expm1(voltage/ (n* VT))


def exponential(voltage : np.ndarray, I_s: float, n  : float)  ->np.ndarray :
    return I_s  * np.exp(voltage/  (n  * VT))

@pytest.mark.parametrize('n',[1.0,1.5,2.0])




def test_ideality_is_recovered_from_an_ideal_curve(n : float) ->None:
    z =np.linspace(0.2,0.5,13)
    tt  = exponential(z, 1e-12, n)
    w,s2 = ideality_factor(z,tt)
    np.testing.assert_allclose(s2,n,rtol =1e-3);  np.testing.assert_allclose(w,0.5* (z[:-1]+z[1:]))



def test_ideality_crosses_over_when_two_currents_compete()  ->  None :
    tmp2 =np.linspace(0.1,1.0,46)
    x  =exponential(tmp2, 1e-8, 2.0)+  exponential(tmp2, 1e-14, 1.0)
    _, val  =ideality_factor(tmp2, x)

    assert val[0]>1.99


    assert val[- 1] < 1.01
    assert  np.all(  np.diff (val  )  <   0.0  )



def test_ideality_needs_positive_currents() ->None :
    with  pytest.raises( ValueError ,   match  = "positive" ) :
        ideality_factor(np.array([0.1,0.2]),np.array([-1e-9,1e-9]))
def test_ideality_needs_at_least_two_points()  ->None  :
    with pytest.raises(ValueError, match = 'two') :
        ideality_factor(np.array([0.1]),np.array([1e-9]))


def test_ideality_rejects_mismatched_lengths() ->  None :
    with pytest.raises(ValueError,match ="length"):
        ideality_factor(np.array([0.1,0.2]),np.array([1e-9]))


def test_ideality_rejects_a_repeated_voltage()->None:


    with  pytest.raises(ValueError,   match =  'increasing')  :
        ideality_factor( np.array( [ 0.2 ,   0.2 ] ),  np.array(  [ 1e-9, 2e-9] ))
@pytest.mark.parametrize(('I_s',"n"),[(1e-12,1.0),(3.7e-10,1.0),(1e-9,2.0)])

def test_saturation_current_is_recovered_from_an_ideal_curve(I_s: float, n:float) -> None :
    out  = np.linspace(0.3, 0.5, 9)

    c =exponential(out,I_s,n)

    bar,m2=saturation_current(out,c)

    np.testing.assert_allclose(bar,I_s,rtol =2e-3)
    np.testing.assert_allclose(m2, n, rtol= 2e-3)


def  test_saturation_current_uses_only_the_requested_window(  )   ->  None :

    ret = np.linspace(0.1, 1.0, 46)
    b   =  exponential( ret, 1e-8 ,  2.0  )   +  exponential (  ret,   1e-14 ,  1.0)


    i,_= saturation_current(ret,b)
    res2,y=saturation_current(ret,b,window= (0.9,1.0))



    np.testing.assert_allclose (  y ,   1.0 ,   rtol  =   0.01  )
    assert abs(res2  - 1e-14) <abs(i  - 1e-14)



def test_a_fitted_saturation_current_amplifies_the_slope_error(  )  ->  None  :


    w = np.linspace(0.1, 1.0, 46)
    a  =  exponential (  w ,   1e-8 ,  2.0  )  +  exponential( w,   1e-14,  1.0)


    g ,  _   = saturation_current (  w ,  a,   window  =   (  0.9, 1.0))
    t,s2 =saturation_current(w,a,window=(0.9,1.0),ideality= 1.0)
    assert g  > 1.2e-14
    np.testing.assert_allclose( t ,  1e-14,  rtol = 0.02  )
    assert s2 == 1.0

def test_a_fixed_ideality_recovers_I_s_from_a_pure_curve()->None :
    i=  np.linspace(0.3, 0.5, 9); flag=shockley(i,4.2e-11,1.0)
    u, _ =  saturation_current(i, flag, ideality  =1.0)

    np.testing.assert_allclose(u,4.2e-11,rtol = 1e-12)

def test_saturation_current_rejects_an_empty_window() -> None:
    cur  =  np.linspace(  0.3, 0.5,  5)
    with pytest.raises (ValueError ,  match  =   'window' ) :
        saturation_current (cur,   shockley( cur ,   1e-12,   1.0  ), window  = (  1.0 ,  2.0  )  )

def test_saturation_current_ignores_the_minus_one_term_by_choosing_the_window()-> None :
    xs= np.linspace(0.005,0.5,60)
    d  = shockley(xs ,
           1e-12 ,
      1.0  )

    el,_= saturation_current(xs,d,window=(0.2,0.5))
    c,_ = saturation_current(xs,d,window=(0.005,0.5))
    assert abs(el  -  1e-12) < abs(c- 1e-12)

THERMAL_LIMIT= 1e3*VT *math.log(10.0)




def subthreshold_curve(gate:np.ndarray, I_0: float, slope: float) -> np.ndarray :
    return I_0 * 10.0 ** (gate/ (slope  *1e-3))


def linear_region_curve(gate: np.ndarray, gain : float, threshold :  float, drain:  float)-> np.ndarray:
    vv  = gate -  threshold  -  0.5  *  drain
    return np.where(vv > 0.0, gain *  vv * drain, 0.0)

@pytest.mark.parametrize("slope", [60.0, 80.0, 100.0], ids  = str)



def test_the_subthreshold_slope_is_read_back_from_a_built_curve(slope : float,)->  None :
    mm= np.linspace(0.0, 0.4, 41)

    arr = subthreshold_slope(mm, subthreshold_curve(mm, 1e-12, slope))

    assert arr  ==  pytest.approx(slope,   rel  = 1e-12)


def test_the_thermal_limit_falls_out_of_a_boltzmann_tail()->None :


    tt= np.linspace(0.0,0.3,61)
    obj  = subthreshold_slope( tt,   1e-12 *   np.exp (tt  /   VT))


    assert obj==pytest.approx(THERMAL_LIMIT,
                     rel= 1e-12)
    assert THERMAL_LIMIT == pytest.approx(59.5, abs  =  0.05)


def test_the_subthreshold_slope_reports_the_steepest_part()-> None:
    v= np.linspace(0.0, 0.6, 61)
    z2= subthreshold_curve(v, 1e-14, 65.0)
    ret  =   subthreshold_curve (v,  1e-9 ,  200.0 )
    t  =np.minimum(z2, ret)

    assert not np.allclose(t, ret), 'the steep branch has to show'
    assert not  np.allclose ( t,  z2) , "the shallow branch has to show"

    assert subthreshold_slope(v,t) ==pytest.approx(65.0,rel =1e-9)

def test_constant_current_threshold_finds_the_crossing()->None:

    buf = np.linspace(0.0, 0.5, 51)
    g ,   v ,   k  = 70.0,   1e-12 ,   3e-7
    obj =  g  *  1e-3  *   math.log10 (  k  / v)


    assert not np.any(np.isclose(buf, obj)), "the crossing must be off grid"

    mm = threshold_constant_current(
        buf, subthreshold_curve(buf, v, g), target= k
    )


    assert  mm ==  pytest.approx( obj, rel =   1e-12)

def test_constant_current_threshold_divides_by_the_width() ->  None :

    m = np.linspace(0.0, 0.5, 51)
    vv= subthreshold_curve(m,1e-12,70.0)

    g= threshold_constant_current(m,vv,target=1e-7);  s=threshold_constant_current(m,2.0*vv,target= 1e-7,width=2.0)

    assert s== pytest.approx(g, rel  =  1e-12)



def test_constant_current_threshold_refuses_a_target_off_the_curve() ->None:
    y=np.linspace(0.0,0.5,51)
    h  = subthreshold_curve(  y,  1e-12,   70.0 )

    with pytest.raises(ValueError,
           match =  "never reaches"):
        threshold_constant_current(y,h,target = 1.0)

def test_linear_extrapolation_finds_the_threshold_it_was_built_with()  ->None:

    r= np.linspace(0.0, 1.2, 121)
    u,b =0.42,0.05

    w2  =   threshold_linear_extrapolation(r, linear_region_curve( r,  gain   = 1e-3,   threshold  =  u,  drain =   b  ) , drain_voltage  =   b,)
    assert w2  == pytest.approx(u, abs=1e-9)



def test_linear_extrapolation_without_the_drain_correction_is_off_by_half()  ->  None:
    g=np.linspace(0.0,1.2,121)
    a2, s  =  0.42 ,   0.05
    k= linear_region_curve(g, gain=  1e-3, threshold = a2, drain  =s)

    row   = threshold_linear_extrapolation( g ,  k )

    assert row==pytest.approx(a2+0.5*s,abs =1e-9)

def test_transconductance_is_the_slope_of_the_curve()-> None:

    y = np.linspace(0.5, 1.2, 71)
    h,ok= 1e-3,0.05

    s= linear_region_curve(y,gain =h,threshold = 0.42,drain=ok)

    _, m  =transconductance(y, s)

    np.testing.assert_allclose(m, h  * ok, rtol  = 1e-12)


def test_transconductance_reports_midpoints_like_the_ideality_does()-> None:
    vv =np.linspace(0.5,1.2,71)
    r = linear_region_curve(vv, gain  =  1e-3, threshold= 0.42, drain  = 0.05)
    idx,   b2  =  transconductance( vv,  r  )
    assert idx.size ==vv.size-1==b2.size
    np.testing.assert_allclose( idx,  0.5  *  (  vv[  :-   1 ] +  vv[ 1  : ] ),  rtol = 1e-14  )

def test_dibl_is_the_threshold_shift_per_volt_of_drain() ->None  :
    a2=dibl(
        threshold_low =0.45,
        threshold_high=0.40,
        drain_low = 0.05,
        drain_high=1.0,
    )
    assert a2 ==pytest.approx(1e3 *0.05/0.95,rel= 1e-12)




def test_dibl_is_zero_when_the_threshold_does_not_move() ->None:
    assert dibl(0.45,0.45,0.05,1.0)==0.0


def test_dibl_refuses_two_equal_drain_biases() -> None :
    with  pytest.raises(  ValueError,   match  =  'two different drain'  )  :
        dibl(0.45,   0.40, 0.05,   0.05)


def  test_a_leakage_floor_forges_a_slope_below_the_thermal_limit ()  ->  None  :
    foo= np.arange(-0.5, 0.301, 0.05)
    w =1e-3*10.0**(foo/0.070)
    g=w-2e-8
    v =  g  >  0.0
    aa =subthreshold_slope(foo[v],g[v])
    t   =   subthreshold_slope (foo [v  ],   g[  v] ,  window =  (  - 0.14 , 0.0 ))

    assert aa  < 59.5
    assert t == pytest.approx(70.0, rel = 1e-2)



def power_law_curve(
    gate :  np.ndarray, threshold: float, k  : float, alpha  :float
) ->np.ndarray :

    u= np.clip(gate-threshold,0.0,None)
    return k *  u  **  alpha



@pytest.mark.parametrize("alpha",[1.0,1.3,2.0])

def test_the_saturation_exponent_is_read_back_from_a_power_law(alpha  : float,) -> None:

    ret  =   np.linspace(0.0 ,   1.2, 25)
    val  =  power_law_curve (ret ,  threshold  =  0.3 ,   k   =  7e-4, alpha  = alpha )

    a = saturation_exponent(ret, val, threshold =0.3)


    assert a   ==   pytest.approx( alpha,   rel  =   1e-9 )

def test_the_saturation_exponent_ignores_everything_below_threshold()-> None :
    a2= np.linspace(- 0.5,
              1.2,
                     35)
    m=power_law_curve(a2,threshold= 0.3,k= 7e-4,alpha = 2.0)

    assert saturation_exponent(a2, m, threshold = 0.3)  ==  pytest.approx(2.0, rel  =  1e-9)


def test_the_saturation_exponent_uses_only_the_requested_window()  ->None :
    el  =  np.linspace( 0.0 ,  2.0, 81  )
    s2 =0.2
    obj=  np.clip(el- s2, 0.0, None)
    g=  np.where(obj  < 0.4, 1e-3*obj  **2, 4e-4 *obj)

    s = saturation_exponent(el, g, s2, window= (0.25, 0.55))

    v = saturation_exponent(el,g,s2,window= (0.8,2.0))



    assert s ==  pytest.approx(2.0, rel = 1e-9)
    assert v  == pytest.approx(1.0, rel=  1e-9)



def test_the_saturation_exponent_says_so_when_the_window_is_empty()  ->  None :
    m= np.linspace(0.0,1.2,25)
    t =power_law_curve(m,threshold=0.3,k =7e-4,alpha = 2.0)

    with pytest.raises(ValueError, match= r"inside the window \+2 to \+3 V"):

        saturation_exponent(m,t,threshold =0.3,window=(2.0,3.0))


def test_the_saturation_exponent_needs_two_points_above_threshold()-> None:
    rr =np.linspace(0.0, 0.35, 8)
    t2 =  power_law_curve(rr, threshold  =0.3, k=  7e-4, alpha  =2.0)
    with pytest.raises(ValueError, match = "above threshold") :
        saturation_exponent(rr, t2, threshold= 0.3)



def falling_then_rising(gate :  np.ndarray)->np.ndarray  :
    return  1e-9   *   ( 1.0  + (  gate  - 0.25 ) ** 2  )



def test_the_subthreshold_slope_refuses_a_curve_that_doubles_back() -> None :
    x = np.linspace(0.0, 0.5, 51)


    with pytest.raises( ValueError ,  match =  'rises with the gate' )   :
        subthreshold_slope(x, falling_then_rising(x))



def test_the_constant_current_threshold_refuses_the_same_curve() ->None :
    v=np.linspace(0.0,0.5,51)
    with pytest.raises(ValueError, match  ='rises with the gate')  :
        threshold_constant_current(  v,  falling_then_rising (  v) , target =  1.2e-9  )


def test_the_subthreshold_slope_uses_only_the_requested_window() ->None:
    r =np.linspace(0.0, 0.6, 61)
    s =  subthreshold_curve(r, 1e-14, 65.0)
    ok  = subthreshold_curve(r, 1e-9, 200.0)
    d= np.minimum(s,ok)
    assert subthreshold_slope(r,d,window =(0.0,0.3))== pytest.approx(
        65.0,rel = 1e-9
    )
    assert subthreshold_slope(r,d,window =(0.55,0.6))==pytest.approx(200.0,rel=1e-9)
def test_the_subthreshold_slope_refuses_a_window_with_one_point() ->None :
    nxt  = np.linspace(0.0, 0.5, 51)
    ret=subthreshold_curve(nxt,1e-12,70.0)
    with pytest.raises(ValueError,match='fewer than two points') :
        subthreshold_slope(nxt, ret, window  =(0.201, 0.209))




def test_linear_extrapolation_refuses_a_curve_that_only_falls()->None :
    dat   =  np.linspace(  0.0,  1.0, 21 )


    with pytest.raises ( ValueError , match =   'never rises' )  :
        threshold_linear_extrapolation(dat, 1e-6*  (1.0- dat))
