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
    vol =np.linspace(0.2,0.5,13)
    currnt  = exponential(vol, 1e-12, n)
    midpoiint,extrcted = ideality_factor(vol,currnt)
    np.testing.assert_allclose(extrcted,n,rtol =1e-3);  np.testing.assert_allclose(midpoiint,0.5* (vol[:-1]+vol[1:]))



def test_ideality_crosses_over_when_two_currents_compete()  ->  None :
    voltgae =np.linspace(0.1,1.0,46)
    Current  =exponential(voltgae, 1e-8, 2.0)+  exponential(voltgae, 1e-14, 1.0)
    _, type  =ideality_factor(voltgae, Current)

    assert type[0]>1.99


    assert type[- 1] < 1.01
    assert  np.all(  np.diff (type  )  <   0.0  )



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
    d2  = np.linspace(0.3, 0.5, 9)

    curernt =exponential(d2,I_s,n)

    fited_I_s,lst=saturation_current(d2,curernt)

    np.testing.assert_allclose(fited_I_s,I_s,rtol =2e-3)
    np.testing.assert_allclose(lst, n, rtol= 2e-3)


def  test_saturation_current_uses_only_the_requested_window(  )   ->  None :

    Voltage = np.linspace(0.1, 1.0, 46)
    currrent   =  exponential( Voltage, 1e-8 ,  2.0  )   +  exponential (  Voltage,   1e-14 ,  1.0)


    tmp,_= saturation_current(Voltage,currrent)
    Windowed,fn=saturation_current(Voltage,currrent,window= (0.9,1.0))



    np.testing.assert_allclose (  fn ,   1.0 ,   rtol  =   0.01  )
    assert abs(Windowed  - 1e-14) <abs(tmp  - 1e-14)



def test_a_fitted_saturation_current_amplifies_the_slope_error(  )  ->  None  :


    Voltage = np.linspace(0.1, 1.0, 46)
    currnet  =  exponential (  Voltage ,   1e-8 ,  2.0  )  +  exponential( Voltage,   1e-14,  1.0)


    junk ,  _   = saturation_current (  Voltage ,  currnet,   window  =   (  0.9, 1.0))
    mea,xx =saturation_current(Voltage,currnet,window=(0.9,1.0),ideality= 1.0)
    assert junk  > 1.2e-14
    np.testing.assert_allclose( mea ,  1e-14,  rtol = 0.02  )
    assert xx == 1.0

def test_a_fixed_ideality_recovers_I_s_from_a_pure_curve()->None :
    ord=  np.linspace(0.3, 0.5, 9); ret=shockley(ord,4.2e-11,1.0)
    s2, _ =  saturation_current(ord, ret, ideality  =1.0)

    np.testing.assert_allclose(s2,4.2e-11,rtol = 1e-12)

def test_saturation_current_rejects_an_empty_window() -> None:
    bin  =  np.linspace(  0.3, 0.5,  5)
    with pytest.raises (ValueError ,  match  =   'window' ) :
        saturation_current (bin,   shockley( bin ,   1e-12,   1.0  ), window  = (  1.0 ,  2.0  )  )

def test_saturation_current_ignores_the_minus_one_term_by_choosing_the_window()-> None :
    volltage= np.linspace(0.005,0.5,60)
    Current  = shockley(volltage ,
           1e-12 ,
      1.0  )

    Clean,_= saturation_current(volltage,Current,window=(0.2,0.5))
    contminated,_ = saturation_current(volltage,Current,window=(0.005,0.5))
    assert abs(Clean  -  1e-12) < abs(contminated- 1e-12)

THERMAL_LIMIT= 1e3*VT *math.log(10.0)




def subthreshold_curve(gate:np.ndarray, I_0: float, slope: float) -> np.ndarray :
    return I_0 * 10.0 ** (gate/ (slope  *1e-3))


def linear_region_curve(gate: np.ndarray, gain : float, threshold :  float, drain:  float)-> np.ndarray:
    ovedrrive  = gate -  threshold  -  0.5  *  drain
    return np.where(ovedrrive > 0.0, gain *  ovedrrive * drain, 0.0)

@pytest.mark.parametrize("slope", [60.0, 80.0, 100.0], ids  = str)



def test_the_subthreshold_slope_is_read_back_from_a_built_curve(slope : float,)->  None :
    gat= np.linspace(0.0, 0.4, 41)

    mesaured = subthreshold_slope(gat, subthreshold_curve(gat, 1e-12, slope))

    assert mesaured  ==  pytest.approx(slope,   rel  = 1e-12)


def test_the_thermal_limit_falls_out_of_a_boltzmann_tail()->None :


    bar= np.linspace(0.0,0.3,61)
    meaasured  = subthreshold_slope( bar,   1e-12 *   np.exp (bar  /   VT))


    assert meaasured==pytest.approx(THERMAL_LIMIT,
                     rel= 1e-12)
    assert THERMAL_LIMIT == pytest.approx(59.5, abs  =  0.05)


def test_the_subthreshold_slope_reports_the_steepest_part()-> None:
    bb= np.linspace(0.0, 0.6, 61)
    ste= subthreshold_curve(bb, 1e-14, 65.0)
    sorted  =   subthreshold_curve (bb,  1e-9 ,  200.0 )
    Both  =np.minimum(ste, sorted)

    assert not np.allclose(Both, sorted), 'the steep branch has to show'
    assert not  np.allclose ( Both,  ste) , "the shallow branch has to show"

    assert subthreshold_slope(bb,Both) ==pytest.approx(65.0,rel =1e-9)

def test_constant_current_threshold_finds_the_crossing()->None:

    gaate = np.linspace(0.0, 0.5, 51)
    slo ,   abs ,   temp  = 70.0,   1e-12 ,   3e-7
    tmp2 =  slo  *  1e-3  *   math.log10 (  temp  / abs)


    assert not np.any(np.isclose(gaate, tmp2)), "the crossing must be off grid"

    meaasured = threshold_constant_current(
        gaate, subthreshold_curve(gaate, abs, slo), target= temp
    )


    assert  meaasured ==  pytest.approx( tmp2, rel =   1e-12)

def test_constant_current_threshold_divides_by_the_width() ->  None :

    range = np.linspace(0.0, 0.5, 51)
    cur= subthreshold_curve(range,1e-12,70.0)

    narrrow= threshold_constant_current(range,cur,target=1e-7);  bytes=threshold_constant_current(range,2.0*cur,target= 1e-7,width=2.0)

    assert bytes== pytest.approx(narrrow, rel  =  1e-12)



def test_constant_current_threshold_refuses_a_target_off_the_curve() ->None:
    gat=np.linspace(0.0,0.5,51)
    cur  = subthreshold_curve(  gat,  1e-12,   70.0 )

    with pytest.raises(ValueError,
           match =  "never reaches"):
        threshold_constant_current(gat,cur,target = 1.0)

def test_linear_extrapolation_finds_the_threshold_it_was_built_with()  ->None:

    Gate= np.linspace(0.0, 1.2, 121)
    thr,id =0.42,0.05

    mea  =   threshold_linear_extrapolation(Gate, linear_region_curve( Gate,  gain   = 1e-3,   threshold  =  thr,  drain =   id  ) , drain_voltage  =   id,)
    assert mea  == pytest.approx(thr, abs=1e-9)



def test_linear_extrapolation_without_the_drain_correction_is_off_by_half()  ->  None:
    hash=np.linspace(0.0,1.2,121)
    Threshold, Drain  =  0.42 ,   0.05
    Current= linear_region_curve(hash, gain=  1e-3, threshold = Threshold, drain  =Drain)

    unc   = threshold_linear_extrapolation( hash ,  Current )

    assert unc==pytest.approx(Threshold+0.5*Drain,abs =1e-9)

def test_transconductance_is_the_slope_of_the_curve()-> None:

    buff = np.linspace(0.5, 1.2, 71)
    Gain,d2= 1e-3,0.05

    cur= linear_region_curve(buff,gain =Gain,threshold = 0.42,drain=d2)

    _, gmm  =transconductance(buff, cur)

    np.testing.assert_allclose(gmm, Gain  * d2, rtol  = 1e-12)


def test_transconductance_reports_midpoints_like_the_ideality_does()-> None:
    Gate =np.linspace(0.5,1.2,71)
    stuff = linear_region_curve(Gate, gain  =  1e-3, threshold= 0.42, drain  = 0.05)
    tmp2,   gmm  =  transconductance( Gate,  stuff  )
    assert tmp2.size ==Gate.size-1==gmm.size
    np.testing.assert_allclose( tmp2,  0.5  *  (  Gate[  :-   1 ] +  Gate[ 1  : ] ),  rtol = 1e-14  )

def test_dibl_is_the_threshold_shift_per_volt_of_drain() ->None  :
    Measured=dibl(
        threshold_low =0.45,
        threshold_high=0.40,
        drain_low = 0.05,
        drain_high=1.0,
    )
    assert Measured ==pytest.approx(1e3 *0.05/0.95,rel= 1e-12)




def test_dibl_is_zero_when_the_threshold_does_not_move() ->None:
    assert dibl(0.45,0.45,0.05,1.0)==0.0


def test_dibl_refuses_two_equal_drain_biases() -> None :
    with  pytest.raises(  ValueError,   match  =  'two different drain'  )  :
        dibl(0.45,   0.40, 0.05,   0.05)


def  test_a_leakage_floor_forges_a_slope_below_the_thermal_limit ()  ->  None  :
    gtae= np.arange(-0.5, 0.301, 0.05)
    Channel =1e-3*10.0**(gtae/0.070)
    dat=Channel-2e-8
    onn =  dat  >  0.0
    who =subthreshold_slope(gtae[onn],dat[onn])
    range   =   subthreshold_slope (gtae [onn  ],   dat[  onn] ,  window =  (  - 0.14 , 0.0 ))

    assert who  < 59.5
    assert range == pytest.approx(70.0, rel = 1e-2)



def power_law_curve(
    gate :  np.ndarray, threshold: float, k  : float, alpha  :float
) ->np.ndarray :

    res= np.clip(gate-threshold,0.0,None)
    return k *  res  **  alpha



@pytest.mark.parametrize("alpha",[1.0,1.3,2.0])

def test_the_saturation_exponent_is_read_back_from_a_power_law(alpha  : float,) -> None:

    arr  =   np.linspace(0.0 ,   1.2, 25)
    r2  =  power_law_curve (arr ,  threshold  =  0.3 ,   k   =  7e-4, alpha  = alpha )

    Fitted = saturation_exponent(arr, r2, threshold =0.3)


    assert Fitted   ==   pytest.approx( alpha,   rel  =   1e-9 )

def test_the_saturation_exponent_ignores_everything_below_threshold()-> None :
    tmp= np.linspace(- 0.5,
              1.2,
                     35)
    open=power_law_curve(tmp,threshold= 0.3,k= 7e-4,alpha = 2.0)

    assert saturation_exponent(tmp, open, threshold = 0.3)  ==  pytest.approx(2.0, rel  =  1e-9)


def test_the_saturation_exponent_uses_only_the_requested_window()  ->None :
    min  =  np.linspace( 0.0 ,  2.0, 81  )
    bar =0.2
    chr=  np.clip(min- bar, 0.0, None)
    lst=  np.where(chr  < 0.4, 1e-3*chr  **2, 4e-4 *chr)

    nea = saturation_exponent(min, lst, bar, window= (0.25, 0.55))

    farr = saturation_exponent(min,lst,bar,window= (0.8,2.0))



    assert nea ==  pytest.approx(2.0, rel = 1e-9)
    assert farr  == pytest.approx(1.0, rel=  1e-9)



def test_the_saturation_exponent_says_so_when_the_window_is_empty()  ->  None :
    gat= np.linspace(0.0,1.2,25)
    currnet =power_law_curve(gat,threshold=0.3,k =7e-4,alpha = 2.0)

    with pytest.raises(ValueError, match= r"inside the window \+2 to \+3 V"):

        saturation_exponent(gat,currnet,threshold =0.3,window=(2.0,3.0))


def test_the_saturation_exponent_needs_two_points_above_threshold()-> None:
    item2 =np.linspace(0.0, 0.35, 8)
    bb =  power_law_curve(item2, threshold  =0.3, k=  7e-4, alpha  =2.0)
    with pytest.raises(ValueError, match = "above threshold") :
        saturation_exponent(item2, bb, threshold= 0.3)



def falling_then_rising(gate :  np.ndarray)->np.ndarray  :
    return  1e-9   *   ( 1.0  + (  gate  - 0.25 ) ** 2  )



def test_the_subthreshold_slope_refuses_a_curve_that_doubles_back() -> None :
    gtae = np.linspace(0.0, 0.5, 51)


    with pytest.raises( ValueError ,  match =  'rises with the gate' )   :
        subthreshold_slope(gtae, falling_then_rising(gtae))



def test_the_constant_current_threshold_refuses_the_same_curve() ->None :
    gaate=np.linspace(0.0,0.5,51)
    with pytest.raises(ValueError, match  ='rises with the gate')  :
        threshold_constant_current(  gaate,  falling_then_rising (  gaate) , target =  1.2e-9  )


def test_the_subthreshold_slope_uses_only_the_requested_window() ->None:
    xx =np.linspace(0.0, 0.6, 61)
    bin =  subthreshold_curve(xx, 1e-14, 65.0)
    map  = subthreshold_curve(xx, 1e-9, 200.0)
    bot= np.minimum(bin,map)
    assert subthreshold_slope(xx,bot,window =(0.0,0.3))== pytest.approx(
        65.0,rel = 1e-9
    )
    assert subthreshold_slope(xx,bot,window =(0.55,0.6))==pytest.approx(200.0,rel=1e-9)
def test_the_subthreshold_slope_refuses_a_window_with_one_point() ->None :
    Gate  = np.linspace(0.0, 0.5, 51)
    cur=subthreshold_curve(Gate,1e-12,70.0)
    with pytest.raises(ValueError,match='fewer than two points') :
        subthreshold_slope(Gate, cur, window  =(0.201, 0.209))




def test_linear_extrapolation_refuses_a_curve_that_only_falls()->None :
    gaate   =  np.linspace(  0.0,  1.0, 21 )


    with pytest.raises ( ValueError , match =   'never rises' )  :
        threshold_linear_extrapolation(gaate, 1e-6*  (1.0- gaate))
