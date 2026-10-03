from __future__ import annotations
from collections.abc import Callable
import numpy as np, pytest
from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors
from ddsim.physics.recombination import (
    AugerRecombination,
    NoRecombination ,
    RecombinationModel,
    SRHRecombination ,
    SumOfRecombination,
    scharfetter_lifetime,
    srh_electron_linearization,
    srh_hole_linearization ,
    srh_rate,
)

TAU_N= 2.0


TAU_P  =  3.0



DENSITY_PAIRS  =[(1e-4, 1e4), (1.0, 1.0), (1e6, 1e-6), (1e6, 1e2), (1e8, 1e8)]



def  as_float (  value  : object  ) ->  float  :
    return  float( np.asarray( value )  )




@pytest.mark.parametrize("n",[1e-6,1e-3,1.0,1e3,1e6,1e10])

def test_rate_is_exactly_zero_at_equilibrium(n  : float) -> None :
    p =   1.0   /  n
    assert srh_rate(n, p, TAU_N, TAU_P) ==  0.0


def test_rate_is_positive_above_equilibrium()->None:
    assert srh_rate(10.0,1.0,TAU_N,TAU_P)>0.0


def test_rate_is_negative_below_equilibrium() -> None :

    assert srh_rate(0.1, 0.1, TAU_N, TAU_P)  <0.0

def test_rate_is_symmetric_under_swapping_the_carriers ( )   ->  None   :


    g =srh_rate(1e4,1e-2,TAU_N,TAU_P)
    res   =   srh_rate(  1e-2 , 1e4 ,   TAU_P,   TAU_N )
    assert g ==res


def test_low_injection_limit_in_n_type( )  ->  None  :
    n= 1e8
    p =   1e-8 +  1e-4
    np.testing.assert_allclose(srh_rate(n, p, TAU_N, TAU_P), 1e-4 / TAU_P, rtol =  1e-6)


def test_low_injection_limit_in_p_type(  )   ->   None  :
    p = 1e8
    n   =  1e-8 +   1e-4
    np.testing.assert_allclose(srh_rate (  n,   p,   TAU_N ,  TAU_P), 1e-4   / TAU_N,  rtol =  1e-6)



def test_intrinsic_material_at_equilibrium_has_no_net_rate (  )  ->   None  :
    assert srh_rate(1.0,1.0,TAU_N,TAU_P)== 0.0



def test_rate_works_on_arrays()-> None:
    n = np.array([1e-3, 1.0, 1e3])
    p  =  1.0/n
    np.testing.assert_array_equal(srh_rate(n, p, TAU_N, TAU_P), np.zeros(3))


def  test_lifetimes_may_vary_per_node()  ->   None :
    n = np.full(3, 1e8)
    p = np.full(3, 1e-8+ 1e-4);f  = np.array ( [ 1.0,   2.0, 4.0 ]  )
    np.testing.assert_allclose(srh_rate(n,p,TAU_N,f),1e-4/f,rtol=1e-6)




def test_scaled_and_physical_routes_agree()-> None :
    a=ScaleFactors.for_silicon()
    res2= C.n_i()

    thing ,   b  =   1e16 , 1e6
    r, m = 1e-5, 3e-6
    v=srh_rate(
        thing,
        b,
        r,
        m,
        ni2=res2 * res2,
        n1 =res2,
        p1= res2,
    )
    val2=srh_rate(thing / a.C_0, b  /a.C_0, r  /  a.t_0, m / a.t_0, ni2 = (res2/ a.C_0)  **2, n1  =  res2 / a.C_0, p1= res2 / a.C_0,)
    np.testing.assert_allclose(val2*a.R_0,v,rtol=1e-12)



def test_rate_against_a_hand_computed_value() -> None :
    m = srh_rate(1e16, 1e12, 1e-5, 3e-6, ni2 = 1e20, n1  = 1e10, p1=  1e10)

    j=1e16 * 1e12- 1e20
    tmp3 =3e-6*(1e16 + 1e10)+1e-5*(1e12+1e10)
    np.testing.assert_allclose(m,j /tmp3,rtol=1e-14)
    np.testing.assert_allclose( m, 3.3322e17,  rtol   =  1e-4)



def complex_step(function : Callable[[complex], complex], x : float)-> float:
    a=1e-30
    return float(np.imag(function(complex(x,
       a)))/a)




@pytest.mark.parametrize ( (  "n", "p"  ),  DENSITY_PAIRS)




def test_dR_dn_against_complex_step(n:float,p: float)-> None :
    r = SRHRecombination(tau_n=  TAU_N,
              tau_p = TAU_P)
    t =complex_step(lambda z:srh_rate(z,p,TAU_N,TAU_P),n)
    np.testing.assert_allclose(as_float(r.d_rate_dn(n, p)), t, rtol=  1e-12)



@pytest.mark.parametrize(('n','p'),DENSITY_PAIRS)



def test_dR_dp_against_complex_step(n :  float, p :  float)-> None  :
    f = SRHRecombination(tau_n = TAU_N, tau_p= TAU_P)
    ok = complex_step(lambda z:srh_rate(n,
                   z,
             TAU_N,
                   TAU_P),
           p)
    np.testing.assert_allclose(as_float(f.d_rate_dp(n,p)),ok,rtol =1e-12)


@pytest.mark.parametrize(  ("n",  "p" ),   DENSITY_PAIRS)

def test_both_derivatives_are_positive(n  : float,   p  :  float )  -> None :
    r= SRHRecombination(tau_n= TAU_N,tau_p=TAU_P)
    assert as_float(r.d_rate_dn(n,p))> 0.0
    assert as_float(  r.d_rate_dp(  n,   p)  ) >  0.0

@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)



def  test_electron_linearization_reproduces_the_rate( n  :   float, p  :  float  )   ->  None :
    val,a=srh_electron_linearization(n,p,TAU_N,TAU_P)
    np.testing.assert_allclose(val*n-a,srh_rate(n,p,TAU_N,TAU_P),rtol=1e-13)




@pytest.mark.parametrize (  ( "n" ,  "p"  ),  DENSITY_PAIRS)


def test_hole_linearization_reproduces_the_rate(n  :float, p : float)->  None :
    m2, s=srh_hole_linearization(n, p, TAU_N, TAU_P)

    np.testing.assert_allclose( m2   * p  - s ,   srh_rate ( n ,  p, TAU_N, TAU_P  ) ,  rtol =  1e-13 )


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_linearization_coefficients_are_non_negative( n :   float , p  :   float  )  ->  None  :
    for d, lst in(
        srh_electron_linearization(n, p, TAU_N, TAU_P),
        srh_hole_linearization(n, p, TAU_N, TAU_P),
    )  :
        assert as_float(d) >=0.0
        assert as_float(lst)>=0.0


def test_linearization_slope_is_not_the_exact_derivative()->None:
    n, p=1e6, 1e2
    t2=SRHRecombination(tau_n= TAU_N, tau_p= TAU_P)
    i, _ = srh_electron_linearization(n, p, TAU_N, TAU_P)

    assert as_float(i) != as_float(t2.d_rate_dn(n, p))


def test_lifetime_is_tau_max_in_undoped_material()-> None :
    assert scharfetter_lifetime(0.0,tau_max=1e-5) ==1e-5


def test_lifetime_is_the_midpoint_at_the_reference_doping() -> None :
    j  =  scharfetter_lifetime(
        C.N_REF_SRH,   tau_max  = 1e-5, tau_min  =  1e-7,  N_ref = C.N_REF_SRH
    )
    np.testing.assert_allclose( j,
                      0.5   * (1e-5  +  1e-7 ),
              rtol  =  1e-14  )



def  test_lifetime_approaches_tau_min_at_high_doping() ->  None  :
    np.testing.assert_allclose(
        scharfetter_lifetime(1e24, tau_max= 1e-5, tau_min = 1e-7), 1e-7, rtol =  1e-5
    )

def test_lifetime_decreases_with_doping()  -> None :
    lst= np.array([0.0, 1e14, 1e16, 1e18, 1e20])
    assert np.all(np.diff(scharfetter_lifetime(lst,tau_max =1e-5)) < 0.0)



def test_lifetime_gamma_sharpens_the_transition() -> None :
    m= scharfetter_lifetime(1e17,tau_max= 1e-5,gamma=1.0)
    ret= scharfetter_lifetime(1e17, tau_max  = 1e-5, gamma  = 2.0)

    assert ret  <m

def test_lifetime_at_1e16_matches_the_documented_defaults()  ->  None:

    np.testing.assert_allclose(
        scharfetter_lifetime(1e16,tau_max= C.TAU_N_MAX),1e-5 / 1.2,rtol = 1e-14
    )



def test_negative_doping_raises() -> None  :
    with pytest.raises (  ValueError ,   match  =   'total'  ) :
        scharfetter_lifetime(-1e16,tau_max=1e-5)


def  test_srh_model_matches_the_free_function( )  ->  None :
    d=SRHRecombination(tau_n= TAU_N,tau_p=TAU_P)
    n, p= 1e6, 1e2

    np.testing.assert_allclose(np.asarray(d.rate(n, p)), srh_rate(n, p, TAU_N, TAU_P), rtol=1e-14)

def test_srh_model_carries_its_own_intrinsic_density()->None :

    arr =SRHRecombination(tau_n= TAU_N, tau_p= TAU_P, ni2= 4.0, n1  = 2.0, p1  = 2.0)
    assert  as_float(arr.rate(  2.0 ,   2.0 ) )  ==   0.0


def test_srh_model_linearizations_match_the_free_functions()  ->  None:

    x2 = SRHRecombination(tau_n= TAU_N, tau_p =TAU_P)
    n, p = 1e6, 1e2

    h,out=x2.electron_linearization(n,p)
    x,j=srh_electron_linearization(n,p,TAU_N,TAU_P)
    np.testing.assert_allclose(np.asarray(h), x, rtol= 1e-14)
    np.testing.assert_allclose(np.asarray(out), j, rtol =1e-14)

    h, out=x2.hole_linearization(n, p)
    x, j  = srh_hole_linearization(n, p, TAU_N, TAU_P)
    np.testing.assert_allclose(np.asarray(h), x, rtol  =  1e-14)
    np.testing.assert_allclose(np.asarray(out), j, rtol =1e-14)




def test_no_recombination_returns_zeros()->None:
    dat=NoRecombination()
    n = np.array([1e6, 1.0, 1e-6])
    p=np.array([1e-6,1.0,1e6])
    c,y= dat.electron_linearization(n,p)
    f,  i =  dat.hole_linearization( n ,   p)
    for u in(
        dat.rate(n,p),
        dat.d_rate_dn(n,p),
        dat.d_rate_dp(n,p),
        c,
        y,
        f,
        i,
    ):
        np.testing.assert_array_equal(np.asarray(u),np.zeros(3))




def test_no_recombination_broadcasts_to_the_input_shape()-> None  :
    x  =   NoRecombination ( )
    assert np.asarray(x.rate(np.zeros(5), np.zeros(5))).shape  == (5, )




def test_a_lifetime_floor_above_the_ceiling_raises()->None:
    with pytest.raises(  ValueError ,   match =   "tau_max")  :

        scharfetter_lifetime(1e16,tau_max=1e-7,tau_min=1e-5)


def test_a_non_positive_reference_doping_raises() ->None:
    with pytest.raises(ValueError, match  ="N_ref"):
        scharfetter_lifetime(1e16, tau_max = 1e-5, N_ref =  0.0)

AUGER_NI2=1.0



def auger():

    return AugerRecombination(C_n =C.AUGER_C_N,C_p = C.AUGER_C_P)

def  test_auger_vanishes_at_equilibrium (  )  ->  None   :
    val  =  auger(  )
    for n in(1e-6,1.0,1e3,1e8):
        assert val.rate(n,AUGER_NI2 /n) == 0.0



def test_auger_recombines_above_equilibrium_and_generates_below()->None:
    k =auger()



    assert k.rate( 1e6,  1e6 ) >  0.0

    assert k.rate(1e-3,1e-3)<0.0



def test_auger_is_cubic_in_the_carrier_density()->None :

    ss =auger()
    t = ss.rate(1e6, 1e6)
    d  =  ss.rate (  1e7,   1e7 )


    assert d/t== pytest.approx(1000.0,rel= 1e-3)

def test_auger_electron_and_hole_channels_are_separately_visible() ->None:
    g=AugerRecombination(C_n=C.AUGER_C_N,C_p = 0.0)
    y= AugerRecombination(C_n =  0.0, C_p  = C.AUGER_C_P)

    assert g.rate(1e6,1e2)>y.rate(1e6,1e2)

@pytest.mark.parametrize ( "n,p", [(  1e6,   1e2 ) ,  (  1e2,   1e6  ) ,  (  1e3 ,   1e3 ),   (  1e-2,  1e-2 )  ])

def test_auger_derivatives_match_complex_step(n:float,p:float)->None:

    a   =  auger( )
    f  =   1e-20
    foo =(a.rate(complex(n,f),p)).imag /f
    k =(a.rate(n, complex(p, f))).imag/ f


    assert  a.d_rate_dn(n ,   p  )  ==   pytest.approx (foo ,  rel  = 1e-12  )
    assert a.d_rate_dp(n, p)==pytest.approx(k, rel =1e-12)


def test_the_auger_electron_tangent_goes_negative_in_depletion() -> None:
    d2= auger()
    assert d2.d_rate_dn(1e-8, 1e-8) < 0.0
    assert d2.d_rate_dn(1e6,1e6)> 0.0



@pytest.mark.parametrize('n,p',[(1e6,1e2),(1e-3,1e-3),(1e3,1e3)])


def test_the_auger_linearization_keeps_both_coefficients_non_negative(n :  float, p :float)-> None:
    stuff = auger()
    for s, j in(stuff.electron_linearization(n, p), stuff.hole_linearization(n, p)) :
        assert s>=0.0


        assert j>=0.0


@pytest.mark.parametrize("n,p",[(1e6,1e2),(1e-3,1e-3),(1e3,1e3)])
def test_the_auger_linearization_is_exact_at_the_current_state(
    n  :  float, p :float
)->None  :
    stuff = auger()


    v, res2  =  stuff.electron_linearization(n,
                   p)
    yy ,   t  =  stuff.hole_linearization (  n ,  p )
    assert v  *  n  -   res2 ==   pytest.approx( stuff.rate (  n, p ), rel  =   1e-12,   abs = 1e-30)
    assert yy *  p - t== pytest.approx(stuff.rate(n, p), rel =  1e-12, abs  =1e-30)




def test_a_sum_of_models_adds_their_rates()->None:
    u  =  SRHRecombination(  tau_n   =  1e3 , tau_p  =   1e3 )
    h=SumOfRecombination((u,auger()))


    n, p = 1e5, 1e4
    assert h.rate(n, p)==pytest.approx(u.rate(n, p)+ auger().rate(n, p), rel=  1e-14)

def test_a_sum_of_models_adds_their_derivatives()->None:


    r=SRHRecombination(tau_n= 1e3,tau_p =1e3)
    t  =SumOfRecombination((r, auger()))


    n, p =  1e5, 1e4
    assert t.d_rate_dn(n, p)== pytest.approx(
        r.d_rate_dn(n, p)  +  auger().d_rate_dn(n, p), rel  = 1e-14
    )
    assert t.d_rate_dp(n, p)  ==pytest.approx(r.d_rate_dp(n, p)  +  auger().d_rate_dp(n, p), rel  =  1e-14)


def test_a_sum_of_models_adds_their_linearizations() -> None  :
    u= SRHRecombination(tau_n =1e3,tau_p=1e3)

    v= SumOfRecombination((u,auger()))
    n,p =1e5,1e4
    ss, y= v.electron_linearization(n, p)
    f, a=u.electron_linearization(n, p)
    mm,r= auger().electron_linearization(n,p)



    assert ss ==pytest.approx(f  +  mm, rel =1e-14)
    assert y == pytest.approx(a + r, rel  = 1e-14)

    assert ss*n- y== pytest.approx(v.rate(n,p),rel= 1e-12)

def test_an_empty_sum_is_no_recombination()->  None:
    jj= SumOfRecombination(())
    assert jj.rate(np.full(4, 1e5), np.full(4, 1e4)).tolist() ==[0.0]*4
    assert jj.d_rate_dn (np.full(  4, 1e5 ) ,   np.full(  4, 1e4)).tolist(  ) == [ 0.0 ]   *   4



def test_a_sum_of_one_model_is_that_model()-> None:
    t= SRHRecombination(tau_n =1e3, tau_p = 1e3)
    f   = SumOfRecombination( (  t, ) )
    assert f.rate(1e5, 1e4) ==  pytest.approx(t.rate(1e5, 1e4), rel =  1e-15)


def test_a_summed_model_satisfies_the_protocol()   ->  None  :
    assert isinstance(SumOfRecombination((auger(),)),RecombinationModel)
    assert isinstance(auger(),RecombinationModel)



def test_a_sum_of_models_adds_their_hole_linearizations() -> None  :

    s= SRHRecombination(tau_n =  1e3, tau_p = 1e3)


    c  = SumOfRecombination(  (s,   auger())  )
    n, p= 1e5, 1e4
    i,   w2  =  c.hole_linearization(  n,   p  )
    arr,w=s.hole_linearization(n,p);  f, res= auger().hole_linearization(n, p)

    assert i==pytest.approx(arr+f,rel= 1e-14)
    assert w2==pytest.approx(w+ res,
          rel =1e-14)
    assert i* p  -w2 == pytest.approx(c.rate(n, p), rel  =1e-12)
