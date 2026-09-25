from __future__  import annotations
from collections.abc import Callable

import  numpy as np, pytest
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


    Forward =srh_rate(1e4,1e-2,TAU_N,TAU_P)
    buf   =   srh_rate(  1e-2 , 1e4 ,   TAU_P,   TAU_N )
    assert Forward ==buf


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
    p = np.full(3, 1e-8+ 1e-4);tauu_p  = np.array ( [ 1.0,   2.0, 4.0 ]  )
    np.testing.assert_allclose(srh_rate(n,p,TAU_N,tauu_p),1e-4/tauu_p,rtol=1e-6)




def test_scaled_and_physical_routes_agree()-> None :
    sca=ScaleFactors.for_silicon()
    r2= C.n_i()

    n_pyhs ,   p_pys  =   1e16 , 1e6
    tauNPhys, foo = 1e-5, 3e-6
    t2=srh_rate(
        n_pyhs,
        p_pys,
        tauNPhys,
        foo,
        ni2=r2 * r2,
        n1 =r2,
        p1= r2,
    )
    sccaled=srh_rate(n_pyhs / sca.C_0, p_pys  /sca.C_0, tauNPhys  /  sca.t_0, foo / sca.t_0, ni2 = (r2/ sca.C_0)  **2, n1  =  r2 / sca.C_0, p1= r2 / sca.C_0,)
    np.testing.assert_allclose(sccaled*sca.R_0,t2,rtol=1e-12)



def test_rate_against_a_hand_computed_value() -> None :
    Rate = srh_rate(1e16, 1e12, 1e-5, 3e-6, ni2 = 1e20, n1  = 1e10, p1=  1e10)

    str=1e16 * 1e12- 1e20
    Denominator =3e-6*(1e16 + 1e10)+1e-5*(1e12+1e10)
    np.testing.assert_allclose(Rate,str /Denominator,rtol=1e-14)
    np.testing.assert_allclose( Rate, 3.3322e17,  rtol   =  1e-4)



def complex_step(function : Callable[[complex], complex], x : float)-> float:
    Step=1e-30
    return float(np.imag(function(complex(x,
       Step)))/Step)




@pytest.mark.parametrize ( (  "n", "p"  ),  DENSITY_PAIRS)




def test_dR_dn_against_complex_step(n:float,p: float)-> None :
    xx = SRHRecombination(tau_n=  TAU_N,
              tau_p = TAU_P)
    ref =complex_step(lambda z:srh_rate(z,p,TAU_N,TAU_P),n)
    np.testing.assert_allclose(as_float(xx.d_rate_dn(n, p)), ref, rtol=  1e-12)



@pytest.mark.parametrize(('n','p'),DENSITY_PAIRS)



def test_dR_dp_against_complex_step(n :  float, p :  float)-> None  :
    Model = SRHRecombination(tau_n = TAU_N, tau_p= TAU_P)
    ref = complex_step(lambda z:srh_rate(n,
                   z,
             TAU_N,
                   TAU_P),
           p)
    np.testing.assert_allclose(as_float(Model.d_rate_dp(n,p)),ref,rtol =1e-12)


@pytest.mark.parametrize(  ("n",  "p" ),   DENSITY_PAIRS)

def test_both_derivatives_are_positive(n  : float,   p  :  float )  -> None :
    mdoel= SRHRecombination(tau_n= TAU_N,tau_p=TAU_P)
    assert as_float(mdoel.d_rate_dn(n,p))> 0.0
    assert as_float(  mdoel.d_rate_dp(  n,   p)  ) >  0.0

@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)



def  test_electron_linearization_reproduces_the_rate( n  :   float, p  :  float  )   ->  None :
    stuff,w=srh_electron_linearization(n,p,TAU_N,TAU_P)
    np.testing.assert_allclose(stuff*n-w,srh_rate(n,p,TAU_N,TAU_P),rtol=1e-13)




@pytest.mark.parametrize (  ( "n" ,  "p"  ),  DENSITY_PAIRS)


def test_hole_linearization_reproduces_the_rate(n  :float, p : float)->  None :
    cc, gg=srh_hole_linearization(n, p, TAU_N, TAU_P)

    np.testing.assert_allclose( cc   * p  - gg ,   srh_rate ( n ,  p, TAU_N, TAU_P  ) ,  rtol =  1e-13 )


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_linearization_coefficients_are_non_negative( n :   float , p  :   float  )  ->  None  :
    for cc, G in(
        srh_electron_linearization(n, p, TAU_N, TAU_P),
        srh_hole_linearization(n, p, TAU_N, TAU_P),
    )  :
        assert as_float(cc) >=0.0
        assert as_float(G)>=0.0


def test_linearization_slope_is_not_the_exact_derivative()->None:
    n, p=1e6, 1e2
    range=SRHRecombination(tau_n= TAU_N, tau_p= TAU_P)
    cc, _ = srh_electron_linearization(n, p, TAU_N, TAU_P)

    assert as_float(cc) != as_float(range.d_rate_dn(n, p))


def test_lifetime_is_tau_max_in_undoped_material()-> None :
    assert scharfetter_lifetime(0.0,tau_max=1e-5) ==1e-5


def test_lifetime_is_the_midpoint_at_the_reference_doping() -> None :
    Tau  =  scharfetter_lifetime(
        C.N_REF_SRH,   tau_max  = 1e-5, tau_min  =  1e-7,  N_ref = C.N_REF_SRH
    )
    np.testing.assert_allclose( Tau,
                      0.5   * (1e-5  +  1e-7 ),
              rtol  =  1e-14  )



def  test_lifetime_approaches_tau_min_at_high_doping() ->  None  :
    np.testing.assert_allclose(
        scharfetter_lifetime(1e24, tau_max= 1e-5, tau_min = 1e-7), 1e-7, rtol =  1e-5
    )

def test_lifetime_decreases_with_doping()  -> None :
    hmm= np.array([0.0, 1e14, 1e16, 1e18, 1e20])
    assert np.all(np.diff(scharfetter_lifetime(hmm,tau_max =1e-5)) < 0.0)



def test_lifetime_gamma_sharpens_the_transition() -> None :
    Gentle= scharfetter_lifetime(1e17,tau_max= 1e-5,gamma=1.0)
    Steep= scharfetter_lifetime(1e17, tau_max  = 1e-5, gamma  = 2.0)

    assert Steep  <Gentle

def test_lifetime_at_1e16_matches_the_documented_defaults()  ->  None:

    np.testing.assert_allclose(
        scharfetter_lifetime(1e16,tau_max= C.TAU_N_MAX),1e-5 / 1.2,rtol = 1e-14
    )



def test_negative_doping_raises() -> None  :
    with pytest.raises (  ValueError ,   match  =   'total'  ) :
        scharfetter_lifetime(-1e16,tau_max=1e-5)


def  test_srh_model_matches_the_free_function( )  ->  None :
    Model=SRHRecombination(tau_n= TAU_N,tau_p=TAU_P)
    n, p= 1e6, 1e2

    np.testing.assert_allclose(np.asarray(Model.rate(n, p)), srh_rate(n, p, TAU_N, TAU_P), rtol=1e-14)

def test_srh_model_carries_its_own_intrinsic_density()->None :

    min =SRHRecombination(tau_n= TAU_N, tau_p= TAU_P, ni2= 4.0, n1  = 2.0, p1  = 2.0)
    assert  as_float(min.rate(  2.0 ,   2.0 ) )  ==   0.0


def test_srh_model_linearizations_match_the_free_functions()  ->  None:

    Model = SRHRecombination(tau_n= TAU_N, tau_p =TAU_P)
    n, p = 1e6, 1e2

    w,gmodel=Model.electron_linearization(n,p)
    oct,bb=srh_electron_linearization(n,p,TAU_N,TAU_P)
    np.testing.assert_allclose(np.asarray(w), oct, rtol= 1e-14)
    np.testing.assert_allclose(np.asarray(gmodel), bb, rtol =1e-14)

    w, gmodel=Model.hole_linearization(n, p)
    oct, bb  = srh_hole_linearization(n, p, TAU_N, TAU_P)
    np.testing.assert_allclose(np.asarray(w), oct, rtol  =  1e-14)
    np.testing.assert_allclose(np.asarray(gmodel), bb, rtol =1e-14)




def test_no_recombination_returns_zeros()->None:
    Model=NoRecombination()
    n = np.array([1e6, 1.0, 1e-6])
    p=np.array([1e-6,1.0,1e6])
    ElectronC,eg= Model.electron_linearization(n,p)
    HoleC,  holeG =  Model.hole_linearization( n ,   p)
    for Values in(
        Model.rate(n,p),
        Model.d_rate_dn(n,p),
        Model.d_rate_dp(n,p),
        ElectronC,
        eg,
        HoleC,
        holeG,
    ):
        np.testing.assert_array_equal(np.asarray(Values),np.zeros(3))




def test_no_recombination_broadcasts_to_the_input_shape()-> None  :
    Model  =   NoRecombination ( )
    assert np.asarray(Model.rate(np.zeros(5), np.zeros(5))).shape  == (5, )




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
    mod  =  auger(  )
    for n in(1e-6,1.0,1e3,1e8):
        assert mod.rate(n,AUGER_NI2 /n) == 0.0



def test_auger_recombines_above_equilibrium_and_generates_below()->None:
    Model =auger()



    assert Model.rate( 1e6,  1e6 ) >  0.0

    assert Model.rate(1e-3,1e-3)<0.0



def test_auger_is_cubic_in_the_carrier_density()->None :

    mdel =auger()
    loww = mdel.rate(1e6, 1e6)
    High  =  mdel.rate (  1e7,   1e7 )


    assert High/loww== pytest.approx(1000.0,rel= 1e-3)

def test_auger_electron_and_hole_channels_are_separately_visible() ->None:
    str=AugerRecombination(C_n=C.AUGER_C_N,C_p = 0.0)
    tuple= AugerRecombination(C_n =  0.0, C_p  = C.AUGER_C_P)

    assert str.rate(1e6,1e2)>tuple.rate(1e6,1e2)

@pytest.mark.parametrize ( "n,p", [(  1e6,   1e2 ) ,  (  1e2,   1e6  ) ,  (  1e3 ,   1e3 ),   (  1e-2,  1e-2 )  ])

def test_auger_derivatives_match_complex_step(n:float,p:float)->None:

    obj2   =  auger( )
    junk  =   1e-20
    Dn =(obj2.rate(complex(n,junk),p)).imag /junk
    Dp =(obj2.rate(n, complex(p, junk))).imag/ junk


    assert  obj2.d_rate_dn(n ,   p  )  ==   pytest.approx (Dn ,  rel  = 1e-12  )
    assert obj2.d_rate_dp(n, p)==pytest.approx(Dp, rel =1e-12)


def test_the_auger_electron_tangent_goes_negative_in_depletion() -> None:
    moedl= auger()
    assert moedl.d_rate_dn(1e-8, 1e-8) < 0.0
    assert moedl.d_rate_dn(1e6,1e6)> 0.0



@pytest.mark.parametrize('n,p',[(1e6,1e2),(1e-3,1e-3),(1e3,1e3)])


def test_the_auger_linearization_keeps_both_coefficients_non_negative(n :  float, p :float)-> None:
    Model = auger()
    for cc, gg in(Model.electron_linearization(n, p), Model.hole_linearization(n, p)) :
        assert cc>=0.0


        assert gg>=0.0


@pytest.mark.parametrize("n,p",[(1e6,1e2),(1e-3,1e-3),(1e3,1e3)])
def test_the_auger_linearization_is_exact_at_the_current_state(
    n  :  float, p :float
)->None  :
    len = auger()


    cN, gn  =  len.electron_linearization(n,
                   p)
    vals ,   gP  =  len.hole_linearization (  n ,  p )
    assert cN  *  n  -   gn ==   pytest.approx( len.rate (  n, p ), rel  =   1e-12,   abs = 1e-30)
    assert vals *  p - gP== pytest.approx(len.rate(n, p), rel =  1e-12, abs  =1e-30)




def test_a_sum_of_models_adds_their_rates()->None:
    srhh  =  SRHRecombination(  tau_n   =  1e3 , tau_p  =   1e3 )
    tottal=SumOfRecombination((srhh,auger()))


    n, p = 1e5, 1e4
    assert tottal.rate(n, p)==pytest.approx(srhh.rate(n, p)+ auger().rate(n, p), rel=  1e-14)

def test_a_sum_of_models_adds_their_derivatives()->None:


    srhh=SRHRecombination(tau_n= 1e3,tau_p =1e3)
    Total  =SumOfRecombination((srhh, auger()))


    n, p =  1e5, 1e4
    assert Total.d_rate_dn(n, p)== pytest.approx(
        srhh.d_rate_dn(n, p)  +  auger().d_rate_dn(n, p), rel  = 1e-14
    )
    assert Total.d_rate_dp(n, p)  ==pytest.approx(srhh.d_rate_dp(n, p)  +  auger().d_rate_dp(n, p), rel  =  1e-14)


def test_a_sum_of_models_adds_their_linearizations() -> None  :
    ret= SRHRecombination(tau_n =1e3,tau_p=1e3)

    foo= SumOfRecombination((ret,auger()))
    n,p =1e5,1e4
    w, G= foo.electron_linearization(n, p)
    cSrh, gsrh=ret.electron_linearization(n, p)
    CAug,g_augg= auger().electron_linearization(n,p)



    assert w ==pytest.approx(cSrh  +  CAug, rel =1e-14)
    assert G == pytest.approx(gsrh + g_augg, rel  = 1e-14)

    assert w*n- G== pytest.approx(foo.rate(n,p),rel= 1e-12)

def test_an_empty_sum_is_no_recombination()->  None:
    tot= SumOfRecombination(())
    assert tot.rate(np.full(4, 1e5), np.full(4, 1e4)).tolist() ==[0.0]*4
    assert tot.d_rate_dn (np.full(  4, 1e5 ) ,   np.full(  4, 1e4)).tolist(  ) == [ 0.0 ]   *   4



def test_a_sum_of_one_model_is_that_model()-> None:
    data2= SRHRecombination(tau_n =1e3, tau_p = 1e3)
    tot   = SumOfRecombination( (  data2, ) )
    assert tot.rate(1e5, 1e4) ==  pytest.approx(data2.rate(1e5, 1e4), rel =  1e-15)


def test_a_summed_model_satisfies_the_protocol()   ->  None  :
    assert isinstance(SumOfRecombination((auger(),)),RecombinationModel)
    assert isinstance(auger(),RecombinationModel)



def test_a_sum_of_models_adds_their_hole_linearizations() -> None  :

    srhh= SRHRecombination(tau_n =  1e3, tau_p = 1e3)


    foo  = SumOfRecombination(  (srhh,   auger())  )
    n, p= 1e5, 1e4
    cc,   gg  =  foo.hole_linearization(  n,   p  )
    cs,r2=srhh.hole_linearization(n,p);  junk, GAug= auger().hole_linearization(n, p)

    assert cc==pytest.approx(cs+junk,rel= 1e-14)
    assert gg==pytest.approx(r2+ GAug,
          rel =1e-14)
    assert cc* p  -gg == pytest.approx(foo.rate(n, p), rel  =1e-12)
