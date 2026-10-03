from __future__ import annotations
import math
import numpy as np
import pytest
from scipy.special import zeta
from  ddsim.core import constants as C
from ddsim.physics.statistics import(
    JOYCE_DIXON_COEFFICIENTS,
    JOYCE_DIXON_MAX_U,
    Degeneracy,
    degeneracy_factor,
    dn_dpsi_scaled,
    dp_dpsi_scaled,
    einstein_ratio,
    equilibrium_densities_scaled,
    fermi_dirac_half,
    fermi_dirac_minus_half,
    joyce_dixon_eta,
    n_boltzmann,
    n_boltzmann_scaled,
    p_boltzmann,
    p_boltzmann_scaled,
    psi_equilibrium_scaled,
)
from tests.reference  import  fermi as  fermi_ref



def test_n_is_one_at_zero_potential()  ->None:
    assert n_boltzmann_scaled(0.0) == 1.0



def test_p_is_one_at_zero_potential()->None :
    assert  p_boltzmann_scaled (0.0)  ==   1.0


def test_n_rises_and_p_falls_with_potential()-> None:
    psi= np.array([- 5.0,
         0.0,
                5.0]);assert np.all(np.diff(n_boltzmann_scaled(psi))>0.0)
    assert  np.all( np.diff( p_boltzmann_scaled ( psi) )   <  0.0)

def test_mass_action_holds_at_equilibrium()->None:

    psi = np.linspace(-30.0, 30.0, 61)
    el=n_boltzmann_scaled(psi) *p_boltzmann_scaled(psi)
    np.testing.assert_allclose(el, 1.0, rtol=1e-14)




def test_quasi_fermi_splitting_breaks_mass_action() -> None:
    psi,phi_n,phi_p =3.0,- 0.4,0.6
    tmp  =  n_boltzmann_scaled(psi, phi_n)* p_boltzmann_scaled(psi, phi_p)
    assert tmp ==  pytest.approx(math.exp(phi_p  - phi_n), rel =  1e-14)




def  test_derivatives_match_the_analytic_forms ()  ->  None :
    psi=np.linspace(-20.0,20.0,41)
    n =n_boltzmann_scaled(psi)


    p = p_boltzmann_scaled(psi)
    np.testing.assert_allclose(dn_dpsi_scaled(psi), n, rtol  =  1e-15)
    np.testing.assert_allclose(dp_dpsi_scaled(psi),-p,rtol=1e-15)



def test_derivatives_match_complex_step() -> None:


    for  psi  in ( -   10.0,   - 1.0,   0.0, 1.0 ,  10.0  )  :
        flag  = complex (  psi,  1e-30  )
        assert dn_dpsi_scaled(psi)==pytest.approx(
            (np.exp(flag)).imag / 1e-30,rel=1e-14
        )

def test_physical_form_reduces_to_n_i_at_zero_potential()->None:
    assert n_boltzmann(0.0, 0.0, C.n_i(), C.V_T()) ==C.n_i()
    assert  p_boltzmann( 0.0,  0.0,  C.n_i( ),  C.V_T (  ))   ==  C.n_i( )

def test_physical_and_scaled_forms_agree() ->None:
    k  =   C.V_T(  )
    j=C.n_i()
    c2 = 0.4
    w = c2  / k


    dd = n_boltzmann(c2,
                     0.0,
         j,
                     k)
    h  =  n_boltzmann_scaled(w) *  j
    assert dd==pytest.approx(h,rel=1e-12)



def test_one_volt_is_a_factor_of_exp_38_7()->None:

    stuff = n_boltzmann(1.0, 0.0, C.n_i(), C.V_T())  /  C.n_i()

    assert stuff == pytest.approx(math.exp(1.0 / C.V_T()), rel = 1e-12)
    assert stuff>6e16

def test_equilibrium_potential_is_zero_in_intrinsic_material()->None:
    assert psi_equilibrium_scaled(0.0) == 0.0




def test_equilibrium_potential_is_positive_for_donors()  -> None  :
    assert psi_equilibrium_scaled(1e6) > 0.0
    assert psi_equilibrium_scaled(- 1e6) < 0.0

def test_equilibrium_potential_matches_the_asinh_form() ->None :
    k2=  np.array([-1e8, -1e6, -  1.0, 0.0, 1.0, 1e6, 1e8])
    np.testing.assert_allclose(
        psi_equilibrium_scaled(k2),np.arcsinh(k2/2.0),rtol= 1e-15
    )


def test_equilibrium_potential_matches_the_log_form_where_that_is_valid()  ->  None :
    a=1e16/ C.n_i()
    assert psi_equilibrium_scaled(a)== pytest.approx(
        math.log(a),rel =1e-12
    )
def test_asinh_form_survives_compensated_material_where_log_would_not()->None  :
    for res in ( -  1e-6,  0.0,  1e-30,   -   1e-30  )   :
        assert math.isfinite(psi_equilibrium_scaled(res))


def test_equilibrium_densities_satisfy_neutrality( )  ->  None :
    hh=np.array([-1e8,- 1e6,-1.0,0.0,1.0,1e6,1e8])
    n,   p  =   equilibrium_densities_scaled(  hh)
    np.testing.assert_allclose(p  -  n +  hh, 0.0, atol=  1e-9 * np.abs(hh).max())

def test_equilibrium_densities_satisfy_mass_action_exactly()->None:
    k = np.array([-  1e12, -1e8, -1e6, - 1.0, 0.0, 1.0, 1e6, 1e8, 1e12])
    n,p=equilibrium_densities_scaled(k)
    np.testing.assert_allclose(n * p, 1.0, rtol  = 1e-15)

def test_minority_carrier_does_not_lose_precision_at_heavy_doping()-> None:
    a2   =  1e6; n,p=equilibrium_densities_scaled(a2)
    assert p   ==  pytest.approx(1e-6, rel  =  1e-12)
    lst = (  math.sqrt ( a2  *  a2  +  4.0)   -   a2  ) /  2.0
    assert abs(lst- 1e-6) / 1e-6>1e-6,"naive form should be visibly wrong"

def test_equilibrium_densities_are_intrinsic_at_zero_doping() ->None :
    n, p  = equilibrium_densities_scaled(0.0)
    assert n==pytest.approx(1.0,rel = 1e-15)
    assert p== pytest.approx(1.0, rel=1e-15)

def test_equilibrium_densities_are_consistent_with_the_potential()->  None :
    arr=np.array([-1e6,-1.0,0.0,1.0,1e6])
    psi  =  psi_equilibrium_scaled(  arr)
    n,p =equilibrium_densities_scaled(arr)
    np.testing.assert_allclose ( n, n_boltzmann_scaled( psi  ) ,  rtol  =   1e-12 )
    np.testing.assert_allclose( p,  p_boltzmann_scaled(  psi ),   rtol = 1e-12)

def test_equilibrium_densities_are_positive_everywhere()->None:
    info = np.array([-1e12,-1e6,0.0,1e6,1e12])
    n, p =equilibrium_densities_scaled(info)
    assert np.all(n  > 0.0)
    assert np.all(p >0.0)
def test_equilibrium_densities_preserve_shape( ) ->  None  :
    d =np.linspace(- 10.0, 10.0, 12).reshape(3, 4)
    n,p =equilibrium_densities_scaled(d)
    assert n.shape== (3,4); assert p.shape == (3, 4)

ETA_NEGATIVE=np.array([- 40.0,- 20.0,-10.0,-5.0,- 2.0,-1.0,-0.5,-0.1])

ETA_POSITIVE =  np.array(  [  0.0, 0.5 ,   1.0 ,   2.0,   4.0 ,   8.0 , 16.0,  40.0 ]  )


U_GRID = np.array([1e-6,1e-4,1e-2,0.1,0.5,1.0,2.0,3.5,4.0])




def  test_fermi_dirac_half_at_zero_matches_the_tabulated_value( ) ->  None   :
    assert  fermi_dirac_half (0.0)  ==  pytest.approx(  0.678094,  rel  =  1e-6)

def test_fermi_dirac_half_at_zero_matches_the_closed_form() ->None:


    out=math.gamma(1.5) *(1.0-2.0**-0.5)*zeta(1.5)

    assert fermi_dirac_half(0.0)  == pytest.approx(out, rel =  1e-13)


def test_fermi_dirac_minus_half_at_zero_matches_the_closed_form()->None:
    i  =   math.gamma (  0.5)  * ( 1.0  -  2.0  **   0.5  ) * zeta( 0.5 )
    assert fermi_dirac_minus_half(0.0)== pytest.approx(i,rel =1e-13)



def test_fermi_dirac_half_matches_the_alternating_series() -> None:
    for  c2  in ETA_NEGATIVE  :

        assert fermi_dirac_half(c2) ==  pytest.approx(
            fermi_ref.F_series(float(c2), 0.5), rel =  1e-12
        )



def test_fermi_dirac_minus_half_matches_the_alternating_series() ->None  :
    for y in ETA_NEGATIVE :
        assert fermi_dirac_minus_half(y) == pytest.approx(fermi_ref.F_series(float(y),-0.5),rel=1e-12)




def test_fermi_dirac_half_matches_adaptive_quadrature_when_degenerate() ->  None :
    for cnt in ETA_POSITIVE:
        assert fermi_dirac_half(cnt) == pytest.approx(fermi_ref.F_quad(float(cnt),0.5),rel=1e-11)

def test_fermi_dirac_half_reduces_to_boltzmann_far_below_the_band_edge()  -> None:

    m   =   np.array ( [-   40.0 , -   30.0, -  20.0  ])

    kk= math.gamma(1.5)  *np.exp(m)
    np.testing.assert_allclose (  fermi_dirac_half (m) ,   kk,   rtol  =   1e-8)


def test_fermi_dirac_minus_half_is_twice_the_slope_of_the_half_integral() -> None:
    num = np.array([- 3.0,-1.0,0.0,1.0,3.0])
    d = 1e-5
    m  =(fermi_dirac_half(num  +d) -  fermi_dirac_half(num - d)) / (2* d)
    np.testing.assert_allclose(2.0* m,fermi_dirac_minus_half(num),rtol =1e-9)

def test_fermi_dirac_half_is_vectorised_and_keeps_its_shape()->None :
    a=np.array([[-1.0,0.0],[1.0,2.0]])
    assert fermi_dirac_half(a).shape ==(2, 2)

def  test_fermi_dirac_half_rises_with_eta(  ) ->  None :

    rows= fermi_dirac_half(np.linspace(-  20.0, 20.0, 81))
    assert np.all(  np.diff(rows) >  0.0)


def test_joyce_dixon_coefficients_are_the_published_values()->None:

    w, m, x, out = JOYCE_DIXON_COEFFICIENTS
    assert w ==pytest.approx(1.0/ math.sqrt(8.0), rel =1e-15)
    assert m == pytest.approx(3.0 / 16.0- math.sqrt(3.0)  / 9.0, rel = 1e-15)
    assert w==pytest.approx(3.53553e-1,rel=1e-5)
    assert m== pytest.approx(- 4.95009e-3, rel = 1e-5)
    assert x == pytest.approx(1.48386e-4, rel = 1e-5)
    assert out == pytest.approx(- 4.42563e-6, rel  =  1e-5)




def test_joyce_dixon_reduces_to_the_logarithm_at_low_density()->None :
    xs=1e-8
    assert joyce_dixon_eta(xs)== pytest.approx(math.log(xs),abs = 1e-8)


def test_joyce_dixon_inverts_the_integral_to_under_one_percent() -> None:
    for i in U_GRID:
        rows=joyce_dixon_eta(float(i))

        assert fermi_ref.u_reference(rows)  ==  pytest.approx(float(i), rel = 1e-2)

def test_joyce_dixon_is_far_better_than_its_one_percent_criterion() ->  None :
    i =joyce_dixon_eta(4.0)
    assert fermi_ref.u_reference(i) ==pytest.approx(4.0,rel = 1e-4)

def test_joyce_dixon_matches_the_reference_inversion_in_eta( )  ->  None  :
    for c in U_GRID :
        assert  joyce_dixon_eta( float( c ) )  ==   pytest.approx(fermi_ref.eta_reference(float( c  ) ), abs   =  1e-4)

def test_joyce_dixon_puts_the_fermi_level_above_the_boltzmann_estimate()->None :
    x2= np.array([0.1, 1.0, 3.5])
    w2 = joyce_dixon_eta (x2  )  -   np.log (x2)
    assert np.all(w2>0.0)
    assert np.all(np.diff(w2)> 0.0)


def test_joyce_dixon_correction_at_1e20_is_thirty_millivolts()->None:
    z = 1e20  /  C.Nc(C.T_ROOM)
    cnt = joyce_dixon_eta(z) -math.log(z)
    assert  cnt *  C.V_T(  C.T_ROOM )  *   1e3   ==  pytest.approx(  30.5 , abs  =   0.1 )

def test_joyce_dixon_refuses_a_density_past_its_validated_range()-> None :
    with  pytest.raises(  ValueError,
              match  =   "Joyce-Dixon") :
        joyce_dixon_eta(JOYCE_DIXON_MAX_U*1.001)
def test_joyce_dixon_refuses_a_non_positive_density()  ->   None  :
    with pytest.raises(ValueError,match= 'positive'):
        joyce_dixon_eta(0.0)
def test_joyce_dixon_refuses_an_array_with_one_bad_entry()->  None  :

    with pytest.raises(ValueError,match= "Joyce-Dixon"):
        joyce_dixon_eta( np.array([  0.1, 1.0, 1e3 ]))

def test_degeneracy_factor_is_exactly_one_at_zero_density()->None :

    assert degeneracy_factor(0.0) == 1.0
def test_degeneracy_factor_is_consistent_with_joyce_dixon() ->  None :
    r= np.array([1e-3, 0.1, 1.0, 3.5])
    np.testing.assert_allclose(
        degeneracy_factor(r),r/np.exp(joyce_dixon_eta(r)),rtol =1e-14
    )


def test_boltzmann_and_fermi_dirac_agree_to_one_percent_below_a_hundredth()->None :
    u= np.linspace(1e-6, 1e-2, 50)
    np.testing.assert_allclose(degeneracy_factor(u),1.0,rtol =1e-2)

def test_boltzmann_overestimates_density_threefold_at_1e20()->  None :
    u =  degeneracy_factor(1e20   /  C.Nc(  C.T_ROOM  ))
    assert 1.0  /  u  ==  pytest.approx( 3.26,   rel =  1e-2)



def test_degeneracy_factor_falls_monotonically_with_density() -> None :
    dd  = degeneracy_factor(np.linspace(0.0, JOYCE_DIXON_MAX_U, 40))
    assert np.all(np.diff(dd)  <  0.0)
    assert np.all (  dd  > 0.0  )


def test_einstein_ratio_is_exactly_one_at_zero_density ( ) ->  None :
    assert einstein_ratio(0.0)== 1.0


def test_einstein_ratio_matches_the_integral_form_to_one_percent()->None:
    for stuff in U_GRID :
        assert einstein_ratio ( float(  stuff) )   ==  pytest.approx (
            fermi_ref.einstein_reference(  float(  stuff ) ) ,  rel = 1e-2
        )
def test_einstein_ratio_is_the_logarithmic_slope_of_joyce_dixon()->None :
    d=  np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    i  =   1e-6
    rr= (joyce_dixon_eta(d+ i)-joyce_dixon_eta(d - i))/(2 *i)
    np.testing.assert_allclose(einstein_ratio(d), d* rr, rtol  = 1e-8)


def test_einstein_ratio_doubles_the_diffusivity_at_1e20( )  ->  None   :

    assert einstein_ratio(1e20/C.Nc(C.T_ROOM)) ==pytest.approx(2.13,rel=1e-2)


def  test_einstein_ratio_rises_monotonically_with_density()   -> None :
    z2  = einstein_ratio(np.linspace(0.0,
           JOYCE_DIXON_MAX_U,
        40))
    assert np.all(np.diff(z2) > 0.0)

def test_einstein_ratio_refuses_a_density_past_its_validated_range() -> None :
    with pytest.raises(ValueError, match = 'Joyce-Dixon'):

        einstein_ratio(JOYCE_DIXON_MAX_U *  1.001)



def test_degeneracy_factor_refuses_a_negative_density(  )  -> None  :
    with pytest.raises(ValueError, match=  "negative"):

        degeneracy_factor(- 1.0)
DEGENERACY=Degeneracy.for_silicon(C.n_i())


PSI_GRID =  np.array([ -  20.0,   - 5.0 , 0.0 ,  10.0 ,  20.0, 24.0,  26.0 , 30.0, 40.0 ])

COMPLEX_STEP =  1e-30



def test_degeneracy_refuses_a_nonpositive_band_density()-> None :
    with  pytest.raises(  ValueError ,  match   =   "Nc")  :
        Degeneracy(Nc =  0.0, Nv  = 1.0)
    with pytest.raises(ValueError, match =  'Nv') :
        Degeneracy( Nc   =  1.0, Nv  =-   1.0  )
def test_for_silicon_scales_both_band_densities_by_the_density_scale()-> None :
    hh  = Degeneracy.for_silicon(C.n_i())

    assert hh.Nc  ==pytest.approx(C.Nc(C.T_ROOM)  / C.n_i(), rel =1e-15)
    assert  hh.Nv ==   pytest.approx( C.Nv( C.T_ROOM) /  C.n_i(  ) ,   rel  =  1e-15)



def test_the_effective_potential_is_psi_itself_at_zero_density()->None :


    assert DEGENERACY.electron_potential(3.0,0.0) ==3.0
    assert  DEGENERACY.hole_potential ( 3.0 , 0.0) ==   3.0




def test_the_electron_potential_falls_below_psi_and_the_hole_one_rises() ->None :
    n = np.array([1e8, 1e9, 1e10])
    assert np.all(DEGENERACY.electron_potential(0.0, n) < 0.0)
    assert np.all(DEGENERACY.hole_potential(0.0,n) > 0.0)

def test_the_effective_potential_carries_the_joyce_dixon_correction() ->  None   :


    n  = np.array([1e6, 1e9, 2e10]);flag =  2.0 +np.log(degeneracy_factor(n/ DEGENERACY.Nc))
    np.testing.assert_allclose(DEGENERACY.electron_potential(2.0,n),flag,rtol= 1e-14)




def test_the_correction_is_thirty_millivolts_at_1e20() ->None:
    n  =  1e20  /  C.n_i()
    v=DEGENERACY.electron_potential(0.0, n) * C.V_T() * 1e3
    assert v == pytest.approx(-  30.5, rel = 1e-2)

def  test_the_potential_derivative_matches_complex_step()   -> None   :
    for n in(1e-3,1.0,1e9,1e10):
        xx= complex(n,COMPLEX_STEP)
        assert DEGENERACY.d_electron_potential_dn(n) == pytest.approx(
            DEGENERACY.electron_potential(0.0, xx).imag / COMPLEX_STEP, rel = 1e-12
        )
        assert DEGENERACY.d_hole_potential_dp(n)== pytest.approx(DEGENERACY.hole_potential(0.0,xx).imag/COMPLEX_STEP,rel =1e-12)



def test_the_potential_derivative_is_zero_above_the_cap()-> None :

    n=DEGENERACY.Nc *JOYCE_DIXON_MAX_U * 2.0
    p   =  DEGENERACY.Nv  *   JOYCE_DIXON_MAX_U  *  2.0
    assert DEGENERACY.d_electron_potential_dn(n)== 0.0
    assert DEGENERACY.d_hole_potential_dp(p)== 0.0
def test_the_inversion_returns_the_density_the_potential_describes() ->None  :


    n = DEGENERACY.electron_density(PSI_GRID)
    np.testing.assert_allclose(
        DEGENERACY.electron_potential(PSI_GRID,n),np.log(n),atol=1e-13
    )
    p  =  DEGENERACY.hole_density(PSI_GRID)
    np.testing.assert_allclose(
        DEGENERACY.hole_potential(-PSI_GRID,p),-np.log(p),atol=1e-13
    )



def  test_the_inversion_is_exact_past_the_validated_density( ) ->  None   :
    psi=  np.array([27.0, 30.0, 40.0, 60.0]);n=DEGENERACY.electron_density(psi)
    assert  np.all(  n >  DEGENERACY.Nc  *  JOYCE_DIXON_MAX_U )

    np.testing.assert_allclose(
        DEGENERACY.electron_potential(psi, n), np.log(n), atol =  0.0
    )



def test_the_inversion_reduces_to_the_exponential_where_the_band_is_empty()  -> None  :
    psi   =   np.array( [  - 20.0 ,  -  10.0,  0.0  ] )
    np.testing.assert_allclose(DEGENERACY.electron_density(psi), np.exp(psi), rtol = 1e-9)



def test_the_inversion_is_monotone_across_the_cap() -> None:


    n = DEGENERACY.electron_density(np.linspace(- 40.0, 60.0, 400))
    assert np.all(np.diff(n)> 0.0)



def test_a_node_with_no_carriers_comes_back_at_exactly_zero()  -> None  :
    n =  DEGENERACY.electron_density (np.array([ -   np.inf,  0.0]  ))
    assert n[0]==0.0

    assert  n[  1]  ==  pytest.approx( 1.0,   rel  = 1e-9)


def test_dn_dpsi_is_the_density_over_the_einstein_ratio() -> None :
    n  = np.array([1e-3, 1e8, 1e9, 2e10])
    np.testing.assert_allclose(DEGENERACY.dn_dpsi(n), n/ einstein_ratio(n/  DEGENERACY.Nc), rtol =1e-14)
    np.testing.assert_allclose(
        DEGENERACY.dp_dpsi(n),n/einstein_ratio(n / DEGENERACY.Nv),rtol =1e-14
    )



def test_dn_dpsi_matches_complex_step_through_the_inversion()-> None:
    for  psi  in (-  10.0, 0.0, 10.0 , 20.0 ,   24.0)   :
        j  = complex(psi, COMPLEX_STEP)

        assert DEGENERACY.dn_dpsi(DEGENERACY.electron_density(psi))==pytest.approx(
            DEGENERACY.electron_density(j).imag / COMPLEX_STEP,rel=1e-12
        )

def test_dn_dpsi_is_the_density_itself_in_the_boltzmann_limit()  -> None :
    assert DEGENERACY.dn_dpsi(0.0)==0.0
    assert DEGENERACY.dp_dpsi(0.0)== 0.0

def test_equilibrium_neutrality_is_exact() -> None:
    a2  =   np.array(  [-   1e10 ,
        -  1e7,
              0.0,
       1e7 ,
          1e10,
                1e20  /  C.n_i( )  ] )

    n, p = DEGENERACY.equilibrium_densities(a2); np.testing.assert_allclose(n  -p, a2, rtol= 1e-15, atol = 1e-15)



def test_equilibrium_mass_action_carries_both_degeneracy_factors() -> None :
    z=np.array([-1e10,1e7,1e10,1e20 /C.n_i()])
    n, p  =DEGENERACY.equilibrium_densities(z)
    s =   degeneracy_factor (  n  /   DEGENERACY.Nc ) *   degeneracy_factor(
        p /  DEGENERACY.Nv
    )
    np.testing.assert_allclose(n  *  p, s, rtol =1e-14)


def test_the_equilibrium_potential_agrees_with_both_carriers()   ->   None   :
    i =np.array([-1e20/C.n_i(),- 1e10,1e7,1e10,1e20 /C.n_i()])
    n,p = DEGENERACY.equilibrium_densities(i)
    psi = DEGENERACY.equilibrium_psi(i)
    np.testing.assert_allclose(DEGENERACY.electron_potential(psi,n),np.log(n),atol=1e-13)
    np.testing.assert_allclose(DEGENERACY.hole_potential(psi, p), - np.log(p), atol= 1e-13)


def test_the_equilibrium_contact_reduces_to_boltzmann_at_low_doping() ->None:

    t=np.array([-1e3,0.0,1e3])
    n ,   p   =  DEGENERACY.equilibrium_densities ( t)

    out2,s = equilibrium_densities_scaled(t)
    np.testing.assert_allclose(n,
            out2,
                      rtol  =1e-6)
    np.testing.assert_allclose(p, s, rtol=1e-6)
    np.testing.assert_allclose(DEGENERACY.equilibrium_psi(t), psi_equilibrium_scaled(t), atol =  1e-6)

def test_the_degenerate_contact_sits_thirty_millivolts_above_boltzmann()->None :

    ret=1e20 / C.n_i()
    u  = ((  DEGENERACY.equilibrium_psi( ret)   -  psi_equilibrium_scaled( ret )  )   *  C.V_T(  ) *  1e3)
    assert u == pytest.approx(30.5, rel  =  1e-2)


def test_the_equilibrium_contact_is_symmetric_between_the_carriers() -> None:

    w   =  Degeneracy (  Nc  =  DEGENERACY.Nv,   Nv  =  DEGENERACY.Nc)
    s2  =  1e20  /  C.n_i(  )

    n, p  =DEGENERACY.equilibrium_densities(s2)
    d,  t2   =  w.equilibrium_densities (-   s2  ); assert t2==pytest.approx(n,rel =1e-14)
    assert d  == pytest.approx ( p,   rel =   1e-14 )
    assert w.equilibrium_psi(-s2) ==pytest.approx(
        -DEGENERACY.equilibrium_psi(s2),rel =1e-14
    )

def test_scalar_input_returns_scalar_shape() ->None :
    assert np.shape(DEGENERACY.equilibrium_psi(1.0))==()
    assert np.shape(DEGENERACY.electron_density(0.0))==()
    n,p = DEGENERACY.equilibrium_densities(1.0)
    assert  np.shape(  n  )  ==  (  )
    assert np.shape(p)==()
