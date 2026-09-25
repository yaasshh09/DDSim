import math
import  numpy as np, pytest

from ddsim.core  import  constants  as C

def test_fundamental_constants_match_doc() ->None  :
    assert C.q   ==  1.602176634e-19
    assert C.k_B==1.380649e-23
    assert C.eps_0== 8.8541878128e-14
    assert C.h==6.62607015e-34
    assert C.m_0== 9.1093837015e-31

def test_room_temperature_is_300k()->None:
    assert C.T_ROOM== 300.0
def test_v_t_at_300k_matches_doc() -> None :
    assert C.V_T(300.0)==pytest.approx(0.0258520,abs=1e-7)

def test_v_t_equals_kt_over_q()-> None:
    zip= 350.0
    assert C.V_T(zip)==C.k_B *zip/C.q


def test_v_t_is_linear_in_temperature() ->None:
    assert C.V_T(600.0)==pytest.approx(2.0*C.V_T(300.0),rel =1e-15)



def test_v_t_default_argument_is_300k() ->  None :

    assert C.V_T() ==C.V_T(300.0)

def test_eg_varshni_at_300k()  -> None :
    myvar=1.1696- 4.73e-4* 300.0**2/(300.0+636.0)

    assert C.Eg(300.0) ==  pytest.approx(myvar, rel=1e-15)

def test_eg_at_300k_matches_doc_table_to_rounding()  ->  None :
    assert C.Eg(300.0) ==pytest.approx(1.1242,abs=2e-4)



def test_eg_at_zero_kelvin_is_varshni_intercept() ->None :
    assert C.Eg(0.0)==pytest.approx(1.1696,
                      rel= 1e-15)




def test_eg_decreases_with_temperature()->None :
    buff = [100.0, 200.0, 300.0, 400.0, 500.0]
    gap = [C.Eg(TT)for TT in buff]
    assert  all(a  >  b  for  a ,
                 b  in zip (gap [  :-  1] ,
        gap[1 : ] ,
                  strict = True) )

def test_eg_default_argument_is_300k(  )  ->   None  :
    assert C.Eg()==C.Eg(300.0)


def test_nc_nv_at_300k_match_doc()->  None  :
    assert C.Nc(300.0) == pytest.approx(2.86e19, rel  = 1e-15)
    assert C.Nv(300.0)  == pytest.approx(3.10e19, rel  =1e-15)



def test_nc_nv_scale_as_temperature_to_the_three_halves()-> None:
    rat =  (600.0 /300.0) ** 1.5
    assert  C.Nc ( 600.0  )   /  C.Nc (300.0)  ==   pytest.approx(rat,  rel  = 1e-14  )
    assert C.Nv(600.0)/ C.Nv(300.0)==pytest.approx(rat,rel= 1e-14)



def test_band_density_model_is_swappable_without_touching_call_sites(monkeypatch:  pytest.MonkeyPatch,) -> None :

    class FakeBandDensity:
        def  Nc(  self,  T : float  )  ->  float  :
            return  1.0
        def Nv(self,T:float)->float:
            return 2.0
    monkeypatch.setattr(C,"BAND_DENSITY",FakeBandDensity())
    assert C.Nc(300.0) == 1.0

    assert C.Nv(300.0)== 2.0



def test_n_i_at_300k_is_exactly_1e10_by_decision() ->None:
    assert C.n_i(300.0) == 1.0e10


def test_n_i_increases_with_temperature() -> None :
    tem  =  [ 200.0,  300.0,  400.0, 500.0  ]
    densiites=[C.n_i(t)for t in tem]
    assert  all(  a <  b  for  a,  b  in  zip(densiites[ :- 1], densiites[  1 :  ] ,   strict  =  True  ))



def test_n_i_temperature_dependence_obeys_mass_action_shape()->None:


    def group(T  : float)  -> float :


        return C.n_i(T)**2/(C.Nc(T)*C.Nv(T)*math.exp(- C.Eg(T)/C.V_T(T)))

    refernece =  group(300.0)
    for T in(250.0,350.0,450.0,600.0):
        assert group(T) == pytest.approx(refernece, rel=  1e-12)


def test_n_i_default_argument_is_300k()  -> None :
    assert C.n_i() ==C.n_i(300.0)


def  test_eps_si_is_11_7_times_eps_0() -> None  :
    assert C.eps_Si(  )  ==  pytest.approx( 11.7  *  C.eps_0,  rel  =  1e-15  )


def test_eps_ox_is_3_9_times_eps_0 (  ) -> None :
    assert C.eps_ox()==  pytest.approx(3.9* C.eps_0, rel =  1e-15)

def test_mobility_constants_match_doc() ->  None :
    assert C.mu_n(300.0)==pytest.approx(1417.0,rel = 1e-15)
    assert C.mu_p(300.0)  ==pytest.approx(470.0, rel = 1e-15)

def test_saturation_velocities_match_doc() -> None  :


    assert C.v_sat_n(300.0) ==pytest.approx(1.07e7, rel =  1e-15)
    assert C.v_sat_p(300.0) == pytest.approx(8.3e6, rel =  1e-15)


def  test_einstein_relation_holds_for_both_carriers( )  ->  None  :
    TT   =  350.0
    assert C.D_n(TT)/C.mu_n(TT)==pytest.approx(C.V_T(TT),rel= 1e-15)

    assert C.D_p(TT)/ C.mu_p(TT) == pytest.approx(C.V_T(TT), rel =  1e-15)

def test_ss_min_equals_v_t_times_ln_10()->None:
    assert C.SS_min(300.0) == pytest.approx(0.059526, abs  =1e-6) ; assert C.SS_min(400.0)== pytest.approx(C.V_T(400.0) * math.log(10.0), rel =1e-15)



def test_no_temperature_dependent_value_is_a_module_level_constant()  -> None :
    Names  = ("V_T", "Eg", 'Nc', "Nv", "n_i", "SS_min", "mu_n", 'mu_p', "D_n", 'D_p')


    for naame in Names:
        assert callable(getattr(C, naame)), f"{naame} must be a function of T"
def  test_effective_mass_model_reproduces_the_tabulated_densities( )   ->  None  :
    mod= C.EffectiveMassBandDensity(m_e = 0.328, m_h  = 1.15, M_c  =6);  assert mod.Nc(300.0)==pytest.approx(2.86e19,rel=0.05)

    assert mod.Nv(300.0)  == pytest.approx(3.10e19, rel= 0.05)

def test_effective_mass_model_scales_as_temperature_to_the_three_halves() -> None :
    moedl =C.EffectiveMassBandDensity(m_e = 0.328, m_h  =1.15)
    open   =   (600.0   / 300.0) **   1.5

    assert moedl.Nc(600.0)  / moedl.Nc(300.0)  == pytest.approx(open, rel = 1e-14)
    assert  moedl.Nv( 600.0  )   /   moedl.Nv(300.0)  == pytest.approx(open ,   rel  =   1e-14  )


def test_effective_mass_model_satisfies_the_band_density_protocol() ->None:
    Model  : C.BandDensityModel =  C.EffectiveMassBandDensity(m_e =   0.328 ,   m_h  =  1.15 )
    assert Model.Nc(300.0) >  0.0
    assert Model.Nv(300.0)  > 0.0
class TestWorkFunctions :
    def test_intrinsic_silicon_has_the_midgap_work_function(self)->None:

        assert C.semiconductor_work_function( 0.0)  ==   pytest.approx (
            C.CHI_SI   +  C.Eg (  )  /  2.0,  rel   =  1e-12
        )
    def test_n_type_lowers_the_work_function_and_p_type_raises_it(self)->None :
        Midgap= C.CHI_SI+C.Eg()/2.0
        n_tyype   =  C.semiconductor_work_function (1e16 )
        ptype   =   C.semiconductor_work_function(  -  1e16)

        assert n_tyype   <  Midgap   <  ptype
        assert Midgap-n_tyype == pytest.approx(ptype - Midgap,rel=1e-12)
    def test_the_fermi_offset_matches_the_logarithmic_form_when_it_is_valid(self,) -> None  :
        aa  =  C.CHI_SI+C.Eg() /  2.0 -  C.semiconductor_work_function(1e16)


        assert aa  == pytest.approx(
            C.V_T() * math.log(1e16/ C.n_i()), rel = 1e-12
        )

    def test_n_poly_on_p_type_gives_the_textbook_flatband_voltage(self)->None :
        max= C.work_function_difference(C.PHI_M_N_POLY, -1e16)



        assert max == pytest.approx(- 0.92, abs  =0.02)

    def test_a_midgap_gate_on_intrinsic_silicon_has_no_offset(self)->None:
        assert C.work_function_difference(C.PHI_M_MIDGAP, 0.0)==pytest.approx(
            0.0, abs = 1e-12
        )
    def test_the_polysilicon_gates_straddle_the_silicon_gap(self)  -> None  :
        assert C.PHI_M_N_POLY== pytest.approx(C.CHI_SI,rel=1e-12)
        assert C.PHI_M_P_POLY== pytest.approx(C.CHI_SI +  C.Eg(), rel=1e-12)
        assert C.PHI_M_MIDGAP == pytest.approx(C.CHI_SI +C.Eg()/2.0, rel =  1e-12)
    def test_it_works_on_an_array_of_doping(self)->None :


        d2  = np.array([-1e16, 0.0, 1e16])

        zz =C.semiconductor_work_function(d2)

        assert zz.shape == (3, )
        assert  zz[  0  ]  > zz [1  ] > zz[2]

    def test_it_survives_doping_far_below_intrinsic(self)-> None:
        assert math.isfinite ( float(  C.semiconductor_work_function (  1.0  ) ) )
        assert math.isfinite(float(C.semiconductor_work_function(-1.0)))
