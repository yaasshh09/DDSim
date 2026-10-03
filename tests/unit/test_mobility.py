from __future__ import annotations
import numpy as np; import pytest
from  ddsim.core  import  constants  as C
from ddsim.physics.mobility import(
    AroraMobility,
    CaugheyThomas,
    ConstantMobility,
    LombardiSurface,
    edge_diffusivity,
)


@pytest.mark.parametrize(  'model',   [AroraMobility.electrons(  ) ,  AroraMobility.holes (  ) ]  )



def test_the_undoped_limit_is_mu_min_plus_mu_d(model : AroraMobility) ->None :
    assert  float( model(0.0) )  == pytest.approx( model.mu_min  +   model.mu_d ,  rel =  1e-14)
@pytest.mark.parametrize( "model" ,  [AroraMobility.electrons ( ) ,   AroraMobility.holes( )])


def test_at_the_reference_doping_the_lattice_term_is_halved(model  :  AroraMobility ,)  -> None :


    assert float(model(model.N_ref))==  pytest.approx(model.mu_min  + model.mu_d  / 2.0, rel  = 1e-14)

@pytest.mark.parametrize("model", [AroraMobility.electrons(), AroraMobility.holes()])
def test_the_heavily_doped_limit_is_mu_min(model: AroraMobility)->None :
    x  =   float(model(  1e25 ) )

    assert x >model.mu_min ; assert x == pytest.approx(model.mu_min,
                     rel =1e-3)



@pytest.mark.parametrize('model',[AroraMobility.electrons(),AroraMobility.holes()])


def test_mobility_falls_monotonically_with_doping(model : AroraMobility) ->  None  :
    c   =   np.logspace( 10, 21, 200)
    assert np.all(np.diff(model(c)) <0.0)


@pytest.mark.parametrize("doping,expected,tolerance", [(1e16,1230.0,0.02), (1e18,280.0,0.05),],)
def test_electron_mobility_matches_the_literature(doping  : float, expected  : float, tolerance  : float) ->None :
    assert float(AroraMobility.electrons()(doping))==pytest.approx(
        expected,rel=tolerance
    )


def test_holes_are_slower_than_electrons_at_every_doping(  )   -> None  :


    i =  np.logspace(  13,  20 , 60  )
    assert np.all(AroraMobility.holes() (i)  <  AroraMobility.electrons()(i))

def test_the_parameters_reduce_to_their_tabulated_values_at_300_K()-> None :
    k  =  AroraMobility.electrons(T =C.T_ROOM)



    assert k.mu_min ==pytest.approx(88.0, rel =  1e-14)
    assert k.mu_d  ==  pytest.approx(1252.0, rel=1e-14)

    assert k.N_ref==pytest.approx(1.432e17,rel= 1e-14)
    assert k.exponent   ==  pytest.approx (0.88 ,  rel = 1e-14  )


def  test_mobility_falls_as_temperature_rises_in_lightly_doped_silicon()   ->   None   :


    c2= 1e14
    h =float(AroraMobility.electrons(T = 250.0)(c2))
    e = float(AroraMobility.electrons(T= 400.0)(c2))

    assert  e  <  h




def  test_the_constant_model_ignores_the_doping()  ->  None  :


    obj  =   ConstantMobility (C.MU_N_300 )
    np.testing.assert_allclose(obj(np.array([0.0, 1e15, 1e20])), C.MU_N_300, rtol=0.0)



def test_the_constant_model_returns_one_value_per_node() -> None :
    z =  ConstantMobility( 1417.0)  ( np.zeros(7  )  )
    assert z.shape == (7, )

def test_edge_diffusivity_averages_the_two_endpoint_values( ) ->  None  :
    s=np.array([1000.0,500.0,100.0])

    z   =  edge_diffusivity( s, V_T   =  0.02585  )


    np.testing.assert_allclose(z, np.array([750.0, 300.0])* 0.02585, rtol = 1e-14)

def test_edge_diffusivity_applies_the_einstein_relation() ->  None  :

    y =edge_diffusivity(np.full(2, 1417.0), V_T =0.02585)
    assert float(y[0]) ==pytest.approx(1417.0* 0.02585,rel =1e-14)



def test_edge_diffusivity_gives_one_value_per_edge()  ->  None  :
    i =  edge_diffusivity( np.ones(11),   V_T   =  0.02585  )
    assert i.shape == (10, )




def test_edge_diffusivity_gathers_the_endpoints_an_edge_list_names (  )  ->  None :

    j  =   np.array ( [ 100.0 , 200.0, 400.0, 800.0  ]  )
    z=np.array([[0,2],[1,3],[3,0]],dtype = np.int64)

    a   =   edge_diffusivity(  j , V_T  = 2.0,   edge_nodes  =  z )
    np.testing.assert_allclose(a, [500.0, 1000.0, 900.0], rtol =1e-14)



def test_the_edge_list_form_reproduces_the_1d_chain_exactly()-> None :
    x=np.linspace(300.0, 1400.0, 9)
    xx= np.column_stack([np.arange(8),np.arange(8)+ 1]).astype(np.int64)


    np.testing.assert_array_equal (
        edge_diffusivity( x ,  V_T  =  0.02585, edge_nodes   =  xx  ),
        edge_diffusivity(  x,   V_T  =  0.02585  ) ,
    )



def test_the_arora_undoped_limit_does_not_match_the_tabulated_mobility()-> None  :
    kk =AroraMobility.electrons()
    b2 = AroraMobility.holes()

    assert float(kk(0.0)) == pytest.approx(1340.0, rel=1e-12)
    assert float(b2(0.0))==pytest.approx(461.3, rel=1e-12)
    assert  float (  kk (  0.0 ) ) /   C.MU_N_300   ==  pytest.approx(0.9456,   rel  =  1e-3 )
    assert float(b2(0.0))/ C.MU_P_300 == pytest.approx(0.9815,rel= 1e-3)


def edges(value :float, count : int= 5):
    return np.full(count,value)
def test_at_zero_field_the_model_returns_the_low_field_value()->None:
    vv = CaugheyThomas(low_field=edges(1.0), v_sat=  0.5, beta = 2.0)

    d= np.full(5, 0.1)
    np.testing.assert_array_equal(vv(np.zeros(5), d), edges(1.0))




def test_the_drift_velocity_saturates_at_v_sat() ->None  :
    y =  0.5
    a   =  CaugheyThomas(low_field  = edges(  1.0 ),  v_sat   = y,   beta   =  2.0 )
    s = np.full(5, 0.1)


    i= np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    m2 =  a (  i, s )   * np.abs(  i)  /  s

    assert m2[- 1]  ==pytest.approx(y, rel= 1e-6)
    assert  np.all (  np.diff(  m2  )   >   0.0 )


def test_the_drift_velocity_never_exceeds_v_sat() -> None:

    tmp3 =  0.5


    for row in(1.0, 2.0) :
        d2= CaugheyThomas(low_field =edges(1.0),v_sat =tmp3,beta=row)
        flag=np.full(5,0.1)
        a= np.array([0.0,1e-3,1.0,1e3,1e9])

        assert np.all(d2(a,flag)* np.abs(a) /flag<=tmp3)



def test_the_knee_is_where_the_low_field_drift_would_reach_v_sat() -> None  :
    for zz in(1.0, 2.0) :
        b= CaugheyThomas(low_field=edges(1.0),v_sat =0.5,beta =zz)
        rows  = np.full(5,
            0.1)
        foo=np.full(5,0.5 * 0.1)
        np.testing.assert_allclose(b(foo,rows),1.0/2.0**(1.0 /zz),rtol=1e-14)


def test_mobility_falls_monotonically_with_field()->None :
    for  i  in(  1.0 ,
             2.0 )  :

        m =CaugheyThomas(low_field= edges(1.0,60),v_sat=0.5,beta= i)

        dat=  np.full(60, 0.1)
        k =  np.linspace(0.0, 30.0, 60)

        assert np.all(np.diff (  m(k,   dat )  ) < 0.0 )
def test_the_model_does_not_care_which_way_the_field_points()->None :
    a  = CaugheyThomas(low_field = edges(1.0), v_sat=0.5, beta= 2.0)
    tmp3 =  np.full(5, 0.1)
    m2=np.array([0.1,1.0,5.0,20.0,100.0])

    np.testing.assert_array_equal(a(m2, tmp3), a(-  m2, tmp3))


def test_electrons_hold_their_mobility_longer_than_holes_do() -> None :
    dat =  np.full(5, 0.1)
    a  =  np.full(  5, 0.1  *  0.5  *  0.1)

    w= CaugheyThomas(low_field=edges(1.0),v_sat=0.5,beta=2.0)
    i = CaugheyThomas(low_field = edges(1.0), v_sat= 0.5, beta = 1.0)


    assert np.all(w(a,dat)>i(a,dat))
def test_a_longer_edge_across_the_same_drop_is_a_weaker_field() -> None:
    z=CaugheyThomas(low_field=edges(1.0),v_sat= 0.5,beta=2.0)
    h   =  np.full(  5,  1.0)
    assert np.all(z(h, np.full(5, 1.0))  >z(h, np.full(5, 0.1)))


def complex_step_dD_dX(model, X, h, step: float = 1e-30) :
    return np.imag( model(X +   1j  * step,
                  h  )  )  /  step

def test_the_tangent_matches_a_complex_step_for_electrons() -> None:
    d  =   CaugheyThomas (  low_field   =  edges (1.3,
                7  ),
          v_sat  = 0.5,
                  beta   =  2.0)
    thing  = np.linspace (  0.05,   0.4, 7 )

    cnt = np.array([-  30.0,
                - 5.0,
               -0.5,
                0.2,
                      3.0,
            12.0,
          200.0])

    np.testing.assert_allclose(d.derivative(cnt, thing), complex_step_dD_dX(d, cnt, thing), rtol = 1e-12)

def  test_the_tangent_matches_a_complex_step_for_holes (  )   ->   None  :

    w=  CaugheyThomas(low_field= edges(0.4, 7), v_sat = 0.3, beta =  1.0)
    k2   =  np.linspace ( 0.05,  0.4 , 7)
    i = np.array([-30.0, -5.0, -  0.5, 0.2, 3.0, 12.0, 200.0])

    np.testing.assert_allclose (w.derivative ( i , k2), complex_step_dD_dX (  w, i ,   k2), rtol  = 1e-12)

def test_the_tangent_is_negative_wherever_the_field_is_positive() ->None :
    tt =CaugheyThomas(low_field = edges(1.0),v_sat=0.5,beta=2.0)
    b2 =np.full(5,0.1); obj =np.array([0.1,1.0,5.0,20.0,100.0])

    assert np.all(tt.derivative(obj,b2)<0.0)
    assert np.all(tt.derivative(-obj,b2) > 0.0)
def test_the_tangent_is_zero_at_zero_field() -> None  :
    for vals in (1.0 , 2.0  )   :
        ss=CaugheyThomas(low_field =edges(1.0),v_sat=0.5,beta= vals)


        np.testing.assert_array_equal(ss.derivative(np.zeros(5), np.full(5, 0.1)), np.zeros(5))


def test_a_complex_step_at_zero_field_takes_the_right_hand_side()-> None :
    c=CaugheyThomas(low_field=edges(1.0),v_sat=0.5,beta=1.0)
    f=  CaugheyThomas(low_field =  edges(1.0), v_sat =0.5, beta= 2.0)
    x2= np.full(5,0.1)
    dat  =complex_step_dD_dX(c, np.zeros(5), x2)

    assert np.all( dat   < 0.0)
    np.testing.assert_allclose(dat, c.derivative(np.full(5, 1e-8), x2), rtol= 1e-6)
    np.testing.assert_allclose(
        complex_step_dD_dX(f,np.zeros(5),x2),np.zeros(5),atol= 1e-30
    )
def test_a_complex_argument_survives_the_model()->None:

    b  =  CaugheyThomas ( low_field  =   edges(  1.0 ) ,   v_sat  =  0.5, beta =   2.0 )

    tt= b(np.full(5,2.0)+1j*1e-30,np.full(5,0.1))
    assert np.iscomplexobj(tt)
    np.testing.assert_allclose(
        np.real(  tt ),  b(  np.full ( 5,   2.0  ), np.full (5 , 0.1  )),   rtol  =  1e-15
    )



def test_a_zero_saturation_velocity_is_refused (  ) -> None  :
    with pytest.raises(ValueError,match='v_sat') :
        CaugheyThomas(  low_field  =   edges(  1.0 ),  v_sat = 0.0 ,  beta  =  2.0  )

def test_a_non_positive_beta_is_refused()-> None :
    with pytest.raises(ValueError, match="beta") :
        CaugheyThomas(low_field=  edges(1.0), v_sat= 0.5, beta =  0.0)

DEVSIM_ELECTRONS  = {
    'B' : 3.61e7,
    "C_ac" : 1.70e4,
    "tau"  : 0.0233,
    'delta'  : 3.58e18,
    'A' : 2.58,
    "alpha" : 6.85e-21,
    "eta" : 0.0767,
    "kappa" : 1.7,
}
DEVSIM_HOLES={"B": 1.51e7, "C_ac": 4.18e3, 'tau' :0.0119, "delta": 4.10e15, 'A':2.18, "alpha" : 7.82e-21, "eta" :0.123, "kappa":0.9,}



@pytest.mark.parametrize(("build",'table'), [(LombardiSurface.electrons,DEVSIM_ELECTRONS), (LombardiSurface.holes,DEVSIM_HOLES),], ids = ['electrons',"holes"],)



def  test_the_parameters_are_the_devsim_ones( build,  table  )  ->   None  :


    g =build(T = C.T_ROOM)
    for cnt,jj in table.items() :
        assert getattr(g, cnt)  == jj



@pytest.mark.parametrize ('build', [LombardiSurface.electrons , LombardiSurface.holes],  ids   =  [  "n",   'p' ])

def test_a_vanishing_normal_field_gives_back_the_bulk_mobility(build)->None:
    y  =  build()
    vals =np.full(4,800.0)
    g = y(vals, E_perp  =  np.zeros(4), total_doping=np.full(4, 1e17), carriers =np.full(4, 1e10),)


    np.testing.assert_allclose(g,vals,rtol= 0.02)
    assert np.all(g< vals),"scattering can only ever subtract"
def test_the_normal_field_is_floored_rather_than_dividing_by_zero()->None :

    g = LombardiSurface.electrons()
    yy =  np.full(3, 800.0)
    arr,b =np.full(3,1e17),np.full(3,1e10)
    flag=g(yy,np.zeros(3),arr,b);  cur  =  g( yy , np.full( 3,  g.E_floor ) , arr, b)
    r2  = g(yy, np.full(3, 1.0), arr, b)

    assert np.all(np.isfinite(flag))
    np.testing.assert_allclose(flag,cur,rtol=1e-14)
    np.testing.assert_allclose(r2, cur, rtol  =  1e-14)




@pytest.mark.parametrize("build", [LombardiSurface.electrons, LombardiSurface.holes], ids  = ["n", 'p'])


def test_the_three_channels_combine_by_reciprocals ( build) -> None   :

    nxt=build()
    r  = np.full(5, 700.0)
    k2  = np.array( [  1e3 ,   1e4, 1e5 , 3e5,   1e6  ]  ); u,s=np.full(5,3e17),np.full(5,5e17)

    ret =nxt(r,k2,u,s)
    idx= nxt.acoustic(k2, u)

    z  =  nxt.roughness(k2 , u,   s  )

    v2   =  1.0   / (1.0  / r   +  1.0  /   idx +  1.0  /  z )
    np.testing.assert_allclose(ret ,   v2,   rtol   =   1e-14)
    assert np.all(ret <  np.minimum(r, np.minimum(idx, z)))

@pytest.mark.parametrize(
    "build", [LombardiSurface.electrons, LombardiSurface.holes], ids =  ['n', "p"]
)

def test_mobility_falls_as_the_normal_field_rises(  build  )  ->  None :
    c =build()
    g = np.logspace(2.0, 6.5, 60)
    arr   =  c( np.full( 60 ,  800.0 ) ,   g, np.full( 60 ,  1e17 ), np.full(  60, 1e18 ) )

    assert  np.all(  np.diff(  arr) < 0.0 )



def test_the_inversion_layer_is_two_to_three_times_slower_than_bulk() ->  None:
    z  =  1e17
    w2=float(AroraMobility.electrons()(z))
    y  =  LombardiSurface.electrons()  (np.full(1,   w2), E_perp  =  np.full(  1, 5e5 ) , total_doping = np.full( 1,   z) , carriers  =  np.full(1, 1e18),)

    j  =   w2  / float (y[ 0])
    assert 2.0<j <3.0,f"surface mobility is {j:.2f}x below bulk"

def test_holes_stay_slower_than_electrons_at_the_surface()  -> None :

    r,z,g =np.full(4,3e5),np.full(4,1e17),np.full(4,1e18)



    kk = LombardiSurface.electrons()(np.full(4,800.0),r,z,g)
    ii =  LombardiSurface.holes() (np.full(4, 300.0), r, z, g)

    assert np.all(ii<kk)

def test_a_heavier_inversion_layer_roughens_the_surface_it_sees()->None:

    i =  LombardiSurface.electrons()
    t2,bar=np.full(3,5e5),np.full(3,1e17)
    a=i.roughness(t2, bar, carriers = np.full(3, 1e14))

    s2 = i.roughness(t2, bar, carriers =  np.full(3, 1e20))
    assert np.all(i.gamma(bar,np.full(3,1e20))>i.gamma(bar,np.full(3,1e14)))
    assert np.all(s2 <  a)

def test_the_exponent_reduces_to_A_with_no_carriers_present()-> None:


    r =LombardiSurface.holes()
    z= r.gamma(np.full(2,1e17),np.zeros(2))

    np.testing.assert_allclose (z, r.A,  rtol  = 1e-15  )

def test_the_acoustic_term_carries_the_temperature_exponent()-> None:
    num   =  LombardiSurface.electrons(  T = 250.0 )
    k =LombardiSurface.electrons(T =  350.0)


    c,item =np.full(3,3e5),np.full(3,1e17)
    assert np.all(k.acoustic(c, item) < num.acoustic(c, item))


def test_a_negative_normal_field_is_refused()->None:

    with pytest.raises(ValueError, match = 'E_perp') :
        LombardiSurface.electrons (  )   (np.full(3 , 800.0  ), np.array ([1e5,   -   1e5 , 1e5]  ) , np.full(  3, 1e17), np.full (  3,  1e18),)




def test_a_vanished_roughness_mobility_leaves_no_mobility_and_no_warning()-> None :
    tt =LombardiSurface.electrons(); b,dat=np.full(2,1e20),np.full(2,8.8e6); y = np.array([1e20, 1.6e36])
    assert  tt.roughness (  dat ,  b,   y)  [  1]  == 0.0
    a2  = tt(np.full(2, 100.0), dat, b, y)
    assert a2[1]  ==  0.0
    assert a2[0] > 0.0
