from __future__ import annotations
import math
import numpy as np, pytest
from scipy.special import erfc
from ddsim.device.doping import(
    Along,
    Coordinates,
    Erfc,
    Gaussian,
    Layers,
    Mirrored,
    Product,
    Step,
    Uniform,
    Window,
    abrupt_junction,
)

MICRON  =  1e-4



def test_uniform_is_constant_everywhere()-> None :
    i  =  Uniform (1e16)
    a=np.linspace(0.0,MICRON,11) ; np.testing.assert_allclose(i(a),1e16)


def test_uniform_accepts_a_scalar_position()  ->None:
    assert Uniform(1e16) (0.5* MICRON)== 1e16



def test_negative_uniform_represents_acceptors()  ->  None:
    assert Uniform(-1e16) (0.0)== -  1e16


def test_step_takes_the_left_value_before_the_position()->None :
    xs =Step(left=- 1e16,right=1e16,position=0.5 *MICRON); assert xs(0.25  *  MICRON) == -  1e16

def test_step_takes_the_right_value_after_the_position()  ->  None   :
    f= Step(left  =-  1e16, right=1e16, position =  0.5 *MICRON)
    assert f(0.75 *MICRON) ==1e16


def test_step_is_right_continuous_at_the_junction()->  None:
    r=Step(left =-1e16,right= 1e16,position=0.5 *MICRON)
    assert r(0.5 *  MICRON) ==1e16




def test_step_changes_sign_across_the_junction()->None:
    val2 =   Step(  left   =-  1e16 ,   right  =  1e16,   position  =  0.5   *  MICRON)
    s  =  np.linspace(0.0,   MICRON, 101  )
    aa=val2(s)
    assert np.any(aa<0.0)
    assert np.any(aa >0.0)




def test_gaussian_peaks_at_its_centre( )   -> None   :
    v= Gaussian(peak= 1e18,centre= 0.3 *MICRON,sigma=0.05 * MICRON)
    assert v(0.3  *MICRON) ==  pytest.approx(1e18, rel= 1e-15)
def  test_gaussian_matches_the_analytic_form(  )  -> None  :
    zz,   yy,  z  =  1e18,   0.3   *  MICRON, 0.05   * MICRON
    u = Gaussian(peak=  zz, centre= yy, sigma = z)
    b= np.linspace(0.0,MICRON,21)
    m  =  zz   *   np.exp (  - (  (b -   yy )   **  2 )  /  (  2.0  *   z ** 2  )  )
    np.testing.assert_allclose (u ( b ),   m,  rtol  =  1e-14)
def test_gaussian_falls_by_one_e_at_one_sigma()  -> None  :
    u = 0.05 *  MICRON
    d  =Gaussian(peak= 1e18, centre= 0.0, sigma = u);  assert d (u  ) == pytest.approx(1e18   *  math.exp(  -  0.5), rel  =  1e-14  )


def  test_gaussian_rejects_non_positive_sigma( )  ->  None  :
    with pytest.raises(ValueError,match= 'sigma'):
        Gaussian(peak=1e18,centre=0.0,sigma=0.0)

def test_erfc_matches_the_analytic_form() ->None :
    out2,i,w= 1e19,0.0,0.02*MICRON
    j=Erfc(peak=out2,position = i,length= w)
    r= np.linspace(0.0, 0.2*  MICRON, 21)
    np.testing.assert_allclose(j(  r  ),  out2   *   erfc((  r   -  i  )  / w  ))

def test_erfc_is_half_the_peak_at_the_position() ->  None :
    ii= Erfc(peak= 1e19,position= 0.0,length=0.02 *MICRON)
    assert ii(0.0) == pytest.approx(1e19, rel =1e-14)


def test_erfc_decays_monotonically() -> None :
    z  = Erfc(peak =  1e19, position =0.0, length  =  0.02 *  MICRON); dat =  np.linspace( 0.0,  0.2   * MICRON, 51 )
    assert np.all(np.diff (z (dat ))   <  0.0 )




def test_erfc_rejects_non_positive_length() -> None :
    with pytest.raises(ValueError,match ='length') :

        Erfc(peak   =   1e19,  position  =  0.0,   length   = 0.0  )

def test_profiles_compose_by_addition()-> None:
    i = Uniform(- 1e16)
    bb =  Gaussian(peak  = 1e18, centre=  0.0, sigma =  0.05* MICRON)
    m2  =i +  bb
    zz= np.array([0.0,0.5*MICRON])
    np.testing.assert_allclose(m2(zz), i(zz)  +  bb(zz), rtol =  1e-15)


def test_composition_is_associative() ->None :
    r,nxt,y = Uniform(1e15),Uniform(2e15),Uniform(3e15)
    val   = np.array( [0.0,  MICRON  ]  )
    np.testing.assert_allclose(((r+ nxt) +  y) (val), (r + (nxt + y)) (val), rtol =1e-15)


def  test_three_way_composition_sums_all_terms(  )  ->   None   :
    d  = Uniform(1e15) + Uniform(2e15) +Uniform(3e15)
    assert  d (0.0  ) ==  pytest.approx(6e15,   rel   =  1e-15  )
def test_profiles_negate()-> None  :
    dat = -Uniform(1e16)
    assert dat(0.0)==pytest.approx(-1e16,rel= 1e-15)

def test_profiles_subtract()->None:
    it  = Uniform (1e16  )   - Uniform (  4e15 )
    assert  it(0.0) ==  pytest.approx(6e15 ,   rel =   1e-15  )




def test_composed_profile_compensates_to_zero_where_terms_cancel()->None:
    k2 =  Uniform(1e16) +Uniform(-  1e16)
    assert k2(0.0) ==0.0
def  test_adding_a_non_profile_raises ( ) ->   None  :
    with  pytest.raises(  TypeError  ) :
        Uniform(1e16) + 5.0
def test_a_profile_gives_the_same_values_on_any_mesh()  ->None :


    foo=  Uniform(- 1e16)+ Gaussian(peak = 1e18, centre = 0.5 *MICRON, sigma =  1e-6)
    thing = np.linspace(0.0, MICRON, 11)
    a  = np.linspace(0.0, MICRON, 1001)
    m = np.intersect1d(thing, a)

    np.testing.assert_allclose(foo(m),
             foo(m),
         rtol =0.0)
    for z in m :
        assert foo(z)==pytest.approx(float(foo(np.array([z])) [0]),rel = 1e-15)

def test_abrupt_junction_is_p_type_on_the_left()-> None:
    tmp = abrupt_junction(Na= 1e16, Nd = 1e16, position =  0.5 * MICRON)

    assert tmp(0.25  * MICRON) ==  pytest.approx(-  1e16, rel  =  1e-15)




def test_abrupt_junction_is_n_type_on_the_right()-> None:
    aa  =abrupt_junction(Na = 1e16, Nd= 1e16, position =0.5 *MICRON)
    assert aa(0.75*MICRON)==pytest.approx(1e16,rel =1e-15)




def test_abrupt_junction_takes_magnitudes_not_signed_values() -> None:

    s  =  abrupt_junction(Na= 2e16, Nd = 5e17, position =  0.5  * MICRON)
    assert s(0.0)== pytest.approx(-2e16,rel=1e-15)
    assert s(MICRON)== pytest.approx(5e17, rel =  1e-15)
def test_abrupt_junction_rejects_negative_concentrations()->None :
    with pytest.raises(ValueError, match ='Na') :
        abrupt_junction(Na  =-  1e16, Nd = 1e16, position= 0.5 *  MICRON)


def  test_abrupt_junction_rejects_a_negative_donor_concentration(  )   ->  None :
    with pytest.raises(ValueError, match =  'Nd') :
        abrupt_junction(Na  = 1e16, Nd =-1e16, position =0.5 *  MICRON)

def test_a_bare_position_is_still_the_x_axis()->None :

    t = np.linspace(0.0,MICRON,5)

    cur   =   Coordinates.of(t  )

    np.testing.assert_array_equal(cur.x,t)
    assert cur.y is None


def test_coordinates_pass_through_unchanged() ->None:
    tmp=Coordinates(np.array([0.0, MICRON]), np.array([0.0, 0.0]))
    assert  Coordinates.of (tmp )  is  tmp




def test_a_scalar_position_still_works() -> None   :
    assert Coordinates.of(0.5 *MICRON).x==pytest.approx(0.5* MICRON)




def test_coordinates_refuse_axes_of_different_lengths() -> None:

    with pytest.raises(ValueError, match = 'same number of positions') :

        Coordinates(np.zeros(5), np.zeros(4))


def test_asking_for_depth_on_a_line_says_what_is_missing()  ->None :
    d = Coordinates.of(np.linspace(0.0,
          MICRON,
               5))

    with pytest.raises(ValueError,match ="no y coordinate") :
        d.axis("y")


def test_coordinates_refuse_an_axis_that_is_not_x_or_y()-> None:
    a2=Coordinates.of(np.linspace(0.0,MICRON,5))

    with pytest.raises (  ValueError ,  match   =  "x or y"  ) :
        a2.axis('z')
def test_along_y_reads_the_depth_coordinate() ->None :

    mm  =  np.linspace ( 0.0 ,   MICRON,   7 )
    cur = Coordinates(np.zeros_like(mm), mm)
    t  =  Gaussian(  peak  =  1e19,
             centre  =  0.0,
                   sigma =  0.05  *   MICRON)

    np.testing.assert_array_equal ( Along ( t , "y" )  (  cur ),  t(  mm  ) )



def test_along_x_is_what_a_profile_already_did()->None:

    jj=Coordinates(np.linspace(0.0,MICRON,7),np.linspace(0.0,MICRON,7))
    hh  =   Gaussian(peak   =   1e19, centre =   0.5  *  MICRON, sigma   =  0.05  *  MICRON)
    np.testing.assert_array_equal (  Along( hh , 'x'  ) (jj ) ,  hh( jj ))


def test_along_y_on_a_line_is_refused ()   ->   None  :
    t = Along(Gaussian(peak=  1e19, centre  =  0.0, sigma = 1e-6), 'y')



    with pytest.raises(ValueError, match = "no y coordinate")  :
        t(np.linspace(0.0,MICRON,5))

def test_along_refuses_an_axis_it_does_not_have()->None :
    tt = Coordinates( np.zeros( 3) ,  np.zeros ( 3 )  )
    with pytest.raises(ValueError, match = "x or y")  :
        Along(Uniform(1e16),'z') (tt)



def test_a_product_is_separable()->None :

    w2=np.array([0.0,0.5*MICRON,MICRON,1.5*MICRON])
    s=np.array([0.0,0.1 * MICRON,0.2*MICRON,0.3 * MICRON]);  info  =  Coordinates(w2 ,  s )



    vals  =   Step(  left  =   1.0,   right  = 0.0,   position   =  MICRON  )
    d =  Gaussian(peak  = 1e20, centre = 0.0, sigma=0.05 * MICRON)
    u  =  Along (  vals , "x"  )   *  Along(d ,   "y" )

    np.testing.assert_allclose(u(info),vals(w2)*d(s),rtol=0.0)



def test_a_product_multiplies_every_factor()  -> None  :
    y2 =  Coordinates.of(np.zeros(3))
    rows=Uniform(2.0) * Uniform(3.0)* Uniform(5.0)

    assert isinstance(rows, Product)
    np.testing.assert_allclose(rows(y2), 30.0, rtol =0.0)



def test_a_lateral_bound_is_two_steps_multiplied()->None:
    w  = np.linspace(0.0, 2.0 * MICRON, 21)
    m = Step(left = 0.0, right  = 1.0, position=  0.5 *  MICRON) *Step(left= 1.0, right =  0.0, position= 1.5 * MICRON)
    v2 = m(w)

    assert np.all(v2[w <0.5 * MICRON] ==0.0) ; assert np.all(v2[(w  >=  0.5 *  MICRON) &(w <1.5 * MICRON)] == 1.0)
    assert np.all(v2[w >=  1.5* MICRON]  ==  0.0)


def  test_multiplying_by_a_number_scales_the_profile() ->  None  :
    c = Gaussian(peak=1.0, centre  =  0.0, sigma =0.05 * MICRON);r2=np.linspace(0.0,MICRON,11)

    np.testing.assert_allclose((c  * 1e20)  (  r2 ),  1e20 *  c(r2  ) ,   rtol  =  1e-15)
    np.testing.assert_allclose((1e20 * c)(r2),1e20 *c(r2),rtol = 1e-15)


def test_multiplying_by_something_that_is_neither_raises() -> None :
    with pytest.raises(TypeError,match ="DopingProfile or a number") :
        Uniform( 1e16  )  *   'half'

def  test_mirroring_reflects_about_a_position ()   ->  None :
    s = Erfc(peak= 0.5,position=0.4*MICRON,length= 0.05* MICRON)

    a  = np.linspace(0.0, 2.0 * MICRON, 21)

    h = MICRON

    np.testing.assert_array_equal(Mirrored(s,about= h) (a),s(2.0*h- a))
def test_mirroring_twice_is_the_original() -> None:
    d  =   Erfc(peak   =   0.5 , position = 0.4  * MICRON, length  =   0.05   * MICRON  )
    d2=np.linspace(0.0,2.0 *MICRON,21)



    out2 =Mirrored(Mirrored(d,about =MICRON),about= MICRON)
    np.testing.assert_allclose( out2 (d2),   d (  d2  ), rtol  =  1e-13 )




def test_mirroring_leaves_the_depth_alone()-> None :
    m =   np.linspace(  0.0,   2.0  *  MICRON,  5  )
    x2 =np.linspace(0.0,0.2 *MICRON,5)
    zz  =   Coordinates(  m,  x2  )
    g =  Erfc(peak= 0.5, position= 0.4 *  MICRON, length = 0.05 *  MICRON)
    k = Gaussian(peak = 1e20, centre=  0.0, sigma =0.05 * MICRON)
    h   =  Along (  g , 'x' )   *   Along(  k ,  "y"  )
    np.testing.assert_array_equal(
        Mirrored(h,about= MICRON)(zz),
        g(2.0* MICRON- m)*k(x2),
    )

def test_mirroring_a_bare_position_reflects_it()   ->   None   :
    a= np.linspace(0.0, MICRON, 5)

    c =Erfc(peak  =  1.0, position =  0.3  * MICRON, length = 0.1* MICRON)
    np.testing.assert_array_equal(
        Along(Mirrored(c, about =0.5 * MICRON), "y")(
            Coordinates(np.zeros_like(a), a)
        ),
        c(MICRON -  a),
    )


def  test_layers_takes_each_region_value_inside_it()   ->   None  :

    k=Layers(boundaries  =  (MICRON, 2.0 * MICRON), values =(- 1e18, 1e14, 1e18))
    e = np.array([0.5, 1.5, 2.5])* MICRON


    np.testing.assert_array_equal(k(e), [-  1e18, 1e14, 1e18])



def test_layers_is_right_continuous_at_every_boundary() -> None :
    e=Layers(boundaries =(MICRON,2.0 *MICRON),values=(- 1.0,2.0,3.0))
    np.testing.assert_array_equal (e(np.array([  MICRON ,   2.0  *  MICRON] )), [ 2.0,  3.0]  )
def test_one_boundary_layers_is_the_abrupt_junction_bit_for_bit()->None:
    info  =   np.linspace(  0.0,  MICRON, 201)
    np.testing.assert_array_equal(Layers( boundaries  =   (  0.5  *  MICRON, ) ,  values = (-  1e16 , 1e16  ) ) (  info  ), abrupt_junction(  Na  =   1e16,  Nd  =  1e16, position =  0.5  *   MICRON)  (  info),)


def test_layers_needs_one_more_value_than_boundaries() ->  None  :
    with pytest.raises(ValueError,match="one more value"):
        Layers(boundaries= (MICRON,),
                values= (1.0,))


def test_layers_needs_increasing_boundaries (  )  ->  None   :
    with pytest.raises(ValueError, match  = 'increasing'):
        Layers(boundaries=(2.0*MICRON,MICRON),values= (1.0,2.0,3.0))


ACROSS =np.linspace(- 0.5  *  MICRON, 1.5 *MICRON, 401)



def test_an_abrupt_window_is_one_inside_and_zero_outside()-> None:
    it=Window(low =0.2 *MICRON,high= 0.6 *MICRON)

    cc  =it(ACROSS);  u=(ACROSS>= 0.2 *MICRON)&(ACROSS<=0.6*MICRON)
    np.testing.assert_array_equal(cc,
       np.where(u,
                      1.0,
                    0.0))

def test_an_abrupt_window_holds_both_of_its_edges() ->None:
    c  = Window ( low   = 0.2  * MICRON,  high  = 0.6 * MICRON  )
    np.testing.assert_array_equal(c(np.array([0.2, 0.6]) * MICRON), [1.0, 1.0])

def test_a_gaussian_window_holds_its_peak_inside_and_falls_off_outside() -> None :
    bar= 0.05 *MICRON;  lst  =Window(low  =0.2  *MICRON, high = 0.6 *  MICRON, edge  = "gaussian", length = bar)
    stuff=lst(ACROSS)
    y=(ACROSS>= 0.2*MICRON)&(ACROSS <= 0.6* MICRON)
    assert np.all(stuff[y]== 1.0)
    f = ACROSS< 0.2 *MICRON


    arr  =  np.exp(-  ((  0.2  *  MICRON  -  ACROSS [  f]  ) **  2)   /  (2.0  * bar  **  2  ))
    np.testing.assert_allclose(stuff[f],arr,rtol=1e-15)
    res2  =ACROSS>0.6*MICRON

    arr=np.exp(-((ACROSS[res2] -0.6 *MICRON)**2) /(2.0*bar**2))

    np.testing.assert_allclose(stuff[res2],arr,rtol=1e-15)

def test_a_gaussian_window_below_an_open_top_is_the_nmos_depth_profile()  ->  None:
    item,row = 1.0*MICRON,0.05*MICRON
    j=  np.linspace(0.0, item, 301)

    vv = Window(low = item, high =math.inf, edge ='gaussian', length= row)
    np.testing.assert_array_equal(
        vv(j  ),  Gaussian(peak  =  1.0,  centre   = item ,   sigma  =  row)  (  j )
    )
def test_an_erfc_window_open_on_the_left_is_the_nmos_lateral_edge() -> None:
    foo =0.046  *  MICRON;  b2  = Window(low=-  math.inf, high  = 0.4  * MICRON, edge  = "erfc", length = foo)
    np.testing.assert_array_equal(
        b2(ACROSS), Erfc(peak= 0.5, position =0.4  *MICRON, length= foo)  (ACROSS)
    )



def test_an_erfc_window_open_on_the_right_is_the_mirrored_edge()  -> None:

    rr,s= 0.046*MICRON,1.8*MICRON
    m   =  Window ( low  =  1.4  * MICRON, high  =  math.inf,  edge  =  'erfc' ,  length   =   rr)
    x =Erfc(peak=0.5,position= 0.4 *MICRON,length=rr)
    np.testing.assert_allclose(
        m(ACROSS),Mirrored(x,0.5*s)(ACROSS),rtol=1e-12,atol =1e-300
    )



def  test_an_erfc_window_is_half_its_peak_at_a_mask_edge(  ) ->   None   :
    m2=0.02* MICRON ; a2 = Window(low=0.2*MICRON,high =1.2*MICRON,edge ='erfc',length= m2)
    np.testing.assert_allclose(a2(np.array([0.2, 1.2]) *  MICRON), [0.5, 0.5], rtol = 1e-12)

    assert a2(np.array([0.7 *MICRON]))[0] ==pytest.approx(1.0,rel =1e-12)



def test_a_narrow_erfc_window_never_reaches_its_peak() ->   None :
    u = Window(
        low   =  0.5  *  MICRON,  high   = 0.51  *   MICRON , edge  =  "erfc", length  =   0.05  * MICRON
    )
    assert u(np.array([0.505 * MICRON])) [0] < 0.2



def test_a_window_open_on_both_sides_is_one_everywhere()-> None :


    j=  Window(low =-math.inf, high=math.inf, edge  = "erfc", length =  0.01 * MICRON)
    np.testing.assert_array_equal(j(ACROSS), 1.0)

def test_a_window_reads_the_axis_it_is_put_along()  ->  None  :
    out2 =   Along ( Window( low  = 0.0,   high = 0.5  * MICRON ), 'y')
    flag =Coordinates(x=np.array([0.0,0.0]),y=np.array([0.2,0.8])* MICRON)
    np.testing.assert_array_equal(out2(flag),[1.0,0.0])

def test_a_window_is_refused_upside_down()->None:
    with pytest.raises(ValueError, match= "low"):
        Window(low  =  0.6  *  MICRON,  high  = 0.2  *  MICRON )


def test_a_soft_window_needs_a_length() ->None :
    with pytest.raises(ValueError,match= 'length'):

        Window(low =  0.2  *  MICRON, high =0.6*  MICRON, edge= "erfc")


def test_a_window_edge_is_one_of_three()  ->None:
    with pytest.raises(ValueError,match= "abrupt, gaussian or erfc"):

        Window(low=  0.2 *  MICRON, high =0.6* MICRON, edge  =  "linear", length = 1e-6)
