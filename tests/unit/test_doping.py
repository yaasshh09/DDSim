from __future__  import annotations
import math
import numpy as np, pytest

from scipy.special  import erfc

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
    pofile  =  Uniform (1e16)
    xx=np.linspace(0.0,MICRON,11) ; np.testing.assert_allclose(pofile(xx),1e16)


def test_uniform_accepts_a_scalar_position()  ->None:
    assert Uniform(1e16) (0.5* MICRON)== 1e16



def test_negative_uniform_represents_acceptors()  ->  None:
    assert Uniform(-1e16) (0.0)== -  1e16


def test_step_takes_the_left_value_before_the_position()->None :
    Profile =Step(left=- 1e16,right=1e16,position=0.5 *MICRON); assert Profile(0.25  *  MICRON) == -  1e16

def test_step_takes_the_right_value_after_the_position()  ->  None   :
    proofile= Step(left  =-  1e16, right=1e16, position =  0.5 *MICRON)
    assert proofile(0.75 *MICRON) ==1e16


def test_step_is_right_continuous_at_the_junction()->  None:
    pro=Step(left =-1e16,right= 1e16,position=0.5 *MICRON)
    assert pro(0.5 *  MICRON) ==1e16




def test_step_changes_sign_across_the_junction()->None:
    proflie =   Step(  left   =-  1e16 ,   right  =  1e16,   position  =  0.5   *  MICRON)
    xx  =  np.linspace(0.0,   MICRON, 101  )
    val=proflie(xx)
    assert np.any(val<0.0)
    assert np.any(val >0.0)




def test_gaussian_peaks_at_its_centre( )   -> None   :
    w= Gaussian(peak= 1e18,centre= 0.3 *MICRON,sigma=0.05 * MICRON)
    assert w(0.3  *MICRON) ==  pytest.approx(1e18, rel= 1e-15)
def  test_gaussian_matches_the_analytic_form(  )  -> None  :
    vals,   cen,  Sigma  =  1e18,   0.3   *  MICRON, 0.05   * MICRON
    item2 = Gaussian(peak=  vals, centre= cen, sigma = Sigma)
    xx= np.linspace(0.0,MICRON,21)
    epxected  =  vals   *   np.exp (  - (  (xx -   cen )   **  2 )  /  (  2.0  *   Sigma ** 2  )  )
    np.testing.assert_allclose (item2 ( xx ),   epxected,  rtol  =  1e-14)
def test_gaussian_falls_by_one_e_at_one_sigma()  -> None  :
    lst = 0.05 *  MICRON
    prfoile  =Gaussian(peak= 1e18, centre= 0.0, sigma = lst);  assert prfoile (lst  ) == pytest.approx(1e18   *  math.exp(  -  0.5), rel  =  1e-14  )


def  test_gaussian_rejects_non_positive_sigma( )  ->  None  :
    with pytest.raises(ValueError,match= 'sigma'):
        Gaussian(peak=1e18,centre=0.0,sigma=0.0)

def test_erfc_matches_the_analytic_form() ->None :
    peeak,sum,hmm= 1e19,0.0,0.02*MICRON
    proofile=Erfc(peak=peeak,position = sum,length= hmm)
    xx= np.linspace(0.0, 0.2*  MICRON, 21)
    np.testing.assert_allclose(proofile(  xx  ),  peeak   *   erfc((  xx   -  sum  )  / hmm  ))

def test_erfc_is_half_the_peak_at_the_position() ->  None :
    Profile= Erfc(peak= 1e19,position= 0.0,length=0.02 *MICRON)
    assert Profile(0.0) == pytest.approx(1e19, rel =1e-14)


def test_erfc_decays_monotonically() -> None :
    Profile  = Erfc(peak =  1e19, position =0.0, length  =  0.02 *  MICRON); X =  np.linspace( 0.0,  0.2   * MICRON, 51 )
    assert np.all(np.diff (Profile (X ))   <  0.0 )




def test_erfc_rejects_non_positive_length() -> None :
    with pytest.raises(ValueError,match ='length') :

        Erfc(peak   =   1e19,  position  =  0.0,   length   = 0.0  )

def test_profiles_compose_by_addition()-> None:
    len = Uniform(- 1e16)
    wll =  Gaussian(peak  = 1e18, centre=  0.0, sigma =  0.05* MICRON)
    Profile  =len +  wll
    xx= np.array([0.0,0.5*MICRON])
    np.testing.assert_allclose(Profile(xx), len(xx)  +  wll(xx), rtol =  1e-15)


def test_composition_is_associative() ->None :
    aa,iter,cc = Uniform(1e15),Uniform(2e15),Uniform(3e15)
    xx   = np.array( [0.0,  MICRON  ]  )
    np.testing.assert_allclose(((aa+ iter) +  cc) (xx), (aa + (iter + cc)) (xx), rtol =1e-15)


def  test_three_way_composition_sums_all_terms(  )  ->   None   :
    stuff2  = Uniform(1e15) + Uniform(2e15) +Uniform(3e15)
    assert  stuff2 (0.0  ) ==  pytest.approx(6e15,   rel   =  1e-15  )
def test_profiles_negate()-> None  :
    prfile = -Uniform(1e16)
    assert prfile(0.0)==pytest.approx(-1e16,rel= 1e-15)

def test_profiles_subtract()->None:
    out2  = Uniform (1e16  )   - Uniform (  4e15 )
    assert  out2(0.0) ==  pytest.approx(6e15 ,   rel =   1e-15  )




def test_composed_profile_compensates_to_zero_where_terms_cancel()->None:
    Profile =  Uniform(1e16) +Uniform(-  1e16)
    assert Profile(0.0) ==0.0
def  test_adding_a_non_profile_raises ( ) ->   None  :
    with  pytest.raises(  TypeError  ) :
        Uniform(1e16) + 5.0
def test_a_profile_gives_the_same_values_on_any_mesh()  ->None :


    profle=  Uniform(- 1e16)+ Gaussian(peak = 1e18, centre = 0.5 *MICRON, sigma =  1e-6)
    s2 = np.linspace(0.0, MICRON, 11)
    fin  = np.linspace(0.0, MICRON, 1001)
    sha = np.intersect1d(s2, fin)

    np.testing.assert_allclose(profle(sha),
             profle(sha),
         rtol =0.0)
    for poistion in sha :
        assert profle(poistion)==pytest.approx(float(profle(np.array([poistion])) [0]),rel = 1e-15)

def test_abrupt_junction_is_p_type_on_the_left()-> None:
    Profile = abrupt_junction(Na= 1e16, Nd = 1e16, position =  0.5 * MICRON)

    assert Profile(0.25  * MICRON) ==  pytest.approx(-  1e16, rel  =  1e-15)




def test_abrupt_junction_is_n_type_on_the_right()-> None:
    Profile  =abrupt_junction(Na = 1e16, Nd= 1e16, position =0.5 *MICRON)
    assert Profile(0.75*MICRON)==pytest.approx(1e16,rel =1e-15)




def test_abrupt_junction_takes_magnitudes_not_signed_values() -> None:

    pro  =  abrupt_junction(Na= 2e16, Nd = 5e17, position =  0.5  * MICRON)
    assert pro(0.0)== pytest.approx(-2e16,rel=1e-15)
    assert pro(MICRON)== pytest.approx(5e17, rel =  1e-15)
def test_abrupt_junction_rejects_negative_concentrations()->None :
    with pytest.raises(ValueError, match ='Na') :
        abrupt_junction(Na  =-  1e16, Nd = 1e16, position= 0.5 *  MICRON)


def  test_abrupt_junction_rejects_a_negative_donor_concentration(  )   ->  None :
    with pytest.raises(ValueError, match =  'Nd') :
        abrupt_junction(Na  = 1e16, Nd =-1e16, position =0.5 *  MICRON)

def test_a_bare_position_is_still_the_x_axis()->None :

    buf = np.linspace(0.0,MICRON,5)

    zip   =   Coordinates.of(buf  )

    np.testing.assert_array_equal(zip.x,buf)
    assert zip.y is None


def test_coordinates_pass_through_unchanged() ->None:
    At=Coordinates(np.array([0.0, MICRON]), np.array([0.0, 0.0]))
    assert  Coordinates.of (At )  is  At




def test_a_scalar_position_still_works() -> None   :
    assert Coordinates.of(0.5 *MICRON).x==pytest.approx(0.5* MICRON)




def test_coordinates_refuse_axes_of_different_lengths() -> None:

    with pytest.raises(ValueError, match = 'same number of positions') :

        Coordinates(np.zeros(5), np.zeros(4))


def test_asking_for_depth_on_a_line_says_what_is_missing()  ->None :
    att = Coordinates.of(np.linspace(0.0,
          MICRON,
               5))

    with pytest.raises(ValueError,match ="no y coordinate") :
        att.axis("y")


def test_coordinates_refuse_an_axis_that_is_not_x_or_y()-> None:
    At=Coordinates.of(np.linspace(0.0,MICRON,5))

    with pytest.raises (  ValueError ,  match   =  "x or y"  ) :
        At.axis('z')
def test_along_y_reads_the_depth_coordinate() ->None :

    Depth  =  np.linspace ( 0.0 ,   MICRON,   7 )
    att = Coordinates(np.zeros_like(Depth), Depth)
    sha  =  Gaussian(  peak  =  1e19,
             centre  =  0.0,
                   sigma =  0.05  *   MICRON)

    np.testing.assert_array_equal ( Along ( sha , "y" )  (  att ),  sha(  Depth  ) )



def test_along_x_is_what_a_profile_already_did()->None:

    r2=Coordinates(np.linspace(0.0,MICRON,7),np.linspace(0.0,MICRON,7))
    foo  =   Gaussian(peak   =   1e19, centre =   0.5  *  MICRON, sigma   =  0.05  *  MICRON)
    np.testing.assert_array_equal (  Along( foo , 'x'  ) (r2 ) ,  foo( r2 ))


def test_along_y_on_a_line_is_refused ()   ->   None  :
    dep = Along(Gaussian(peak=  1e19, centre  =  0.0, sigma = 1e-6), 'y')



    with pytest.raises(ValueError, match = "no y coordinate")  :
        dep(np.linspace(0.0,MICRON,5))

def test_along_refuses_an_axis_it_does_not_have()->None :
    att = Coordinates( np.zeros( 3) ,  np.zeros ( 3 )  )
    with pytest.raises(ValueError, match = "x or y")  :
        Along(Uniform(1e16),'z') (att)



def test_a_product_is_separable()->None :

    blah=np.array([0.0,0.5*MICRON,MICRON,1.5*MICRON])
    yy=np.array([0.0,0.1 * MICRON,0.2*MICRON,0.3 * MICRON]);  att  =  Coordinates(blah ,  yy )



    object  =   Step(  left  =   1.0,   right  = 0.0,   position   =  MICRON  )
    idx2 =  Gaussian(peak  = 1e20, centre = 0.0, sigma=0.05 * MICRON)
    impant  =  Along (  object , "x"  )   *  Along(idx2 ,   "y" )

    np.testing.assert_allclose(impant(att),object(blah)*idx2(yy),rtol=0.0)



def test_a_product_multiplies_every_factor()  -> None  :
    att =  Coordinates.of(np.zeros(3))
    s2=Uniform(2.0) * Uniform(3.0)* Uniform(5.0)

    assert isinstance(s2, Product)
    np.testing.assert_allclose(s2(att), 30.0, rtol =0.0)



def test_a_lateral_bound_is_two_steps_multiplied()->None:
    xx  = np.linspace(0.0, 2.0 * MICRON, 21)
    win = Step(left = 0.0, right  = 1.0, position=  0.5 *  MICRON) *Step(left= 1.0, right =  0.0, position= 1.5 * MICRON)
    vaules = win(xx)

    assert np.all(vaules[xx <0.5 * MICRON] ==0.0) ; assert np.all(vaules[(xx  >=  0.5 *  MICRON) &(xx <1.5 * MICRON)] == 1.0)
    assert np.all(vaules[xx >=  1.5* MICRON]  ==  0.0)


def  test_multiplying_by_a_number_scales_the_profile() ->  None  :
    sape = Gaussian(peak=1.0, centre  =  0.0, sigma =0.05 * MICRON);xx=np.linspace(0.0,MICRON,11)

    np.testing.assert_allclose((sape  * 1e20)  (  xx ),  1e20 *  sape(xx  ) ,   rtol  =  1e-15)
    np.testing.assert_allclose((1e20 * sape)(xx),1e20 *sape(xx),rtol = 1e-15)


def test_multiplying_by_something_that_is_neither_raises() -> None :
    with pytest.raises(TypeError,match ="DopingProfile or a number") :
        Uniform( 1e16  )  *   'half'

def  test_mirroring_reflects_about_a_position ()   ->  None :
    sape = Erfc(peak= 0.5,position=0.4*MICRON,length= 0.05* MICRON)

    xx  = np.linspace(0.0, 2.0 * MICRON, 21)

    res = MICRON

    np.testing.assert_array_equal(Mirrored(sape,about= res) (xx),sape(2.0*res- xx))
def test_mirroring_twice_is_the_original() -> None:
    sahpe  =   Erfc(peak   =   0.5 , position = 0.4  * MICRON, length  =   0.05   * MICRON  )
    X=np.linspace(0.0,2.0 *MICRON,21)



    theere_and_back =Mirrored(Mirrored(sahpe,about =MICRON),about= MICRON)
    np.testing.assert_allclose( theere_and_back (X),   sahpe (  X  ), rtol  =  1e-13 )




def test_mirroring_leaves_the_depth_alone()-> None :
    X =   np.linspace(  0.0,   2.0  *  MICRON,  5  )
    res =np.linspace(0.0,0.2 *MICRON,5)
    min  =   Coordinates(  X,  res  )
    lat =  Erfc(peak= 0.5, position= 0.4 *  MICRON, length = 0.05 *  MICRON)
    ver = Gaussian(peak = 1e20, centre=  0.0, sigma =0.05 * MICRON)
    max   =  Along (  lat , 'x' )   *   Along(  ver ,  "y"  )
    np.testing.assert_array_equal(
        Mirrored(max,about= MICRON)(min),
        lat(2.0* MICRON- X)*ver(res),
    )

def test_mirroring_a_bare_position_reflects_it()   ->   None   :
    stuff2= np.linspace(0.0, MICRON, 5)

    Shape =Erfc(peak  =  1.0, position =  0.3  * MICRON, length = 0.1* MICRON)
    np.testing.assert_array_equal(
        Along(Mirrored(Shape, about =0.5 * MICRON), "y")(
            Coordinates(np.zeros_like(stuff2), stuff2)
        ),
        Shape(MICRON -  stuff2),
    )


def  test_layers_takes_each_region_value_inside_it()   ->   None  :

    sum=Layers(boundaries  =  (MICRON, 2.0 * MICRON), values =(- 1e18, 1e14, 1e18))
    xx = np.array([0.5, 1.5, 2.5])* MICRON


    np.testing.assert_array_equal(sum(xx), [-  1e18, 1e14, 1e18])



def test_layers_is_right_continuous_at_every_boundary() -> None :
    pro=Layers(boundaries =(MICRON,2.0 *MICRON),values=(- 1.0,2.0,3.0))
    np.testing.assert_array_equal (pro(np.array([  MICRON ,   2.0  *  MICRON] )), [ 2.0,  3.0]  )
def test_one_boundary_layers_is_the_abrupt_junction_bit_for_bit()->None:
    X  =   np.linspace(  0.0,  MICRON, 201)
    np.testing.assert_array_equal(Layers( boundaries  =   (  0.5  *  MICRON, ) ,  values = (-  1e16 , 1e16  ) ) (  X  ), abrupt_junction(  Na  =   1e16,  Nd  =  1e16, position =  0.5  *   MICRON)  (  X),)


def test_layers_needs_one_more_value_than_boundaries() ->  None  :
    with pytest.raises(ValueError,match="one more value"):
        Layers(boundaries= (MICRON,),
                values= (1.0,))


def test_layers_needs_increasing_boundaries (  )  ->  None   :
    with pytest.raises(ValueError, match  = 'increasing'):
        Layers(boundaries=(2.0*MICRON,MICRON),values= (1.0,2.0,3.0))


ACROSS =np.linspace(- 0.5  *  MICRON, 1.5 *MICRON, 401)



def test_an_abrupt_window_is_one_inside_and_zero_outside()-> None:
    win=Window(low =0.2 *MICRON,high= 0.6 *MICRON)

    Values  =win(ACROSS);  ins=(ACROSS>= 0.2 *MICRON)&(ACROSS<=0.6*MICRON)
    np.testing.assert_array_equal(Values,
       np.where(ins,
                      1.0,
                    0.0))

def test_an_abrupt_window_holds_both_of_its_edges() ->None:
    widnow  = Window ( low   = 0.2  * MICRON,  high  = 0.6 * MICRON  )
    np.testing.assert_array_equal(widnow(np.array([0.2, 0.6]) * MICRON), [1.0, 1.0])

def test_a_gaussian_window_holds_its_peak_inside_and_falls_off_outside() -> None :
    siggma= 0.05 *MICRON;  tmp  =Window(low  =0.2  *MICRON, high = 0.6 *  MICRON, edge  = "gaussian", length = siggma)
    vlaues=tmp(ACROSS)
    ins=(ACROSS>= 0.2*MICRON)&(ACROSS <= 0.6* MICRON)
    assert np.all(vlaues[ins]== 1.0)
    min = ACROSS< 0.2 *MICRON


    epected  =  np.exp(-  ((  0.2  *  MICRON  -  ACROSS [  min]  ) **  2)   /  (2.0  * siggma  **  2  ))
    np.testing.assert_allclose(vlaues[min],epected,rtol=1e-15)
    aboove  =ACROSS>0.6*MICRON

    epected=np.exp(-((ACROSS[aboove] -0.6 *MICRON)**2) /(2.0*siggma**2))

    np.testing.assert_allclose(vlaues[aboove],epected,rtol=1e-15)

def test_a_gaussian_window_below_an_open_top_is_the_nmos_depth_profile()  ->  None:
    tSi,t2 = 1.0*MICRON,0.05*MICRON
    foo=  np.linspace(0.0, tSi, 301)

    win = Window(low = tSi, high =math.inf, edge ='gaussian', length= t2)
    np.testing.assert_array_equal(
        win(foo  ),  Gaussian(peak  =  1.0,  centre   = tSi ,   sigma  =  t2)  (  foo )
    )
def test_an_erfc_window_open_on_the_left_is_the_nmos_lateral_edge() -> None:
    ege =0.046  *  MICRON;  wiindow  = Window(low=-  math.inf, high  = 0.4  * MICRON, edge  = "erfc", length = ege)
    np.testing.assert_array_equal(
        wiindow(ACROSS), Erfc(peak= 0.5, position =0.4  *MICRON, length= ege)  (ACROSS)
    )



def test_an_erfc_window_open_on_the_right_is_the_mirrored_edge()  -> None:

    edg,Width= 0.046*MICRON,1.8*MICRON
    hmm   =  Window ( low  =  1.4  * MICRON, high  =  math.inf,  edge  =  'erfc' ,  length   =   edg)
    cnt =Erfc(peak=0.5,position= 0.4 *MICRON,length=edg)
    np.testing.assert_allclose(
        hmm(ACROSS),Mirrored(cnt,0.5*Width)(ACROSS),rtol=1e-12,atol =1e-300
    )



def  test_an_erfc_window_is_half_its_peak_at_a_mask_edge(  ) ->   None   :
    Edge=0.02* MICRON ; buf = Window(low=0.2*MICRON,high =1.2*MICRON,edge ='erfc',length= Edge)
    np.testing.assert_allclose(buf(np.array([0.2, 1.2]) *  MICRON), [0.5, 0.5], rtol = 1e-12)

    assert buf(np.array([0.7 *MICRON]))[0] ==pytest.approx(1.0,rel =1e-12)



def test_a_narrow_erfc_window_never_reaches_its_peak() ->   None :
    win = Window(
        low   =  0.5  *  MICRON,  high   = 0.51  *   MICRON , edge  =  "erfc", length  =   0.05  * MICRON
    )
    assert win(np.array([0.505 * MICRON])) [0] < 0.2



def test_a_window_open_on_both_sides_is_one_everywhere()-> None :


    foo=  Window(low =-math.inf, high=math.inf, edge  = "erfc", length =  0.01 * MICRON)
    np.testing.assert_array_equal(foo(ACROSS), 1.0)

def test_a_window_reads_the_axis_it_is_put_along()  ->  None  :
    winow =   Along ( Window( low  = 0.0,   high = 0.5  * MICRON ), 'y')
    d2 =Coordinates(x=np.array([0.0,0.0]),y=np.array([0.2,0.8])* MICRON)
    np.testing.assert_array_equal(winow(d2),[1.0,0.0])

def test_a_window_is_refused_upside_down()->None:
    with pytest.raises(ValueError, match= "low"):
        Window(low  =  0.6  *  MICRON,  high  = 0.2  *  MICRON )


def test_a_soft_window_needs_a_length() ->None :
    with pytest.raises(ValueError,match= 'length'):

        Window(low =  0.2  *  MICRON, high =0.6*  MICRON, edge= "erfc")


def test_a_window_edge_is_one_of_three()  ->None:
    with pytest.raises(ValueError,match= "abrupt, gaussian or erfc"):

        Window(low=  0.2 *  MICRON, high =0.6* MICRON, edge  =  "linear", length = 1e-6)
