'''Tests for physics/mobility.py, the Arora doping dependent model.

docs/01-physics.md puts this in Phase 3 and is blunt about why it matters:
"Mobility is where a device simulator earns or loses its quantitative accuracy.
The PDE solve can be perfect and the answer still wrong by 3x if mobility is
wrong."

The model has four analytic limits and they are all checkable by hand:

    N -> 0      mu -> mu_min + mu_d
    N = N_ref   mu = mu_min + mu_d/2
    N -> inf    mu -> mu_min
    dmu/dN < 0  everywhere

Those pin the shape. Two literature values pin the magnitude, because the shape
would be right for any parameter set.
'''
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
    """N -> 0 kills the denominator's second term."""
    assert  float( model(0.0) )  == pytest.approx( model.mu_min  +   model.mu_d ,  rel =  1e-14)
@pytest.mark.parametrize( "model" ,  [AroraMobility.electrons ( ) ,   AroraMobility.holes( )])


def test_at_the_reference_doping_the_lattice_term_is_halved(model  :  AroraMobility ,)  -> None :
    """N = N_ref makes (N/N_ref)^A exactly 1, whatever A is."""


    assert float(model(model.N_ref))==  pytest.approx(model.mu_min  + model.mu_d  / 2.0, rel  = 1e-14)

@pytest.mark.parametrize("model", [AroraMobility.electrons(), AroraMobility.holes()])
def test_the_heavily_doped_limit_is_mu_min(model: AroraMobility)->None :
    """Ionized impurity scattering saturates. Approached, never reached."""
    hea  =   float(model(  1e25 ) )

    assert hea >model.mu_min ; assert hea == pytest.approx(model.mu_min,
                     rel =1e-3)



@pytest.mark.parametrize('model',[AroraMobility.electrons(),AroraMobility.holes()])


def test_mobility_falls_monotonically_with_doping(model : AroraMobility) ->  None  :
    """More scattering centres, less mobility. No exceptions in this model."""
    len   =   np.logspace( 10, 21, 200)
    assert np.all(np.diff(model(len)) <0.0)


@pytest.mark.parametrize("doping,expected,tolerance", [(1e16,1230.0,0.02), (1e18,280.0,0.05),],)
def test_electron_mobility_matches_the_literature(doping  : float, expected  : float, tolerance  : float) ->None :
    """Silicon n-type at two dopings, against the values everyone quotes.

    The four limits above fix the shape and would pass for any parameter set
    with the right form. These fix the numbers.
    """
    assert float(AroraMobility.electrons()(doping))==pytest.approx(
        expected,rel=tolerance
    )


def test_holes_are_slower_than_electrons_at_every_doping(  )   -> None  :
    """True in silicon at every doping, and a sign check on the parameters."""


    out2 =  np.logspace(  13,  20 , 60  )
    assert np.all(AroraMobility.holes() (out2)  <  AroraMobility.electrons()(out2))

def test_the_parameters_reduce_to_their_tabulated_values_at_300_K()-> None :
    """Every parameter carries a (T/300)^k factor which must vanish at 300."""
    Electrons  =  AroraMobility.electrons(T =C.T_ROOM)



    assert Electrons.mu_min ==pytest.approx(88.0, rel =  1e-14)
    assert Electrons.mu_d  ==  pytest.approx(1252.0, rel=1e-14)

    assert Electrons.N_ref==pytest.approx(1.432e17,rel= 1e-14)
    assert Electrons.exponent   ==  pytest.approx (0.88 ,  rel = 1e-14  )


def  test_mobility_falls_as_temperature_rises_in_lightly_doped_silicon()   ->   None   :


    """Phonon scattering dominates there, and it grows with temperature.

    Both mu_min and mu_d carry negative exponents, so this is really a check
    that neither sign was transcribed backwards.
    """
    open= 1e14
    coold =float(AroraMobility.electrons(T = 250.0)(open))
    hott = float(AroraMobility.electrons(T= 400.0)(open))

    assert  hott  <  coold




def  test_the_constant_model_ignores_the_doping()  ->  None  :


    """The Phase 1 and 2 behaviour, kept so the seam has one shape."""
    moodel  =   ConstantMobility (C.MU_N_300 )
    np.testing.assert_allclose(moodel(np.array([0.0, 1e15, 1e20])), C.MU_N_300, rtol=0.0)



def test_the_constant_model_returns_one_value_per_node() -> None :
    """A bare scalar would broadcast and then silently collapse an average."""
    Got =  ConstantMobility( 1417.0)  ( np.zeros(7  )  )
    assert Got.shape == (7, )

def test_edge_diffusivity_averages_the_two_endpoint_values( ) ->  None  :
    """Mobility is a nodal quantity and the flux needs it on the edge.

    The arithmetic mean of the two endpoints, which is what a Scharfetter-
    Gummel edge wants: the flux derivation assumes the coefficient is constant
    along the edge, so the edge value is the one thing being approximated and
    the mean is the honest choice.
    """
    moiblity=np.array([1000.0,500.0,100.0])

    yy   =  edge_diffusivity( moiblity, V_T   =  0.02585  )


    np.testing.assert_allclose(yy, np.array([750.0, 300.0])* 0.02585, rtol = 1e-14)

def test_edge_diffusivity_applies_the_einstein_relation() ->  None  :
    """D = V_T * mu. Getting this wrong scales every current by 40."""

    gott =edge_diffusivity(np.full(2, 1417.0), V_T =0.02585)
    assert float(gott[0]) ==pytest.approx(1417.0* 0.02585,rel =1e-14)



def test_edge_diffusivity_gives_one_value_per_edge()  ->  None  :
    arr =  edge_diffusivity( np.ones(11),   V_T   =  0.02585  )
    assert arr.shape == (10, )




def test_edge_diffusivity_gathers_the_endpoints_an_edge_list_names (  )  ->  None :

    """A 2D mesh has to say which two nodes an edge joins.

    Its edges are not contiguous and a column of it is not a slice, so the
    1D branch would average node 3 with node 4 when the edge in question runs
    from node 3 to node 8. Nothing downstream would notice: the array is the
    right dtype and, on a square mesh, very nearly the right length.
    """
    Mobility  =   np.array ( [ 100.0 , 200.0, 400.0, 800.0  ]  )
    buf=np.array([[0,2],[1,3],[3,0]],dtype = np.int64)

    bb   =   edge_diffusivity(  Mobility , V_T  = 2.0,   edge_nodes  =  buf )
    np.testing.assert_allclose(bb, [500.0, 1000.0, 900.0], rtol =1e-14)



def test_the_edge_list_form_reproduces_the_1d_chain_exactly()-> None :
    '''Bit for bit, because the 1D chain is one particular edge list.

    Every current measured in Phases 1 to 3 went through the slice branch, so
    the two have to be the same arithmetic and not merely the same answer.
    '''
    mobbility=np.linspace(300.0, 1400.0, 9)
    cha= np.column_stack([np.arange(8),np.arange(8)+ 1]).astype(np.int64)


    np.testing.assert_array_equal (
        edge_diffusivity( mobbility ,  V_T  =  0.02585, edge_nodes   =  cha  ),
        edge_diffusivity(  mobbility,   V_T  =  0.02585  ) ,
    )



def test_the_arora_undoped_limit_does_not_match_the_tabulated_mobility()-> None  :
    '''Documented rather than reconciled, because both numbers are measured.

    docs/06-constants.md lists mu_n = 1417 for undoped silicon and separately
    gives Arora parameters whose N -> 0 limit is 88 + 1252 = 1340. They
    disagree by 5.4 percent for electrons and 1.8 percent for holes. Neither
    is wrong: they come from different fits to different data, and Arora is
    fitted over the doped range where it is used rather than at the intrinsic
    limit where nothing is ever measured.

    It matters because switching a device from the constant model to Arora
    moves its current by that much even at doping low enough that the model
    should not be doing anything, and that would otherwise read as a bug.
    '''
    eletcrons =AroraMobility.electrons()
    out2 = AroraMobility.holes()

    assert float(eletcrons(0.0)) == pytest.approx(1340.0, rel=1e-12)
    assert float(out2(0.0))==pytest.approx(461.3, rel=1e-12)
    assert  float (  eletcrons (  0.0 ) ) /   C.MU_N_300   ==  pytest.approx(0.9456,   rel  =  1e-3 )
    assert float(out2(0.0))/ C.MU_P_300 == pytest.approx(0.9815,rel= 1e-3)


def edges(value :float, count : int= 5):
    """A low field diffusivity on `count` edges."""
    return np.full(count,value)
def test_at_zero_field_the_model_returns_the_low_field_value()->None:
    """Exactly, not nearly. A model that shaved a fraction off at zero field
    would move every result taken before it existed."""
    moddel = CaugheyThomas(low_field=edges(1.0), v_sat=  0.5, beta = 2.0)

    all= np.full(5, 0.1)
    np.testing.assert_array_equal(moddel(np.zeros(5), all), edges(1.0))




def test_the_drift_velocity_saturates_at_v_sat() ->None  :
    """The whole point of the model. mu falls exactly fast enough that mu*E
    approaches a constant, and that constant is v_sat."""
    VSat =  0.5
    Model   =  CaugheyThomas(low_field  = edges(  1.0 ),  v_sat   = VSat,   beta   =  2.0 )
    q = np.full(5, 0.1)


    t2= np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    Velocity =  Model (  t2, q )   * np.abs(  t2)  /  q

    assert Velocity[- 1]  ==pytest.approx(VSat, rel= 1e-6)
    assert  np.all (  np.diff(  Velocity  )   >   0.0 )


def test_the_drift_velocity_never_exceeds_v_sat() -> None:
    '''At any field at all, which is a stronger statement than the limit.'''

    r2 =  0.5


    for thing in(1.0, 2.0) :
        mod= CaugheyThomas(low_field =edges(1.0),v_sat =r2,beta=thing)
        obj2=np.full(5,0.1)
        x= np.array([0.0,1e-3,1.0,1e3,1e9])

        assert np.all(mod(x,obj2)* np.abs(x) /obj2<=r2)



def test_the_knee_is_where_the_low_field_drift_would_reach_v_sat() -> None  :
    """The one point of the curve that is closed form: at mu_0 E = v_sat the
    bracket is exactly 2, so mu is mu_0 over the beta-th root of 2."""
    for idx2 in(1.0, 2.0) :
        arr= CaugheyThomas(low_field=edges(1.0),v_sat =0.5,beta =idx2)
        H  = np.full(5,
            0.1)
        XX=np.full(5,0.5 * 0.1)
        np.testing.assert_allclose(arr(XX,H),1.0/2.0**(1.0 /idx2),rtol=1e-14)


def test_mobility_falls_monotonically_with_field()->None :
    for  bet  in(  1.0 ,
             2.0 )  :

        moddel =CaugheyThomas(low_field= edges(1.0,60),v_sat=0.5,beta= bet)

        hh=  np.full(60, 0.1)
        x =  np.linspace(0.0, 30.0, 60)

        assert np.all(np.diff (  moddel(x,   hh )  ) < 0.0 )
def test_the_model_does_not_care_which_way_the_field_points()->None :
    '''A carrier slows down in a strong field whichever direction it runs, so
    only the magnitude of the potential drop enters.'''
    x2  = CaugheyThomas(low_field = edges(1.0), v_sat=0.5, beta= 2.0)
    arr =  np.full(5, 0.1)
    x=np.array([0.1,1.0,5.0,20.0,100.0])

    np.testing.assert_array_equal(x2(x, arr), x2(-  x, arr))


def test_electrons_hold_their_mobility_longer_than_holes_do() -> None :
    """beta = 2 against beta = 1 at the same v_sat. Below the knee the larger
    exponent keeps the bracket nearer to 1, so the electron curve stays flat
    and then turns while the hole curve starts falling straight away."""
    hh =  np.full(5, 0.1)
    x  =  np.full(  5, 0.1  *  0.5  *  0.1)

    ele= CaugheyThomas(low_field=edges(1.0),v_sat=0.5,beta=2.0)
    hol = CaugheyThomas(low_field = edges(1.0), v_sat= 0.5, beta = 1.0)


    assert np.all(ele(x,hh)>hol(x,hh))
def test_a_longer_edge_across_the_same_drop_is_a_weaker_field() -> None:
    """The field is the drop divided by the length, so the edge length is not
    decoration. Dropping it would make the model depend on how the mesh was
    graded rather than on the physics."""
    dat=CaugheyThomas(low_field=edges(1.0),v_sat= 0.5,beta=2.0)
    XX   =  np.full(  5,  1.0)
    assert np.all(dat(XX, np.full(5, 1.0))  >dat(XX, np.full(5, 0.1)))


def complex_step_dD_dX(model, X, h, step: float = 1e-30) :
    """dD/dX by complex step, exact to machine precision.

    The technique phases/PHASE-3.md makes a permanent CI requirement for the
    coupled Jacobian, applied here to one term of it on its own.
    """
    return np.imag( model(X +   1j  * step,
                  h  )  )  /  step

def test_the_tangent_matches_a_complex_step_for_electrons() -> None:
    mod  =   CaugheyThomas (  low_field   =  edges (1.3,
                7  ),
          v_sat  = 0.5,
                  beta   =  2.0)
    hh  = np.linspace (  0.05,   0.4, 7 )

    zz = np.array([-  30.0,
                - 5.0,
               -0.5,
                0.2,
                      3.0,
            12.0,
          200.0])

    np.testing.assert_allclose(mod.derivative(zz, hh), complex_step_dD_dX(mod, zz, hh), rtol = 1e-12)

def  test_the_tangent_matches_a_complex_step_for_holes (  )   ->   None  :

    """beta = 1, where the model has an absolute value in it and the tangent
    on the negative side is the one that would be easiest to get wrong."""
    moddel=  CaugheyThomas(low_field= edges(0.4, 7), v_sat = 0.3, beta =  1.0)
    hh   =  np.linspace ( 0.05,  0.4 , 7)
    XX = np.array([-30.0, -5.0, -  0.5, 0.2, 3.0, 12.0, 200.0])

    np.testing.assert_allclose (moddel.derivative ( XX , hh), complex_step_dD_dX (  moddel, XX ,   hh), rtol  = 1e-12)

def test_the_tangent_is_negative_wherever_the_field_is_positive() ->None :
    """More field, less mobility. A sign is what a Jacobian gets wrong
    silently: Newton still converges with the wrong sign on a small term, just
    slowly, and to the same answer."""
    temp =CaugheyThomas(low_field = edges(1.0),v_sat=0.5,beta=2.0)
    hh =np.full(5,0.1); XX =np.array([0.1,1.0,5.0,20.0,100.0])

    assert np.all(temp.derivative(XX,hh)<0.0)
    assert np.all(temp.derivative(-XX,hh) > 0.0)
def test_the_tangent_is_zero_at_zero_field() -> None  :
    """Zero from both sides for electrons, and by choice for holes.

    beta = 2 is smooth at the origin and its derivative there really is zero.
    beta = 1 is not: the model has |E| in it, so the two one sided derivatives
    are equal and opposite and there is no two sided one. Zero is the value
    halfway between them, and it is the right choice for a Jacobian, because
    either one sided value would claim the mobility falls when the potential
    is raised and rises when it is lowered, which is half of the truth stated
    as the whole of it.
    """
    for bet in (1.0 , 2.0  )   :
        Model=CaugheyThomas(low_field =edges(1.0),v_sat=0.5,beta= bet)


        np.testing.assert_array_equal(Model.derivative(np.zeros(5), np.full(5, 0.1)), np.zeros(5))


def test_a_complex_step_at_zero_field_takes_the_right_hand_side()-> None :
    """Which is where the Jacobian and its verification part company, and the
    only place they do.

    A complex step approaches the origin along the imaginary axis, and the
    square root of a square resolves that onto the positive branch, so it
    reports the derivative from the right. For electrons that is zero and
    agrees. For holes it is the one sided value, against zero in the Jacobian.
    Both are defensible readings of a point where the model has a corner, and
    the disagreement exists on a set of measure zero: a device with a hole
    current has a field somewhere, and an edge sitting at exactly 0.0 has no
    current across it to get wrong.
    """
    vals=CaugheyThomas(low_field=edges(1.0),v_sat=0.5,beta=1.0)
    Electrons=  CaugheyThomas(low_field =  edges(1.0), v_sat =0.5, beta= 2.0)
    round= np.full(5,0.1)
    ftr  =complex_step_dD_dX(vals, np.zeros(5), round)

    assert np.all( ftr   < 0.0)
    np.testing.assert_allclose(ftr, vals.derivative(np.full(5, 1e-8), round), rtol= 1e-6)
    np.testing.assert_allclose(
        complex_step_dD_dX(Electrons,np.zeros(5),round),np.zeros(5),atol= 1e-30
    )
def test_a_complex_argument_survives_the_model()->None:

    """The block verification differentiates the coupled residual by complex
    step, so everything the residual calls has to stay analytic. abs() of a
    complex number is not, which is why the magnitude is a square root of a
    square instead."""
    moodel  =  CaugheyThomas ( low_field  =   edges(  1.0 ) ,   v_sat  =  0.5, beta =   2.0 )

    data2= moodel(np.full(5,2.0)+1j*1e-30,np.full(5,0.1))
    assert np.iscomplexobj(data2)
    np.testing.assert_allclose(
        np.real(  data2 ),  moodel(  np.full ( 5,   2.0  ), np.full (5 , 0.1  )),   rtol  =  1e-15
    )



def test_a_zero_saturation_velocity_is_refused (  ) -> None  :
    """It divides the field, and a carrier that cannot move at all is not a
    slow carrier, it is a different model."""
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


    '''Pins the provenance. Every one of these came out of the reference's own
    source, and a fit parameter that drifts is not visible in any result until
    a DEVSIM regression fails for a reason nobody can locate.'''
    hmm =build(T = C.T_ROOM)
    for ord,val in table.items() :
        assert getattr(hmm, ord)  == val



@pytest.mark.parametrize ('build', [LombardiSurface.electrons , LombardiSurface.holes],  ids   =  [  "n",   'p' ])

def test_a_vanishing_normal_field_gives_back_the_bulk_mobility(build)->None:
    '''Far from the interface there is no normal field, so there must be no
    surface scattering. This is what lets the model be applied over a whole
    region instead of inside a layer whose thickness somebody has to choose.

    Both surface terms diverge as E_perp falls, so their reciprocals vanish
    and Matthiessen leaves mu_bulk alone. Not exactly, because the floor stops
    E_perp at 1e2 V/cm, so the assertion is a percent rather than an equality.
    '''
    mod  =  build()
    mubulk =np.full(4,800.0)
    Mu = mod(mubulk, E_perp  =  np.zeros(4), total_doping=np.full(4, 1e17), carriers =np.full(4, 1e10),)


    np.testing.assert_allclose(Mu,mubulk,rtol= 0.02)
    assert np.all(Mu< mubulk),"scattering can only ever subtract"
def test_the_normal_field_is_floored_rather_than_dividing_by_zero()->None :
    '''Both terms divide by E_perp, so an unfloored zero is an inf in mu_ac
    and a nan the moment it meets the reciprocal sum. DEVSIM floors it at
    1e2 V/cm and the floor is copied rather than invented.'''

    mod = LombardiSurface.electrons()
    mubulk =  np.full(3, 800.0)
    dop,foo =np.full(3,1e17),np.full(3,1e10)
    atZero=mod(mubulk,np.zeros(3),dop,foo);  AtFloor  =  mod( mubulk , np.full( 3,  mod.E_floor ) , dop, foo)
    beow  = mod(mubulk, np.full(3, 1.0), dop, foo)

    assert np.all(np.isfinite(atZero))
    np.testing.assert_allclose(atZero,AtFloor,rtol=1e-14)
    np.testing.assert_allclose(beow, AtFloor, rtol  =  1e-14)




@pytest.mark.parametrize("build", [LombardiSurface.electrons, LombardiSurface.holes], ids  = ["n", 'p'])


def test_the_three_channels_combine_by_reciprocals ( build) -> None   :
    """1/mu = 1/mu_bulk + 1/mu_ac + 1/mu_sr, which is the whole content of the
    model. Asserted against the components the model reports separately, so a
    sign or an association error in the combining line has somewhere to show."""

    mod=build()
    mubulk  = np.full(5, 700.0)
    EE  = np.array( [  1e3 ,   1e4, 1e5 , 3e5,   1e6  ]  ); dop,Carriers=np.full(5,3e17),np.full(5,5e17)

    Mu =mod(mubulk,EE,dop,Carriers)
    muac= mod.acoustic(EE, dop)

    musr  =  mod.roughness(EE , dop,   Carriers  )

    xx   =  1.0   / (1.0  / mubulk   +  1.0  /   muac +  1.0  /  musr )
    np.testing.assert_allclose(Mu ,   xx,   rtol   =   1e-14)
    assert np.all(Mu <  np.minimum(mubulk, np.minimum(muac, musr)))

@pytest.mark.parametrize(
    "build", [LombardiSurface.electrons, LombardiSurface.holes], ids =  ['n', "p"]
)

def test_mobility_falls_as_the_normal_field_rises(  build  )  ->  None :
    """More field pulls the carrier harder against the interface, so it
    scatters off it more. Monotone with no turning point anywhere in the range
    a MOSFET occupies."""
    mod =build()
    temp = np.logspace(2.0, 6.5, 60)
    Mu   =  mod( np.full( 60 ,  800.0 ) ,   temp, np.full( 60 ,  1e17 ), np.full(  60, 1e18 ) )

    assert  np.all(  np.diff(  Mu) < 0.0 )



def test_the_inversion_layer_is_two_to_three_times_slower_than_bulk() ->  None:
    """The acceptance test for the whole model, and the reason Phase 5 calls
    it not optional.

    docs/01-physics.md: "Without this your inversion-layer mobility is too
    high by a factor of 2 to 3 and your Id is correspondingly wrong." The
    condition is a real one: a 1e17 channel under a normal field of 5e5 V/cm,
    which is what a 1 um NMOS sees at a volt of overdrive through 10 nm of
    oxide, carrying the surface density an inversion layer holds.

    The bulk mobility is Arora's own answer at that doping rather than a round
    number, so the ratio is between two things the code computes.
    """
    lst  =  1e17
    mb=float(AroraMobility.electrons()(lst))
    Mu  =  LombardiSurface.electrons()  (np.full(1,   mb), E_perp  =  np.full(  1, 5e5 ) , total_doping = np.full( 1,   lst) , carriers  =  np.full(1, 1e18),)

    bar  =   mb  / float (Mu[ 0])
    assert 2.0<bar <3.0,f"surface mobility is {bar:.2f}x below bulk"

def test_holes_stay_slower_than_electrons_at_the_surface()  -> None :
    """True in the bulk and it had better survive the surface terms, which
    carry their own separate parameter set and could reorder them."""

    len,Doping,val =np.full(4,3e5),np.full(4,1e17),np.full(4,1e18)



    next = LombardiSurface.electrons()(np.full(4,800.0),len,Doping,val)
    myvar =  LombardiSurface.holes() (np.full(4, 300.0), len, Doping, val)

    assert np.all(myvar<next)

def test_a_heavier_inversion_layer_roughens_the_surface_it_sees()->None:

    """The exponent gamma is what makes this the enhanced model rather than
    the 1988 one, and it is the only place a carrier density enters. More
    carriers in the layer means a larger exponent, so mu_sr falls faster with
    field. With gamma a constant this test cannot pass."""
    mod =  LombardiSurface.electrons()
    e,dpoing=np.full(3,5e5),np.full(3,1e17)
    tmp=mod.roughness(e, dpoing, carriers = np.full(3, 1e14))

    Heavy = mod.roughness(e, dpoing, carriers =  np.full(3, 1e20))
    assert np.all(mod.gamma(dpoing,np.full(3,1e20))>mod.gamma(dpoing,np.full(3,1e14)))
    assert np.all(Heavy <  tmp)

def test_the_exponent_reduces_to_A_with_no_carriers_present()-> None:


    """gamma = A + alpha*(n + p)*N^(-eta), so an empty band leaves A. That is
    the 1988 Lombardi exponent, and the two models agree there."""
    Model =LombardiSurface.holes()
    t2= Model.gamma(np.full(2,1e17),np.zeros(2))

    np.testing.assert_allclose (t2, Model.A,  rtol  = 1e-15  )

def test_the_acoustic_term_carries_the_temperature_exponent()-> None:
    """mu_ac divides its second term by (T/300)^kappa, so a hotter lattice
    scatters more. kappa differs between the carriers, 1.7 against 0.9, which
    is why it is a parameter and not a shared constant."""
    idx2   =  LombardiSurface.electrons(  T = 250.0 )
    hott =LombardiSurface.electrons(T =  350.0)


    e,val =np.full(3,3e5),np.full(3,1e17)
    assert np.all(hott.acoustic(e, val) < idx2.acoustic(e, val))


def test_a_negative_normal_field_is_refused()->None:
    """E_perp is a magnitude by construction, so a negative one means the
    caller took a signed difference and forgot the absolute value. Quietly
    taking abs() here would hide that: the exponents would accept it and the
    mobility would come out looking plausible."""

    with pytest.raises(ValueError, match = 'E_perp') :
        LombardiSurface.electrons (  )   (np.full(3 , 800.0  ), np.array ([1e5,   -   1e5 , 1e5]  ) , np.full(  3, 1e17), np.full (  3,  1e18),)




def test_a_vanished_roughness_mobility_leaves_no_mobility_and_no_warning()-> None :
    """gamma carries the carrier density linearly, so a state with n far above
    its own doping raises E_perp to a power that underflows mu_sr to exactly
    zero. Matthiessen's rule says the answer there is zero, and it is: no
    surface roughness channel means no conduction. What it must not do is
    reach that zero through 1/mu_sr, because an infinity on the way costs a
    divide by zero warning and `filterwarnings = error` turns that into a
    failure.

    The state itself is not physical. It is the cold guess a MOSFET solve
    starts from: `initial_state` puts n = n_i exp((psi - phi_n)/V_T) at the
    drain, and at a volt of drain bias that is 1e36 cm^-3 before Newton has
    taken a single step. A model handed an iterate has no say in which iterate
    it is handed.
    """
    mod =LombardiSurface.electrons(); thing,EE=np.full(2,1e20),np.full(2,8.8e6); car = np.array([1e20, 1.6e36])
    assert  mod.roughness (  EE ,  thing,   car)  [  1]  == 0.0
    stuff2  = mod(np.full(2, 100.0), EE, thing, car)
    assert stuff2[1]  ==  0.0
    assert stuff2[0] > 0.0
