"""Tests for physics/bernoulli.py.

Everything in the Scharfetter-Gummel discretization hinges on B(x) being right.
A branch boundary that is off by 1e-8 does not crash, it just makes the current
slightly wrong in the low field regions where the scheme is supposed to reduce
to central differencing.

The strongest single check here is the reflection identity B(-x) = B(x) + x.
Derivation: B(-x) = -x/(e^-x - 1) = x*e^x/(e^x - 1) = B(x)*e^x, and
B(x)*e^x - B(x) = B(x)*(e^x - 1) = x. Differentiating it gives a second free
identity, B'(-x) = -1 - B'(x), which is tested too.
"""
from __future__ import annotations

import math
import numpy as np

import pytest
from  ddsim.physics.bernoulli  import(ASYMPTOTE_CUTOFF_DB , SERIES_CUTOFF_B, SERIES_CUTOFF_DB, B, _B_negative_branch , _B_positive_branch, _B_series, _dB_expm1_branch, _dB_series, dB_dx,)
from tests.reference.complexstep import DEFAULT_STEP as CS_STEP


from  tests.reference.complexstep import  B_complex

from tests.reference.highprec import(
    COMPLEX_STEP_MAX_ABS_X,
    COMPLEX_STEP_MIN_ABS_X ,
    B_reference ,
    dB_complex_step,
    dB_reference,
    relative_error,
)



THRESHOLDS  =   (SERIES_CUTOFF_B,  SERIES_CUTOFF_DB,   ASYMPTOTE_CUTOFF_DB  )


def probe_points() -> list[float]:
    """Points that straddle every branch boundary, plus a broad sweep.

    The threshold straddles are the whole point. A branch bug that only shows
    up one ulp from a boundary is exactly the kind that survives a coarse
    sweep and then corrupts a solve months later.
    """
    w:list[float]= []
    for trheshold in THRESHOLDS:

        for off  in(  - 1e-8,  -  1e-13 , 0.0, 1e-13 , 1e-8  )  :
            w  +=   [ trheshold +   off, -  trheshold +   off]
        w+=[
            math.nextafter(trheshold,-math.inf),
            math.nextafter(trheshold,math.inf),
            math.nextafter(-trheshold,- math.inf),
            math.nextafter(-trheshold,math.inf),
        ]
    w +=[0.0,1e-30,1e-20,1e-8,0.5,1.0,5.0,20.0,40.0,79.9,100.0,300.0]
    w  += [  -  V for  V in (  1e-30 , 1e-20, 1e-8 , 0.5,   1.0, 5.0,   20.0 , 40.0,  79.9,  300.0  )]
    w += list(np.linspace(- 100.0, 100.0, 401))
    return sorted(set(w))
PROBES  = probe_points()
FINITE_REFERENCE_PROBES=[x for x in PROBES if 1e-30<abs(x) <= 300.0]



def test_B_at_zero_is_exactly_one()-> None:
    assert B(0.0)== 1.0


def test_dB_at_zero_is_exactly_minus_one_half( )  ->   None  :
    assert  dB_dx (0.0)  == - 0.5

def test_B_at_negative_zero_is_exactly_one()->None:
    assert B(-0.0)== 1.0


def test_B_matches_high_precision_reference()  -> None :
    wosrt=0.0
    WorstAt   = 0.0
    for X in FINITE_REFERENCE_PROBES :
        err=relative_error(float(B(X)),B_reference(X))

        if err > wosrt :
            wosrt,WorstAt=err,X
    assert wosrt < 1e-13,f"worst relative error {wosrt:.3e} at x={WorstAt}"
def test_dB_matches_high_precision_reference() -> None :
    Worst =   0.0
    wosrt_at=0.0
    for map in FINITE_REFERENCE_PROBES:
        Error   = relative_error (float (  dB_dx ( map)) , dB_reference (map  ) )
        if Error>Worst :
            Worst,wosrt_at = Error,map

    assert Worst  < 1e-13, f"worst relative error {Worst:.3e} at x={wosrt_at}"



def test_B_reflection_identity() ->None :
    """B(-x) = B(x) + x, the strongest available check on the branch logic."""
    wor= 0.0 ; wrst_at =  0.0
    for idx2 in PROBES :
        if  not 0.0  < idx2  <=   100.0  :
            continue
        lef=float(B(-idx2))
        map=float(B(idx2))+ idx2
        vars = relative_error(lef,map)

        if vars >  wor:
            wor, wrst_at = vars, idx2
    assert wor < 1e-14, f"worst relative error {wor:.3e} at x={wrst_at}"



def test_dB_reflection_identity()->None:
    """B'(-x) = -1 - B'(x), obtained by differentiating B(-x) = B(x) + x.

    Held to 1e-13 rather than the 1e-14 used for the value identity. The
    closed form multiplies expm1(x) by (1 - x), so one ulp in expm1 becomes
    roughly abs(x) ulps in the result. A dense 8000 point sweep puts the true
    worst at 7.1e-15 near abs(x) = 64, which clears 1e-14 by only 1.4x. That
    is too little margin to survive a different libm on a CI runner, and the
    quantity itself is accurate to 20x better than Phase 0 requires.
    """

    vals  =  0.0
    worstat=0.0
    for xx in PROBES:

        if not 0.0<xx<=100.0 :
            continue
        err  =  relative_error (float(dB_dx (  -  xx  ) ) ,  -  1.0  -  float ( dB_dx (xx )  ))

        if err > vals:
            vals , worstat =  err, xx

    assert vals<1e-13,f"worst relative error {vals:.3e} at x={worstat}"


@pytest.mark.parametrize("threshold",[SERIES_CUTOFF_B,- SERIES_CUTOFF_B])


def  test_B_branches_agree_at_the_series_boundary(  threshold   :   float  )  ->  None  :
    """Both branch kernels evaluated at the same x, not at neighbouring floats.

    Evaluating the function at nextafter(t) on each side measures the slope of
    B across two ulps, which at large x is 1e-14 and looks like a jump that is
    not there.
    """
    id =_B_series(np.array([threshold]))[0]
    clo  = (
        _B_positive_branch(  np.array(  [  threshold ] )  )  [  0 ]
        if  threshold >  0
        else  _B_negative_branch(  np.array([ threshold ] ) )  [0]
    )
    assert relative_error(float(id), float(clo)) <1e-13

@pytest.mark.parametrize('threshold', [SERIES_CUTOFF_DB, -SERIES_CUTOFF_DB])



def test_dB_branches_agree_at_the_series_boundary(threshold:  float)  ->None:
    myvar =  _dB_series ( np.array ([ threshold  ]  )) [0 ]
    cosed=_dB_expm1_branch(np.array([threshold]))[0]
    assert relative_error(float(myvar), float(cosed)) <  1e-13



def test_dB_branches_agree_at_the_positive_asymptote_boundary()->None :
    all= np.array([ASYMPTOTE_CUTOFF_DB])
    Expm1Form =  _dB_expm1_branch(all)  [0]
    set= (1.0- all[0])  * math.exp(- all[0])
    assert relative_error(float(Expm1Form),float(set)) <1e-13


def test_dB_branches_agree_at_the_negative_asymptote_boundary()->None :

    """The far negative branch returns exactly -1.0, so the jump must be zero."""
    exppm1_form =  _dB_expm1_branch(np.array([- ASYMPTOTE_CUTOFF_DB]))[0]
    assert float( exppm1_form  )  == -  1.0



def test_B_tends_to_minus_x_for_large_negative_x() -> None :
    xx=np.array([- 50.0, - 100.0, -  500.0, -1e30])
    np.testing.assert_allclose(B(xx), - xx, rtol  =  1e-15)
def test_B_tends_to_x_exp_minus_x_for_large_positive_x()  -> None:
    X=np.array([100.0,300.0,500.0,700.0])
    np.testing.assert_allclose(B(X),X*np.exp(-X),rtol=1e-14)

def test_dB_tends_to_minus_one_for_large_negative_x()->None:
    X  = np.array ( [- 50.0, -  100.0, - 500.0,  -  1e30 ] )
    np.testing.assert_array_equal(dB_dx(X),
                  np.full(X.shape,
      -1.0))



def test_dB_tends_to_one_minus_x_times_exp_minus_x_for_large_positive_x() -> None :
    X = np.array([100.0, 300.0, 500.0, 700.0])
    np.testing.assert_allclose (dB_dx (  X) , (  1.0 -  X  )  *   np.exp( -  X  ) , rtol = 1e-14 )


def test_B_underflows_to_zero_rather_than_overflowing( )   ->   None  :

    """x/expm1(x) overflows above x = 709. The form used here does not.

    docs/02-numerics.md says to return 0.0 above x = 80, which discards every
    value from B(80) = 1.44e-33 down to B(745) = 3.7e-321. Those are all
    representable, so we keep them.
    """
    assert B(80.0)==pytest.approx(1.443881e-33, rel = 1e-6)
    assert  B(  700.0)  > 0.0

    assert B(745.0)>0.0

    assert B(760.0)  ==0.0



def test_B_is_finite_across_the_whole_double_range()->None :

    yy=np.array([- 1e300,- 1e30,- 1e3,0.0,1e3,1e30,1e300])
    list=B(yy)
    assert np.all (  np.isfinite (  list))
def test_dB_is_finite_across_the_whole_double_range (  )  -> None  :
    X=np.array([-1e300,- 1e30,-1e3,0.0,1e3,1e30,1e300])
    resuult=dB_dx(X)
    assert np.all(np.isfinite(resuult))


def test_no_divide_overflow_or_invalid_warnings()  -> None  :


    """Underflow is deliberate and excluded. The other three are always bugs."""
    tmp2=np.array(PROBES+[- 1e300,1e300,709.0,710.0,745.0,760.0])
    with  np.errstate( divide  =  'raise',  over   = "raise",   invalid  =  'raise'  ) :
        B(tmp2)
        dB_dx(tmp2)

def  test_B_is_monotonically_decreasing (  )  ->   None  :
    xx = np.linspace(-100.0, 100.0, 2001)
    assert  np.all(np.diff (B(  xx  )  )  <  0.0 )



def test_B_is_positive_everywhere() ->None:
    xx= np.linspace(- 200.0, 200.0, 2001)


    assert np.all(B(xx)>0.0)

def test_dB_is_negative_everywhere() ->None :
    xx  = np.linspace (- 100.0 ,  100.0 ,   2001)
    assert np.all(dB_dx(xx ) < 0.0  )


def test_dB_never_becomes_positive_even_in_the_tails()-> None:

    xx = np.array(  [- 1e300 ,   -  500.0,   0.0,  500.0, 1e300 ]  )
    assert np.all( dB_dx(  xx )   <=  0.0 )


def test_B_accepts_a_python_float_and_returns_a_float() ->None:
    rsult  =  B(1.0)
    assert  isinstance(  rsult,   float)




def test_dB_accepts_a_python_float_and_returns_a_float() ->None:
    assert  isinstance(dB_dx(  1.0  ) ,  float  )




def test_B_preserves_input_shape( )   ->  None   :
    xx = np.linspace(-  5.0, 5.0, 12).reshape(3, 4)
    assert B(xx).shape == (3,4)

def  test_dB_preserves_input_shape ( )   ->  None  :
    xx  = np.linspace (-   5.0,  5.0,   12).reshape (  3,  4 )
    assert dB_dx(xx).shape == (3, 4)


def test_vectorized_matches_scalar_evaluation()-> None:
    xx =np.array(PROBES)
    next= B(xx)
    Scalar = np.array([B(float(idx2)) for idx2 in xx])
    np.testing.assert_array_equal(next, Scalar)




def test_vectorized_derivative_matches_scalar_evaluation()->None :
    X =np.array(PROBES)
    vecotr=dB_dx(X)
    Scalar  =   np.array( [  dB_dx(  float( vv  ) ) for vv in  X  ] )
    np.testing.assert_array_equal(vecotr, Scalar)

def test_dB_matches_complex_step_differentiation() -> None:
    '''The acceptance criterion, over the range where complex step is exact.'''
    wrost =0.0
    WorstAt =   0.0
    for X in PROBES:
        if  not COMPLEX_STEP_MIN_ABS_X  <= abs( X)  <=  COMPLEX_STEP_MAX_ABS_X  :
            continue
        data2 =relative_error(float(dB_dx(X)), dB_complex_step(X))
        if data2 > wrost :

            wrost, WorstAt =data2, X
    assert wrost<  1e-13, f"worst relative error {wrost:.3e} at x={WorstAt}"

def test_complex_step_reference_is_itself_accurate_where_it_is_used()->None :
    """Guards the guard.

    Complex step is not exact for B near the origin, so its usable range is
    restricted. This pins that restriction down so nobody later widens the
    range and spends a day debugging a correct implementation.
    """
    for X in(0.1,0.5,1.0,10.0,100.0,300.0,-0.1,-1.0,-100.0):


        err=relative_error(dB_complex_step(X),dB_reference(X))

        assert err<1e-14,f"reference itself is off by {err:.3e} at x={X}"




def test_complex_step_is_untrustworthy_below_the_documented_cutoff()->None:
    """Documents why COMPLEX_STEP_MIN_ABS_X exists, with a measurement.

    If this ever starts passing, complex step got better and the cutoff can be
    lowered. Until then it stands as the reason the range is restricted.
    """
    erorr = relative_error(dB_complex_step(1e-6),dB_reference(1e-6))
    assert erorr >  1e-13




def test_B_preserves_a_complex_dtype()  ->  None  :
    '''The continuity residual is verified by complex step, and B is in it.

    Without this, the Scharfetter-Gummel residual discards the imaginary part
    and the complex step Jacobian of every continuity block comes back as
    exactly zero, which reads as agreement rather than as breakage.
    '''
    gott =  B(np.array([0.5 +1e-20j, -  0.5+  1e-20j]))
    assert  np.iscomplexobj(gott)


def test_B_on_a_real_valued_complex_array_matches_the_real_branch()->None :
    """A zero imaginary part must not change the answer."""
    temp2  =np.array([-300.0, - 37.0, -  1.0, -  0.05, 0.0, 0.05, 1.0, 37.0, 300.0])
    gott=B(temp2.astype(np.complex128))


    np.testing.assert_allclose(
        np.asarray(  gott).real,   np.asarray(  B (temp2) ),   rtol  = 2e-15,   atol  =  0.0
    )




def test_B_on_complex_input_matches_the_independent_reference() -> None :

    """Checked against the closed form reference, which shares no branch."""
    dat  =  np.array(  [  - 300.0,  -  37.0,  -  1.0, -  0.05,   0.0,   0.05, 1.0 ,   37.0, 300.0] )
    Z=dat+1e-20j
    gott= np.asarray(B(Z))
    Expected=B_complex(Z)

    np.testing.assert_allclose(gott.real,Expected.real,rtol=2e-15,atol =0.0);  np.testing.assert_allclose(gott.imag, Expected.imag, rtol =2e-13, atol= 0.0)

def test_B_complex_does_not_overflow_in_the_positive_tail()->None:
    """The real branch avoids exp(x) past 710 and the complex one must too."""
    Got  =  np.asarray(B(  np.array( [700.0   +  1e-20j ]) )) [  0]

    assert math.isfinite(Got.real)
    assert math.isfinite (Got.imag  )

def test_complex_step_through_B_recovers_dB_dx_at_the_origin() ->  None :
    '''B'(0) = -1/2, the case the whole complex path exists to make work.'''
    ste= CS_STEP
    foo = np.asarray(B(np.array([complex(0.0,ste)]))) [0].imag /ste
    assert foo==pytest.approx(- 0.5, rel=  1e-14)


@pytest.mark.parametrize(  'x' ,  [- 300.0,   -  37.0, -  1.0,  -  0.1, 0.1 ,   1.0,  37.0,   300.0  ]  )




def test_complex_step_through_B_recovers_dB_dx(x:float)->None :
    """The Phase 3 harness path, end to end, over the trustworthy range."""
    Got= np.asarray(B(np.array([complex(x, CS_STEP)])))  [0].imag  /CS_STEP

    assert relative_error(Got,dB_reference(x))<1e-13
