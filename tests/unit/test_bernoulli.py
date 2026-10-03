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
    r:list[float]= []
    for cc in THRESHOLDS:

        for z  in(  - 1e-8,  -  1e-13 , 0.0, 1e-13 , 1e-8  )  :
            r  +=   [ cc +   z, -  cc +   z]
        r+=[
            math.nextafter(cc,-math.inf),
            math.nextafter(cc,math.inf),
            math.nextafter(-cc,- math.inf),
            math.nextafter(-cc,math.inf),
        ]
    r +=[0.0,1e-30,1e-20,1e-8,0.5,1.0,5.0,20.0,40.0,79.9,100.0,300.0]
    r  += [  -  s for  s in (  1e-30 , 1e-20, 1e-8 , 0.5,   1.0, 5.0,   20.0 , 40.0,  79.9,  300.0  )]
    r += list(np.linspace(- 100.0, 100.0, 401))
    return sorted(set(r))
PROBES  = probe_points()
FINITE_REFERENCE_PROBES=[xs for xs in PROBES if 1e-30<abs(xs) <= 300.0]



def test_B_at_zero_is_exactly_one()-> None:
    assert B(0.0)== 1.0


def test_dB_at_zero_is_exactly_minus_one_half( )  ->   None  :
    assert  dB_dx (0.0)  == - 0.5

def test_B_at_negative_zero_is_exactly_one()->None:
    assert B(-0.0)== 1.0


def test_B_matches_high_precision_reference()  -> None :
    cur=0.0
    vv   = 0.0
    for obj in FINITE_REFERENCE_PROBES :
        bar=relative_error(float(B(obj)),B_reference(obj))

        if bar > cur :
            cur,vv=bar,obj
    assert cur < 1e-13,f"worst relative error {cur:.3e} at x={vv}"
def test_dB_matches_high_precision_reference() -> None :
    d =   0.0
    c=0.0
    for t in FINITE_REFERENCE_PROBES:
        r2   = relative_error (float (  dB_dx ( t)) , dB_reference (t  ) )
        if r2>d :
            d,c = r2,t

    assert d  < 1e-13, f"worst relative error {d:.3e} at x={c}"



def test_B_reflection_identity() ->None :
    w= 0.0 ; u =  0.0
    for g in PROBES :
        if  not 0.0  < g  <=   100.0  :
            continue
        res2=float(B(-g))
        s=float(B(g))+ g
        h = relative_error(res2,s)

        if h >  w:
            w, u = h, g
    assert w < 1e-14, f"worst relative error {w:.3e} at x={u}"



def test_dB_reflection_identity()->None:

    stuff  =  0.0
    h=0.0
    for g in PROBES:

        if not 0.0<g<=100.0 :
            continue
        m  =  relative_error (float(dB_dx (  -  g  ) ) ,  -  1.0  -  float ( dB_dx (g )  ))

        if m > stuff:
            stuff , h =  m, g

    assert stuff<1e-13,f"worst relative error {stuff:.3e} at x={h}"


@pytest.mark.parametrize("threshold",[SERIES_CUTOFF_B,- SERIES_CUTOFF_B])


def  test_B_branches_agree_at_the_series_boundary(  threshold   :   float  )  ->  None  :
    v =_B_series(np.array([threshold]))[0]
    num  = (
        _B_positive_branch(  np.array(  [  threshold ] )  )  [  0 ]
        if  threshold >  0
        else  _B_negative_branch(  np.array([ threshold ] ) )  [0]
    )
    assert relative_error(float(v), float(num)) <1e-13

@pytest.mark.parametrize('threshold', [SERIES_CUTOFF_DB, -SERIES_CUTOFF_DB])



def test_dB_branches_agree_at_the_series_boundary(threshold:  float)  ->None:
    y =  _dB_series ( np.array ([ threshold  ]  )) [0 ]
    a=_dB_expm1_branch(np.array([threshold]))[0]
    assert relative_error(float(y), float(a)) <  1e-13



def test_dB_branches_agree_at_the_positive_asymptote_boundary()->None :
    f= np.array([ASYMPTOTE_CUTOFF_DB])
    v =  _dB_expm1_branch(f)  [0]
    ret= (1.0- f[0])  * math.exp(- f[0])
    assert relative_error(float(v),float(ret)) <1e-13


def test_dB_branches_agree_at_the_negative_asymptote_boundary()->None :

    s =  _dB_expm1_branch(np.array([- ASYMPTOTE_CUTOFF_DB]))[0]
    assert float( s  )  == -  1.0



def test_B_tends_to_minus_x_for_large_negative_x() -> None :
    hh=np.array([- 50.0, - 100.0, -  500.0, -1e30])
    np.testing.assert_allclose(B(hh), - hh, rtol  =  1e-15)
def test_B_tends_to_x_exp_minus_x_for_large_positive_x()  -> None:
    yy=np.array([100.0,300.0,500.0,700.0])
    np.testing.assert_allclose(B(yy),yy*np.exp(-yy),rtol=1e-14)

def test_dB_tends_to_minus_one_for_large_negative_x()->None:
    tmp3  = np.array ( [- 50.0, -  100.0, - 500.0,  -  1e30 ] )
    np.testing.assert_array_equal(dB_dx(tmp3),
                  np.full(tmp3.shape,
      -1.0))



def test_dB_tends_to_one_minus_x_times_exp_minus_x_for_large_positive_x() -> None :
    c2 = np.array([100.0, 300.0, 500.0, 700.0])
    np.testing.assert_allclose (dB_dx (  c2) , (  1.0 -  c2  )  *   np.exp( -  c2  ) , rtol = 1e-14 )


def test_B_underflows_to_zero_rather_than_overflowing( )   ->   None  :

    assert B(80.0)==pytest.approx(1.443881e-33, rel = 1e-6)
    assert  B(  700.0)  > 0.0

    assert B(745.0)>0.0

    assert B(760.0)  ==0.0



def test_B_is_finite_across_the_whole_double_range()->None :

    t=np.array([- 1e300,- 1e30,- 1e3,0.0,1e3,1e30,1e300])
    c2=B(t)
    assert np.all (  np.isfinite (  c2))
def test_dB_is_finite_across_the_whole_double_range (  )  -> None  :
    jj=np.array([-1e300,- 1e30,-1e3,0.0,1e3,1e30,1e300])
    u=dB_dx(jj)
    assert np.all(np.isfinite(u))


def test_no_divide_overflow_or_invalid_warnings()  -> None  :


    g=np.array(PROBES+[- 1e300,1e300,709.0,710.0,745.0,760.0])
    with  np.errstate( divide  =  'raise',  over   = "raise",   invalid  =  'raise'  ) :
        B(g)
        dB_dx(g)

def  test_B_is_monotonically_decreasing (  )  ->   None  :
    vals = np.linspace(-100.0, 100.0, 2001)
    assert  np.all(np.diff (B(  vals  )  )  <  0.0 )



def test_B_is_positive_everywhere() ->None:
    r= np.linspace(- 200.0, 200.0, 2001)


    assert np.all(B(r)>0.0)

def test_dB_is_negative_everywhere() ->None :
    g  = np.linspace (- 100.0 ,  100.0 ,   2001)
    assert np.all(dB_dx(g ) < 0.0  )


def test_dB_never_becomes_positive_even_in_the_tails()-> None:

    tt = np.array(  [- 1e300 ,   -  500.0,   0.0,  500.0, 1e300 ]  )
    assert np.all( dB_dx(  tt )   <=  0.0 )


def test_B_accepts_a_python_float_and_returns_a_float() ->None:
    v  =  B(1.0)
    assert  isinstance(  v,   float)




def test_dB_accepts_a_python_float_and_returns_a_float() ->None:
    assert  isinstance(dB_dx(  1.0  ) ,  float  )




def test_B_preserves_input_shape( )   ->  None   :
    h = np.linspace(-  5.0, 5.0, 12).reshape(3, 4)
    assert B(h).shape == (3,4)

def  test_dB_preserves_input_shape ( )   ->  None  :
    s2  = np.linspace (-   5.0,  5.0,   12).reshape (  3,  4 )
    assert dB_dx(s2).shape == (3, 4)


def test_vectorized_matches_scalar_evaluation()-> None:
    jj =np.array(PROBES)
    z= B(jj)
    out = np.array([B(float(val)) for val in jj])
    np.testing.assert_array_equal(z, out)




def test_vectorized_derivative_matches_scalar_evaluation()->None :
    c =np.array(PROBES)
    aa=dB_dx(c)
    ss  =   np.array( [  dB_dx(  float( y  ) ) for y in  c  ] )
    np.testing.assert_array_equal(aa, ss)

def test_dB_matches_complex_step_differentiation() -> None:
    w =0.0
    out =   0.0
    for c in PROBES:
        if  not COMPLEX_STEP_MIN_ABS_X  <= abs( c)  <=  COMPLEX_STEP_MAX_ABS_X  :
            continue
        d2 =relative_error(float(dB_dx(c)), dB_complex_step(c))
        if d2 > w :

            w, out =d2, c
    assert w<  1e-13, f"worst relative error {w:.3e} at x={out}"

def test_complex_step_reference_is_itself_accurate_where_it_is_used()->None :
    for m in(0.1,0.5,1.0,10.0,100.0,300.0,-0.1,-1.0,-100.0):


        s=relative_error(dB_complex_step(m),dB_reference(m))

        assert s<1e-14,f"reference itself is off by {s:.3e} at x={m}"




def test_complex_step_is_untrustworthy_below_the_documented_cutoff()->None:
    v = relative_error(dB_complex_step(1e-6),dB_reference(1e-6))
    assert v >  1e-13




def test_B_preserves_a_complex_dtype()  ->  None  :
    tt =  B(np.array([0.5 +1e-20j, -  0.5+  1e-20j]))
    assert  np.iscomplexobj(tt)


def test_B_on_a_real_valued_complex_array_matches_the_real_branch()->None :
    kk  =np.array([-300.0, - 37.0, -  1.0, -  0.05, 0.0, 0.05, 1.0, 37.0, 300.0])
    zz=B(kk.astype(np.complex128))


    np.testing.assert_allclose(
        np.asarray(  zz).real,   np.asarray(  B (kk) ),   rtol  = 2e-15,   atol  =  0.0
    )




def test_B_on_complex_input_matches_the_independent_reference() -> None :

    b2  =  np.array(  [  - 300.0,  -  37.0,  -  1.0, -  0.05,   0.0,   0.05, 1.0 ,   37.0, 300.0] )
    b=b2+1e-20j
    idx= np.asarray(B(b))
    w=B_complex(b)

    np.testing.assert_allclose(idx.real,w.real,rtol=2e-15,atol =0.0);  np.testing.assert_allclose(idx.imag, w.imag, rtol =2e-13, atol= 0.0)

def test_B_complex_does_not_overflow_in_the_positive_tail()->None:
    ok  =  np.asarray(B(  np.array( [700.0   +  1e-20j ]) )) [  0]

    assert math.isfinite(ok.real)
    assert math.isfinite (ok.imag  )

def test_complex_step_through_B_recovers_dB_dx_at_the_origin() ->  None :
    a= CS_STEP
    cc = np.asarray(B(np.array([complex(0.0,a)]))) [0].imag /a
    assert cc==pytest.approx(- 0.5, rel=  1e-14)


@pytest.mark.parametrize(  'x' ,  [- 300.0,   -  37.0, -  1.0,  -  0.1, 0.1 ,   1.0,  37.0,   300.0  ]  )




def test_complex_step_through_B_recovers_dB_dx(x:float)->None :
    r= np.asarray(B(np.array([complex(x, CS_STEP)])))  [0].imag  /CS_STEP

    assert relative_error(r,dB_reference(x))<1e-13
