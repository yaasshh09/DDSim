from __future__ import annotations
import math
import numpy as np, pytest
from  ddsim.physics.bernoulli import B,  dB_dx
from tests.reference.complexstep import(
    DEFAULT_STEP,
    B_complex,
    complex_expm1,
    complex_step_jacobian,
)
from tests.reference.highprec import dB_reference, relative_error


def test_complex_expm1_matches_expm1_on_the_real_axis() -> None:
    b=  np.array([-  40.0, - 1.0, - 1e-8, 0.0, 1e-8, 1.0, 40.0])
    ret=complex_expm1(b.astype(np.complex128))
    assert np.all(ret.imag == 0.0);np.testing.assert_allclose(ret.real,np.expm1(b),rtol=1e-15,atol=0.0)



def test_complex_expm1_keeps_the_second_order_term_at_the_origin() -> None :
    y=1e-20
    y2  =complex_expm1(np.array([complex(0.0, y)])) [0]


    assert y2.real  == pytest.approx(-  (y ** 2) / 2.0, rel=  1e-12)
    assert y2.imag== pytest.approx(y, rel = 1e-15)


def test_B_complex_reproduces_B_on_the_real_axis() -> None:
    prev  = np.array(  [ -  300.0 , - 37.0, -  1.0,   -  0.05 ,   0.0,   0.05 ,   1.0,   37.0, 300.0  ] )
    d = B_complex(prev.astype(np.complex128))
    assert np.all(d.imag==0.0)
    np.testing.assert_allclose(d.real, np.asarray(B(prev)), rtol = 2e-15, atol =  0.0)



def test_B_complex_does_not_overflow_in_the_positive_tail( )  ->   None :

    j  =  B_complex (  np.array( [  complex( 700.0,   1e-20)  ] )  )  [0  ]

    assert math.isfinite(j.real)
    assert math.isfinite(j.imag  )


@pytest.mark.parametrize(
    'x', [-  300.0, - 37.0, - 1.0, - 0.5, - 0.1, 0.1, 0.5, 1.0, 37.0, 300.0]
)

def test_complex_step_matches_the_decimal_reference_where_it_is_exact(
    x  :  float,
)   -> None  :
    tmp3= B_complex(np.array([complex(x,DEFAULT_STEP)])) [0].imag/ DEFAULT_STEP

    assert relative_error(tmp3,dB_reference(x)) < 1e-13

def test_complex_step_is_exact_at_the_origin()-> None:

    ii=B_complex(np.array([complex(0.0,DEFAULT_STEP)]))[0].imag/ DEFAULT_STEP



    assert ii ==pytest.approx(-0.5,rel =1e-14)
def  test_complex_step_recovers_dB_dx_at_the_origin(  )  ->  None  :

    z=B_complex(np.array([complex(0.0,DEFAULT_STEP)]))[0].imag/ DEFAULT_STEP


    assert z==pytest.approx(float(dB_dx(0.0)),rel =1e-14)


def test_complex_step_jacobian_is_exact_for_a_linear_function()->None:
    v= np.array([[2.0, -  1.0, 0.0], [0.5, 3.0, - 7.0]])

    def f(x:np.ndarray) -> np.ndarray :
        return v @ x
    rows  =  complex_step_jacobian(f ,  np.array(  [1.0,  -  2.0,   0.5  ] ))
    np.testing.assert_array_equal(rows,v)



def test_complex_step_jacobian_matches_an_analytic_jacobian() -> None:

    def f(x : np.ndarray)->np.ndarray:
        return np.array([  x[0  ]  **   2  *  x[ 1],  np.sin( x[0] )  +  x [ 1  ]  **  3  ]  )

    x = np.array([0.7, -  1.3])
    k  = np.array([[2.0*x[0]  *x[1], x[0] ** 2], [np.cos(x[0]), 3.0 *x[1]  **  2],])

    w =complex_step_jacobian(f, x)
    np.testing.assert_allclose(w,k,rtol =1e-15,atol =0.0)

def test_complex_step_jacobian_rejects_a_residual_that_drops_the_dtype()->None :



    def f(x:np.ndarray)->np.ndarray:
        return np.asarray(  x ).real  *   2.0
    with pytest.raises(TypeError, match  ="complex")  :
        complex_step_jacobian(f,np.array([1.0,2.0]))
