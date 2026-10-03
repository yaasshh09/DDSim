from __future__ import annotations
from dataclasses import replace
import numpy as np, pytest
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.equilibrium import MAX_PSI_STEP;from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import(TransportError, _check_positive, initial_state, poisson_block, solve_bias,)
MICRON= 1e-4



def diode() :
    return pn_diode(
        Na  = 1e16,
        Nd =  1e16,
        length  =  12  * MICRON,
        junction  = 6*  MICRON,
        n_nodes = 101,
        h_min =5e-7,
    )



def shifted_state(device, shift :float) :

    i=initial_state(device)
    return  replace (
        i,
        psi =  Field(
            i.psi.data  +   shift,
            'V' ,
            ScalingState.SCALED,
            Location.NODE ,
            name = 'psi',
        ) ,
    )


def test_a_stalled_poisson_block_raises_with_the_reason() -> None :
    dat=diode()
    v   =  shifted_state( dat ,  400.0)


    with pytest.raises(TransportError,match='Poisson block did not converge'):
        poisson_block(dat)(v)


def test_a_stalled_block_carries_the_state_that_failed() -> None:
    val  =  diode(  )

    buf = shifted_state(val, 400.0)
    try :
        poisson_block(val) (buf)
    except  TransportError  as v  :
        assert v.state is buf
    else :
        pytest.fail('the block should not have converged')

def test_solve_bias_reports_a_stalled_block_rather_than_raising()->None:
    info =  diode()
    t = solve_bias(info, guess =shifted_state(info, 400.0))
    assert t.gummel is not None
    assert not t.gummel.converged
    assert 'Poisson' in t.gummel.message



def  test_a_non_positive_density_is_refused_rather_than_clamped( )   -> None :
    thing =diode()

    d = initial_state(thing)

    tmp2 =  np.full(thing.mesh.n_nodes, 1.0)
    tmp2[ 7  ] = -   3.5e-9

    with pytest.raises(TransportError, match ="node 7"):
        _check_positive(tmp2, "n", d)

def test_a_positive_density_passes_the_check() ->  None  :
    t=diode()
    a= initial_state(t)

    _check_positive(np.full(t.mesh.n_nodes,
          1e-30),
              "n",
               a)


def test_the_refusal_names_the_carrier()->None:
    w=diode()
    x= initial_state(w)
    flag= np.full(w.mesh.n_nodes,1.0)
    flag[0]=0.0

    with pytest.raises(TransportError,match ='^p came out'):
        _check_positive(flag, 'p', x)
def test_the_state_repr_reports_the_ranges() -> None :
    m2 =   repr( initial_state (  diode( )  )  )
    assert '101 nodes' in m2
    assert "psi" in m2
    assert "n" in m2

def test_the_overflow_guard_is_unreachable_by_arithmetic()-> None  :

    num  =  MAX_PSI_STEP  *   50
    assert np.exp(num) *  1e10 < np.finfo(np.float64).max
