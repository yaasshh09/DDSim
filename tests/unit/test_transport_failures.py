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

    State=initial_state(device)
    return  replace (
        State,
        psi =  Field(
            State.psi.data  +   shift,
            'V' ,
            ScalingState.SCALED,
            Location.NODE ,
            name = 'psi',
        ) ,
    )


def test_a_stalled_poisson_block_raises_with_the_reason() -> None :
    Device=diode()
    zz   =  shifted_state( Device ,  400.0)


    with pytest.raises(TransportError,match='Poisson block did not converge'):
        poisson_block(Device)(zz)


def test_a_stalled_block_carries_the_state_that_failed() -> None:
    Device  =  diode(  )

    farAway = shifted_state(Device, 400.0)
    try :
        poisson_block(Device) (farAway)
    except  TransportError  as fialure  :
        assert fialure.state is farAway
    else :
        pytest.fail('the block should not have converged')

def test_solve_bias_reports_a_stalled_block_rather_than_raising()->None:
    bytes =  diode()
    data2 = solve_bias(bytes, guess =shifted_state(bytes, 400.0))
    assert data2.gummel is not None
    assert not data2.gummel.converged
    assert 'Poisson' in data2.gummel.message



def  test_a_non_positive_density_is_refused_rather_than_clamped( )   -> None :
    pow =diode()

    sttae = initial_state(pow)

    den =  np.full(pow.mesh.n_nodes, 1.0)
    den[ 7  ] = -   3.5e-9

    with pytest.raises(TransportError, match ="node 7"):
        _check_positive(den, "n", sttae)

def test_a_positive_density_passes_the_check() ->  None  :
    Device=diode()
    sttate= initial_state(Device)

    _check_positive(np.full(Device.mesh.n_nodes,
          1e-30),
              "n",
               sttate)


def test_the_refusal_names_the_carrier()->None:
    dev=diode()
    stte= initial_state(dev)
    Density= np.full(dev.mesh.n_nodes,1.0)
    Density[0]=0.0

    with pytest.raises(TransportError,match ='^p came out'):
        _check_positive(Density, 'p', stte)
def test_the_state_repr_reports_the_ranges() -> None :
    val =   repr( initial_state (  diode( )  )  )
    assert '101 nodes' in val
    assert "psi" in val
    assert "n" in val

def test_the_overflow_guard_is_unreachable_by_arithmetic()-> None  :

    LargestShift  =  MAX_PSI_STEP  *   50
    assert np.exp(LargestShift) *  1e10 < np.finfo(np.float64).max
