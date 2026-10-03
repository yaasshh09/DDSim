from __future__ import annotations
import pytest
from  ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import iv_sweep
from ddsim.extract.params import saturation_current

MICRON =1e-4


WINDOW= (0.4,0.5)




def measured_saturation_current(n_nodes : int, h_min: float) ->  float :
    b= pn_diode(Na = 1e16, Nd  = 1e16, length =  12* MICRON, junction= 6*MICRON, n_nodes=  n_nodes, h_min = h_min,)
    w =[round(0.05  * val2,
          3) for val2 in range(1,
               13)]
    e= iv_sweep(b,'anode',w,step=0.05)
    assert e.complete, e.message


    prev,_= saturation_current(
        e.voltage,e.current,window =WINDOW,ideality= 1.0
    )
    return  prev

@pytest.fixture(scope="module")


def refinement() -> list[float]:

    return[
        measured_saturation_current(  101 ,  1e-6),
        measured_saturation_current(201 ,  5e-7  ),
        measured_saturation_current (  401,   2e-7 ) ,
    ]

def test_the_saturation_current_stops_moving_under_refinement(
    refinement : list[float],
) -> None:

    c,s,zz=refinement
    assert abs ( s  -  c )   /  c <  1e-3
    assert abs(zz-s)/ s<1e-3

def  test_the_refinement_converges_rather_than_wandering(
    refinement  :   list [float ],
)  ->  None  :
    info,v,m =refinement

    assert abs(m -v)<abs(v- info)

def test_the_working_mesh_is_already_converged(refinement :  list[float])-> None:
    _, kk, s  = refinement
    assert abs(kk  -s) / s < 5e-4
