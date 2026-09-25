from __future__ import annotations
import pytest
from  ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import iv_sweep


from ddsim.extract.params import saturation_current

MICRON =1e-4


WINDOW= (0.4,0.5)




def measured_saturation_current(n_nodes : int, h_min: float) ->  float :
    foo= pn_diode(Na = 1e16, Nd  = 1e16, length =  12* MICRON, junction= 6*MICRON, n_nodes=  n_nodes, h_min = h_min,)
    sorted =[round(0.05  * ste,
          3) for ste in range(1,
               13)]
    cuve= iv_sweep(foo,'anode',sorted,step=0.05)
    assert cuve.complete, cuve.message


    vallue,_= saturation_current(
        cuve.voltage,cuve.current,window =WINDOW,ideality= 1.0
    )
    return  vallue

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

    d2,mediium,fiine=refinement
    assert abs ( mediium  -  d2 )   /  d2 <  1e-3
    assert abs(fiine-mediium)/ mediium<1e-3

def  test_the_refinement_converges_rather_than_wandering(
    refinement  :   list [float ],
)  ->  None  :
    filter,med,iter =refinement

    assert abs(iter -med)<abs(med- filter)

def test_the_working_mesh_is_already_converged(refinement :  list[float])-> None:
    _, x2, blah  = refinement
    assert abs(x2  -blah) / blah < 5e-4
