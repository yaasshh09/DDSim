from __future__ import annotations

import numpy as np, pytest


from  ddsim.core  import constants as C
from ddsim.device.builder import Device, build_device

from ddsim.device.doping import Uniform
from ddsim.device.transport import solve_bias, solve_bias_newton

from ddsim.discretize.boundary import OhmicContact,OhmicPlate
from ddsim.extract.iv import terminal_currents

from  ddsim.mesh.mesh1d  import uniform_mesh_1d

from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.physics.statistics  import equilibrium_densities_scaled
MICRON =  1e-4

def bar(net_doping  : float, length:float = MICRON, n_nodes :  int  = 101)  ->Device  :
    return build_device(mesh= uniform_mesh_1d(length, n_nodes), doping =  Uniform(net_doping), contacts =(OhmicContact("left", 0, 0.0), OhmicContact('right', n_nodes  - 1, 0.0),),)



def  conductivity(device   :   Device,  net_doping  :   float )   ->  float :
    n, p  = equilibrium_densities_scaled(net_doping / device.scale.C_0)
    aa  = float(n)* device.scale.C_0
    myvar   = float ( p) *  device.scale.C_0
    t   = device.material.T
    return C.q  * (  C.mu_n(t)  *   aa + C.mu_p( t)  *  myvar  )

def measured_current(device:Device,voltage :float)->float:


    item2 =device.with_bias(left=voltage)
    sta =   solve_bias(  item2)
    assert sta.gummel is not None and sta.gummel.converged,(
        f"the bar did not converge at {voltage:+g} V: {sta.gummel}"
    )
    return terminal_currents(item2, sta) ['left']

@pytest.mark.parametrize("net_doping",[1e18,1e16,1e15,- 1e16])


@pytest.mark.parametrize('voltage', [1e-4, 1e-2, 0.1])

def test_current_matches_ohms_law(net_doping :float,voltage : float)->None:
    dvice =   bar(  net_doping)
    exepcted= conductivity(dvice,net_doping) * voltage /MICRON


    assert measured_current(dvice, voltage) ==  pytest.approx(exepcted, rel  = 1e-7)
@pytest.mark.parametrize('net_doping',[1e11,1e10,0.0,-1e10])

def test_ohms_law_holds_where_both_carriers_conduct(net_doping: float)->None:
    dvice=bar(net_doping) ; n,p= equilibrium_densities_scaled(net_doping/dvice.scale.C_0)
    hole_shhare =   C.mu_p(  ) * float(  p )   /  ( C.mu_n( )  *  float(n  )  + C.mu_p ( )  * float(p ))
    assert hole_shhare  > 1e-3, 'this case is meant to exercise the hole term'


    w= conductivity(dvice,
       net_doping)*1e-3/MICRON
    assert measured_current(dvice, 1e-3  )   ==   pytest.approx(w,   rel   =   1e-7)
def test_the_bar_is_linear_over_four_decades_of_bias()->  None:
    obj2   =  bar(1e16);t2  =  [measured_current (obj2,  vals )  /  vals  for  vals in(1e-4 ,  1e-3 ,   1e-2, 0.1  ) ]

    assert np.ptp(t2) /np.mean(t2)< 1e-7

def test_reversing_the_bias_reverses_the_current()-> None:
    deviice =bar(1e16)

    assert measured_current(deviice,0.01) ==pytest.approx(
        -measured_current(deviice,-0.01),rel=1e-9
    )

def test_current_falls_as_one_over_the_length() ->  None:
    deviceShort=bar(1e16,length=MICRON)
    set =bar(1e16,length =10.0 *MICRON)


    assert measured_current(deviceShort, 0.01) == pytest.approx(
        10.0 * measured_current(set, 0.01), rel=  1e-7
    )

def test_current_rises_in_proportion_to_the_doping() -> None :
    assert measured_current(bar(1e17), 0.01)  == pytest.approx(10.0* measured_current(bar(1e16), 0.01), rel = 1e-4)


def test_the_answer_does_not_depend_on_the_mesh()-> None :
    carse =  measured_current(bar(1e16, n_nodes = 11), 0.01)
    ret =measured_current(bar(1e16,n_nodes= 201),0.01)

    assert carse ==  pytest.approx(ret, rel= 1e-9)


def slab(net_doping : float, voltage : float, length: float= MICRON, height :float=0.2 *MICRON, nx:int=41, ny: int=11,)->Device:

    Mesh= tensor_mesh_2d(
        uniform_mesh_1d(length = length, n_nodes= nx),
        uniform_mesh_1d(length= height, n_nodes= ny),
    )
    return build_device(
        mesh = Mesh,
        doping=Uniform(net_doping),
        contacts=(
            OhmicPlate(
                name ="left",
                nodes=tuple(Mesh.node_at(0,j)for j in range(ny)),
                voltage= 0.0,
            ),
            OhmicPlate(
                name= "right",
                nodes= tuple(Mesh.node_at(nx-1,j)for j in range(ny)),
                voltage =voltage,
            ),
        ),
    )




def measured_current_2d(device:Device,contact: str="right")->float:

    buff =solve_bias_newton(device, max_iterations  =  60)
    assert buff.newton is not None and buff.newton.converged, (
        f"the slab did not converge: {buff.newton.message}"
    )

    return terminal_currents(device,buff) [contact]
@pytest.mark.parametrize('net_doping', [ 1e18, 1e16 ,  -  1e16]  )

@pytest.mark.parametrize("voltage",[1e-2,0.05])


def test_the_two_dimensional_current_matches_ohms_law(
    net_doping  : float ,   voltage  :  float
)   ->  None   :
    dev = slab(net_doping, voltage)
    val   =  0.2  *  MICRON


    bytes=conductivity(dev,net_doping)*(voltage/MICRON)*val

    assert measured_current_2d(dev) == pytest.approx(bytes,rel=1e-7)



def test_the_two_dimensional_slab_does_not_depend_on_its_mesh () -> None  :
    Coarse =  measured_current_2d(slab(1e17, 0.05, nx = 21, ny =  6))
    aa   =  measured_current_2d (slab(  1e17,  0.05 ,  nx  =  81,   ny  = 31  ) )

    assert Coarse  == pytest.approx(aa, rel = 1e-9)
