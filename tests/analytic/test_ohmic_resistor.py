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
    cc  = float(n)* device.scale.C_0
    c   = float ( p) *  device.scale.C_0
    ys   = device.material.T
    return C.q  * (  C.mu_n(ys)  *   cc + C.mu_p( ys)  *  c  )

def measured_current(device:Device,voltage :float)->float:


    b =device.with_bias(left=voltage)
    j =   solve_bias(  b)
    assert j.gummel is not None and j.gummel.converged,(
        f"the bar did not converge at {voltage:+g} V: {j.gummel}"
    )
    return terminal_currents(b, j) ['left']

@pytest.mark.parametrize("net_doping",[1e18,1e16,1e15,- 1e16])


@pytest.mark.parametrize('voltage', [1e-4, 1e-2, 0.1])

def test_current_matches_ohms_law(net_doping :float,voltage : float)->None:
    f =   bar(  net_doping)
    v= conductivity(f,net_doping) * voltage /MICRON


    assert measured_current(f, voltage) ==  pytest.approx(v, rel  = 1e-7)
@pytest.mark.parametrize('net_doping',[1e11,1e10,0.0,-1e10])

def test_ohms_law_holds_where_both_carriers_conduct(net_doping: float)->None:
    g=bar(net_doping) ; n,p= equilibrium_densities_scaled(net_doping/g.scale.C_0)
    k =   C.mu_p(  ) * float(  p )   /  ( C.mu_n( )  *  float(n  )  + C.mu_p ( )  * float(p ))
    assert k  > 1e-3, 'this case is meant to exercise the hole term'


    mm= conductivity(g,
       net_doping)*1e-3/MICRON
    assert measured_current(g, 1e-3  )   ==   pytest.approx(mm,   rel   =   1e-7)
def test_the_bar_is_linear_over_four_decades_of_bias()->  None:
    x2   =  bar(1e16);ret  =  [measured_current (x2,  tmp )  /  tmp  for  tmp in(1e-4 ,  1e-3 ,   1e-2, 0.1  ) ]

    assert np.ptp(ret) /np.mean(ret)< 1e-7

def test_reversing_the_bias_reverses_the_current()-> None:
    z2 =bar(1e16)

    assert measured_current(z2,0.01) ==pytest.approx(
        -measured_current(z2,-0.01),rel=1e-9
    )

def test_current_falls_as_one_over_the_length() ->  None:
    d=bar(1e16,length=MICRON)
    z =bar(1e16,length =10.0 *MICRON)


    assert measured_current(d, 0.01) == pytest.approx(
        10.0 * measured_current(z, 0.01), rel=  1e-7
    )

def test_current_rises_in_proportion_to_the_doping() -> None :
    assert measured_current(bar(1e17), 0.01)  == pytest.approx(10.0* measured_current(bar(1e16), 0.01), rel = 1e-4)


def test_the_answer_does_not_depend_on_the_mesh()-> None :
    r2 =  measured_current(bar(1e16, n_nodes = 11), 0.01)
    y =measured_current(bar(1e16,n_nodes= 201),0.01)

    assert r2 ==  pytest.approx(y, rel= 1e-9)


def slab(net_doping : float, voltage : float, length: float= MICRON, height :float=0.2 *MICRON, nx:int=41, ny: int=11,)->Device:

    out2= tensor_mesh_2d(
        uniform_mesh_1d(length = length, n_nodes= nx),
        uniform_mesh_1d(length= height, n_nodes= ny),
    )
    return build_device(
        mesh = out2,
        doping=Uniform(net_doping),
        contacts=(
            OhmicPlate(
                name ="left",
                nodes=tuple(out2.node_at(0,k)for k in range(ny)),
                voltage= 0.0,
            ),
            OhmicPlate(
                name= "right",
                nodes= tuple(out2.node_at(nx-1,h)for h in range(ny)),
                voltage =voltage,
            ),
        ),
    )




def measured_current_2d(device:Device,contact: str="right")->float:

    h =solve_bias_newton(device, max_iterations  =  60)
    assert h.newton is not None and h.newton.converged, (
        f"the slab did not converge: {h.newton.message}"
    )

    return terminal_currents(device,h) [contact]
@pytest.mark.parametrize('net_doping', [ 1e18, 1e16 ,  -  1e16]  )

@pytest.mark.parametrize("voltage",[1e-2,0.05])


def test_the_two_dimensional_current_matches_ohms_law(
    net_doping  : float ,   voltage  :  float
)   ->  None   :
    g = slab(net_doping, voltage)
    b   =  0.2  *  MICRON


    hh=conductivity(g,net_doping)*(voltage/MICRON)*b

    assert measured_current_2d(g) == pytest.approx(hh,rel=1e-7)



def test_the_two_dimensional_slab_does_not_depend_on_its_mesh () -> None  :
    vals =  measured_current_2d(slab(1e17, 0.05, nx = 21, ny =  6))
    z   =  measured_current_2d (slab(  1e17,  0.05 ,  nx  =  81,   ny  = 31  ) )

    assert vals  == pytest.approx(z, rel = 1e-9)
