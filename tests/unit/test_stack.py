from __future__ import annotations
import numpy as np;import pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.stack import DOPING_RANGE,Region,stack

MICRON = 1e-4




def net_doping(device) -> np.ndarray :
    return device.doping(  device.mesh.x  )



def test_the_default_stack_is_the_phase_2_diode_bit_for_bit() -> None  :
    d,ii=stack(),pn_diode()
    np.testing.assert_array_equal(d.mesh.x,ii.mesh.x)


    np.testing.assert_array_equal(net_doping(d),net_doping(ii))
def test_a_stack_has_a_contact_at_each_end() ->None:


    d =  stack(left_voltage= 0.3, right_voltage =-  0.1)
    s, x  = d.contacts
    assert(s.name, s.nodes, s.voltage)== ("left", (0, ), 0.3)

    e= d.mesh.n_nodes- 1
    assert(x.name,x.nodes,x.voltage)==("right",(e,),-0.1)




def test_each_region_holds_its_own_doping() ->None :
    b   =   (
        Region( 'p' ,   0.2  *  MICRON,  1e18 ),
        Region ("n",   1.0 *  MICRON,   1e14),
        Region( "n", 0.2  *  MICRON,   1e18  ),
    )
    s=stack(b)
    z = s.mesh.x; w=net_doping(s)
    assert np.all(w[z< 0.2*MICRON]== - 1e18)
    assert np.all(w[(z>= 0.2  * MICRON) &  (z <1.2 * MICRON)] == 1e14)
    assert  np.all(  w[ z  >=  1.2  *   MICRON ]  == 1e18)



@pytest.mark.parametrize("regions", [(Region("p",0.2*MICRON,1e18), Region("n",1.0*MICRON,1e14), Region('n',0.2 *MICRON,1e18),), (Region("n",0.3 * MICRON,1e18), Region('p',0.2 *MICRON,1e17), Region("n",0.5 *MICRON,1e16),), (Region("p",0.1*MICRON,1e17), Region('n',0.1 * MICRON,1e17), Region("p",1.0*MICRON,1e15), Region("n",0.1* MICRON,1e17),),], ids =['pin',"npn",'four regions'],)
def test_every_junction_is_a_node_at_the_finest_spacing(regions)  ->  None :
    rows =   1e-7
    s   =  stack(regions ,   h_min  =  rows)
    f  =  s.mesh
    for u in np.cumsum([w.length for w in regions])[:-1]:
        v  = int(np.argmin(np.abs(f.x -u)))
        assert f.x[v] ==  u
        assert max(f.h[v -1],f.h[v]) <1.5* rows
    c= f.h[1:] / f.h[:-1]
    assert max(c.max(),(1.0/c).max()) <=1.5


def test_the_node_count_is_the_one_asked_for() ->None :
    a   =   (Region("n",  0.3 *   MICRON,  1e18  ), Region(  "p",  0.2  *  MICRON,   1e17  ), Region("n",  0.5   *  MICRON,  1e16 ) ,)
    assert stack(a,
      n_nodes= 301).mesh.n_nodes==301




def test_more_nodes_never_breaks_a_stack_with_a_short_segment() ->  None :
    d =(Region("p", 10* MICRON, 1e15), Region("n", 0.01 * MICRON, 1e17), Region("p", 0.01 *  MICRON, 1e17), Region("n", 10*MICRON, 1e15),)
    for u in(101, 201, 401)  :
        assert stack(d,n_nodes = u).mesh.n_nodes==u

def test_more_nodes_than_the_stack_holds_at_h_min_is_refused()  ->  None:
    j  = ( Region(  "p",   5e-6 ,   1e17) , Region(  "n" ,  5e-6 ,  1e17  )  )
    with pytest.raises(ValueError, match = "n_nodes") as a:
        stack(j, n_nodes  =  1001, h_min =1e-7)
    assert  'h_min' in str(  a.value  )


def test_a_region_the_graded_mesh_leaves_too_few_nodes_in_is_refused (  )   ->  None :

    it=(
        Region("n",MICRON,1e16),
        Region("p",3e-7,1e18),
        Region("n",MICRON,1e16),
    )
    with pytest.raises(ValueError, match= "region 2.*holds 1 mesh nodes") :
        stack(it,  h_min  = 1e-7 )


def test_a_boundary_between_two_equal_regions_is_not_graded_towards()->None :
    buf= (Region('p', 0.2* MICRON, 1e16), Region("p", 0.3 *MICRON, 1e16), Region('n', 0.5 * MICRON, 1e16),)

    np.testing.assert_array_equal(stack(buf).mesh.x, pn_diode().mesh.x)


def test_a_stack_with_no_junction_is_refused()->None:
    with  pytest.raises(  ValueError, match   =  'no junction' )  :
        stack((Region('n', MICRON, 1e16), ))


def test_a_stack_of_equal_regions_has_no_junction_either() -> None :
    with pytest.raises(ValueError,match = "no junction") :
        stack((Region('n',MICRON,1e16),Region('n',MICRON,1e16)))



def test_an_empty_stack_is_refused()->None:
    with  pytest.raises( ValueError,  match   =   "at least two regions" ) :
        stack (  ())



@pytest.mark.parametrize("concentration", [1e13, 1e20, 0.0])



def test_a_doping_outside_the_model_range_is_refused(  concentration)   ->   None   :
    nxt,cur= DOPING_RANGE

    m = (Region('p',MICRON,1e16),Region('n',MICRON,concentration))
    with  pytest.raises( ValueError,   match  =   "region 2")   as a :
        stack(m)
    assert f"{nxt:g}" in str(a.value) and f"{cur:g}" in str(a.value)

    assert 'references/physics.md' in str(a.value)



def test_the_ends_of_the_doping_range_are_accepted() -> None:
    obj,  x2   = DOPING_RANGE
    stack((Region("p", MICRON, x2), Region('n', MICRON, obj)), n_nodes  = 301)

def test_a_dopant_that_is_not_n_or_p_is_refused() ->None:
    with pytest.raises(ValueError,match = "region 1.*'i'"):
        stack((Region('i',MICRON,1e16),Region('n',MICRON,1e16)))

def test_a_region_with_no_length_is_refused() ->None:
    with pytest.raises(ValueError,match= "region 2.*length") :
        stack((Region("p", MICRON, 1e16), Region('n', 0.0, 1e16)))




def test_a_region_shorter_than_the_mesh_resolves_is_refused() -> None:
    b  =   (
        Region('n', MICRON,   1e16 ),
        Region("p" ,   0.5e-7,   1e18 ),
        Region( "n",  MICRON,  1e16 ) ,
    )
    with pytest.raises(ValueError,match= "region 2")as yy:
        stack(b,h_min=1e-7)
    assert  'h_min'  in str (yy.value) and  'n_nodes'  in str(yy.value)
