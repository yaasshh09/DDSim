from __future__ import annotations
from  types import  SimpleNamespace
import numpy as np ; import pytest
from  ddsim.device.builder  import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.transport import TransportModels, solve_bias
from ddsim.discretize.boundary import OhmicContact
from  ddsim.extract  import  iv
from ddsim.extract.iv import current_densities,node_current_density;from ddsim.mesh.mesh1d import  uniform_mesh_1d
from ddsim.physics.recombination  import NoRecombination
from tests.invariant.test_current_continuity import solved as solved_1d
from tests.invariant.test_current_continuity_2d import (
    capped_diode_2d ,
    cut_currents,
    diode_2d,
    solved ,
)

@pytest.fixture(scope ="module")




def  plain( )  :
    d2 = diode_2d();  el, u =solved(d2)

    return d2, el, u


def test_on_a_diode_uniform_in_y_the_current_points_along_x(plain) ->None:

    out2, d, r=  plain
    a, bb = node_current_density(  out2,   d, r )


    assert np.max(np.abs(bb)) <1e-9 *  np.max(np.abs(a))


def test_every_row_carries_the_same_x_current(plain)  ->None:

    z,  obj,  c   =   plain

    r, _ = node_current_density(z, obj, c)
    row= r.reshape(z.mesh.ny,z.mesh.nx)

    np.testing.assert_allclose( row ,  np.broadcast_to(  row[0  ], row.shape ) ,   rtol =   1e-9)

def test_a_column_of_node_current_integrates_to_the_cut_current (plain)  -> None  :
    w2,h,mm=plain
    d,  _  = node_current_density(w2 ,  h ,   mm)
    u= w2.mesh
    flag =np.asarray(u.y_axis.volume)
    v2= d.reshape(u.ny,u.nx)[:,u.nx//2]

    r =cut_currents(w2,h,mm) [u.nx// 2]
    assert  float( np.sum(v2  * flag  ))  == pytest.approx( r ,  rel =  1e-6)

def test_a_vertical_edge_lands_on_the_nodes_above_and_below_it(monkeypatch) ->  None :

    g  =diode_2d()
    rr = g.mesh
    flag, t2= 3, 2

    lst=rr.n_horizontal + t2 * rr.nx+ flag



    w =  np.zeros(rr.n_edges)
    w [  lst  ]  =  1.0
    h =SimpleNamespace(data= w)
    b =SimpleNamespace(data=np.zeros(rr.n_edges))
    monkeypatch.setattr(iv,'current_densities',lambda*args : (h,b))
    u,c =iv.node_current_density(g,None)
    assert  np.all (u  ==   0.0  )
    assert set(np.flatnonzero(c)) =={rr.node_at(flag,t2),rr.node_at(flag,t2+ 1)}

def test_on_a_1d_diode_every_node_carries_the_edge_current()-> None:

    ss,tt,y2= solved_1d(0.5,NoRecombination())
    m,vals= current_densities(ss,tt,y2)
    g ,   t   =  node_current_density(ss , tt, y2  )
    assert  g.shape == t.shape == (  ss.mesh.n_nodes,   )
    assert np.all(t== 0.0)

    np.testing.assert_allclose (  g , np.mean(m.data   +   vals.data ) ,  rtol   =  1e-6)
def test_the_oxide_carries_no_current()-> None :
    t  =  capped_diode_2d(  )
    w,  out   = solved( t)
    nxt,r =node_current_density(t,w,out)
    j =t.regions.oxide_nodes
    assert np.all(nxt[j]  == 0.0)
    assert np.all(r[j]==0.0)
def test_a_degenerate_junction_at_rest_carries_no_current() ->None :
    xx= 201
    k = build_device(mesh=uniform_mesh_1d(length =  1e-4, n_nodes=  xx), doping =abrupt_junction(Na=  1e17, Nd= 1e20, position =  0.5e-4), contacts = (OhmicContact(name  ='anode', node = 0, voltage =0.0), OhmicContact(name  = 'cathode', node = xx -  1, voltage= 0.0),), degenerate = True,)
    cur =   TransportModels.for_device (k  )
    mm= solve_bias(k,
          models = cur)

    aa, xs  =  current_densities( k ,
                 mm,
                cur )
    assert np.max(np.abs(aa.data + xs.data)) <1e-3
