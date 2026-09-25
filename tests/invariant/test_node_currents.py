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
    dev = diode_2d();  State, Models =solved(dev)

    return dev, State, Models


def test_on_a_diode_uniform_in_y_the_current_points_along_x(plain) ->None:

    any, staate, modeels=  plain
    chr, jy = node_current_density(  any,   staate, modeels )


    assert np.max(np.abs(jy)) <1e-9 *  np.max(np.abs(chr))


def test_every_row_carries_the_same_x_current(plain)  ->None:

    dvice,  hash,  moels   =   plain

    jx, _ = node_current_density(dvice, hash, moels)
    rws= jx.reshape(dvice.mesh.ny,dvice.mesh.nx)

    np.testing.assert_allclose( rws ,  np.broadcast_to(  rws[0  ], rws.shape ) ,   rtol =   1e-9)

def test_a_column_of_node_current_integrates_to_the_cut_current (plain)  -> None  :
    dev,sta,mod=plain
    Jxx,  _  = node_current_density(dev ,  sta ,   mod)
    mes= dev.mesh
    hights =np.asarray(mes.y_axis.volume)
    clumn= Jxx.reshape(mes.ny,mes.nx)[:,mes.nx//2]

    Expected =cut_currents(dev,sta,mod) [mes.nx// 2]
    assert  float( np.sum(clumn  * hights  ))  == pytest.approx( Expected ,  rel =  1e-6)

def test_a_vertical_edge_lands_on_the_nodes_above_and_below_it(monkeypatch) ->  None :

    Device  =diode_2d()
    stuff2 = Device.mesh
    ii, jj= 3, 2

    edg=stuff2.n_horizontal + jj * stuff2.nx+ ii



    Total =  np.zeros(stuff2.n_edges)
    Total [  edg  ]  =  1.0
    Fake =SimpleNamespace(data= Total)
    format =SimpleNamespace(data=np.zeros(stuff2.n_edges))
    monkeypatch.setattr(iv,'current_densities',lambda*args : (Fake,format))
    Jxx,Jyy =iv.node_current_density(Device,None)
    assert  np.all (Jxx  ==   0.0  )
    assert set(np.flatnonzero(Jyy)) =={stuff2.node_at(ii,jj),stuff2.node_at(ii,jj+ 1)}

def test_on_a_1d_diode_every_node_carries_the_edge_current()-> None:

    dveice,State,moedls= solved_1d(0.5,NoRecombination())
    jn,Jpp= current_densities(dveice,State,moedls)
    slice ,   next   =  node_current_density(dveice , State, moedls  )
    assert  slice.shape == next.shape == (  dveice.mesh.n_nodes,   )
    assert np.all(next== 0.0)

    np.testing.assert_allclose (  slice , np.mean(jn.data   +   Jpp.data ) ,  rtol   =  1e-6)
def test_the_oxide_carries_no_current()-> None :
    dev  =  capped_diode_2d(  )
    State,  mod   = solved( dev)
    Jxx,jy =node_current_density(dev,State,mod)
    oxi =dev.regions.oxide_nodes
    assert np.all(Jxx[oxi]  == 0.0)
    assert np.all(jy[oxi]==0.0)
def test_a_degenerate_junction_at_rest_carries_no_current() ->None :
    nNodes= 201
    devce = build_device(mesh=uniform_mesh_1d(length =  1e-4, n_nodes=  nNodes), doping =abrupt_junction(Na=  1e17, Nd= 1e20, position =  0.5e-4), contacts = (OhmicContact(name  ='anode', node = 0, voltage =0.0), OhmicContact(name  = 'cathode', node = nNodes -  1, voltage= 0.0),), degenerate = True,)
    moedls =   TransportModels.for_device (devce  )
    bb= solve_bias(devce,
          models = moedls)

    Jnn, jp  =  current_densities( devce ,
                 bb,
                moedls )
    assert np.max(np.abs(Jnn.data + jp.data)) <1e-3
