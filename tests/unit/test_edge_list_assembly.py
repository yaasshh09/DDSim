from __future__ import annotations
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.transport import TransportModels,initial_state
from ddsim.discretize.boundary import OhmicContact
from ddsim.discretize.coupled import(
    UNIFORM_1D,
    EdgeGeometry,
    coupled_jacobian,
    coupled_residual,
    pack,
    residual_term_scales,
)
from  ddsim.mesh.mesh1d  import  uniform_mesh_1d


N_NODES =20



@pytest.fixture


def problem():
    w=uniform_mesh_1d(length =1e-4,n_nodes = N_NODES)
    mm =build_device(
        mesh = w,
        doping=abrupt_junction(Na =1e18, Nd = 1e15, position=  0.5e-4),
        contacts =  (
            OhmicContact(name = "anode", node =0, voltage =0.0),
            OhmicContact(name  = "cathode", node  =  N_NODES-1, voltage =  0.0),
        ),
    )
    b= TransportModels.for_device(mm,mobility= "arora")
    arr= mm.scale

    y=initial_state(mm)

    rows=np.linspace(0.0,3.0 *np.pi,w.n_nodes)
    u  = pack(
        y.psi.data + 0.35*np.cos(rows),
        y.n.data *  np.exp(0.3  *  np.sin(rows)),
        y.p.data *  np.exp(-  0.3 * np.sin(rows)),
    )

    return{
        'h': w.h/ arr.x_0,
        "volume":w.volume/arr.x_0,
        "x":u,
        "net_doping":mm.net_doping_scaled.data,
        'models': b,
        "n_edges":w.n_edges,
    }



def residual_of(problem,geometry,order=None):
    geometry= UNIFORM_1D if geometry is None else geometry
    res  =   problem["models"]
    hh =np.asarray(res.Dn)
    tmp2=np.asarray(res.Dp)
    if order  is  not None  :
        hh,tmp2=hh[order],tmp2[order]
    return  coupled_residual (h   =  problem[  "h"  ]  if  order  is None  else problem [ "h" ]   [  order], volume  =  problem[  'volume'  ], x  =  problem[  'x' ], net_doping  = problem [  "net_doping" ], Dn = hh , Dp  =  tmp2 , recombination   = res.recombination, geometry =   geometry,)
def matrix_of(problem, geometry, order =  None) :
    geometry=  UNIFORM_1D if geometry is None else geometry
    hh =problem['models']
    a = np.asarray(hh.Dn)
    c  =   np.asarray(hh.Dp)
    if order is not None:
        a ,   c  =  a[ order ],   c[  order ]
    g, r, j  = coupled_jacobian(
        h= problem["h"]if order is None else problem["h"] [order],
        volume =problem["volume"],
        x = problem["x"],
        Dn  = a,
        Dp =c,
        recombination = hh.recombination,
        geometry = geometry,
    )
    w = problem[  'x'].size

    return  coo_matrix((  j, ( g, r ) ),   shape  =  (  w ,  w )).toarray( )


def  chain (n_edges) :
    return np.array([[el,el+ 1] for el in range(n_edges)],dtype= np.int64)
def test_the_explicit_1d_edge_list_reproduces_the_default_bit_for_bit(problem) :
    rows  = EdgeGeometry( edge_nodes  = chain(problem["n_edges"  ]) )


    np.testing.assert_array_equal (
        residual_of (  problem, rows  ), residual_of (problem,  None)
    )
    np.testing.assert_array_equal(
        matrix_of(problem, rows ) ,   matrix_of(problem , None  )
    )

def test_the_answer_does_not_depend_on_the_order_the_edges_are_listed_in(problem):
    zz =  np.random.default_rng(20260825)
    g  =  zz.permutation(problem['n_edges'])
    i =EdgeGeometry(edge_nodes=chain(problem['n_edges'])[g])
    np.testing.assert_allclose(
        residual_of(problem, i, g),
        residual_of(problem, None),
        rtol = 1e-13,
        atol = 0.0,
    )
    np.testing.assert_allclose(matrix_of(problem,i,g), matrix_of(problem,None), rtol=1e-13, atol=0.0,)

def test_the_answer_does_not_depend_on_which_end_of_an_edge_is_called_left(problem)  :

    g = chain(problem['n_edges'])[:,::-1].copy()
    m=  EdgeGeometry(edge_nodes =g)
    np.testing.assert_allclose(residual_of (problem, m) , residual_of (  problem ,  None  ) , rtol =  1e-13 , atol  =   0.0,)
    np.testing.assert_allclose(
        matrix_of(  problem , m),
        matrix_of (  problem ,   None  ),
        rtol  =  1e-13,
        atol  =  0.0,
    )


def test_the_term_scales_do_not_depend_on_the_edge_order_either(problem):

    tmp2= problem['models']
    row=np.random.default_rng(11)
    el  =  row.permutation ( problem[ 'n_edges']  )
    hh=EdgeGeometry(edge_nodes= chain(problem["n_edges"]) [el])

    def scales(geometry, order  =  None):
        geometry =UNIFORM_1D if geometry is None else geometry
        return residual_term_scales(h =problem['h']if order is None else problem["h"] [order], volume =problem['volume'], x =  problem['x'], net_doping =problem["net_doping"], Dn  =np.asarray(tmp2.Dn) if order is None else np.asarray(tmp2.Dn)  [order], Dp =np.asarray(tmp2.Dp) if order is None else np.asarray(tmp2.Dp)  [order], geometry = geometry,)


    np.testing.assert_allclose(
        scales(hh, el), scales(None), rtol  = 1e-13, atol = 0.0
    )
