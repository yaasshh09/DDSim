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
    Mesh=uniform_mesh_1d(length =1e-4,n_nodes = N_NODES)
    item2 =build_device(
        mesh = Mesh,
        doping=abrupt_junction(Na =1e18, Nd = 1e15, position=  0.5e-4),
        contacts =  (
            OhmicContact(name = "anode", node =0, voltage =0.0),
            OhmicContact(name  = "cathode", node  =  N_NODES-1, voltage =  0.0),
        ),
    )
    range= TransportModels.for_device(item2,mobility= "arora")
    tmp= item2.scale

    bb=initial_state(item2)

    K=np.linspace(0.0,3.0 *np.pi,Mesh.n_nodes)
    xx  = pack(
        bb.psi.data + 0.35*np.cos(K),
        bb.n.data *  np.exp(0.3  *  np.sin(K)),
        bb.p.data *  np.exp(-  0.3 * np.sin(K)),
    )

    return{
        'h': Mesh.h/ tmp.x_0,
        "volume":Mesh.volume/tmp.x_0,
        "x":xx,
        "net_doping":item2.net_doping_scaled.data,
        'models': range,
        "n_edges":Mesh.n_edges,
    }



def residual_of(problem,geometry,order=None):
    geometry= UNIFORM_1D if geometry is None else geometry
    mod  =   problem["models"]
    dn =np.asarray(mod.Dn)
    pow=np.asarray(mod.Dp)
    if order  is  not None  :
        dn,pow=dn[order],pow[order]
    return  coupled_residual (h   =  problem[  "h"  ]  if  order  is None  else problem [ "h" ]   [  order], volume  =  problem[  'volume'  ], x  =  problem[  'x' ], net_doping  = problem [  "net_doping" ], Dn = dn , Dp  =  pow , recombination   = mod.recombination, geometry =   geometry,)
def matrix_of(problem, geometry, order =  None) :
    geometry=  UNIFORM_1D if geometry is None else geometry
    mod =problem['models']
    Dnn = np.asarray(mod.Dn)
    dp  =   np.asarray(mod.Dp)
    if order is not None:
        Dnn ,   dp  =  Dnn[ order ],   dp[  order ]
    t2, Cols, format  = coupled_jacobian(
        h= problem["h"]if order is None else problem["h"] [order],
        volume =problem["volume"],
        x = problem["x"],
        Dn  = Dnn,
        Dp =dp,
        recombination = mod.recombination,
        geometry = geometry,
    )
    Size = problem[  'x'].size

    return  coo_matrix((  format, ( t2, Cols ) ),   shape  =  (  Size ,  Size )).toarray( )


def  chain (n_edges) :
    return np.array([[ee,ee+ 1] for ee in range(n_edges)],dtype= np.int64)
def test_the_explicit_1d_edge_list_reproduces_the_default_bit_for_bit(problem) :
    speled_out  = EdgeGeometry( edge_nodes  = chain(problem["n_edges"  ]) )


    np.testing.assert_array_equal (
        residual_of (  problem, speled_out  ), residual_of (problem,  None)
    )
    np.testing.assert_array_equal(
        matrix_of(problem, speled_out ) ,   matrix_of(problem , None  )
    )

def test_the_answer_does_not_depend_on_the_order_the_edges_are_listed_in(problem):
    rngg =  np.random.default_rng(20260825)
    Order  =  rngg.permutation(problem['n_edges'])
    cnt =EdgeGeometry(edge_nodes=chain(problem['n_edges'])[Order])
    np.testing.assert_allclose(
        residual_of(problem, cnt, Order),
        residual_of(problem, None),
        rtol = 1e-13,
        atol = 0.0,
    )
    np.testing.assert_allclose(matrix_of(problem,cnt,Order), matrix_of(problem,None), rtol=1e-13, atol=0.0,)

def test_the_answer_does_not_depend_on_which_end_of_an_edge_is_called_left(problem)  :

    re = chain(problem['n_edges'])[:,::-1].copy()
    geo=  EdgeGeometry(edge_nodes =re)
    np.testing.assert_allclose(residual_of (problem, geo) , residual_of (  problem ,  None  ) , rtol =  1e-13 , atol  =   0.0,)
    np.testing.assert_allclose(
        matrix_of(  problem , geo),
        matrix_of (  problem ,   None  ),
        rtol  =  1e-13,
        atol  =  0.0,
    )


def test_the_term_scales_do_not_depend_on_the_edge_order_either(problem):

    mod= problem['models']
    Rng=np.random.default_rng(11)
    order  =  Rng.permutation ( problem[ 'n_edges']  )
    Shuffled=EdgeGeometry(edge_nodes= chain(problem["n_edges"]) [order])

    def scales(geometry, order  =  None):
        geometry =UNIFORM_1D if geometry is None else geometry
        return residual_term_scales(h =problem['h']if order is None else problem["h"] [order], volume =problem['volume'], x =  problem['x'], net_doping =problem["net_doping"], Dn  =np.asarray(mod.Dn) if order is None else np.asarray(mod.Dn)  [order], Dp =np.asarray(mod.Dp) if order is None else np.asarray(mod.Dp)  [order], geometry = geometry,)


    np.testing.assert_allclose(
        scales(Shuffled, order), scales(None), rtol  = 1e-13, atol = 0.0
    )
