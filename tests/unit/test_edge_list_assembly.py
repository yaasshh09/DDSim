"""The assemblies must depend on the edge list, not on the array order.

Passing the existing 1D suite only proves the refactor did not break anything.
It does not prove the assemblies actually read the edge list, because the 1D
default happens to agree with the slicing they used to do. These are the tests
that prove it, and they are the ones that will still mean something in 2D:

- spelling the 1D edge list out by hand must reproduce the default exactly,
- shuffling the order the edges are listed in must not change the answer,
- flipping which end of an edge is called left must not change the answer.

The last one is the sharpest. Reversing an edge flips the sign of
X = psi_right - psi_left, swaps B(X) with B(-X), swaps which node each factor
multiplies, and flips the sign of the scatter. All four have to be right for
the residual to come back unchanged. Get any one of them wrong and this test
fails while every 1D test in the suite still passes, because the 1D default
never reverses an edge.
"""
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
    '''A 1e18 / 1e15 diode, perturbed off the solution manifold.

    Asymmetric on purpose, so that Arora gives a genuinely varying per edge
    diffusivity and a reversed or reordered edge cannot hide behind a constant.
    '''
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
    """The coupled residual under one edge geometry. None means the default."""
    geometry= UNIFORM_1D if geometry is None else geometry
    mod  =   problem["models"]
    dn =np.asarray(mod.Dn)
    pow=np.asarray(mod.Dp)
    if order  is  not None  :
        dn,pow=dn[order],pow[order]
    return  coupled_residual (h   =  problem[  "h"  ]  if  order  is None  else problem [ "h" ]   [  order], volume  =  problem[  'volume'  ], x  =  problem[  'x' ], net_doping  = problem [  "net_doping" ], Dn = dn , Dp  =  pow , recombination   = mod.recombination, geometry =   geometry,)
def matrix_of(problem, geometry, order =  None) :
    """The coupled Jacobian as a dense array, under one edge geometry."""
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
    """The 1D edge list written out by hand."""
    return np.array([[ee,ee+ 1] for ee in range(n_edges)],dtype= np.int64)
def test_the_explicit_1d_edge_list_reproduces_the_default_bit_for_bit(problem) :
    """Not close to. Identical.

    The default and the spelled out list describe the same mesh, gather the
    same values in the same order and scatter them in the same order, so there
    is no reordering for the floating point sum to notice. Anything less than
    exact equality here would mean the two paths are not the same arithmetic.
    """
    speled_out  = EdgeGeometry( edge_nodes  = chain(problem["n_edges"  ]) )


    np.testing.assert_array_equal (
        residual_of (  problem, speled_out  ), residual_of (problem,  None)
    )
    np.testing.assert_array_equal(
        matrix_of(problem, speled_out ) ,   matrix_of(problem , None  )
    )

def test_the_answer_does_not_depend_on_the_order_the_edges_are_listed_in(problem):
    """Shuffle the edge list. The physics cannot care.

    Not bit exact, and it should not be: the scatter adds each node's incoming
    fluxes in whatever order the edges appear, and floating point addition is
    not associative. The tolerance is there for that and nothing else.
    """
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
    """Reverse every edge. The device is unchanged, so the residual must be.

    This is the one that catches a half finished generalization. Four separate
    things flip when an edge is reversed and they have to cancel exactly:
    the sign of X, which Bernoulli factor is which, which node each factor
    multiplies, and the sign of the scatter into the two nodes.
    """

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

    """Row scaling reads the same fluxes, so it inherits the same invariance."""
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
