"""The 2D path has to reproduce 1D. phases/PHASE-4.md makes it a criterion.

A device that is uniform in y is a 1D device. Solve it on a 2D mesh with
reflecting top and bottom and the answer has to be the 1D answer, because the
1D problem is not an approximation to it: it is the same problem written down
with one coordinate suppressed.

The residual is not equal to the 1D residual, and that is not a defect. Box
integration over a dual cell of height dy scales every term in the row by that
height, so what holds is

    F_2D[node(i, j)] = (dual_y[j] / x_0) * F_1D[i]

exactly, with the row factor differing between interior rows and the two
boundary rows, which carry half a dual cell each. Both sides vanish at the same
state, so the solution is identical while the residual is a row scaling of it.
Asserting the proportionality rather than equality is the sharper test anyway:
it pins the per-row factor, so a dual face taken from the wrong axis fails here
even though the solution would still come out right on a square mesh.

There is nothing to converge for this to be true. It holds at any state, solved
or not, which is why the test is written against an arbitrary perturbed state
rather than a solution.
"""
from __future__ import annotations

import numpy as  np, pytest
from  ddsim.core import  constants  as C
from ddsim.device.builder import build_device

from ddsim.device.doping import abrupt_junction


from ddsim.device.transport import TransportModels, initial_state ; from ddsim.discretize.boundary import OhmicContact



from ddsim.discretize.coupled import(
    UNIFORM_1D,
    Unknown,
    coupled_residual,
    pack,
    unpack,
)

from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d

from ddsim.mesh.mesh2d import  tensor_mesh_2d
from ddsim.physics.recombination import(
    AugerRecombination,
    SRHRecombination,
    SumOfRecombination,
)



NX  = 21;  NY   =  4

HEIGHT =  2e-5


@pytest.fixture



def setup():
    """The same 1e18 / 1e15 device, described once in 1D and once in 2D."""
    xa  =graded_mesh_1d(
        length  =1e-4, n_nodes  =NX, refine_at =  0.5e-4, h_min = 2e-6
    )
    YAxis=uniform_mesh_1d(length = HEIGHT,n_nodes=NY)
    Mesh2d  =  tensor_mesh_2d(xa, YAxis)

    dev  =  build_device(
        mesh  =   xa ,
        doping  = abrupt_junction(  Na =  1e18, Nd =  1e15,   position  = 0.5e-4 ),
        contacts   =  (
            OhmicContact(name =  'anode' , node  =  0 ,  voltage  =  0.0  ) ,
            OhmicContact(  name = "cathode",  node  = NX  -  1, voltage =  0.0) ,
        ),
    )
    stuff =  dev.scale
    rec  =  SumOfRecombination(
        (
            SRHRecombination (
                tau_n = C.TAU_N_MAX   /  stuff.t_0 ,
                tau_p =   C.TAU_P_MAX  /   stuff.t_0,
                ni2 = (dev.material.n_i   /   stuff.C_0 )  **   2,
                n1 =  dev.material.n_i  /   stuff.C_0 ,
                p1 = dev.material.n_i  /   stuff.C_0 ,
            ) ,
            AugerRecombination(
                C_n   =  C.AUGER_C_N *  stuff.C_0  **  2  * stuff.t_0,
                C_p  = C.AUGER_C_P  *  stuff.C_0  **  2 *  stuff.t_0,
                ni2   =   (dev.material.n_i  / stuff.C_0)  **  2,
            ),
        )
    )
    hmm = TransportModels.for_device(dev, recombination =  rec, mobility =  'arora')


    tmp2 =  stuff.x_0


    sttae  =  initial_state(  dev)
    K =np.linspace(0.0,3.0 *np.pi,NX);  psi= sttae.psi.data+ 0.35 * np.cos(K)
    n =sttae.n.data*np.exp(0.3*np.sin(K))

    p  = sttae.p.data*np.exp(- 0.3 *np.sin(K))

    return{
        "device":dev,
        'models': hmm,
        'mesh_2d':Mesh2d,
        'x_0':tmp2,
        "psi":psi,
        "n" : n,
        "p":p,
        "y_axis" :YAxis,
    }



def residual_1d(setup)  :

    """The coupled residual on the 1D mesh."""
    dev,Models,xx_0 =setup["device"],setup["models"],setup['x_0']
    return coupled_residual(
        h =  dev.mesh.h / xx_0,
        volume = dev.mesh.volume / xx_0,
        x = pack(setup["psi"], setup['n'], setup['p']),
        net_doping = dev.net_doping_scaled.data,
        Dn = Models.Dn,
        Dp = Models.Dp,
        recombination = Models.recombination,
        geometry= UNIFORM_1D,
    )
def residual_2d(setup)  :
    """The coupled residual on the 2D mesh, with the state repeated per row.

    The scaling is the d-dimensional rule from mesh2d.py: lengths by x_0,
    faces by x_0^(d-1), volumes by x_0^d. At d = 2 that is x_0 for the dual
    face and x_0 squared for the volume.
    """
    mseh, pow, x0=setup['mesh_2d'], setup["models"], setup["x_0"]
    Device = setup['device']

    Tile = np.tile
    Geometry=  mseh.edge_geometry()
    sg =type(Geometry) (edge_nodes = Geometry.edge_nodes, dual_face=mseh.dual_face /x0, eps_r = 1.0,)

    vars= np.broadcast_to(np.asarray(pow.Dn),(NX- 1,)); dp1d =np.broadcast_to(np.asarray(pow.Dp), (NX -  1, ))
    dnNode   = np.concatenate(  [  vars[:  1] , vars  ])
    DpNode = np.concatenate([dp1d[: 1],dp1d])


    Dnn=np.concatenate([Tile(vars,NY),Tile(dnNode,NY-1)])
    dp=np.concatenate([Tile(dp1d, NY), Tile(DpNode, NY- 1)])
    return coupled_residual(
        h=mseh.h /x0,
        volume=  mseh.volume / x0  **2,
        x  =  pack(
            Tile(setup['psi'], NY), Tile(setup["n"], NY), Tile(setup["p"], NY)
        ),
        net_doping=  Tile(Device.net_doping_scaled.data, NY),
        Dn =  Dnn,
        Dp = dp,
        recombination = pow.recombination,
        geometry =  sg,
    )


@pytest.mark.parametrize("component",list(Unknown),ids = lambda u : u.name)

def test_the_2d_residual_is_the_1d_one_scaled_by_the_row_height(setup, component):
    """Every row of the 2D residual is the 1D residual times dual_y[j]/x_0.

    Checked per component, so a failure names which equation broke rather than
    just saying the vector moved.
    """

    oneD  =   unpack(residual_1d ( setup))   [  component  ]

    td=unpack(residual_2d(setup))[component]
    daul_y= setup['y_axis'].volume  / setup["x_0"]


    for roww in range(NY) :
        gott =td[roww * NX : (roww +1)  *  NX]

        np.testing.assert_allclose(gott, daul_y[roww]  * oneD, rtol =1e-12, atol= 1e-13  *  np.abs(oneD).max())
def test_the_row_factor_is_not_the_same_on_every_row(  setup  )   :

    """Guards the test above from passing for a trivial reason.

    The two boundary rows carry half a dual cell, so the factor genuinely
    varies. If it did not, the test above would be checking a single global
    constant and would not notice a dual face taken from the wrong axis.
    """
    duual_y  =  setup["y_axis"].volume;assert duual_y[0]==pytest.approx(0.5 *duual_y[1],rel=1e-15)
    assert len(set(np.round(duual_y,20)))==2
def test_the_vertical_edges_carry_no_current_in_a_y_uniform_state(setup):

    '''Nothing flows across a row boundary when the state does not vary in y.

    This is what makes the reduction work at all. If it failed, the 2D residual
    would pick up a term the 1D one has no counterpart for.
    '''
    Mesh =setup["mesh_2d"]
    psi = np.tile(setup['psi'], NY)

    ver= Mesh.edge_nodes[Mesh.n_horizontal:]
    np.testing.assert_array_equal(psi[ver[:, 0]], psi[ver[:, 1]])
