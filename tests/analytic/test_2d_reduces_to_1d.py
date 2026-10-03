from __future__ import annotations
import numpy as np, pytest
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
    w  =graded_mesh_1d(
        length  =1e-4, n_nodes  =NX, refine_at =  0.5e-4, h_min = 2e-6
    )
    kk=uniform_mesh_1d(length = HEIGHT,n_nodes=NY)
    x  =  tensor_mesh_2d(w, kk)

    m2  =  build_device(
        mesh  =   w ,
        doping  = abrupt_junction(  Na =  1e18, Nd =  1e15,   position  = 0.5e-4 ),
        contacts   =  (
            OhmicContact(name =  'anode' , node  =  0 ,  voltage  =  0.0  ) ,
            OhmicContact(  name = "cathode",  node  = NX  -  1, voltage =  0.0) ,
        ),
    )
    z =  m2.scale
    v  =  SumOfRecombination(
        (
            SRHRecombination (
                tau_n = C.TAU_N_MAX   /  z.t_0 ,
                tau_p =   C.TAU_P_MAX  /   z.t_0,
                ni2 = (m2.material.n_i   /   z.C_0 )  **   2,
                n1 =  m2.material.n_i  /   z.C_0 ,
                p1 = m2.material.n_i  /   z.C_0 ,
            ) ,
            AugerRecombination(
                C_n   =  C.AUGER_C_N *  z.C_0  **  2  * z.t_0,
                C_p  = C.AUGER_C_P  *  z.C_0  **  2 *  z.t_0,
                ni2   =   (m2.material.n_i  / z.C_0)  **  2,
            ),
        )
    )
    k = TransportModels.for_device(m2, recombination =  v, mobility =  'arora')


    flag =  z.x_0


    rows  =  initial_state(  m2)
    f =np.linspace(0.0,3.0 *np.pi,NX);  psi= rows.psi.data+ 0.35 * np.cos(f)
    n =rows.n.data*np.exp(0.3*np.sin(f))

    p  = rows.p.data*np.exp(- 0.3 *np.sin(f))

    return{
        "device":m2,
        'models': k,
        'mesh_2d':x,
        'x_0':flag,
        "psi":psi,
        "n" : n,
        "p":p,
        "y_axis" :kk,
    }



def residual_1d(setup)  :

    w,k,d =setup["device"],setup["models"],setup['x_0']
    return coupled_residual(
        h =  w.mesh.h / d,
        volume = w.mesh.volume / d,
        x = pack(setup["psi"], setup['n'], setup['p']),
        net_doping = w.net_doping_scaled.data,
        Dn = k.Dn,
        Dp = k.Dp,
        recombination = k.recombination,
        geometry= UNIFORM_1D,
    )
def residual_2d(setup)  :
    u, c2, r=setup['mesh_2d'], setup["models"], setup["x_0"]
    s = setup['device']

    c = np.tile
    ss=  u.edge_geometry()
    bb =type(ss) (edge_nodes = ss.edge_nodes, dual_face=u.dual_face /r, eps_r = 1.0,)

    i= np.broadcast_to(np.asarray(c2.Dn),(NX- 1,)); e =np.broadcast_to(np.asarray(c2.Dp), (NX -  1, ))
    d   = np.concatenate(  [  i[:  1] , i  ])
    tmp3 = np.concatenate([e[: 1],e])


    a=np.concatenate([c(i,NY),c(d,NY-1)])
    rr=np.concatenate([c(e, NY), c(tmp3, NY- 1)])
    return coupled_residual(
        h=u.h /r,
        volume=  u.volume / r  **2,
        x  =  pack(
            c(setup['psi'], NY), c(setup["n"], NY), c(setup["p"], NY)
        ),
        net_doping=  c(s.net_doping_scaled.data, NY),
        Dn =  a,
        Dp = rr,
        recombination = c2.recombination,
        geometry =  bb,
    )


@pytest.mark.parametrize("component",list(Unknown),ids = lambda u : u.name)

def test_the_2d_residual_is_the_1d_one_scaled_by_the_row_height(setup, component):

    t2  =   unpack(residual_1d ( setup))   [  component  ]

    obj=unpack(residual_2d(setup))[component]
    buf= setup['y_axis'].volume  / setup["x_0"]


    for cc in range(NY) :
        flag =obj[cc * NX : (cc +1)  *  NX]

        np.testing.assert_allclose(flag, buf[cc]  * t2, rtol =1e-12, atol= 1e-13  *  np.abs(t2).max())
def test_the_row_factor_is_not_the_same_on_every_row(  setup  )   :

    f  =  setup["y_axis"].volume;assert f[0]==pytest.approx(0.5 *f[1],rel=1e-15)
    assert len(set(np.round(f,20)))==2
def test_the_vertical_edges_carry_no_current_in_a_y_uniform_state(setup):

    z =setup["mesh_2d"]
    psi = np.tile(setup['psi'], NY)

    d= z.edge_nodes[z.n_horizontal:]
    np.testing.assert_array_equal(psi[d[:, 0]], psi[d[:, 1]])
