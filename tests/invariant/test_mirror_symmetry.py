from __future__ import annotations
import numpy as np, pytest
from ddsim.device.builder import Device,build_device
from ddsim.device.doping import Step
from ddsim.device.transport import TransportModels, solve_bias; from ddsim.discretize.boundary import OhmicContact
from ddsim.extract.iv import current_densities, terminal_currents
from ddsim.mesh.mesh1d import graded_mesh_1d
from tests.invariant.test_current_continuity import EPS, largest_flux_term


LENGTH= 1e-4


JUNCTION  = 0.5e-4

H_MIN= 1e-7

N_NODES  = 201

DOPING  =  1e16

OFF_NODE=H_MIN /3.0


def mesh():
    return graded_mesh_1d(LENGTH, N_NODES, refine_at = JUNCTION, h_min=H_MIN)



def p_on_the_left(offset: float)-> Device:
    return build_device(mesh  =  mesh(), doping =Step(left =-  DOPING, right  =  DOPING, position =JUNCTION +offset), contacts=(OhmicContact("anode", 0, 0.0), OhmicContact("cathode", N_NODES - 1, 0.0),),)




def n_on_the_left(offset :float)  -> Device :
    return  build_device(
        mesh   = mesh(),
        doping  = Step (
            left =  DOPING ,   right  =-  DOPING,  position  =   LENGTH  - ( JUNCTION +  offset)
        ),
        contacts = (
            OhmicContact( "cathode", 0, 0.0),
            OhmicContact( 'anode',   N_NODES  - 1 ,  0.0  ) ,
        ),
    )

def anode_current(device: Device, voltage :float)-> float  :

    xs=device.with_bias(anode =voltage)
    t =  solve_bias(xs)

    assert t.gummel is not None and t.gummel.converged,(
        f"the solve at {voltage:+g} V did not converge: {t.gummel}"
    )
    return terminal_currents( xs , t) [ "anode"  ]



def test_the_graded_mesh_is_symmetric_about_a_central_refinement() ->  None :
    z  =  mesh().x
    np.testing.assert_allclose(z,LENGTH-z[::-1],rtol= 0.0,atol= 1e-18)

def test_the_dual_cells_mirror_too()-> None :
    d2=mesh().volume
    b =   8.0 *  EPS *  LENGTH  / d2.min ( )
    idx  = float(np.max(np.abs(d2- d2[::-  1]) /d2));  assert idx < b,f"worst {idx:.3e}, accumulation floor {b:.3e}"



@pytest.mark.parametrize( "voltage",   [ - 1.0,   -   0.2 ,   0.0 , 0.2,  0.4,   0.5 ])


def test_a_mirrored_diode_gives_the_same_terminal_current(voltage: float) ->None :
    out2= anode_current(p_on_the_left(OFF_NODE),voltage)
    z2  =  anode_current( n_on_the_left(OFF_NODE),   voltage)

    assert out2 ==   pytest.approx( z2,   rel =  1e-12),   (
        f"at {voltage:+g} V the p-on-the-left diode gives {out2:.9e} and the "
        f"n-on-the-left diode gives {z2:.9e}. The mesh and the doping "
        "mirror exactly, so a difference here is a left-right bias in the "
        'assembly, the boundary conditions or the flux.'
    )


def  test_the_built_in_potential_mirrors_and_changes_sign ( )   ->  None   :
    rows = solve_bias(p_on_the_left(OFF_NODE))
    cnt   =  solve_bias(  n_on_the_left( OFF_NODE  ) )
    m = float(rows.psi.data[-1]  - rows.psi.data[0]); a2 =float(cnt.psi.data[-1]-cnt.psi.data[0])
    assert m > 0.0
    assert a2==  pytest.approx(-  m, rel =1e-12)


def test_the_solved_profiles_are_reflections_of_each_other()   -> None :
    t=solve_bias(p_on_the_left(OFF_NODE).with_bias(anode= 0.4))
    kk= solve_bias(n_on_the_left(OFF_NODE).with_bias(anode=0.4))

    np.testing.assert_allclose(
        t.n.data,kk.n.data[::- 1],rtol=1e-11,atol=0.0
    )
    np.testing.assert_allclose(t.p.data , kk.p.data[::-  1 ], rtol  =  1e-11,   atol =  0.0)




def test_the_current_densities_mirror_with_a_sign_change()-> None:

    g  =  p_on_the_left ( OFF_NODE  ).with_bias(  anode  =  0.4 )
    prev = n_on_the_left(OFF_NODE).with_bias(anode =0.4)


    jj  =  solve_bias( g);s=TransportModels.for_device(g)
    num, tmp= current_densities(g, jj, s)
    m, r   =  current_densities(  prev,  solve_bias ( prev )  )
    c=3.0 * EPS *largest_flux_term(g,jj,s)/(np.abs(num.data).max()/g.scale.J_0)

    for v, rows, dat in((num.data, m.data, "Jn"), (tmp.data, r.data, 'Jp'),):
        buf   = float (
            np.max ( np.abs (v  +  rows [  ::-   1 ])  /  np.abs(v ).max(  )  )
        )
        assert buf <  c, (
            f"{dat} disagrees with its reflection by {buf:.3e}, above the "
            f"cancellation floor of {c:.3e}. A difference above the floor "
            'is a left-right bias in the flux, not arithmetic.'
        )


def test_a_node_on_the_step_is_the_only_thing_that_breaks_the_mirror()-> None :
    d=  p_on_the_left(0.0).net_doping.data
    s = n_on_the_left(0.0).net_doping.data

    assert np.count_nonzero(d != s[::-1])== 1
    row = p_on_the_left(OFF_NODE).net_doping.data
    e =n_on_the_left(OFF_NODE).net_doping.data
    np.testing.assert_array_equal(row,e[::-1])



def test_one_cell_of_base_width_is_worth_about_a_tenth_of_a_percent() ->None :
    obj =0.5
    w =  anode_current( p_on_the_left(0.0) ,  obj  )
    y=anode_current(p_on_the_left(H_MIN),obj)
    c=anode_current(n_on_the_left(0.0),obj)


    v= (y- w)/w
    num=(c-w)/w
    assert v <0.0,'widening the p side base must lower the current'
    assert abs(v) ==  pytest.approx(abs(num), rel = 1e-6)
    assert 5e-4  <   abs(  num)   < 5e-3
