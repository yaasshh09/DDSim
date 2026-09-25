from __future__ import annotations
import numpy as np, pytest

from ddsim.device.builder import Device,build_device

from ddsim.device.doping import Step


from ddsim.device.transport import  TransportModels,   solve_bias; from ddsim.discretize.boundary import OhmicContact

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

    bia=device.with_bias(anode =voltage)
    foo =  solve_bias(bia)

    assert foo.gummel is not None and foo.gummel.converged,(
        f"the solve at {voltage:+g} V did not converge: {foo.gummel}"
    )
    return terminal_currents( bia , foo) [ "anode"  ]



def test_the_graded_mesh_is_symmetric_about_a_central_refinement() ->  None :
    xx  =  mesh().x
    np.testing.assert_allclose(xx,LENGTH-xx[::-1],rtol= 0.0,atol= 1e-18)

def test_the_dual_cells_mirror_too()-> None :
    vloume=mesh().volume
    data2 =   8.0 *  EPS *  LENGTH  / vloume.min ( )
    object  = float(np.max(np.abs(vloume- vloume[::-  1]) /vloume));  assert object < data2,f"worst {object:.3e}, accumulation floor {data2:.3e}"



@pytest.mark.parametrize( "voltage",   [ - 1.0,   -   0.2 ,   0.0 , 0.2,  0.4,   0.5 ])


def test_a_mirrored_diode_gives_the_same_terminal_current(voltage: float) ->None :
    vars= anode_current(p_on_the_left(OFF_NODE),voltage)
    bac  =  anode_current( n_on_the_left(OFF_NODE),   voltage)

    assert vars ==   pytest.approx( bac,   rel =  1e-12),   (
        f"at {voltage:+g} V the p-on-the-left diode gives {vars:.9e} and the "
        f"n-on-the-left diode gives {bac:.9e}. The mesh and the doping "
        "mirror exactly, so a difference here is a left-right bias in the "
        'assembly, the boundary conditions or the flux.'
    )


def  test_the_built_in_potential_mirrors_and_changes_sign ( )   ->  None   :
    dir = solve_bias(p_on_the_left(OFF_NODE))
    bac   =  solve_bias(  n_on_the_left( OFF_NODE  ) )
    Rise = float(dir.psi.data[-1]  - dir.psi.data[0]); Fall =float(bac.psi.data[-1]-bac.psi.data[0])
    assert Rise > 0.0
    assert Fall==  pytest.approx(-  Rise, rel =1e-12)


def test_the_solved_profiles_are_reflections_of_each_other()   -> None :
    foorward=solve_bias(p_on_the_left(OFF_NODE).with_bias(anode= 0.4))
    bac= solve_bias(n_on_the_left(OFF_NODE).with_bias(anode=0.4))

    np.testing.assert_allclose(
        foorward.n.data,bac.n.data[::- 1],rtol=1e-11,atol=0.0
    )
    np.testing.assert_allclose(foorward.p.data , bac.p.data[::-  1 ], rtol  =  1e-11,   atol =  0.0)




def test_the_current_densities_mirror_with_a_sign_change()-> None:

    w  =  p_on_the_left ( OFF_NODE  ).with_bias(  anode  =  0.4 )
    len = n_on_the_left(OFF_NODE).with_bias(anode =0.4)


    State  =  solve_bias( w);Models=TransportModels.for_device(w)
    Jnn, type= current_densities(w, State, Models)
    idx2, jm   =  current_densities(  len,  solve_bias ( len )  )
    yy=3.0 * EPS *largest_flux_term(w,State,Models)/(np.abs(Jnn.data).max()/w.scale.J_0)

    for mea, buff, nme in((Jnn.data, idx2.data, "Jn"), (type.data, jm.data, 'Jp'),):
        wor   = float (
            np.max ( np.abs (mea  +  buff [  ::-   1 ])  /  np.abs(mea ).max(  )  )
        )
        assert wor <  yy, (
            f"{nme} disagrees with its reflection by {wor:.3e}, above the "
            f"cancellation floor of {yy:.3e}. A difference above the floor "
            'is a left-right bias in the flux, not arithmetic.'
        )


def test_a_node_on_the_step_is_the_only_thing_that_breaks_the_mirror()-> None :
    onnode=  p_on_the_left(0.0).net_doping.data
    OnNodeMirror = n_on_the_left(0.0).net_doping.data

    assert np.count_nonzero(onnode != OnNodeMirror[::-1])== 1
    bar = p_on_the_left(OFF_NODE).net_doping.data
    OffNodeMirror =n_on_the_left(OFF_NODE).net_doping.data
    np.testing.assert_array_equal(bar,OffNodeMirror[::-1])



def test_one_cell_of_base_width_is_worth_about_a_tenth_of_a_percent() ->None :
    Voltage =0.5
    Unshifted =  anode_current( p_on_the_left(0.0) ,  Voltage  )
    Shifted=anode_current(p_on_the_left(H_MIN),Voltage)
    junk=anode_current(n_on_the_left(0.0),Voltage)


    tmp= (Shifted- Unshifted)/Unshifted
    bytes=(junk-Unshifted)/Unshifted
    assert tmp <0.0,'widening the p side base must lower the current'
    assert abs(tmp) ==  pytest.approx(abs(bytes), rel = 1e-6)
    assert 5e-4  <   abs(  bytes)   < 5e-3
