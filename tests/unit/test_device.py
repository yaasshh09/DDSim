"""Tests for device/builder.py, device/pn_diode.py and device/equilibrium.py.

device/ composes: geometry and doping in, a Device out that knows its mesh, its
material, its contacts and how to evaluate its doping anywhere. It holds no
solver state of its own, which is why solve_equilibrium returns a DeviceState
rather than mutating the Device.
"""


from __future__ import annotations
import numpy as np, pytest
from ddsim.core.field import Location,ScalingState
from ddsim.device.builder import Device,Material,build_device



from ddsim.device.doping  import  Uniform
from  ddsim.device.equilibrium  import solve_equilibrium

from  ddsim.device.pn_diode  import  pn_diode

from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import uniform_mesh_1d


MICRON =  1e-4

"""One micron [cm]."""


def  test_silicon_material_matches_the_constants_doc()  ->  None :
    Silicon  =  Material.silicon()
    assert Silicon.T ==300.0
    assert Silicon.n_i==1.0e10
    assert Silicon.eps == pytest.approx(11.7*8.8541878128e-14)


def test_material_at_another_temperature_moves_n_i()->None:
    assert Material.silicon(  T  =   400.0).n_i  >  Material.silicon(  T  =  300.0).n_i



def test_build_device_evaluates_doping_on_the_mesh()->None :
    tmp=uniform_mesh_1d(MICRON,11)
    aa  = build_device(
        mesh  = tmp,
        doping =  Uniform (1e16  ),
        contacts  =   (  OhmicContact( "anode",   0 ,   0.0) ,  OhmicContact ('cathode', 10, 0.0  )  ),
    )
    np.testing.assert_allclose(aa.net_doping.data,1e16)


def test_net_doping_is_a_physical_node_field()->None  :
    hash   =   pn_diode(  )
    assert hash.net_doping.unit ==  "cm^-3"
    assert hash.net_doping.scaling is ScalingState.PHYSICAL
    assert hash.net_doping.location is Location.NODE


def test_net_doping_scaled_divides_by_c_0() ->  None :
    dev   = pn_diode(  Na  =   1e16,  Nd   =  1e16  )
    saled  =  dev.net_doping_scaled
    assert saled.scaling is ScalingState.SCALED
    np.testing.assert_allclose(
        saled.data, dev.net_doping.data/ dev.scale.C_0, rtol = 1e-14
    )



def test_device_rejects_a_contact_outside_the_mesh()-> None:
    meesh  =  uniform_mesh_1d (  MICRON ,  11)
    with pytest.raises( IndexError,   match   =   "node")  :
        build_device(mesh=meesh, doping = Uniform(1e16), contacts =  (OhmicContact("anode", 99, 0.0), ),)


def  test_device_requires_at_least_one_contact ( ) -> None  :


    meesh=uniform_mesh_1d(MICRON,11)

    with pytest.raises(ValueError, match  ="contact"):

        build_device(mesh= meesh,
             doping= Uniform(1e16),
              contacts= ())

def test_device_is_immutable()->  None :
    dev  =  pn_diode ( )
    with pytest.raises(AttributeError) :
        dev.mesh=None

def test_doping_stays_re_evaluable_after_construction()->None:
    """The profile is kept, not just its values on this mesh.

    Phase 5 refines adaptively, so the Device has to be able to produce doping
    on a mesh that did not exist when it was built.
    """
    dev  = pn_diode(Na  = 1e16, Nd = 1e17, length =MICRON, junction =0.5 * MICRON)
    item2 =np.linspace(0.0,MICRON,1001)
    Values =dev.doping(item2)

    assert Values[  0 ] ==   pytest.approx( -  1e16)
    assert  Values[- 1  ]  ==  pytest.approx(1e17)

def test_pn_diode_is_p_type_on_the_left_and_n_type_on_the_right()  ->  None  :

    dev =pn_diode(Na= 1e16, Nd=1e16);assert dev.net_doping.data[0]< 0.0
    assert dev.net_doping.data[-1]>0.0

def test_pn_diode_has_two_named_contacts_at_the_ends()-> None:
    dvice=pn_diode()
    assert[cc.name for cc in dvice.contacts]==["anode","cathode"]
    assert dvice.contacts[0].node==0
    assert dvice.contacts[1].node ==dvice.mesh.n_nodes- 1
def  test_pn_diode_refines_the_mesh_at_the_junction()   ->   None  :
    deice=pn_diode(junction=0.5 * MICRON,h_min=1e-7)
    Finest= int(np.argmin(deice.mesh.h))
    mid = 0.5* (deice.mesh.x[Finest] + deice.mesh.x[Finest + 1])
    assert abs(mid -  0.5  *  MICRON  ) <  2e-7

def test_pn_diode_contacts_default_to_zero_bias() ->None :

    devce= pn_diode()
    assert all(contact.voltage==0.0 for contact in devce.contacts)


def  test_equilibrium_solve_converges ()   ->  None   :

    sttate =solve_equilibrium(pn_diode())
    assert sttate.newton.converged

def test_equilibrium_solve_converges_in_under_ten_iterations()->None:
    """The acceptance criterion in phases/PHASE-1.md."""
    State  =   solve_equilibrium(  pn_diode( ) )
    assert State.newton.iterations< 10,State.newton.residual_history




def test_equilibrium_state_fields_are_scaled_node_fields() -> None :
    State = solve_equilibrium(pn_diode())
    for feld in(  State.psi,
                 State.n,
                   State.p  )  :

        assert feld.scaling is ScalingState.SCALED
        assert feld.location is Location.NODE



def test_equilibrium_psi_can_be_converted_to_volts() -> None:


    min  =   pn_diode ()

    sta  =  solve_equilibrium(min)

    vol= sta.psi.to_physical(min.scale)
    assert  vol.scaling is  ScalingState.PHYSICAL
    assert np.max(np.abs(vol.data))<2.0

def  test_equilibrium_pins_psi_at_the_contacts(  )  ->   None   :
    from ddsim.discretize.boundary import ohmic_psi_scaled


    id= pn_diode(Na=1e16,Nd = 1e16)
    State= solve_equilibrium(id)
    bb  =id.net_doping_scaled.data
    for acc in id.contacts :

        Expected  =  ohmic_psi_scaled(float (  bb[ acc.node]  ) ,   0.0)
        assert State.psi.data[  acc.node ]   ==   pytest.approx(  Expected,  rel  =  1e-10 )



def test_equilibrium_is_flat_in_uniform_material()->None:

    '''No junction means no field. The initial guess is already the answer.'''
    Mesh = uniform_mesh_1d(MICRON,51)
    devce =build_device(mesh= Mesh, doping  = Uniform(1e16), contacts=  (OhmicContact('left', 0, 0.0), OhmicContact("right", 50, 0.0)),)
    State=solve_equilibrium(devce)
    assert np.ptp(State.psi.data)< 1e-9


def test_equilibrium_reports_the_residual_history() -> None :
    """phases/PHASE-1.md wants the history logged so the tail can be inspected."""


    bar = solve_equilibrium(pn_diode())
    assert len(bar.newton.residual_history)== bar.newton.iterations  +1
    assert bar.newton.residual_history[- 1  ]  < bar.newton.residual_history[0]



def test_equilibrium_raises_if_it_fails_to_converge() -> None :
    """A silently unconverged solve is the worst possible outcome here."""
    with pytest.raises (RuntimeError,
                match   =  "converge")   :
        solve_equilibrium(pn_diode(),max_iterations=1)




def test_equilibrium_accepts_a_device_with_a_bias_applied()->None:
    """No continuation yet, but a contact voltage must still be honoured."""
    dvice  = pn_diode(anode_voltage  =-  1.0)
    State = solve_equilibrium(dvice)
    assert State.newton.converged
    ret = State.psi.to_physical (  dvice.scale  ).data
    assert ret[0]<ret[-1]



def test_device_state_is_immutable()-> None :


    range= solve_equilibrium(pn_diode())
    with pytest.raises(  AttributeError  )  :
        range.psi= None

def test_solve_does_not_mutate_the_device() ->None :
    arr=pn_diode()
    bef = arr.net_doping.data.copy()
    solve_equilibrium(arr)
    np.testing.assert_array_equal( arr.net_doping.data , bef  )

def test_device_repr_is_informative(  )  ->  None :

    tex=repr(pn_diode())
    assert "Device" in tex
    assert "nodes"  in tex
def test_build_device_defaults_to_silicon() ->None  :
    dev  =  build_device(
        mesh   =   uniform_mesh_1d ( MICRON,   11),
        doping   = Uniform ( 1e16) ,
        contacts   =  ( OhmicContact ('a',   0 , 0.0  ) ,   ),
    )
    assert isinstance(dev,Device)

    assert dev.material.n_i == 1.0e10



def test_device_rejects_duplicate_contact_names() ->None :
    stuff =uniform_mesh_1d(MICRON, 11)
    with pytest.raises(ValueError,match="unique"):
        build_device(mesh=stuff, doping=Uniform(1e16), contacts= (OhmicContact("a",0,0.0),OhmicContact("a",10,0.0)),)
