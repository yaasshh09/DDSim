from __future__ import annotations
import numpy as np, pytest
from ddsim.core.field import Location,ScalingState
from ddsim.device.builder import Device,Material,build_device
from ddsim.device.doping import Uniform
from  ddsim.device.equilibrium  import solve_equilibrium
from  ddsim.device.pn_diode  import  pn_diode
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import uniform_mesh_1d


MICRON =  1e-4


def  test_silicon_material_matches_the_constants_doc()  ->  None :
    kk  =  Material.silicon()
    assert kk.T ==300.0
    assert kk.n_i==1.0e10
    assert kk.eps == pytest.approx(11.7*8.8541878128e-14)


def test_material_at_another_temperature_moves_n_i()->None:
    assert Material.silicon(  T  =   400.0).n_i  >  Material.silicon(  T  =  300.0).n_i



def test_build_device_evaluates_doping_on_the_mesh()->None :
    e=uniform_mesh_1d(MICRON,11)
    x  = build_device(
        mesh  = e,
        doping =  Uniform (1e16  ),
        contacts  =   (  OhmicContact( "anode",   0 ,   0.0) ,  OhmicContact ('cathode', 10, 0.0  )  ),
    )
    np.testing.assert_allclose(x.net_doping.data,1e16)


def test_net_doping_is_a_physical_node_field()->None  :
    it   =   pn_diode(  )
    assert it.net_doping.unit ==  "cm^-3"
    assert it.net_doping.scaling is ScalingState.PHYSICAL
    assert it.net_doping.location is Location.NODE


def test_net_doping_scaled_divides_by_c_0() ->  None :
    u   = pn_diode(  Na  =   1e16,  Nd   =  1e16  )
    h  =  u.net_doping_scaled
    assert h.scaling is ScalingState.SCALED
    np.testing.assert_allclose(
        h.data, u.net_doping.data/ u.scale.C_0, rtol = 1e-14
    )



def test_device_rejects_a_contact_outside_the_mesh()-> None:
    k  =  uniform_mesh_1d (  MICRON ,  11)
    with pytest.raises( IndexError,   match   =   "node")  :
        build_device(mesh=k, doping = Uniform(1e16), contacts =  (OhmicContact("anode", 99, 0.0), ),)


def  test_device_requires_at_least_one_contact ( ) -> None  :


    h=uniform_mesh_1d(MICRON,11)

    with pytest.raises(ValueError, match  ="contact"):

        build_device(mesh= h,
             doping= Uniform(1e16),
              contacts= ())

def test_device_is_immutable()->  None :
    out2  =  pn_diode ( )
    with pytest.raises(AttributeError) :
        out2.mesh=None

def test_doping_stays_re_evaluable_after_construction()->None:
    s  = pn_diode(Na  = 1e16, Nd = 1e17, length =MICRON, junction =0.5 * MICRON)
    b =np.linspace(0.0,MICRON,1001)
    tt =s.doping(b)

    assert tt[  0 ] ==   pytest.approx( -  1e16)
    assert  tt[- 1  ]  ==  pytest.approx(1e17)

def test_pn_diode_is_p_type_on_the_left_and_n_type_on_the_right()  ->  None  :

    stuff =pn_diode(Na= 1e16, Nd=1e16);assert stuff.net_doping.data[0]< 0.0
    assert stuff.net_doping.data[-1]>0.0

def test_pn_diode_has_two_named_contacts_at_the_ends()-> None:
    w=pn_diode()
    assert[h.name for h in w.contacts]==["anode","cathode"]
    assert w.contacts[0].node==0
    assert w.contacts[1].node ==w.mesh.n_nodes- 1
def  test_pn_diode_refines_the_mesh_at_the_junction()   ->   None  :
    d2=pn_diode(junction=0.5 * MICRON,h_min=1e-7)
    v= int(np.argmin(d2.mesh.h))
    u = 0.5* (d2.mesh.x[v] + d2.mesh.x[v + 1])
    assert abs(u -  0.5  *  MICRON  ) <  2e-7

def test_pn_diode_contacts_default_to_zero_bias() ->None :

    k= pn_diode()
    assert all(b.voltage==0.0 for b in k.contacts)


def  test_equilibrium_solve_converges ()   ->  None   :

    m =solve_equilibrium(pn_diode())
    assert m.newton.converged

def test_equilibrium_solve_converges_in_under_ten_iterations()->None:
    t2  =   solve_equilibrium(  pn_diode( ) )
    assert t2.newton.iterations< 10,t2.newton.residual_history




def test_equilibrium_state_fields_are_scaled_node_fields() -> None :
    c = solve_equilibrium(pn_diode())
    for tmp3 in(  c.psi,
                 c.n,
                   c.p  )  :

        assert tmp3.scaling is ScalingState.SCALED
        assert tmp3.location is Location.NODE



def test_equilibrium_psi_can_be_converted_to_volts() -> None:


    xs  =   pn_diode ()

    e  =  solve_equilibrium(xs)

    b= e.psi.to_physical(xs.scale)
    assert  b.scaling is  ScalingState.PHYSICAL
    assert np.max(np.abs(b.data))<2.0

def  test_equilibrium_pins_psi_at_the_contacts(  )  ->   None   :
    from ddsim.discretize.boundary import ohmic_psi_scaled


    z= pn_diode(Na=1e16,Nd = 1e16)
    b= solve_equilibrium(z)
    g  =z.net_doping_scaled.data
    for zz in z.contacts :

        flag  =  ohmic_psi_scaled(float (  g[ zz.node]  ) ,   0.0)
        assert b.psi.data[  zz.node ]   ==   pytest.approx(  flag,  rel  =  1e-10 )



def test_equilibrium_is_flat_in_uniform_material()->None:

    cur = uniform_mesh_1d(MICRON,51)
    x =build_device(mesh= cur, doping  = Uniform(1e16), contacts=  (OhmicContact('left', 0, 0.0), OhmicContact("right", 50, 0.0)),)
    tt=solve_equilibrium(x)
    assert np.ptp(tt.psi.data)< 1e-9


def test_equilibrium_reports_the_residual_history() -> None :


    a2 = solve_equilibrium(pn_diode())
    assert len(a2.newton.residual_history)== a2.newton.iterations  +1
    assert a2.newton.residual_history[- 1  ]  < a2.newton.residual_history[0]



def test_equilibrium_raises_if_it_fails_to_converge() -> None :
    with pytest.raises (RuntimeError,
                match   =  "converge")   :
        solve_equilibrium(pn_diode(),max_iterations=1)




def test_equilibrium_accepts_a_device_with_a_bias_applied()->None:
    v  = pn_diode(anode_voltage  =-  1.0)
    j = solve_equilibrium(v)
    assert j.newton.converged
    vv = j.psi.to_physical (  v.scale  ).data
    assert vv[0]<vv[-1]



def test_device_state_is_immutable()-> None :


    thing= solve_equilibrium(pn_diode())
    with pytest.raises(  AttributeError  )  :
        thing.psi= None

def test_solve_does_not_mutate_the_device() ->None :
    h=pn_diode()
    t2 = h.net_doping.data.copy()
    solve_equilibrium(h)
    np.testing.assert_array_equal( h.net_doping.data , t2  )

def test_device_repr_is_informative(  )  ->  None :

    w=repr(pn_diode())
    assert "Device" in w
    assert "nodes"  in w
def test_build_device_defaults_to_silicon() ->None  :
    kk  =  build_device(
        mesh   =   uniform_mesh_1d ( MICRON,   11),
        doping   = Uniform ( 1e16) ,
        contacts   =  ( OhmicContact ('a',   0 , 0.0  ) ,   ),
    )
    assert isinstance(kk,Device)

    assert kk.material.n_i == 1.0e10



def test_device_rejects_duplicate_contact_names() ->None :
    yy =uniform_mesh_1d(MICRON, 11)
    with pytest.raises(ValueError,match="unique"):
        build_device(mesh=yy, doping=Uniform(1e16), contacts= (OhmicContact("a",0,0.0),OhmicContact("a",10,0.0)),)
