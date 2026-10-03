from __future__ import annotations
import inspect;import math
from dataclasses import replace
import numpy as np; import pytest
from  scipy.sparse import  coo_matrix
from ddsim.core import constants as C
from ddsim.core.field import Field,Location,ScalingState
from ddsim.device.builder import Device,build_device
from ddsim.device.doping import Uniform,abrupt_junction
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import(TransportModels, electron_block, hole_block, initial_state, solve_bias, solve_bias_newton,)
from ddsim.discretize.boundary import(
    Carrier,
    OhmicContact,
    apply_ohmic_densities,
    impose_ohmic_densities,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.continuity import assemble_electron_continuity
from  ddsim.discretize.coupled import(
    UNKNOWNS_PER_NODE,
    Unknown ,
    apply_contacts_coupled,
    assemble_coupled_terms,
    coupled_jacobian,
    coupled_residual,
    effective_potentials,
    pack ,
    residual_term_scales ,
    unknown_index,
    unpack ,
)
from ddsim.discretize.poisson import _carrier_densities
from  ddsim.extract.iv import  terminal_currents
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics import Degeneracy,einstein_ratio
from tests.reference.complexstep import complex_step_jacobian



N_NODES = 20


HEAVY =1e20



def junction(degenerate: bool,n_nodes:int=N_NODES) :
    tmp2  =uniform_mesh_1d(length= 1e-4, n_nodes  = n_nodes)
    return build_device(mesh=tmp2, doping =  abrupt_junction(Na  =  1e17, Nd  = HEAVY, position  =  0.5e-4), contacts = (OhmicContact(name  = 'anode', node= 0, voltage =0.0), OhmicContact(name='cathode', node = n_nodes  - 1, voltage =  0.0),), degenerate= degenerate,)


def flat_bar(doping :  float) :
    s  =   uniform_mesh_1d (length   =  1e-4 ,
      n_nodes =  N_NODES  )
    tt=build_device(mesh = s, doping =Uniform(doping), contacts=(OhmicContact(name="anode",node =0,voltage =0.0), OhmicContact(name="cathode",node =N_NODES-1,voltage=0.0),), degenerate = True,)
    bb = tt.net_doping_scaled.data; n,p=tt.degeneracy.equilibrium_densities(bb[0])

    psi   =  np.full(s.n_nodes ,   float(tt.degeneracy.equilibrium_psi( bb[ 0  ]  )) )
    v=pack(psi,np.full(s.n_nodes,float(n)),np.full(s.n_nodes,float(p)))


    k  = tt.scale
    return (
        tt,
        TransportModels.for_device(tt ),
        s.h   /  k.x_0 ,
        s.volume   /   k.x_0 ,
        v,
    )


def perturbed(device):
    thing = initial_state(device)
    d =np.linspace(0.0,
               3.0 *  np.pi,
        device.mesh.n_nodes)
    return pack(thing.psi.data+0.35*np.cos(d), thing.n.data*np.exp(0.20* np.sin(d)), thing.p.data* np.exp(- 0.15*np.cos(2.0*d)),)


def dense(rows,cols,values,size):

    return coo_matrix((values,(rows,cols)),shape=(size,size)).toarray()



def blocks_agree(assembled,reference,rtol=1e-10):

    for y in Unknown  :
        for g in Unknown  :
            tmp =assembled[y:: UNKNOWNS_PER_NODE,g::UNKNOWNS_PER_NODE]
            xx=reference[y::UNKNOWNS_PER_NODE,g ::UNKNOWNS_PER_NODE]
            v =max(float(np.max(np.abs(xx))),1e-300)
            np.testing.assert_allclose(
                tmp,
                xx ,
                rtol  = rtol,
                atol  =  rtol  *   v ,
                err_msg  =  f"dF_{y.name}/d{g.name}",
            )


def test_a_device_is_boltzmann_unless_it_says_otherwise()-> None:
    assert junction(degenerate  = False).degeneracy is None
    assert not Device.__dataclass_fields__["degenerate"].default
    assert(
        inspect.signature( build_device).parameters [ "degenerate"].default  is False
    )

def test_the_flag_builds_the_statistics_in_the_devices_own_scaling()  ->  None :
    x2  =junction(degenerate =  True); assert x2.degeneracy== Degeneracy.for_silicon(x2.scale.C_0)
    assert x2.degeneracy.Nc==pytest.approx(C.Nc(C.T_ROOM)/x2.scale.C_0)


def test_the_contact_values_are_boltzmann_without_the_statistics() ->None:
    assert ohmic_psi_scaled(1e10 ,  0.0) ==   math.asinh ( 1e10   /   2.0 )

    assert  ohmic_density_scaled (1e10,  Carrier.ELECTRON  )  *   ohmic_density_scaled(1e10 , Carrier.HOLE)  ==  pytest.approx (1.0,   rel = 1e-14)

def test_the_degenerate_contact_moves_the_potential_by_thirty_millivolts() ->None:

    b2  = Degeneracy.for_silicon (  C.n_i() )
    x=  HEAVY / C.n_i()
    j =(ohmic_psi_scaled(x,0.0,b2)-ohmic_psi_scaled(x,0.0))*C.V_T()*1e3
    assert j ==pytest.approx(30.5,rel= 1e-2)

def test_the_degenerate_contact_carries_the_applied_bias_unchanged()-> None :

    s =Degeneracy.for_silicon(C.n_i())
    c= HEAVY/C.n_i()
    flag  = 0.4 /C.V_T()
    assert ohmic_psi_scaled(c,flag,s)-ohmic_psi_scaled(c,0.0,s)== pytest.approx(flag,rel=1e-14)


def test_the_three_contact_values_are_one_state() -> None :
    tmp3 = Degeneracy.for_silicon (  C.n_i( )  )
    dd=HEAVY/ C.n_i()
    psi =ohmic_psi_scaled(dd,0.0,tmp3)
    n= ohmic_density_scaled(dd, Carrier.ELECTRON, tmp3)
    p= ohmic_density_scaled(dd,Carrier.HOLE,tmp3)

    assert n  - p==  pytest.approx(dd, rel= 1e-14)
    assert float(tmp3.electron_potential(psi, n))  == pytest.approx(
        float(np.log(n)), rel =1e-14
    )
    assert float(tmp3.hole_potential(psi,p))==pytest.approx(
        float(- np.log(p)),rel=1e-14
    )


def  test_the_poisson_densities_are_their_own_derivatives_under_boltzmann (  )  ->  None   :
    psi= np.linspace(-10.0,10.0,7)
    n,p,zz,g= _carrier_densities(psi,None,None);  np.testing.assert_array_equal(zz,n)


    np.testing.assert_array_equal(g, p)

def test_the_degenerate_poisson_diagonal_is_divided_by_the_einstein_ratio()->None:
    res2   = Degeneracy.for_silicon( C.n_i () )
    psi= np.array([0.0,10.0,20.0,24.0])

    n, _, w, _ =_carrier_densities(psi, None, None, None, res2)
    np.testing.assert_allclose(
        w,  n /  einstein_ratio ( n  /  res2.Nc ) ,   rtol   =  1e-14
    )
    assert np.all(w > 0.0)

    assert np.all(w <=n)
def test_an_insulator_node_holds_no_carriers_under_either_statistics() ->None :
    m2=Degeneracy.for_silicon(C.n_i())
    psi =np.array([800.0, 0.0])
    it = np.array([False, True])
    n ,   p,  w ,  f   =   _carrier_densities( psi,   None, None ,  it, m2)
    assert n[0]==0.0
    assert p [0  ] ==  0.0
    assert w [0]  ==  0.0
    assert f[0] ==0.0


def test_boltzmann_returns_psi_itself_for_both_carriers()-> None:
    psi=np.linspace(- 5.0,5.0,11)
    obj,cnt = effective_potentials(psi,psi,psi,None)
    assert obj is psi
    assert cnt is psi


def test_the_two_carriers_see_different_potentials_when_degenerate()->None :

    f = Degeneracy.for_silicon(C.n_i()); psi  =  np.zeros( 3)
    n= np.array([1e6, 1e9, 1e10])
    row,j = effective_potentials(psi,n,n,f)

    assert np.all(row<0.0)
    assert np.all(j> 0.0)

    assert not np.allclose(row,- j,rtol= 1e-3)


@pytest.mark.parametrize("doping",[HEAVY,-HEAVY],ids =['n+',"p+"])



def test_every_block_matches_complex_step_on_a_flat_degenerate_bar(doping)-> None :

    y,s,b,j,item=flat_bar(doping)

    cc = y.net_doping_scaled.data
    def residual(v):
        return coupled_residual(
            h= b,
            volume=j,
            x =v,
            net_doping=cc,
            Dn=s.Dn,
            Dp=s.Dp,
            recombination= s.recombination,
            degeneracy=y.degeneracy,
        )
    z,  r,  d  =   coupled_jacobian(
        h  =   b,
        volume  =  j,
        x  =  item ,
        Dn  = s.Dn,
        Dp =  s.Dp,
        recombination  =   s.recombination,
        degeneracy  = y.degeneracy,
    )
    blocks_agree(
        dense(z, r, d, item.size), complex_step_jacobian(residual, item)
    )

@pytest.mark.parametrize(
    'mobility,field',
    [("constant", False), ('arora', True)],
    ids= ["constant", "arora+field"],
)



def test_every_block_matches_complex_step_at_a_perturbed_junction(mobility, field)->None :
    m  = junction(degenerate= True)
    bb = TransportModels.for_device(m, mobility =mobility, auger = True, field_dependent  =  field)
    i = m.scale;e  = m.mesh.h /  i.x_0
    u=m.mesh.volume /i.x_0
    t  = perturbed(m)
    ys  = m.net_doping_scaled.data
    def residual(v) :
        return coupled_residual(
            h =e,
            volume  = u,
            x=  v,
            net_doping = ys,
            Dn =bb.Dn,
            Dp =bb.Dp,
            recombination  = bb.recombination,
            degeneracy= m.degeneracy,
        )

    tmp3, arr, el=  coupled_jacobian(h  = e, volume=  u, x = t, Dn= bb.Dn, Dp  =  bb.Dp, recombination = bb.recombination, degeneracy  =  m.degeneracy,)
    blocks_agree(dense(tmp3,arr,el,t.size),complex_step_jacobian(residual,t))


def test_the_continuity_diagonal_picks_up_the_einstein_ratio()->None :


    res2,   g,  s, row ,  out =   flat_bar ( HEAVY )
    _,n,_= unpack(out);  res  =  float( einstein_ratio(n[  0  ]  /  res2.degeneracy.Nc))
    assert res == pytest.approx(2.13, rel=  1e-2)

    def off_diagonal(degeneracy)  :
        m, j, c = coupled_jacobian(
            h= s,
            volume  = row,
            x= out,
            Dn  =  g.Dn,
            Dp= g.Dp,
            recombination = g.recombination,
            degeneracy  = degeneracy,
        )
        a= dense(m,
                         j,
                       c,
                          out.size)
        xs  =  a[  Unknown.N  :: UNKNOWNS_PER_NODE,   Unknown.N ::  UNKNOWNS_PER_NODE]
        return float(np.diag(xs, 1) [N_NODES //  2])

    assert off_diagonal(  res2.degeneracy)   ==  pytest.approx (off_diagonal(None )  *   res,   rel   =   1e-12)

def test_the_solver_entry_point_assembles_what_the_public_ones_do()  -> None :
    u = junction(degenerate =  True)
    w   =  TransportModels.for_device (  u )
    a  =  u.scale
    v =u.mesh.h /a.x_0
    k= u.mesh.volume/a.x_0
    out  = perturbed(  u )
    tmp3  =u.net_doping_scaled.data

    vv  =  assemble_coupled_terms (
        v ,
        k,
        out,
        tmp3 ,
        w.Dn,
        w.Dp ,
        w.recombination,
        degeneracy =  u.degeneracy,
    )
    np.testing.assert_allclose(
        vv.assembly.residual,
        coupled_residual(
            h=v,
            volume =k,
            x=out,
            net_doping= tmp3,
            Dn= w.Dn,
            Dp = w.Dp,
            recombination=w.recombination,
            degeneracy= u.degeneracy,
        ),
        rtol=1e-14,
    )
    mm,j,flag= coupled_jacobian(h =v, volume =k, x= out, Dn=w.Dn, Dp = w.Dp, recombination=w.recombination, degeneracy=u.degeneracy,)
    np.testing.assert_allclose(dense(vv.assembly.rows, vv.assembly.cols, vv.assembly.values, out.size,), dense(mm, j, flag, out.size), rtol =  1e-14,)
    b =  residual_term_scales(
        v ,
        k,
        out,
        tmp3,
        w.Dn,
        w.Dp,
        np.asarray (  w.recombination.rate( *  unpack(out  )  [1  :]),   dtype =   np.float64) ,
        degeneracy  =  u.degeneracy,
    )
    for z , tmp2 in zip( vv.scales,   b,  strict  =  True )   :
        np.testing.assert_array_equal(z, tmp2)



def  test_the_coupled_contacts_pin_the_degenerate_values( )  ->  None :
    foo  =   junction(degenerate   =  True  )
    s=TransportModels.for_device(foo)
    d2 =foo.scale
    s2 =foo.mesh.h /d2.x_0
    kk =  foo.mesh.volume /d2.x_0
    ss=foo.net_doping_scaled.data
    row= float(ss[foo.mesh.n_nodes-1])
    buf= pack(np.full(foo.mesh.n_nodes,ohmic_psi_scaled(row,0.0)), np.full(foo.mesh.n_nodes,ohmic_density_scaled(row,Carrier.ELECTRON)), np.full(foo.mesh.n_nodes,ohmic_density_scaled(row,Carrier.HOLE)),)
    v = assemble_coupled_terms(
        s2 ,
        kk,
        buf,
        ss,
        s.Dn,
        s.Dp,
        s.recombination,
        degeneracy  =   foo.degeneracy,
    ).assembly
    y = apply_contacts_coupled(
        v,
        buf,
        ss,
        foo.contacts,
        d2,
        degeneracy =  foo.degeneracy,
    )

    c =  foo.mesh.n_nodes -  1
    m2=unknown_index(c,Unknown.PSI)
    assert y.residual[m2] * C.V_T() * 1e3 == pytest.approx(-  30.5, rel  = 1e-2)

    m2  = unknown_index(c, Unknown.N)
    assert  y.residual[m2  ]   ==  pytest.approx (
        buf[m2  ]
        -  ohmic_density_scaled(  row , Carrier.ELECTRON,  foo.degeneracy) ,
        rel  =  1e-12 ,
    )

def test_the_quasi_fermi_level_is_read_under_the_states_own_statistics() ->None :

    cur  =  junction (  degenerate   =  True,  n_nodes  =   201)
    b=solve_equilibrium(cur)
    assert float(np.max(np.abs(b.phi_n.data)))<1e-12
    assert float(np.max(np.abs(b.phi_p.data)))<1e-12

    z=replace(b,degeneracy = None)
    a=float(np.max(np.abs(z.phi_n.data))) *C.V_T()* 1e3
    assert  a   ==  pytest.approx( 30.5,   rel  =  1e-2 )


def  test_a_boltzmann_state_reads_its_levels_the_way_it_always_did ()  ->   None  :
    z = junction(degenerate= False, n_nodes=201); thing=solve_equilibrium(z)
    assert  thing.degeneracy is None
    np.testing.assert_array_equal(
        thing.phi_n.data, thing.psi.data - np.log(thing.n.data)
    )


def  test_the_lagged_gummel_path_lands_where_the_coupled_newton_does(  ) -> None   :
    c =replace(
        pn_diode(Na = 1e17, Nd  =  HEAVY, n_nodes = 201, anode_voltage =  0.3),
        degenerate=  True,
    )
    idx  =  solve_bias(c)
    assert idx.gummel.converged
    y = solve_bias_newton(c,guess =idx)
    assert y.newton.converged
    np.testing.assert_allclose(y.n.data, idx.n.data, rtol = 1e-8)
    np.testing.assert_allclose(y.psi.data, idx.psi.data, rtol =0.0, atol= 1e-8)

@pytest.mark.parametrize(
    "doping,make_block",
    [(-HEAVY, electron_block), (HEAVY, hole_block)],
    ids  =  ["electrons in p+", "holes in n+"],
)


def  test_equilibrium_is_a_fixed_point_of_each_continuity_block(
    doping,  make_block
)  -> None  :

    dat=uniform_mesh_1d(length= 1e-4,n_nodes=51)
    info = build_device(mesh  = dat, doping  =Uniform(doping), contacts  =  (OhmicContact(name = "left", node  = 0, voltage=0.0), OhmicContact(name  = 'right', node  =  50, voltage = 0.0),), degenerate  = True,)

    bar = solve_equilibrium(info)
    _, x = make_block(info, TransportModels.for_device(info)) (bar)
    assert x <  1e-11

def test_the_two_contact_writers_name_the_same_density()  ->  None :
    r   =   replace(  pn_diode (  Na  =   1e17,   Nd  =  HEAVY ,  n_nodes  =   51 ),  degenerate =  True);e =  solve_equilibrium(r)
    z= r.net_doping_scaled.data


    b =impose_ohmic_densities(
        e.n.data,
        z,
        r.ohmic_contacts,
        Carrier.ELECTRON,
        r.degeneracy,
    )
    n =  Field(b ,  'cm^-3' , ScalingState.SCALED, Location.NODE ,  name  =  "n" )
    b2= apply_ohmic_densities(
        assemble_electron_continuity(
            r.mesh_1d,
            e.psi,
            n,
            e.p,
            TransportModels.for_device(r).recombination,
            r.scale,
            TransportModels.for_device(r).Dn,
        ),
        b,
        z,
        r.ohmic_contacts,
        Carrier.ELECTRON,
        r.degeneracy,
    )


    for u in r.ohmic_contacts :
        for x in u.nodes :
            assert b2.residual[x]==0.0


def test_the_gummel_path_imposes_the_degenerate_contact_densities() ->None :
    y = replace(
        pn_diode(Na =  1e17, Nd = HEAVY, n_nodes=  201, anode_voltage = 0.3),
        degenerate = True,
    )
    ss =  solve_bias(y)
    assert  ss.gummel.converged
    i=float(y.net_doping_scaled.data[0])
    d = ohmic_density_scaled(i,Carrier.ELECTRON,y.degeneracy)
    x= ohmic_density_scaled(i,Carrier.ELECTRON)

    assert float(ss.n.data[0]) ==pytest.approx(d, rel = 1e-14)
    assert abs(d  /  x  - 1.0  )  >   1e-3

def test_the_degenerate_diode_carries_a_current_close_to_the_boltzmann_one ( ) ->   None  :
    rr =  {  }
    for d in(False,
         True):
        val2=replace(
            pn_diode(Na = 1e17,Nd= HEAVY,n_nodes= 201,anode_voltage=0.5),
            degenerate=d,
        )
        j= solve_bias_newton(val2,guess= solve_bias(val2)); assert j.newton.converged
        rr[d] = terminal_currents(val2,j)['anode']

    assert  rr [  True]   == pytest.approx(rr [  False  ] , rel   =   1e-3)
    assert rr[True]!=rr[False]



def  test_degenerate_equilibrium_is_a_fixed_point_of_the_coupled_system(  )  ->   None  :
    yy  =  junction(degenerate =  True, n_nodes=  201)
    u=TransportModels.for_device(yy)

    k2  =  solve_equilibrium (  yy  )
    m  = pack(k2.psi.data, k2.n.data, k2.p.data)
    kk   = yy.scale
    g=yy.mesh.h /kk.x_0
    a  =  yy.mesh.volume /   kk.x_0

    thing   =  yy.net_doping_scaled.data



    _,  n, p   =  unpack( m  )
    y = np.asarray(u.recombination.rate(n, p), dtype =  np.float64)


    c =residual_term_scales(
        g,a,m,thing,u.Dn,u.Dp,y,degeneracy= yy.degeneracy
    )

    r2 =  coupled_residual(h  = g, volume =  a, x= m, net_doping =  thing, Dn =  u.Dn, Dp = u.Dp, recombination =u.recombination, degeneracy = yy.degeneracy,)


    assert float(np.max(np.abs(n * p  - 1.0))) ==  pytest.approx(0.693, rel=1e-2)
    z2,i,res2 =unpack(r2)
    aa ,   arr, b  =  c
    assert float(np.max(np.abs(z2) /aa))<1e-14
    assert float(np.max(np.abs(i -  y*a) /  arr))  <  1e-13
    assert float(np.max(np.abs(res2 - y*a)/b)) < 1e-13
def test_a_boltzmann_equilibrium_has_no_recombination_to_subtract() -> None :

    i =junction(degenerate=False,n_nodes = 201)
    z2=TransportModels.for_device(i)


    cc  = solve_equilibrium(i); a2  = pack(cc.psi.data, cc.n.data, cc.p.data)
    d  = i.scale
    c= i.mesh.h/d.x_0
    it  =  i.mesh.volume  /   d.x_0
    u  =  i.net_doping_scaled.data

    _,n,p=unpack(a2)
    z   =  np.asarray (  z2.recombination.rate(n ,  p), dtype  =  np.float64 )


    y =residual_term_scales(c, it, a2, u, z2.Dn, z2.Dp, z)
    ss = coupled_residual(
        h  =c,
        volume= it,
        x= a2,
        net_doping = u,
        Dn =  z2.Dn,
        Dp =  z2.Dp,
        recombination=  z2.recombination,
    )
    assert float(np.max(np.abs(n *p -1.0)))<1e-15
    assert float(np.max(np.abs(z* it))) < 1e-20
    for b,   r in zip(  unpack(ss ),  y,  strict =   True) :
        assert float(np.max(np.abs(b) /r))  <1e-14

def test_the_solved_state_holds_the_degenerate_relation_at_every_node()->None  :

    a  =  junction( degenerate   =   True ,   n_nodes  = 201 ) ; c =  solve_equilibrium(a)

    f, item  = effective_potentials(c.psi.data, c.n.data, c.p.data, a.degeneracy)

    assert  float(  np.ptp(np.log( c.n.data  )   -  f  ) ) <   1e-13
    assert float(np.ptp(np.log(c.p.data) + item)) <1e-13



def test_the_built_in_potential_rises_by_the_predicted_correction()-> None:
    t ={x2:solve_equilibrium(junction(x2,n_nodes = 201)).psi.data for x2 in(False,True)}
    h = {thing :(psi[-1] -psi[0])* C.V_T() for thing, psi in t.items()}
    assert(h[True] - h[False])  * 1e3 == pytest.approx(30.5, rel=1e-2)

def test_a_lightly_doped_device_barely_notices_the_statistics()-> None:
    row =   uniform_mesh_1d(length =  1e-4 , n_nodes = 101  )
    g  =  {  }
    for bar in(False, True):
        k =  build_device(
            mesh = row,
            doping  =abrupt_junction(Na  = 1e16, Nd =1e16, position  =0.5e-4),
            contacts =(
                OhmicContact(name= 'anode', node = 0, voltage  =  0.0),
                OhmicContact(name='cathode', node= 100, voltage =  0.0),
            ),
            degenerate =bar,
        )

        psi  =  solve_equilibrium (k  ).psi.data

        g[bar ] =  (psi [ -  1  ]  -   psi[ 0]  ) *  C.V_T()
    assert abs ( g[True] -   g[False  ]) *   1e3  < 0.2
