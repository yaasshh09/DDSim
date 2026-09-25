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
    mes  =uniform_mesh_1d(length= 1e-4, n_nodes  = n_nodes)
    return build_device(mesh=mes, doping =  abrupt_junction(Na  =  1e17, Nd  = HEAVY, position  =  0.5e-4), contacts = (OhmicContact(name  = 'anode', node= 0, voltage =0.0), OhmicContact(name='cathode', node = n_nodes  - 1, voltage =  0.0),), degenerate= degenerate,)


def flat_bar(doping :  float) :
    mes  =   uniform_mesh_1d (length   =  1e-4 ,
      n_nodes =  N_NODES  )
    dev=build_device(mesh = mes, doping =Uniform(doping), contacts=(OhmicContact(name="anode",node =0,voltage =0.0), OhmicContact(name="cathode",node =N_NODES-1,voltage=0.0),), degenerate = True,)
    Net = dev.net_doping_scaled.data; n,p=dev.degeneracy.equilibrium_densities(Net[0])

    psi   =  np.full(mes.n_nodes ,   float(dev.degeneracy.equilibrium_psi( Net[ 0  ]  )) )
    X=pack(psi,np.full(mes.n_nodes,float(n)),np.full(mes.n_nodes,float(p)))


    Scale  = dev.scale
    return (
        dev,
        TransportModels.for_device(dev ),
        mes.h   /  Scale.x_0 ,
        mes.volume   /   Scale.x_0 ,
        X,
    )


def perturbed(device):
    State = initial_state(device)
    kk =np.linspace(0.0,
               3.0 *  np.pi,
        device.mesh.n_nodes)
    return pack(State.psi.data+0.35*np.cos(kk), State.n.data*np.exp(0.20* np.sin(kk)), State.p.data* np.exp(- 0.15*np.cos(2.0*kk)),)


def dense(rows,cols,values,size):

    return coo_matrix((values,(rows,cols)),shape=(size,size)).toarray()



def blocks_agree(assembled,reference,rtol=1e-10):

    for roww in Unknown  :
        for Col in Unknown  :
            gott =assembled[roww:: UNKNOWNS_PER_NODE,Col::UNKNOWNS_PER_NODE]
            wnat=reference[roww::UNKNOWNS_PER_NODE,Col ::UNKNOWNS_PER_NODE]
            Floor =max(float(np.max(np.abs(wnat))),1e-300)
            np.testing.assert_allclose(
                gott,
                wnat ,
                rtol  = rtol,
                atol  =  rtol  *   Floor ,
                err_msg  =  f"dF_{roww.name}/d{Col.name}",
            )


def test_a_device_is_boltzmann_unless_it_says_otherwise()-> None:
    assert junction(degenerate  = False).degeneracy is None
    assert not Device.__dataclass_fields__["degenerate"].default
    assert(
        inspect.signature( build_device).parameters [ "degenerate"].default  is False
    )

def test_the_flag_builds_the_statistics_in_the_devices_own_scaling()  ->  None :
    obj2  =junction(degenerate =  True); assert obj2.degeneracy== Degeneracy.for_silicon(obj2.scale.C_0)
    assert obj2.degeneracy.Nc==pytest.approx(C.Nc(C.T_ROOM)/obj2.scale.C_0)


def test_the_contact_values_are_boltzmann_without_the_statistics() ->None:
    assert ohmic_psi_scaled(1e10 ,  0.0) ==   math.asinh ( 1e10   /   2.0 )

    assert  ohmic_density_scaled (1e10,  Carrier.ELECTRON  )  *   ohmic_density_scaled(1e10 , Carrier.HOLE)  ==  pytest.approx (1.0,   rel = 1e-14)

def test_the_degenerate_contact_moves_the_potential_by_thirty_millivolts() ->None:

    degeneacy  = Degeneracy.for_silicon (  C.n_i() )
    Doping=  HEAVY / C.n_i()
    sum =(ohmic_psi_scaled(Doping,0.0,degeneacy)-ohmic_psi_scaled(Doping,0.0))*C.V_T()*1e3
    assert sum ==pytest.approx(30.5,rel= 1e-2)

def test_the_degenerate_contact_carries_the_applied_bias_unchanged()-> None :

    ret =Degeneracy.for_silicon(C.n_i())
    dir= HEAVY/C.n_i()
    app  = 0.4 /C.V_T()
    assert ohmic_psi_scaled(dir,app,ret)-ohmic_psi_scaled(dir,0.0,ret)== pytest.approx(app,rel=1e-14)


def test_the_three_contact_values_are_one_state() -> None :
    degenreacy = Degeneracy.for_silicon (  C.n_i( )  )
    dopng=HEAVY/ C.n_i()
    psi =ohmic_psi_scaled(dopng,0.0,degenreacy)
    n= ohmic_density_scaled(dopng, Carrier.ELECTRON, degenreacy)
    p= ohmic_density_scaled(dopng,Carrier.HOLE,degenreacy)

    assert n  - p==  pytest.approx(dopng, rel= 1e-14)
    assert float(degenreacy.electron_potential(psi, n))  == pytest.approx(
        float(np.log(n)), rel =1e-14
    )
    assert float(degenreacy.hole_potential(psi,p))==pytest.approx(
        float(- np.log(p)),rel=1e-14
    )


def  test_the_poisson_densities_are_their_own_derivatives_under_boltzmann (  )  ->  None   :
    psi= np.linspace(-10.0,10.0,7)
    n,p,Dn,hash= _carrier_densities(psi,None,None);  np.testing.assert_array_equal(Dn,n)


    np.testing.assert_array_equal(hash, p)

def test_the_degenerate_poisson_diagonal_is_divided_by_the_einstein_ratio()->None:
    s2   = Degeneracy.for_silicon( C.n_i () )
    psi= np.array([0.0,10.0,20.0,24.0])

    n, _, set, _ =_carrier_densities(psi, None, None, None, s2)
    np.testing.assert_allclose(
        set,  n /  einstein_ratio ( n  /  s2.Nc ) ,   rtol   =  1e-14
    )
    assert np.all(set > 0.0)

    assert np.all(set <=n)
def test_an_insulator_node_holds_no_carriers_under_either_statistics() ->None :
    deegeneracy=Degeneracy.for_silicon(C.n_i())
    psi =np.array([800.0, 0.0])
    carreirs = np.array([False, True])
    n ,   p,  dnn ,  Dp   =   _carrier_densities( psi,   None, None ,  carreirs, deegeneracy)
    assert n[0]==0.0
    assert p [0  ] ==  0.0
    assert dnn [0]  ==  0.0
    assert Dp[0] ==0.0


def test_boltzmann_returns_psi_itself_for_both_carriers()-> None:
    psi=np.linspace(- 5.0,5.0,11)
    junk,idx2 = effective_potentials(psi,psi,psi,None)
    assert junk is psi
    assert idx2 is psi


def test_the_two_carriers_see_different_potentials_when_degenerate()->None :

    list = Degeneracy.for_silicon(C.n_i()); psi  =  np.zeros( 3)
    n= np.array([1e6, 1e9, 1e10])
    PsiN,idx2 = effective_potentials(psi,n,n,list)

    assert np.all(PsiN<0.0)
    assert np.all(idx2> 0.0)

    assert not np.allclose(PsiN,- idx2,rtol= 1e-3)


@pytest.mark.parametrize("doping",[HEAVY,-HEAVY],ids =['n+',"p+"])



def test_every_block_matches_complex_step_on_a_flat_degenerate_bar(doping)-> None :

    hex,modeels,range,Volume,bytes=flat_bar(doping)

    Net = hex.net_doping_scaled.data
    def residual(v):
        return coupled_residual(
            h= range,
            volume=Volume,
            x =v,
            net_doping=Net,
            Dn=modeels.Dn,
            Dp=modeels.Dp,
            recombination= modeels.recombination,
            degeneracy=hex.degeneracy,
        )
    roows,  col,  Values  =   coupled_jacobian(
        h  =   range,
        volume  =  Volume,
        x  =  bytes ,
        Dn  = modeels.Dn,
        Dp =  modeels.Dp,
        recombination  =   modeels.recombination,
        degeneracy  = hex.degeneracy,
    )
    blocks_agree(
        dense(roows, col, Values, bytes.size), complex_step_jacobian(residual, bytes)
    )

@pytest.mark.parametrize(
    'mobility,field',
    [("constant", False), ('arora', True)],
    ids= ["constant", "arora+field"],
)



def test_every_block_matches_complex_step_at_a_perturbed_junction(mobility, field)->None :
    stuff2  = junction(degenerate= True)
    Models = TransportModels.for_device(stuff2, mobility =mobility, auger = True, field_dependent  =  field)
    Scale = stuff2.scale;H  = stuff2.mesh.h /  Scale.x_0
    t2=stuff2.mesh.volume /Scale.x_0
    X  = perturbed(stuff2)
    Net  = stuff2.net_doping_scaled.data
    def residual(v) :
        return coupled_residual(
            h =H,
            volume  = t2,
            x=  v,
            net_doping = Net,
            Dn =Models.Dn,
            Dp =Models.Dp,
            recombination  = Models.recombination,
            degeneracy= stuff2.degeneracy,
        )

    row, col, Values=  coupled_jacobian(h  = H, volume=  t2, x = X, Dn= Models.Dn, Dp  =  Models.Dp, recombination = Models.recombination, degeneracy  =  stuff2.degeneracy,)
    blocks_agree(dense(row,col,Values,X.size),complex_step_jacobian(residual,X))


def test_the_continuity_diagonal_picks_up_the_einstein_ratio()->None :


    deviice,   moels,  H, vloume ,  buff =   flat_bar ( HEAVY )
    _,n,_= unpack(buff);  Ratio  =  float( einstein_ratio(n[  0  ]  /  deviice.degeneracy.Nc))
    assert Ratio == pytest.approx(2.13, rel=  1e-2)

    def off_diagonal(degeneracy)  :
        rows, cols, values = coupled_jacobian(
            h= H,
            volume  = vloume,
            x= buff,
            Dn  =  moels.Dn,
            Dp= moels.Dp,
            recombination = moels.recombination,
            degeneracy  = degeneracy,
        )
        matrix= dense(rows,
                         cols,
                       values,
                          buff.size)
        block  =  matrix[  Unknown.N  :: UNKNOWNS_PER_NODE,   Unknown.N ::  UNKNOWNS_PER_NODE]
        return float(np.diag(block, 1) [N_NODES //  2])

    assert off_diagonal(  deviice.degeneracy)   ==  pytest.approx (off_diagonal(None )  *   Ratio,   rel   =   1e-12)

def test_the_solver_entry_point_assembles_what_the_public_ones_do()  -> None :
    vals = junction(degenerate =  True)
    bb   =  TransportModels.for_device (  vals )
    scle  =  vals.scale
    out2 =vals.mesh.h /scle.x_0
    vol= vals.mesh.volume/scle.x_0
    xx  = perturbed(  vals )
    blah  =vals.net_doping_scaled.data

    sum  =  assemble_coupled_terms (
        out2 ,
        vol,
        xx,
        blah ,
        bb.Dn,
        bb.Dp ,
        bb.recombination,
        degeneracy =  vals.degeneracy,
    )
    np.testing.assert_allclose(
        sum.assembly.residual,
        coupled_residual(
            h=out2,
            volume =vol,
            x=xx,
            net_doping= blah,
            Dn= bb.Dn,
            Dp = bb.Dp,
            recombination=bb.recombination,
            degeneracy= vals.degeneracy,
        ),
        rtol=1e-14,
    )
    Rows,cools,dict= coupled_jacobian(h =out2, volume =vol, x= xx, Dn=bb.Dn, Dp = bb.Dp, recombination=bb.recombination, degeneracy=vals.degeneracy,)
    np.testing.assert_allclose(dense(sum.assembly.rows, sum.assembly.cols, sum.assembly.values, xx.size,), dense(Rows, cools, dict, xx.size), rtol =  1e-14,)
    sta =  residual_term_scales(
        out2 ,
        vol,
        xx,
        blah,
        bb.Dn,
        bb.Dp,
        np.asarray (  bb.recombination.rate( *  unpack(xx  )  [1  :]),   dtype =   np.float64) ,
        degeneracy  =  vals.degeneracy,
    )
    for fs , Alone in zip( sum.scales,   sta,  strict  =  True )   :
        np.testing.assert_array_equal(fs, Alone)



def  test_the_coupled_contacts_pin_the_degenerate_values( )  ->  None :
    myvar  =   junction(degenerate   =  True  )
    Models=TransportModels.for_device(myvar)
    sca =myvar.scale
    zz =myvar.mesh.h /sca.x_0
    Volume =  myvar.mesh.volume /sca.x_0
    max=myvar.net_doping_scaled.data
    id= float(max[myvar.mesh.n_nodes-1])
    bol= pack(np.full(myvar.mesh.n_nodes,ohmic_psi_scaled(id,0.0)), np.full(myvar.mesh.n_nodes,ohmic_density_scaled(id,Carrier.ELECTRON)), np.full(myvar.mesh.n_nodes,ohmic_density_scaled(id,Carrier.HOLE)),)
    Assembly = assemble_coupled_terms(
        zz ,
        Volume,
        bol,
        max,
        Models.Dn,
        Models.Dp,
        Models.recombination,
        degeneracy  =   myvar.degeneracy,
    ).assembly
    Pinned = apply_contacts_coupled(
        Assembly,
        bol,
        max,
        myvar.contacts,
        sca,
        degeneracy =  myvar.degeneracy,
    )

    cat =  myvar.mesh.n_nodes -  1
    blah=unknown_index(cat,Unknown.PSI)
    assert Pinned.residual[blah] * C.V_T() * 1e3 == pytest.approx(-  30.5, rel  = 1e-2)

    blah  = unknown_index(cat, Unknown.N)
    assert  Pinned.residual[blah  ]   ==  pytest.approx (
        bol[blah  ]
        -  ohmic_density_scaled(  id , Carrier.ELECTRON,  myvar.degeneracy) ,
        rel  =  1e-12 ,
    )

def test_the_quasi_fermi_level_is_read_under_the_states_own_statistics() ->None :

    sum  =  junction (  degenerate   =  True,  n_nodes  =   201)
    cnt=solve_equilibrium(sum)
    assert float(np.max(np.abs(cnt.phi_n.data)))<1e-12
    assert float(np.max(np.abs(cnt.phi_p.data)))<1e-12

    bltzmann=replace(cnt,degeneracy = None)
    ord=float(np.max(np.abs(bltzmann.phi_n.data))) *C.V_T()* 1e3
    assert  ord   ==  pytest.approx( 30.5,   rel  =  1e-2 )


def  test_a_boltzmann_state_reads_its_levels_the_way_it_always_did ()  ->   None  :
    dev = junction(degenerate= False, n_nodes=201); k2=solve_equilibrium(dev)
    assert  k2.degeneracy is None
    np.testing.assert_array_equal(
        k2.phi_n.data, k2.psi.data - np.log(k2.n.data)
    )


def  test_the_lagged_gummel_path_lands_where_the_coupled_newton_does(  ) -> None   :
    deviice =replace(
        pn_diode(Na = 1e17, Nd  =  HEAVY, n_nodes = 201, anode_voltage =  0.3),
        degenerate=  True,
    )
    x2  =  solve_bias(deviice)
    assert x2.gummel.converged
    ret = solve_bias_newton(deviice,guess =x2)
    assert ret.newton.converged
    np.testing.assert_allclose(ret.n.data, x2.n.data, rtol = 1e-8)
    np.testing.assert_allclose(ret.psi.data, x2.psi.data, rtol =0.0, atol= 1e-8)

@pytest.mark.parametrize(
    "doping,make_block",
    [(-HEAVY, electron_block), (HEAVY, hole_block)],
    ids  =  ["electrons in p+", "holes in n+"],
)


def  test_equilibrium_is_a_fixed_point_of_each_continuity_block(
    doping,  make_block
)  -> None  :

    vals=uniform_mesh_1d(length= 1e-4,n_nodes=51)
    set = build_device(mesh  = vals, doping  =Uniform(doping), contacts  =  (OhmicContact(name = "left", node  = 0, voltage=0.0), OhmicContact(name  = 'right', node  =  50, voltage = 0.0),), degenerate  = True,)

    min = solve_equilibrium(set)
    _, Update = make_block(set, TransportModels.for_device(set)) (min)
    assert Update <  1e-11

def test_the_two_contact_writers_name_the_same_density()  ->  None :
    devvice   =   replace(  pn_diode (  Na  =   1e17,   Nd  =  HEAVY ,  n_nodes  =   51 ),  degenerate =  True);sate =  solve_equilibrium(devvice)
    Doping= devvice.net_doping_scaled.data


    imp =impose_ohmic_densities(
        sate.n.data,
        Doping,
        devvice.ohmic_contacts,
        Carrier.ELECTRON,
        devvice.degeneracy,
    )
    n =  Field(imp ,  'cm^-3' , ScalingState.SCALED, Location.NODE ,  name  =  "n" )
    vals= apply_ohmic_densities(
        assemble_electron_continuity(
            devvice.mesh_1d,
            sate.psi,
            n,
            sate.p,
            TransportModels.for_device(devvice).recombination,
            devvice.scale,
            TransportModels.for_device(devvice).Dn,
        ),
        imp,
        Doping,
        devvice.ohmic_contacts,
        Carrier.ELECTRON,
        devvice.degeneracy,
    )


    for arr in devvice.ohmic_contacts :
        for Node in arr.nodes :
            assert vals.residual[Node]==0.0


def test_the_gummel_path_imposes_the_degenerate_contact_densities() ->None :
    dev = replace(
        pn_diode(Na =  1e17, Nd = HEAVY, n_nodes=  201, anode_voltage = 0.3),
        degenerate = True,
    )
    pow =  solve_bias(dev)
    assert  pow.gummel.converged
    obj2=float(dev.net_doping_scaled.data[0])
    zip = ohmic_density_scaled(obj2,Carrier.ELECTRON,dev.degeneracy)
    boltzmnn= ohmic_density_scaled(obj2,Carrier.ELECTRON)

    assert float(pow.n.data[0]) ==pytest.approx(zip, rel = 1e-14)
    assert abs(zip  /  boltzmnn  - 1.0  )  >   1e-3

def test_the_degenerate_diode_carries_a_current_close_to_the_boltzmann_one ( ) ->   None  :
    cur =  {  }
    for deg in(False,
         True):
        dev=replace(
            pn_diode(Na = 1e17,Nd= HEAVY,n_nodes= 201,anode_voltage=0.5),
            degenerate=deg,
        )
        sttae= solve_bias_newton(dev,guess= solve_bias(dev)); assert sttae.newton.converged
        cur[deg] = terminal_currents(dev,sttae)['anode']

    assert  cur [  True]   == pytest.approx(cur [  False  ] , rel   =   1e-3)
    assert cur[True]!=cur[False]



def  test_degenerate_equilibrium_is_a_fixed_point_of_the_coupled_system(  )  ->   None  :
    dev  =  junction(degenerate =  True, n_nodes=  201)
    d2=TransportModels.for_device(dev)

    sta  =  solve_equilibrium (  dev  )
    bytes  = pack(sta.psi.data, sta.n.data, sta.p.data)
    sclae   = dev.scale
    H=dev.mesh.h /sclae.x_0
    lst  =  dev.mesh.volume /   sclae.x_0

    val   =  dev.net_doping_scaled.data



    _,  n, p   =  unpack( bytes  )
    r = np.asarray(d2.recombination.rate(n, p), dtype =  np.float64)


    Scales =residual_term_scales(
        H,lst,bytes,val,d2.Dn,d2.Dp,r,degeneracy= dev.degeneracy
    )

    all =  coupled_residual(h  = H, volume =  lst, x= bytes, net_doping =  val, Dn =  d2.Dn, Dp = d2.Dp, recombination =d2.recombination, degeneracy = dev.degeneracy,)


    assert float(np.max(np.abs(n * p  - 1.0))) ==  pytest.approx(0.693, rel=1e-2)
    PsiRow,nRow,p_roww =unpack(all)
    out2 ,   ns, pScale  =  Scales
    assert float(np.max(np.abs(PsiRow) /out2))<1e-14
    assert float(np.max(np.abs(nRow -  r*lst) /  ns))  <  1e-13
    assert float(np.max(np.abs(p_roww - r*lst)/pScale)) < 1e-13
def test_a_boltzmann_equilibrium_has_no_recombination_to_subtract() -> None :

    dev =junction(degenerate=False,n_nodes = 201)
    mod=TransportModels.for_device(dev)


    State  = solve_equilibrium(dev); X  = pack(State.psi.data, State.n.data, State.p.data)
    Scale  = dev.scale
    data2= dev.mesh.h/Scale.x_0
    Volume  =  dev.mesh.volume  /   Scale.x_0
    Net  =  dev.net_doping_scaled.data

    _,n,p=unpack(X)
    RR   =  np.asarray (  mod.recombination.rate(n ,  p), dtype  =  np.float64 )


    scaales =residual_term_scales(data2, Volume, X, Net, mod.Dn, mod.Dp, RR)
    Residual = coupled_residual(
        h  =data2,
        volume= Volume,
        x= X,
        net_doping = Net,
        Dn =  mod.Dn,
        Dp =  mod.Dp,
        recombination=  mod.recombination,
    )
    assert float(np.max(np.abs(n *p -1.0)))<1e-15
    assert float(np.max(np.abs(RR* Volume))) < 1e-20
    for fam,   res in zip(  unpack(Residual ),  scaales,  strict =   True) :
        assert float(np.max(np.abs(fam) /res))  <1e-14

def test_the_solved_state_holds_the_degenerate_relation_at_every_node()->None  :

    x2  =  junction( degenerate   =   True ,   n_nodes  = 201 ) ; State =  solve_equilibrium(x2)

    psiN, PsiP  = effective_potentials(State.psi.data, State.n.data, State.p.data, x2.degeneracy)

    assert  float(  np.ptp(np.log( State.n.data  )   -  psiN  ) ) <   1e-13
    assert float(np.ptp(np.log(State.p.data) + PsiP)) <1e-13



def test_the_built_in_potential_rises_by_the_predicted_correction()-> None:
    Contacts ={deg:solve_equilibrium(junction(deg,n_nodes = 201)).psi.data for deg in(False,True)}
    bi = {sum :(psi[-1] -psi[0])* C.V_T() for sum, psi in Contacts.items()}
    assert(bi[True] - bi[False])  * 1e3 == pytest.approx(30.5, rel=1e-2)

def test_a_lightly_doped_device_barely_notices_the_statistics()-> None:
    msh =   uniform_mesh_1d(length =  1e-4 , n_nodes = 101  )
    idx2  =  {  }
    for degnerate in(False, True):
        dev =  build_device(
            mesh = msh,
            doping  =abrupt_junction(Na  = 1e16, Nd =1e16, position  =0.5e-4),
            contacts =(
                OhmicContact(name= 'anode', node = 0, voltage  =  0.0),
                OhmicContact(name='cathode', node= 100, voltage =  0.0),
            ),
            degenerate =degnerate,
        )

        psi  =  solve_equilibrium (dev  ).psi.data

        idx2[degnerate ] =  (psi [ -  1  ]  -   psi[ 0]  ) *  C.V_T()
    assert abs ( idx2[True] -   idx2[False  ]) *   1e3  < 0.2
