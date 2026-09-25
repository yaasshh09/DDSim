from  __future__ import  annotations

import  numpy as  np, pytest


from scipy.sparse import coo_matrix


from ddsim.core.field  import  Field,  Location, ScalingState

from ddsim.device.builder import build_device

from ddsim.device.doping import Uniform,abrupt_junction
from ddsim.device.transport import TransportModels ,  initial_state

from ddsim.discretize.boundary import(
    Carrier,
    OhmicContact,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)



from  ddsim.discretize.continuity  import(
    electron_continuity_residual,
    hole_continuity_residual,
)

from ddsim.discretize.coupled  import(
    UNKNOWNS_PER_NODE ,
    Unknown,
    apply_contacts_coupled,
    assemble_coupled,
    assemble_coupled_arrays,
    assemble_coupled_terms,
    coupled_jacobian ,
    coupled_residual,
    coupled_update_by_family,
    coupled_update_norm,
    edge_drop ,
    pack,
    residual_measure,
    residual_term_scales ,
    row_weights,
    scale_rows ,
    unknown_index,
    unpack ,
)
from ddsim.discretize.poisson  import  poisson_residual
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.mobility import diffusivity_at

from ddsim.solve.linear import  SparseLU
from tests.reference.complexstep import complex_step_jacobian
N_NODES  =  20


@pytest.fixture
def device():

    mes = uniform_mesh_1d (  length  =  1e-4 ,   n_nodes   = N_NODES  )
    return build_device(
        mesh =mes,
        doping=abrupt_junction(Na=1e16,Nd = 1e16,position= 0.5e-4),
        contacts= (
            OhmicContact(name='anode',node=0,voltage=0.0),
            OhmicContact(name = "cathode",node =N_NODES-1,voltage=0.0),
        ),
    )

@pytest.fixture(
    params  = [
        ('constant', False, False),
        ("arora", False, False),
        ('constant', True, False),
        ("arora", True, False),
        ("constant", False, True),
        ('arora', True, True),
    ],
    ids  = [
        'constant',
        'arora',
        "constant+auger",
        'arora+auger',
        "constant+field",
        'arora+auger+field',
    ],
)

def models(request,  device)   :
    mob ,   auer, fie   = request.param
    return TransportModels.for_device(
        device,mobility=mob,auger = auer,field_dependent=fie
    )


@pytest.fixture

def geometry(device)  :
    sca=device.scale;  return device.mesh.h /sca.x_0,device.mesh.volume/sca.x_0


@pytest.fixture

def equilibrium_x(device) :
    satte= initial_state(device)
    return  pack(satte.psi.data ,  satte.n.data, satte.p.data  )
@pytest.fixture


def perturbed_x(device):
    State=initial_state(device)
    foo  =  np.linspace(0.0 ,   3.0  *  np.pi, device.mesh.n_nodes)
    psi=  State.psi.data  +  0.35  *  np.cos(foo)


    n=State.n.data*np.exp(0.20*np.sin(foo))
    p  = State.p.data *  np.exp(- 0.15   *   np.cos(  2.0 *  foo ) )

    return  pack( psi, n, p  )
@pytest.fixture(  params   =  ["equilibrium", "perturbed" ])

def state_x(request, equilibrium_x, perturbed_x)  :
    return  equilibrium_x  if request.param   ==   "equilibrium"  else perturbed_x


def residual_at(geometry,device,models) :
    input,   vol   =   geometry
    def evaluate(x) :
        return coupled_residual(h   =  input , volume   =  vol , x   =  x , net_doping  =   device.net_doping_scaled.data, Dn  =  models.Dn, Dp  = models.Dp, recombination  =   models.recombination,)


    return  evaluate
def dense_jacobian(geometry,x,models):
    hh, Volume= geometry
    Rows, bb, Values = coupled_jacobian(h= hh, volume = Volume, x  =x, Dn = models.Dn, Dp =  models.Dp, recombination=  models.recombination,)
    siz=x.size ; return  coo_matrix(  ( Values, ( Rows,   bb  )  ) ,  shape  = (  siz ,   siz  )  ).toarray()



def block(matrix,
         row: Unknown,
    col: Unknown):
    return matrix[row  ::UNKNOWNS_PER_NODE, col ::UNKNOWNS_PER_NODE]


def node_field(values,unit,name):
    return Field(
        values.copy(), unit, ScalingState.SCALED, Location.NODE, name =  name
    )

def assemble_state(device, models, x) :

    psi,n,p= unpack(x)

    return assemble_coupled (
        mesh  =   device.mesh,
        psi  =  node_field( psi,  'V' , 'psi'  ),
        n   =  node_field(  n, "cm^-3", 'n'),
        p   =  node_field ( p, 'cm^-3',   "p" ),
        net_doping  =  device.net_doping_scaled,
        recombination =  models.recombination ,
        scale  =  device.scale ,
        Dn   = models.Dn,
        Dp  =   models.Dp ,
    )


def test_pack_and_unpack_round_trip() :

    psi= np.array([1.0, 2.0, 3.0])
    n =np.array([4.0, 5.0, 6.0])
    p=np.array([7.0,8.0,9.0])
    gotPsi,GotN,dat= unpack(pack(psi,n,p))
    np.testing.assert_array_equal(gotPsi,psi)


    np.testing.assert_array_equal(GotN, n)
    np.testing.assert_array_equal(dat, p)



def test_ordering_is_interleaved_by_node() :
    psi  =   np.array ([  1.0,   2.0 ,  3.0 ]) ; n=np.array([4.0,5.0,6.0])
    p=np.array([7.0,8.0,9.0])
    bb=pack(psi,n,p)
    np.testing.assert_array_equal(bb, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])

def test_unpack_returns_views_not_copies (  )   :
    xx  =  np.arange(9.0)
    psi,n,p =unpack(xx)

    psi[0] = -  1.0
    assert xx[0]  == -  1.0



def test_pack_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match= 'same length')  :
        pack(np.zeros(3),np.zeros(4),np.zeros(3))
def test_psi_rows_match_the_poisson_residual(device, geometry, models) :
    hh,idx2= geometry
    satte= initial_state(device)
    psi =satte.psi.data
    phi_n =satte.phi_n.data
    phi_p= satte.phi_p.data
    n  = np.exp ( psi   -  phi_n)

    p=np.exp(phi_p- psi)



    s2  =  coupled_residual(h =   hh, volume   =  idx2, x  = pack (  psi, n , p  ) , net_doping   =   device.net_doping_scaled.data, Dn =  models.Dn, Dp  =  models.Dp , recombination   =   models.recombination ,)
    expeected =poisson_residual(
        hh,idx2,psi,device.net_doping_scaled.data,phi_n,phi_p
    )

    np.testing.assert_allclose(s2 [ Unknown.PSI  ::   UNKNOWNS_PER_NODE] ,   expeected , rtol  =  1e-13, atol  =  0.0)

def test_electron_rows_match_the_uncoupled_continuity_residual(
    device,geometry,models,perturbed_x
) :

    hh,  Volume  =  geometry
    psi, n, p = unpack(perturbed_x)
    RR=np.asarray(models.recombination.rate(n, p), dtype = np.float64)
    d2= diffusivity_at(models.Dn, edge_drop(psi), hh)
    gott  =   residual_at( geometry ,  device,  models)  (  perturbed_x  )

    str =electron_continuity_residual(hh,Volume,d2,psi,n,RR)


    np.testing.assert_allclose(gott [  Unknown.N  ::  UNKNOWNS_PER_NODE], str,  rtol =  1e-13 ,  atol  =  0.0)


def test_hole_rows_match_the_uncoupled_continuity_residual(
    device, geometry, models, perturbed_x
):


    x2,  vol   =  geometry
    psi,n,p =unpack(perturbed_x)
    RR=np.asarray(models.recombination.rate(n,p),dtype= np.float64)

    Dpp= diffusivity_at(models.Dp,edge_drop(psi),x2)



    slice=residual_at(geometry,device,models)(perturbed_x)

    Expected = hole_continuity_residual(x2, vol, Dpp, psi, p, RR)
    np.testing.assert_allclose(slice[Unknown.P ::UNKNOWNS_PER_NODE],Expected,rtol= 1e-13,atol =0.0)

def test_residual_preserves_a_complex_dtype(device,geometry,models,perturbed_x):
    gott  = residual_at(  geometry, device,   models  )   (
        perturbed_x.astype(np.complex128)
    )


    assert np.iscomplexobj(gott)


@pytest.fixture


def flat_bar() :


    Mesh  =uniform_mesh_1d(length  = 1e-4, n_nodes= N_NODES)
    stuff2  =   build_device (
        mesh = Mesh ,
        doping   =   Uniform(  1e16),
        contacts  =  (
            OhmicContact( name  =  'anode',  node   =   0 ,   voltage =  0.0  ),
            OhmicContact(name = "cathode",   node   = N_NODES  -   1,  voltage  =   0.0  ) ,
        ) ,
    )
    mod  = TransportModels.for_device(stuff2)
    Scale  =   stuff2.scale


    xx =stuff2.net_doping_scaled.data
    psi =  np.full ( Mesh.n_nodes , np.arcsinh (  xx[0 ]  /   2.0) )
    acc2 =pack(psi, np.exp(psi), np.exp(-  psi))

    return stuff2, mod, (Mesh.h  / Scale.x_0, Mesh.volume  / Scale.x_0), acc2

ALL_BLOCKS =[(row,col)for row in Unknown for col in Unknown]
@pytest.mark.parametrize(  'row,col', ALL_BLOCKS,   ids   = lambda u  :  u.name  )


def test_every_jacobian_block_matches_complex_step(device, geometry, models, state_x, row, col):
    Reference =complex_step_jacobian(
        residual_at(geometry,device,models),state_x
    )
    ass   =  dense_jacobian(geometry ,   state_x,  models )

    gott=block(ass,row,col)
    exp  =  block( Reference,  row, col )
    vars  = np.max(np.abs(exp))

    np.testing.assert_allclose(
        gott,exp,rtol=1e-10,atol= 1e-10 *max(vars,1e-300)
    )




@pytest.mark.parametrize("row,col", ALL_BLOCKS, ids = lambda u : u.name)



def test_every_jacobian_block_matches_complex_step_at_a_flat_potential(flat_bar, row, col)  :

    Device, mod, Geometry,  xx  = flat_bar
    psi,_,_ = unpack(xx)
    assert  np.all( psi[1  :]   - psi [  :- 1]  ==  0.0),  'the fixture is not flat'

    sum = complex_step_jacobian(residual_at(Geometry,Device,mod),xx)
    ass =   dense_jacobian(Geometry ,  xx ,  mod )
    tuple   =  block(  ass,   row ,   col)
    exp = block(  sum,   row ,   col)
    Floor  =  np.max ( np.abs (exp)  )


    np.testing.assert_allclose(
        tuple,exp,rtol=1e-10,atol =1e-10* max(Floor,1e-300)
    )



@pytest.fixture




def lopsided_bar():

    meesh  =uniform_mesh_1d(length=1e-4, n_nodes  = N_NODES)
    Device=build_device(
        mesh= meesh,
        doping=abrupt_junction(Na =1e18,Nd=1e15,position=0.5e-4),
        contacts =(
            OhmicContact(name ='anode',node=0,voltage=0.0),
            OhmicContact(name= "cathode",node=N_NODES -1,voltage=0.0),
        ),
    )
    dat=TransportModels.for_device(Device,mobility ='arora')
    Scale   = Device.scale

    Geometry =(meesh.h/ Scale.x_0,meesh.volume /Scale.x_0)

    sttae= initial_state(Device)
    kk=np.linspace(0.0,3.0 * np.pi,meesh.n_nodes)
    x2 =pack(sttae.psi.data +0.35*np.cos(kk), sttae.n.data*np.exp(0.3*np.sin(kk)), sttae.p.data *np.exp(-0.3* np.sin(kk)),)
    return Device,dat,Geometry,x2

@pytest.mark.parametrize('row,col',ALL_BLOCKS,ids=lambda u:u.name)

def test_every_jacobian_block_matches_complex_step_per_edge_diffusivity(lopsided_bar,row,col):

    tmp,ord,Geometry,q=lopsided_bar
    filter  =  np.asarray(ord.Dn)
    assert filter.size==tmp.mesh.n_edges,"Dn is not per edge"
    assert  np.unique(  filter  ).size >  1 ,  'Dn does not vary, so alignment is untested'
    assert not np.array_equal(filter,filter[::-1]),'Dn is reversal symmetric'

    open=complex_step_jacobian(residual_at(Geometry,tmp,ord),q)
    ass  = dense_jacobian (Geometry ,  q,  ord)

    gott= block(ass,row,col)


    aa  = block(open, row, col); foor  =   np.max(np.abs( aa)  )



    np.testing.assert_allclose(
        gott, aa, rtol = 1e-10, atol  = 1e-10 * max(foor, 1e-300)
    )

def test_the_psi_block_carries_no_boltzmann_charge_term (
    device ,  geometry, models,   perturbed_x
) :
    H,voulme=geometry
    Assembled  =dense_jacobian(geometry, perturbed_x, models)
    id =block(Assembled,Unknown.PSI,Unknown.PSI)

    exected =np.zeros((device.mesh.n_nodes, device.mesh.n_nodes))
    hash=1.0 /H

    for  E  in range( H.size )   :

        exected[E,E]+= hash[E]
        exected[  E  +  1, E  + 1 ]  += hash[E  ]
        exected[ E ,   E   +  1  ]   -=   hash[  E  ]


        exected[E  +  1, E] -=hash[E]

    np.testing.assert_allclose(id,
          exected,
      rtol = 1e-14,
            atol = 0.0)

def test_the_charge_blocks_are_plus_and_minus_the_cell_volume(
    device ,  geometry,   models, perturbed_x
)   :
    _,voolume = geometry
    yy  = dense_jacobian(geometry, perturbed_x, models)

    np.testing.assert_allclose(block(yy,Unknown.PSI,Unknown.N),np.diag(voolume),atol=0.0)
    np.testing.assert_allclose(
        block(yy, Unknown.PSI, Unknown.P), np.diag(- voolume), atol =0.0
    )


def test_the_recombination_cross_blocks_are_diagonal(
    device,geometry,models,perturbed_x
):
    Assembled =dense_jacobian(geometry,perturbed_x,models)
    for Row,q in((Unknown.N,Unknown.P),(Unknown.P,Unknown.N)):
        idx2 =block(Assembled,Row,q);  assert np.count_nonzero(idx2 -  np.diag(np.diag(idx2))) == 0




def test_the_jacobian_sparsity_is_block_tridiagonal (  geometry ,   models,   perturbed_x)  :
    vals ,   voolume  =   geometry
    rws,Cols,_ = coupled_jacobian(
        h = vals,
        volume=voolume,
        x=perturbed_x,
        Dn=models.Dn,
        Dp= models.Dp,
        recombination = models.recombination,
    )
    oct =rws  //  UNKNOWNS_PER_NODE

    colNode =Cols  // UNKNOWNS_PER_NODE
    assert  np.all (np.abs( oct  -  colNode )  <=  1)

def test_assemble_coupled_agrees_with_the_array_level_functions(device ,   geometry,  models , perturbed_x)  :
    ass =  assemble_state(device, models, perturbed_x)


    Expected=residual_at(geometry,device,models)(perturbed_x)


    assert ass.shape   ==  (UNKNOWNS_PER_NODE  *   device.mesh.n_nodes, UNKNOWNS_PER_NODE  *  device.mesh.n_nodes,)
    np.testing.assert_array_equal(ass.residual,Expected)


def test_assemble_coupled_rejects_a_physical_field(  device , models, perturbed_x ) :

    psi,n,p=unpack(perturbed_x)

    with pytest.raises(ValueError,
                  match ="SCALED"):
        assemble_coupled(
            mesh   =  device.mesh,
            psi  =   Field(
                psi.copy(),   'V', ScalingState.PHYSICAL,  Location.NODE, name  =  'psi'
            ) ,
            n =  Field(  n.copy (),  'cm^-3', ScalingState.SCALED , Location.NODE, name  =  'n' ),
            p  =   Field (p.copy(  ), 'cm^-3', ScalingState.SCALED , Location.NODE,  name  =  'p') ,
            net_doping   = device.net_doping_scaled,
            recombination   =  models.recombination,
            scale   =  device.scale ,
            Dn   =  models.Dn,
            Dp   =   models.Dp ,
        )

def test_unknown_index_agrees_with_the_packed_layout(device):
    psi  =  np.arange( 4.0  )
    n= np.arange(4.0)+10.0
    p  = np.arange(4.0)  + 20.0
    x2= pack(psi,n,p)

    for  ndoe in  range( 4)   :

        assert x2[unknown_index(ndoe, Unknown.PSI)]  == psi[ndoe]


        assert x2[unknown_index( ndoe,   Unknown.N )  ]  ==   n [ndoe]
        assert x2[unknown_index(ndoe, Unknown.P)] ==  p[ndoe]


def test_assemble_coupled_rejects_an_edge_field(device,models,perturbed_x):

    psi,n,p=unpack(perturbed_x)


    with pytest.raises(ValueError,
                match= "NODE"):
        assemble_coupled(mesh  =device.mesh, psi  = Field(psi.copy(), "V", ScalingState.SCALED, Location.EDGE, name= "psi"), n =Field(n.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name = "n"), p=  Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name ='p'), net_doping = device.net_doping_scaled, recombination= models.recombination, scale =device.scale, Dn =models.Dn, Dp = models.Dp,)


def test_assemble_coupled_rejects_a_field_of_the_wrong_length(
    device,models,perturbed_x
) :
    psi,n,p =unpack(perturbed_x)
    with pytest.raises(  ValueError ,  match   =  'length'  )  :
        assemble_coupled(
            mesh =device.mesh,
            psi =  Field(
                psi[:- 1].copy(), "V", ScalingState.SCALED, Location.NODE, name= "psi"
            ),
            n  = Field(n.copy(), 'cm^-3', ScalingState.SCALED, Location.NODE, name =  "n"),
            p =  Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name='p'),
            net_doping=device.net_doping_scaled,
            recombination = models.recombination,
            scale  =device.scale,
            Dn= models.Dn,
            Dp  = models.Dp,
        )


def  test_contacts_pin_all_three_unknowns_at_the_contact_node(
    device,   models, perturbed_x
)   :
    zz  =  assemble_state(device, models, perturbed_x)
    piinned =apply_contacts_coupled(
        zz,
        perturbed_x,
        device.net_doping_scaled.data,
        device.contacts,
        device.scale,
    )

    sol = SparseLU(); sol.factorize(piinned.rows, piinned.cols, piinned.values, piinned.shape)
    Delta=sol.solve(-piinned.residual)
    input  =  perturbed_x  +  Delta

    doing  =  device.net_doping_scaled.data
    for contct in device.contacts:
        nod  = contct.node

        assert input[unknown_index(nod, Unknown.PSI)] == pytest.approx(
            ohmic_psi_scaled(
                float(doing[nod]), contct.voltage  /  device.scale.psi_0
            ),
            rel  =1e-14,
        )

        assert input[unknown_index(nod, Unknown.N)] == pytest.approx(
            ohmic_density_scaled(float(doing[nod]), Carrier.ELECTRON), rel =  1e-14
        )
        assert input[unknown_index(nod, Unknown.P)] == pytest.approx(ohmic_density_scaled(float(doing[nod]), Carrier.HOLE), rel = 1e-14)

def test_the_contact_densities_agree_with_the_contact_potential ( device  )  :
    ret =  device.with_bias(anode  = 0.35,   cathode  =   0.0 )
    arr =ret.net_doping_scaled.data
    for  conttact in ret.contacts   :

        k2 =  conttact.voltage  /  ret.scale.psi_0


        psi =  ohmic_psi_scaled( float (arr[ conttact.node]),   k2  )
        n = ohmic_density_scaled(float(arr[conttact.node]), Carrier.ELECTRON)
        p = ohmic_density_scaled(float(arr[conttact.node]),Carrier.HOLE)
        assert n  ==  pytest.approx(np.exp( psi   -  k2 ) ,   rel  =  1e-12  )
        assert p== pytest.approx(np.exp(k2 -  psi), rel  =1e-12)
        assert n   *  p  == pytest.approx(1.0,  rel  =  1e-12  )




def test_the_applied_bias_moves_psi_and_leaves_the_densities_alone(device):
    oct  =  device.net_doping_scaled.data[ 0  ]
    unbased =  ohmic_psi_scaled( float(  oct ),  0.0)
    bia =  ohmic_psi_scaled (float( oct ) ,
           1.0  )


    assert bia -  unbased  ==  pytest.approx(1.0, rel = 1e-14)
    assert ohmic_density_scaled(float(oct), Carrier.ELECTRON) == ohmic_density_scaled(float(oct), Carrier.ELECTRON)


def test_two_contacts_sharing_a_name_are_rejected(device, models, perturbed_x):
    ass= assemble_state(device, models, perturbed_x)

    with  pytest.raises (ValueError, match  =   "unique"  )  :
        apply_contacts_coupled(
            ass,
            perturbed_x,
            device.net_doping_scaled.data,
            (
                OhmicContact(name =  "anode", node=  0, voltage =0.0),
                OhmicContact(name =  'anode', node = N_NODES  -1, voltage= 0.0),
            ),
            device.scale,
        )

def test_the_poisson_term_scale_counts_the_carriers_not_only_the_doping():
    H = np.full(4, 0.1)
    Volume =np.full(5,0.1)
    X = pack(np.zeros(5), np.ones(5), np.ones(5))

    temp2,_,_ =residual_term_scales(H,Volume,X,np.zeros(5),Dn = 1.0,Dp = 1.0)

    assert np.all(temp2>0.0)


def test_every_term_scale_is_strictly_positive_on_a_real_device(device,   geometry, models,   perturbed_x) :
    H, Volume =geometry
    sccales=residual_term_scales(H, Volume, perturbed_x, device.net_doping_scaled.data, models.Dn, models.Dp)

    assert all(np.all(scale>0.0) for scale in sccales)

def  test_row_scaling_keeps_the_system_finite(  device,   geometry , models,  perturbed_x  )  :
    hh, out2  = geometry
    thing=residual_term_scales(
        hh,out2,perturbed_x,device.net_doping_scaled.data,models.Dn,models.Dp
    )
    System = assemble_coupled_arrays(h =hh, volume =out2, x= perturbed_x, net_doping=device.net_doping_scaled.data, Dn= models.Dn, Dp = models.Dp, recombination=models.recombination,)



    sccaled=scale_rows(System,row_weights(thing,device.mesh.n_nodes))

    assert np.all(np.isfinite(sccaled.residual))
    assert np.all(np.isfinite(sccaled.values))
def test_a_state_with_no_carriers_anywhere_is_refused()  :

    hh  = np.full(4, 0.1)
    vol=  np.full(5, 0.1); X   =  pack (  np.zeros(5 ),   np.zeros( 5  ),  np.zeros(  5))


    with pytest.raises(ValueError, match = "no terms") :

        residual_term_scales(hh, vol, X, np.zeros(5), Dn  =1.0, Dp= 1.0)




def test_a_state_whose_terms_overflow_is_a_diverged_iterate_not_a_bad_state():

    hh  = np.full(4, 0.1)
    Volume =  np.full(5, 0.1)
    xx= pack(np.zeros(5),np.full(5,np.inf),np.ones(5))


    with  np.errstate ( invalid  =   'ignore' , over   =  'ignore' )  :

        with pytest.raises(FloatingPointError,match="diverged"):
            residual_term_scales(hh, Volume, xx, np.zeros(5), Dn=  1.0, Dp = 1.0)



def test_the_shared_path_reproduces_the_standalone_functions_exactly(device, geometry, models, perturbed_x) :
    H,type=geometry
    dop =  device.net_doping_scaled.data

    shhared =assemble_coupled_terms(H,type,perturbed_x,dop,models.Dn,models.Dp,models.recombination).assembly
    acc=coupled_residual(H,type,perturbed_x,dop,models.Dn,models.Dp,models.recombination)
    Rows,Cols,Values =coupled_jacobian(
        H,type,perturbed_x,models.Dn,models.Dp,models.recombination
    )

    np.testing.assert_array_equal(shhared.residual,acc); np.testing.assert_array_equal (  shhared.rows ,   Rows  )
    np.testing.assert_array_equal(shhared.cols,Cols)
    np.testing.assert_array_equal(shhared.values, Values)


def test_the_shared_path_scales_match_the_standalone_scales(device, geometry, models, perturbed_x)  :
    zz,Volume = geometry

    Doping=device.net_doping_scaled.data

    psi,n,p= unpack(perturbed_x)

    _,sha=assemble_coupled_terms(zz,Volume,perturbed_x,Doping,models.Dn,models.Dp,models.recombination)
    standalnoe=residual_term_scales(
        zz,
        Volume,
        perturbed_x,
        Doping,
        models.Dn,
        models.Dp,
        R =np.asarray(models.recombination.rate(n,p),dtype = np.float64),
    )
    for fs, hash  in zip(  sha ,   standalnoe , strict  = True )  :
        np.testing.assert_array_equal(fs,hash)
def  test_a_term_scale_of_the_wrong_length_is_named_rather_than_broadcast ( )  :
    NNodes =   5
    goood  = np.ones(NNodes)
    r2   =  np.ones (NNodes   -  1)

    with pytest.raises(ValueError,match='N term scale has 4 entries'):
        row_weights((goood,r2,goood),NNodes)


def test_a_row_with_no_terms_in_it_is_skipped_rather_than_dividing_by_zero() :
    nnodes=  3
    stuff2=np.full(nnodes,2.0)

    nScale = np.array([1.0,0.0,4.0])
    val =np.full(nnodes,8.0)
    sca  =  ( stuff2,  nScale,   val)

    q =row_weights(sca,nnodes)
    raww  =  np.zeros(  UNKNOWNS_PER_NODE   *  nnodes)
    raww[unknown_index(1, Unknown.N)]=  1e30
    raww [unknown_index(2, Unknown.N) ]   =  2.0

    Measured  = residual_measure (  raww  /  q ,   sca,   nnodes )
    assert Measured== pytest.approx(0.5)




def test_a_row_whose_terms_collapsed_is_skipped_like_one_with_none()  -> None:


    NNodes   =  3
    myvar =np.full(NNodes,2.0)
    nscale = np.array([1.0, 1e-20, 4.0]) ; PScale =  np.full(NNodes, 8.0)
    scalles  =(myvar, nscale, PScale)



    wei=row_weights(scalles, NNodes)
    Raw   =  np.zeros ( UNKNOWNS_PER_NODE  *   NNodes  )
    Raw [unknown_index(1 , Unknown.N  ) ] = 1e-18
    Raw[unknown_index(2, Unknown.N)] = 2.0
    mea  =   residual_measure(Raw / wei , scalles,   NNodes  )
    assert mea==pytest.approx(0.5)

def test_a_row_just_above_the_floor_still_counts()->None:
    NNodes = 2

    oct= float(np.finfo(np.float64).eps)* 4.0
    ns =  np.array([oct * 10.0, 4.0])
    Scales  = (np.full(NNodes,   2.0), ns ,   np.full (  NNodes, 8.0 ) )
    Weights=row_weights(Scales,NNodes)

    out2 =np.zeros(UNKNOWNS_PER_NODE  *  NNodes)
    out2[unknown_index(0,Unknown.N)]= oct *10.0*0.25

    bb=residual_measure(out2 /Weights,Scales,NNodes)



    assert  bb   == pytest.approx(0.25)

def test_the_update_split_puts_each_family_under_its_own_name() -> None  :
    X=pack(np.zeros(3),np.array([9.0,1.0,3.0]),np.array([0.0,4.0,1.0]))
    dellta  = pack(np.array([0.0, - 0.25, 0.1]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -  8.0]),)

    slit =coupled_update_by_family(dellta,X)
    assert slit =={'psi':0.25,'n':0.5,"p": 4.0}
    assert  coupled_update_norm(  dellta ,   X ) ==  4.0
