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

    res2 = uniform_mesh_1d (  length  =  1e-4 ,   n_nodes   = N_NODES  )
    return build_device(
        mesh =res2,
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
    jj ,   g, xs   = request.param
    return TransportModels.for_device(
        device,mobility=jj,auger = g,field_dependent=xs
    )


@pytest.fixture

def geometry(device)  :
    x2=device.scale;  return device.mesh.h /x2.x_0,device.mesh.volume/x2.x_0


@pytest.fixture

def equilibrium_x(device) :
    m= initial_state(device)
    return  pack(m.psi.data ,  m.n.data, m.p.data  )
@pytest.fixture


def perturbed_x(device):
    g=initial_state(device)
    z2  =  np.linspace(0.0 ,   3.0  *  np.pi, device.mesh.n_nodes)
    psi=  g.psi.data  +  0.35  *  np.cos(z2)


    n=g.n.data*np.exp(0.20*np.sin(z2))
    p  = g.p.data *  np.exp(- 0.15   *   np.cos(  2.0 *  z2 ) )

    return  pack( psi, n, p  )
@pytest.fixture(  params   =  ["equilibrium", "perturbed" ])

def state_x(request, equilibrium_x, perturbed_x)  :
    return  equilibrium_x  if request.param   ==   "equilibrium"  else perturbed_x


def residual_at(geometry,device,models) :
    i,   c   =   geometry
    def evaluate(x) :
        return coupled_residual(h   =  i , volume   =  c , x   =  x , net_doping  =   device.net_doping_scaled.data, Dn  =  models.Dn, Dp  = models.Dp, recombination  =   models.recombination,)


    return  evaluate
def dense_jacobian(geometry,x,models):
    f, val2= geometry
    k, u, b = coupled_jacobian(h= f, volume = val2, x  =x, Dn = models.Dn, Dp =  models.Dp, recombination=  models.recombination,)
    out=x.size ; return  coo_matrix(  ( b, ( k,   u  )  ) ,  shape  = (  out ,   out  )  ).toarray()



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
    g,bb,rr= unpack(pack(psi,n,p))
    np.testing.assert_array_equal(g,psi)


    np.testing.assert_array_equal(bb, n)
    np.testing.assert_array_equal(rr, p)



def test_ordering_is_interleaved_by_node() :
    psi  =   np.array ([  1.0,   2.0 ,  3.0 ]) ; n=np.array([4.0,5.0,6.0])
    p=np.array([7.0,8.0,9.0])
    zz=pack(psi,n,p)
    np.testing.assert_array_equal(zz, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])

def test_unpack_returns_views_not_copies (  )   :
    a  =  np.arange(9.0)
    psi,n,p =unpack(a)

    psi[0] = -  1.0
    assert a[0]  == -  1.0



def test_pack_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match= 'same length')  :
        pack(np.zeros(3),np.zeros(4),np.zeros(3))
def test_psi_rows_match_the_poisson_residual(device, geometry, models) :
    a,k2= geometry
    ss= initial_state(device)
    psi =ss.psi.data
    phi_n =ss.phi_n.data
    phi_p= ss.phi_p.data
    n  = np.exp ( psi   -  phi_n)

    p=np.exp(phi_p- psi)



    z  =  coupled_residual(h =   a, volume   =  k2, x  = pack (  psi, n , p  ) , net_doping   =   device.net_doping_scaled.data, Dn =  models.Dn, Dp  =  models.Dp , recombination   =   models.recombination ,)
    rr =poisson_residual(
        a,k2,psi,device.net_doping_scaled.data,phi_n,phi_p
    )

    np.testing.assert_allclose(z [ Unknown.PSI  ::   UNKNOWNS_PER_NODE] ,   rr , rtol  =  1e-13, atol  =  0.0)

def test_electron_rows_match_the_uncoupled_continuity_residual(
    device,geometry,models,perturbed_x
) :

    a,  nxt  =  geometry
    psi, n, p = unpack(perturbed_x)
    b=np.asarray(models.recombination.rate(n, p), dtype = np.float64)
    i= diffusivity_at(models.Dn, edge_drop(psi), a)
    a2  =   residual_at( geometry ,  device,  models)  (  perturbed_x  )

    c =electron_continuity_residual(a,nxt,i,psi,n,b)


    np.testing.assert_allclose(a2 [  Unknown.N  ::  UNKNOWNS_PER_NODE], c,  rtol =  1e-13 ,  atol  =  0.0)


def test_hole_rows_match_the_uncoupled_continuity_residual(
    device, geometry, models, perturbed_x
):


    tt,  j   =  geometry
    psi,n,p =unpack(perturbed_x)
    u=np.asarray(models.recombination.rate(n,p),dtype= np.float64)

    nxt= diffusivity_at(models.Dp,edge_drop(psi),tt)



    rr=residual_at(geometry,device,models)(perturbed_x)

    xs = hole_continuity_residual(tt, j, nxt, psi, p, u)
    np.testing.assert_allclose(rr[Unknown.P ::UNKNOWNS_PER_NODE],xs,rtol= 1e-13,atol =0.0)

def test_residual_preserves_a_complex_dtype(device,geometry,models,perturbed_x):
    d  = residual_at(  geometry, device,   models  )   (
        perturbed_x.astype(np.complex128)
    )


    assert np.iscomplexobj(d)


@pytest.fixture


def flat_bar() :


    y  =uniform_mesh_1d(length  = 1e-4, n_nodes= N_NODES)
    g  =   build_device (
        mesh = y ,
        doping   =   Uniform(  1e16),
        contacts  =  (
            OhmicContact( name  =  'anode',  node   =   0 ,   voltage =  0.0  ),
            OhmicContact(name = "cathode",   node   = N_NODES  -   1,  voltage  =   0.0  ) ,
        ) ,
    )
    v  = TransportModels.for_device(g)
    k  =   g.scale


    z =g.net_doping_scaled.data
    psi =  np.full ( y.n_nodes , np.arcsinh (  z[0 ]  /   2.0) )
    yy =pack(psi, np.exp(psi), np.exp(-  psi))

    return g, v, (y.h  / k.x_0, y.volume  / k.x_0), yy

ALL_BLOCKS =[(tmp,b)for tmp in Unknown for b in Unknown]
@pytest.mark.parametrize(  'row,col', ALL_BLOCKS,   ids   = lambda u  :  u.name  )


def test_every_jacobian_block_matches_complex_step(device, geometry, models, state_x, row, col):
    tmp2 =complex_step_jacobian(
        residual_at(geometry,device,models),state_x
    )
    g   =  dense_jacobian(geometry ,   state_x,  models )

    a=block(g,row,col)
    j  =  block( tmp2,  row, col )
    idx  = np.max(np.abs(j))

    np.testing.assert_allclose(
        a,j,rtol=1e-10,atol= 1e-10 *max(idx,1e-300)
    )




@pytest.mark.parametrize("row,col", ALL_BLOCKS, ids = lambda u : u.name)



def test_every_jacobian_block_matches_complex_step_at_a_flat_potential(flat_bar, row, col)  :

    r, s2, a2,  ret  = flat_bar
    psi,_,_ = unpack(ret)
    assert  np.all( psi[1  :]   - psi [  :- 1]  ==  0.0),  'the fixture is not flat'

    t = complex_step_jacobian(residual_at(a2,r,s2),ret)
    kk =   dense_jacobian(a2 ,  ret ,  s2 )
    tt   =  block(  kk,   row ,   col)
    c = block(  t,   row ,   col)
    res  =  np.max ( np.abs (c)  )


    np.testing.assert_allclose(
        tt,c,rtol=1e-10,atol =1e-10* max(res,1e-300)
    )



@pytest.fixture




def lopsided_bar():

    w  =uniform_mesh_1d(length=1e-4, n_nodes  = N_NODES)
    j=build_device(
        mesh= w,
        doping=abrupt_junction(Na =1e18,Nd=1e15,position=0.5e-4),
        contacts =(
            OhmicContact(name ='anode',node=0,voltage=0.0),
            OhmicContact(name= "cathode",node=N_NODES -1,voltage=0.0),
        ),
    )
    m=TransportModels.for_device(j,mobility ='arora')
    prev   = j.scale

    ss =(w.h/ prev.x_0,w.volume /prev.x_0)

    ok= initial_state(j)
    b=np.linspace(0.0,3.0 * np.pi,w.n_nodes)
    f =pack(ok.psi.data +0.35*np.cos(b), ok.n.data*np.exp(0.3*np.sin(b)), ok.p.data *np.exp(-0.3* np.sin(b)),)
    return j,m,ss,f

@pytest.mark.parametrize('row,col',ALL_BLOCKS,ids=lambda u:u.name)

def test_every_jacobian_block_matches_complex_step_per_edge_diffusivity(lopsided_bar,row,col):

    i,b2,tmp3,t=lopsided_bar
    b  =  np.asarray(b2.Dn)
    assert b.size==i.mesh.n_edges,"Dn is not per edge"
    assert  np.unique(  b  ).size >  1 ,  'Dn does not vary, so alignment is untested'
    assert not np.array_equal(b,b[::-1]),'Dn is reversal symmetric'

    z=complex_step_jacobian(residual_at(tmp3,i,b2),t)
    d2  = dense_jacobian (tmp3 ,  t,  b2)

    xx= block(d2,row,col)


    k  = block(z, row, col); s  =   np.max(np.abs( k)  )



    np.testing.assert_allclose(
        xx, k, rtol = 1e-10, atol  = 1e-10 * max(s, 1e-300)
    )

def test_the_psi_block_carries_no_boltzmann_charge_term (
    device ,  geometry, models,   perturbed_x
) :
    res2,info=geometry
    x2  =dense_jacobian(geometry, perturbed_x, models)
    h =block(x2,Unknown.PSI,Unknown.PSI)

    out =np.zeros((device.mesh.n_nodes, device.mesh.n_nodes))
    d=1.0 /res2

    for  w  in range( res2.size )   :

        out[w,w]+= d[w]
        out[  w  +  1, w  + 1 ]  += d[w  ]
        out[ w ,   w   +  1  ]   -=   d[  w  ]


        out[w  +  1, w] -=d[w]

    np.testing.assert_allclose(h,
          out,
      rtol = 1e-14,
            atol = 0.0)

def test_the_charge_blocks_are_plus_and_minus_the_cell_volume(
    device ,  geometry,   models, perturbed_x
)   :
    _,d = geometry
    j  = dense_jacobian(geometry, perturbed_x, models)

    np.testing.assert_allclose(block(j,Unknown.PSI,Unknown.N),np.diag(d),atol=0.0)
    np.testing.assert_allclose(
        block(j, Unknown.PSI, Unknown.P), np.diag(- d), atol =0.0
    )


def test_the_recombination_cross_blocks_are_diagonal(
    device,geometry,models,perturbed_x
):
    z =dense_jacobian(geometry,perturbed_x,models)
    for s,v in((Unknown.N,Unknown.P),(Unknown.P,Unknown.N)):
        e =block(z,s,v);  assert np.count_nonzero(e -  np.diag(np.diag(e))) == 0




def test_the_jacobian_sparsity_is_block_tridiagonal (  geometry ,   models,   perturbed_x)  :
    c ,   s  =   geometry
    ii,cur,_ = coupled_jacobian(
        h = c,
        volume=s,
        x=perturbed_x,
        Dn=models.Dn,
        Dp= models.Dp,
        recombination = models.recombination,
    )
    dat =ii  //  UNKNOWNS_PER_NODE

    f =cur  // UNKNOWNS_PER_NODE
    assert  np.all (np.abs( dat  -  f )  <=  1)

def test_assemble_coupled_agrees_with_the_array_level_functions(device ,   geometry,  models , perturbed_x)  :
    d =  assemble_state(device, models, perturbed_x)


    a=residual_at(geometry,device,models)(perturbed_x)


    assert d.shape   ==  (UNKNOWNS_PER_NODE  *   device.mesh.n_nodes, UNKNOWNS_PER_NODE  *  device.mesh.n_nodes,)
    np.testing.assert_array_equal(d.residual,a)


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
    obj= pack(psi,n,p)

    for  e in  range( 4)   :

        assert obj[unknown_index(e, Unknown.PSI)]  == psi[e]


        assert obj[unknown_index( e,   Unknown.N )  ]  ==   n [e]
        assert obj[unknown_index(e, Unknown.P)] ==  p[e]


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
    val2  =  assemble_state(device, models, perturbed_x)
    d =apply_contacts_coupled(
        val2,
        perturbed_x,
        device.net_doping_scaled.data,
        device.contacts,
        device.scale,
    )

    x = SparseLU(); x.factorize(d.rows, d.cols, d.values, d.shape)
    a2=x.solve(-d.residual)
    j  =  perturbed_x  +  a2

    xx  =  device.net_doping_scaled.data
    for h in device.contacts:
        a  = h.node

        assert j[unknown_index(a, Unknown.PSI)] == pytest.approx(
            ohmic_psi_scaled(
                float(xx[a]), h.voltage  /  device.scale.psi_0
            ),
            rel  =1e-14,
        )

        assert j[unknown_index(a, Unknown.N)] == pytest.approx(
            ohmic_density_scaled(float(xx[a]), Carrier.ELECTRON), rel =  1e-14
        )
        assert j[unknown_index(a, Unknown.P)] == pytest.approx(ohmic_density_scaled(float(xx[a]), Carrier.HOLE), rel = 1e-14)

def test_the_contact_densities_agree_with_the_contact_potential ( device  )  :
    res =  device.with_bias(anode  = 0.35,   cathode  =   0.0 )
    r2 =res.net_doping_scaled.data
    for  r in res.contacts   :

        buf =  r.voltage  /  res.scale.psi_0


        psi =  ohmic_psi_scaled( float (r2[ r.node]),   buf  )
        n = ohmic_density_scaled(float(r2[r.node]), Carrier.ELECTRON)
        p = ohmic_density_scaled(float(r2[r.node]),Carrier.HOLE)
        assert n  ==  pytest.approx(np.exp( psi   -  buf ) ,   rel  =  1e-12  )
        assert p== pytest.approx(np.exp(buf -  psi), rel  =1e-12)
        assert n   *  p  == pytest.approx(1.0,  rel  =  1e-12  )




def test_the_applied_bias_moves_psi_and_leaves_the_densities_alone(device):
    j  =  device.net_doping_scaled.data[ 0  ]
    g =  ohmic_psi_scaled( float(  j ),  0.0)
    x =  ohmic_psi_scaled (float( j ) ,
           1.0  )


    assert x -  g  ==  pytest.approx(1.0, rel = 1e-14)
    assert ohmic_density_scaled(float(j), Carrier.ELECTRON) == ohmic_density_scaled(float(j), Carrier.ELECTRON)


def test_two_contacts_sharing_a_name_are_rejected(device, models, perturbed_x):
    u= assemble_state(device, models, perturbed_x)

    with  pytest.raises (ValueError, match  =   "unique"  )  :
        apply_contacts_coupled(
            u,
            perturbed_x,
            device.net_doping_scaled.data,
            (
                OhmicContact(name =  "anode", node=  0, voltage =0.0),
                OhmicContact(name =  'anode', node = N_NODES  -1, voltage= 0.0),
            ),
            device.scale,
        )

def test_the_poisson_term_scale_counts_the_carriers_not_only_the_doping():
    b = np.full(4, 0.1)
    x =np.full(5,0.1)
    xs = pack(np.zeros(5), np.ones(5), np.ones(5))

    dd,_,_ =residual_term_scales(b,x,xs,np.zeros(5),Dn = 1.0,Dp = 1.0)

    assert np.all(dd>0.0)


def test_every_term_scale_is_strictly_positive_on_a_real_device(device,   geometry, models,   perturbed_x) :
    m, r =geometry
    a=residual_term_scales(m, r, perturbed_x, device.net_doping_scaled.data, models.Dn, models.Dp)

    assert all(np.all(cur>0.0) for cur in a)

def  test_row_scaling_keeps_the_system_finite(  device,   geometry , models,  perturbed_x  )  :
    u, k  = geometry
    g=residual_term_scales(
        u,k,perturbed_x,device.net_doping_scaled.data,models.Dn,models.Dp
    )
    t = assemble_coupled_arrays(h =u, volume =k, x= perturbed_x, net_doping=device.net_doping_scaled.data, Dn= models.Dn, Dp = models.Dp, recombination=models.recombination,)



    f=scale_rows(t,row_weights(g,device.mesh.n_nodes))

    assert np.all(np.isfinite(f.residual))
    assert np.all(np.isfinite(f.values))
def test_a_state_with_no_carriers_anywhere_is_refused()  :

    b  = np.full(4, 0.1)
    g=  np.full(5, 0.1); u   =  pack (  np.zeros(5 ),   np.zeros( 5  ),  np.zeros(  5))


    with pytest.raises(ValueError, match = "no terms") :

        residual_term_scales(b, g, u, np.zeros(5), Dn  =1.0, Dp= 1.0)




def test_a_state_whose_terms_overflow_is_a_diverged_iterate_not_a_bad_state():

    kk  = np.full(4, 0.1)
    r =  np.full(5, 0.1)
    s= pack(np.zeros(5),np.full(5,np.inf),np.ones(5))


    with  np.errstate ( invalid  =   'ignore' , over   =  'ignore' )  :

        with pytest.raises(FloatingPointError,match="diverged"):
            residual_term_scales(kk, r, s, np.zeros(5), Dn=  1.0, Dp = 1.0)



def test_the_shared_path_reproduces_the_standalone_functions_exactly(device, geometry, models, perturbed_x) :
    cnt,el=geometry
    item =  device.net_doping_scaled.data

    t =assemble_coupled_terms(cnt,el,perturbed_x,item,models.Dn,models.Dp,models.recombination).assembly
    d=coupled_residual(cnt,el,perturbed_x,item,models.Dn,models.Dp,models.recombination)
    x,lst,z =coupled_jacobian(
        cnt,el,perturbed_x,models.Dn,models.Dp,models.recombination
    )

    np.testing.assert_array_equal(t.residual,d); np.testing.assert_array_equal (  t.rows ,   x  )
    np.testing.assert_array_equal(t.cols,lst)
    np.testing.assert_array_equal(t.values, z)


def test_the_shared_path_scales_match_the_standalone_scales(device, geometry, models, perturbed_x)  :
    buf,s2 = geometry

    stuff=device.net_doping_scaled.data

    psi,n,p= unpack(perturbed_x)

    _,yy=assemble_coupled_terms(buf,s2,perturbed_x,stuff,models.Dn,models.Dp,models.recombination)
    d2=residual_term_scales(
        buf,
        s2,
        perturbed_x,
        stuff,
        models.Dn,
        models.Dp,
        R =np.asarray(models.recombination.rate(n,p),dtype = np.float64),
    )
    for jj, s  in zip(  yy ,   d2 , strict  = True )  :
        np.testing.assert_array_equal(jj,s)
def  test_a_term_scale_of_the_wrong_length_is_named_rather_than_broadcast ( )  :
    ys =   5
    stuff  = np.ones(ys)
    e   =  np.ones (ys   -  1)

    with pytest.raises(ValueError,match='N term scale has 4 entries'):
        row_weights((stuff,e,stuff),ys)


def test_a_row_with_no_terms_in_it_is_skipped_rather_than_dividing_by_zero() :
    vv=  3
    x=np.full(vv,2.0)

    z = np.array([1.0,0.0,4.0])
    aa =np.full(vv,8.0)
    f  =  ( x,  z,   aa)

    xx =row_weights(f,vv)
    w2  =  np.zeros(  UNKNOWNS_PER_NODE   *  vv)
    w2[unknown_index(1, Unknown.N)]=  1e30
    w2 [unknown_index(2, Unknown.N) ]   =  2.0

    g  = residual_measure (  w2  /  xx ,   f,   vv )
    assert g== pytest.approx(0.5)




def test_a_row_whose_terms_collapsed_is_skipped_like_one_with_none()  -> None:


    y2   =  3
    v =np.full(y2,2.0)
    s = np.array([1.0, 1e-20, 4.0]) ; c =  np.full(y2, 8.0)
    z  =(v, s, c)



    aa=row_weights(z, y2)
    bb   =  np.zeros ( UNKNOWNS_PER_NODE  *   y2  )
    bb [unknown_index(1 , Unknown.N  ) ] = 1e-18
    bb[unknown_index(2, Unknown.N)] = 2.0
    f  =   residual_measure(bb / aa , z,   y2  )
    assert f==pytest.approx(0.5)

def test_a_row_just_above_the_floor_still_counts()->None:
    cc = 2

    t= float(np.finfo(np.float64).eps)* 4.0
    y =  np.array([t * 10.0, 4.0])
    res  = (np.full(cc,   2.0), y ,   np.full (  cc, 8.0 ) )
    lst=row_weights(res,cc)

    u =np.zeros(UNKNOWNS_PER_NODE  *  cc)
    u[unknown_index(0,Unknown.N)]= t *10.0*0.25

    r=residual_measure(u /lst,res,cc)



    assert  r   == pytest.approx(0.25)

def test_the_update_split_puts_each_family_under_its_own_name() -> None  :
    i=pack(np.zeros(3),np.array([9.0,1.0,3.0]),np.array([0.0,4.0,1.0]))
    t2  = pack(np.array([0.0, - 0.25, 0.1]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -  8.0]),)

    b =coupled_update_by_family(t2,i)
    assert b =={'psi':0.25,'n':0.5,"p": 4.0}
    assert  coupled_update_norm(  t2 ,   i ) ==  4.0
